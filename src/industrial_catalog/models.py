"""Canonical, evidence-first data models for industrial catalog extraction.

The extraction pipeline is intentionally lossless at the field boundary: every
parsed field keeps the source text in ``raw`` and stores any cleaned or typed
representation separately in ``normalized``.  Evidence is attached to the
individual field rather than only to the product so that downstream RAG answers
can cite an exact page and bounding box.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list[Any] | dict[str, Any]


def _utc_now() -> datetime:
    return datetime.now(UTC)


class CatalogModel(BaseModel):
    """Base configuration shared by persisted catalog models."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=False,
        populate_by_name=True,
    )


class BoundingBox(CatalogModel):
    """A rectangular page region.

    ``pixel`` and ``point`` coordinates use the rendered page's coordinate
    system. ``normalized`` coordinates are constrained to the inclusive 0..1
    range. A four-item list/tuple is accepted for easy handoff from VLM/OCR
    output and is interpreted as ``[x0, y0, x1, y1]``.
    """

    x0: float
    y0: float
    x1: float
    y1: float
    coordinate_space: Literal["pixel", "point", "normalized"] = "pixel"
    page_width: float | None = Field(default=None, gt=0)
    page_height: float | None = Field(default=None, gt=0)

    @model_validator(mode="before")
    @classmethod
    def accept_common_bbox_shapes(cls, value: Any) -> Any:
        if isinstance(value, (list, tuple)) and len(value) == 4:
            return {"x0": value[0], "y0": value[1], "x1": value[2], "y1": value[3]}
        if isinstance(value, Mapping):
            value = dict(value)
            aliases = {
                "left": "x0",
                "top": "y0",
                "right": "x1",
                "bottom": "y1",
            }
            for alias, canonical in aliases.items():
                if canonical not in value and alias in value:
                    value[canonical] = value.pop(alias)
            return value
        return value

    @model_validator(mode="after")
    def validate_geometry(self) -> BoundingBox:
        if self.x1 < self.x0:
            raise ValueError("bbox x1 must be greater than or equal to x0")
        if self.y1 < self.y0:
            raise ValueError("bbox y1 must be greater than or equal to y0")
        if self.coordinate_space == "normalized":
            if any(value < 0 or value > 1 for value in (self.x0, self.y0, self.x1, self.y1)):
                raise ValueError("normalized bbox coordinates must be between 0 and 1")
        return self

    def as_list(self) -> list[float]:
        return [self.x0, self.y0, self.x1, self.y1]


class Evidence(CatalogModel):
    """Evidence supporting one extracted field."""

    source_document_id: str = Field(
        min_length=1,
        validation_alias=AliasChoices("source_document_id", "document_id"),
    )
    source_path: str | None = None
    page_number: int = Field(ge=1)
    element_id: str | None = None
    bbox: BoundingBox | None = None
    text: str = Field(
        min_length=1,
        validation_alias=AliasChoices("text", "excerpt"),
    )
    extraction_method: str | None = None
    model: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("source_document_id")
    @classmethod
    def document_id_cannot_be_whitespace(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("source_document_id cannot be blank")
        return value


class ExtractedValue[T](CatalogModel):
    """A value with lossless source text, normalized value, and provenance."""

    raw: str | None = None
    normalized: T | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    evidence: list[Evidence] = Field(default_factory=list)
    normalization_method: str | None = None

    @model_validator(mode="after")
    def require_a_value(self) -> ExtractedValue[T]:
        if self.raw is None and self.normalized is None:
            raise ValueError("an extracted value requires raw or normalized content")
        return self

    @property
    def effective_confidence(self) -> float | None:
        """Return explicit confidence, or the strongest evidence confidence."""

        if self.confidence is not None:
            return self.confidence
        evidence_confidences = [
            item.confidence for item in self.evidence if item.confidence is not None
        ]
        return max(evidence_confidences, default=None)


type TextValue = ExtractedValue[str]
type FlexibleValue = ExtractedValue[JsonValue]


class Specification(CatalogModel):
    """One named technical specification associated with a product."""

    specification_id: str = Field(default_factory=lambda: str(uuid4()), min_length=1)
    name: TextValue
    value: FlexibleValue
    unit: TextValue | None = None
    category: TextValue | None = None
    qualifier: TextValue | None = None


class ExtractionMetadata(CatalogModel):
    """Reproducibility metadata for the LLM parsing stage."""

    run_id: str | None = None
    parser_model: str | None = None
    parser_model_version: str | None = None
    prompt_version: str | None = None
    pipeline_version: str | None = None
    extracted_at: datetime = Field(default_factory=_utc_now)
    record_confidence: float | None = Field(default=None, ge=0, le=1)

    @field_validator("extracted_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("extracted_at must include a timezone")
        return value


class SourceDocument(CatalogModel):
    """Manifest entry for a source catalog."""

    source_document_id: str = Field(
        min_length=1,
        validation_alias=AliasChoices("source_document_id", "document_id"),
    )
    source_path: str
    filename: str | None = None
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-fA-F]{64}$")
    page_count: int | None = Field(default=None, ge=1)
    manufacturer_hint: str | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def derive_filename(self) -> SourceDocument:
        if self.filename is None:
            self.filename = Path(self.source_path).name
        return self


ReviewStatus = Literal["unreviewed", "needs_review", "approved", "rejected"]

# Bounded wire collections prevent a dense page from expanding without limit in
# one structured-output request. Pages that need more rows should be segmented
# by the runner instead of silently increasing the LLM completion budget.
MAX_PRODUCTS_PER_BATCH = 32
MAX_SPECIFICATIONS_PER_PRODUCT = 48
MAX_OTHER_DETAILS_PER_PRODUCT = 24
MAX_EVIDENCE_REFERENCES_PER_FIELD = 8


class ProductRecord(CatalogModel):
    """Canonical representation of one product parsed from a catalog."""

    schema_version: str = "1.0"
    record_id: str = Field(default_factory=lambda: str(uuid4()), min_length=1)
    source_document_id: str = Field(
        min_length=1,
        validation_alias=AliasChoices("source_document_id", "document_id"),
    )
    source_path: str | None = None

    manufacturer: TextValue | None = None
    part_name: TextValue | None = None
    part_number: TextValue | None = None
    category: TextValue | None = None
    description: TextValue | None = None
    specifications: list[Specification] = Field(default_factory=list)

    materials: list[TextValue] = Field(default_factory=list)
    certifications: list[TextValue] = Field(default_factory=list)
    operating_conditions: list[Specification] = Field(default_factory=list)
    compatible_parts: list[TextValue] = Field(default_factory=list)
    aliases_and_cross_references: list[TextValue] = Field(default_factory=list)
    other_details: dict[str, FlexibleValue] = Field(default_factory=dict)

    extraction: ExtractionMetadata = Field(default_factory=ExtractionMetadata)
    review_status: ReviewStatus = "unreviewed"
    validation_warnings: list[str] = Field(default_factory=list)

    @field_validator("source_document_id")
    @classmethod
    def product_document_id_cannot_be_whitespace(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("source_document_id cannot be blank")
        return value

    def iter_field_evidence(self) -> Iterator[tuple[str, Evidence]]:
        """Yield ``(field_path, evidence)`` pairs for persistence/citation."""

        simple_fields = ("manufacturer", "part_name", "part_number", "category", "description")
        for field_name in simple_fields:
            value = getattr(self, field_name)
            if value is not None:
                for evidence in value.evidence:
                    yield field_name, evidence

        for collection_name in (
            "materials",
            "certifications",
            "compatible_parts",
            "aliases_and_cross_references",
        ):
            for index, value in enumerate(getattr(self, collection_name)):
                for evidence in value.evidence:
                    yield f"{collection_name}.{index}", evidence

        for collection_name in ("specifications", "operating_conditions"):
            for index, specification in enumerate(getattr(self, collection_name)):
                for component in ("name", "value", "unit", "category", "qualifier"):
                    value = getattr(specification, component)
                    if value is not None:
                        for evidence in value.evidence:
                            yield f"{collection_name}.{index}.{component}", evidence

        for key, value in self.other_details.items():
            for evidence in value.evidence:
                yield f"other_details.{key}", evidence

    def all_evidence(self) -> list[Evidence]:
        """Return de-duplicated evidence in stable field traversal order."""

        output: list[Evidence] = []
        seen: set[str] = set()
        for _, item in self.iter_field_evidence():
            identity = item.model_dump_json(exclude_none=True)
            if identity not in seen:
                output.append(item)
                seen.add(identity)
        return output


class ProductBatch(CatalogModel):
    """Versioned envelope for products parsed from one source document.

    The batch is the canonical interchange unit between the NVIDIA structured
    parser and persistence layer.  A model validator prevents accidental
    cross-document evidence/product mixing.
    """

    schema_version: str = "1.0"
    batch_id: str = Field(default_factory=lambda: str(uuid4()), min_length=1)
    source_document_id: str = Field(
        min_length=1,
        validation_alias=AliasChoices("source_document_id", "document_id"),
    )
    source_path: str | None = None
    pipeline_version: str | None = None
    parser_schema_sha256: str | None = None
    created_at: datetime = Field(default_factory=_utc_now)
    products: list[ProductRecord] = Field(default_factory=list)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("created_at")
    @classmethod
    def batch_timestamp_requires_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must include a timezone")
        return value

    @model_validator(mode="after")
    def products_must_match_source(self) -> ProductBatch:
        mismatches = [
            product.record_id
            for product in self.products
            if product.source_document_id != self.source_document_id
        ]
        if mismatches:
            raise ValueError(
                "products reference a different source document: " + ", ".join(mismatches)
            )
        return self


def product_batch_guided_json_schema() -> dict[str, Any]:
    """Return a compact vLLM/OpenAI ``response_format`` schema.

    The public Nano model is asked only for exact raw facts and compact source
    element IDs. Normalized values, confidence, document IDs, record IDs,
    extraction metadata, and review state are deliberately added by trusted
    application code. Optional facts are omitted rather than emitted as verbose
    ``null`` placeholders.

    ``other_details`` is an array instead of a dynamic-key JSON object because
    fixed object grammars are substantially more reliable across vLLM structured
    output backends. The evidence hydration step converts it to the canonical
    mapping used by :class:`ProductRecord`. Every emitted product must have a
    source-backed manufacturer and part name; when either identity is absent,
    the model must leave ``products`` empty instead of creating text-block
    candidates. Part number remains optional because some catalogs identify a
    family without listing an orderable code on the current page.
    """

    evidence_reference: dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "properties": {"element_id": {"type": "string", "minLength": 4, "maxLength": 128}},
        "required": ["element_id"],
    }
    value = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "raw": {"type": "string", "minLength": 1, "maxLength": 2048},
            "evidence": {
                "type": "array",
                "items": evidence_reference,
                "minItems": 1,
                "maxItems": MAX_EVIDENCE_REFERENCES_PER_FIELD,
            },
        },
        "required": ["raw", "evidence"],
    }
    specification = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "name": value,
            "value": value,
            "unit": value,
        },
        "required": ["name", "value"],
    }
    other_detail = {
        "type": "object",
        "additionalProperties": False,
        "properties": {"name": value, "value": value},
        "required": ["name", "value"],
    }
    product = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "manufacturer": value,
            "part_name": value,
            "part_number": value,
            "category": value,
            "description": value,
            "specifications": {
                "type": "array",
                "items": specification,
                "maxItems": MAX_SPECIFICATIONS_PER_PRODUCT,
            },
            "other_details": {
                "type": "array",
                "items": other_detail,
                "maxItems": MAX_OTHER_DETAILS_PER_PRODUCT,
            },
        },
        "required": [
            "manufacturer",
            "part_name",
            "specifications",
            "other_details",
        ],
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "IndustrialCatalogProductBatch",
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "products": {
                "type": "array",
                "items": product,
                "maxItems": MAX_PRODUCTS_PER_BATCH,
            }
        },
        "required": ["products"],
    }


# Friendly aliases used by ingestion and API layers.
CatalogProduct = ProductRecord
FieldEvidence = Evidence
FieldValue = ExtractedValue


__all__ = [
    "BoundingBox",
    "CatalogModel",
    "CatalogProduct",
    "Evidence",
    "ExtractedValue",
    "ExtractionMetadata",
    "FieldEvidence",
    "FieldValue",
    "FlexibleValue",
    "JsonScalar",
    "JsonValue",
    "MAX_EVIDENCE_REFERENCES_PER_FIELD",
    "MAX_OTHER_DETAILS_PER_PRODUCT",
    "MAX_PRODUCTS_PER_BATCH",
    "MAX_SPECIFICATIONS_PER_PRODUCT",
    "ProductRecord",
    "ProductBatch",
    "ReviewStatus",
    "SourceDocument",
    "Specification",
    "TextValue",
    "product_batch_guided_json_schema",
]
