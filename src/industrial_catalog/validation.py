"""Normalization and quality checks for extracted product records.

Normalization is non-destructive: source strings in ``raw`` are never changed.
The functions here only populate or clean the separate ``normalized`` value.
Schema failures and extraction-quality issues are returned as structured data so
batch jobs can route questionable records to human review without crashing.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Iterable, Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .models import (
    MAX_OTHER_DETAILS_PER_PRODUCT,
    MAX_SPECIFICATIONS_PER_PRODUCT,
    Evidence,
    ExtractedValue,
    ProductRecord,
    Specification,
    TextValue,
)

Severity = Literal["info", "warning", "error"]


class ValidationIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    message: str
    path: str = ""
    severity: Severity = "warning"


class ValidationReport(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    valid: bool
    issues: list[ValidationIssue] = Field(default_factory=list)
    product: ProductRecord | None = None

    @property
    def errors(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "warning"]


class EvidenceResolution(BaseModel):
    """Canonicalized product payload plus source-linkage issues."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    payload: dict[str, Any]
    issues: list[ValidationIssue] = Field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)


class PreparedProduct(BaseModel):
    """One independently prepared item from an untrusted LLM product batch."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    ordinal: int = Field(ge=0)
    raw_payload: Any
    hydrated_payload: dict[str, Any] | None = None
    product: ProductRecord | None = None
    issues: list[ValidationIssue] = Field(default_factory=list)

    @property
    def accepted(self) -> bool:
        return self.product is not None and not any(
            issue.severity == "error" for issue in self.issues
        )


class ProductValidationError(ValueError):
    """Raised by :func:`require_valid_product` for a rejected record."""

    def __init__(self, report: ValidationReport):
        self.report = report
        summary = "; ".join(f"{item.path or '<root>'}: {item.message}" for item in report.errors)
        super().__init__(summary or "product validation failed")


_WHITESPACE = re.compile(r"\s+")
_DASH_TRANSLATION = str.maketrans(
    {
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
    }
)
_UNIT_ALIASES = {
    "degrees c": "°C",
    "degree c": "°C",
    "deg c": "°C",
    "celsius": "°C",
    "degrees f": "°F",
    "degree f": "°F",
    "deg f": "°F",
    "fahrenheit": "°F",
    "millimeter": "mm",
    "millimeters": "mm",
    "millimetre": "mm",
    "millimetres": "mm",
    "centimeter": "cm",
    "centimeters": "cm",
    "meter": "m",
    "meters": "m",
    "kilogram": "kg",
    "kilograms": "kg",
    "gram": "g",
    "grams": "g",
    "volts": "V",
    "volt": "V",
    "amps": "A",
    "amp": "A",
    "ampere": "A",
    "amperes": "A",
    "watts": "W",
    "watt": "W",
    "kilowatts": "kW",
    "kilowatt": "kW",
    "hertz": "Hz",
    "megapascal": "MPa",
    "megapascals": "MPa",
    "kilopascal": "kPa",
    "kilopascals": "kPa",
    "pounds per square inch": "psi",
}


def normalize_whitespace(value: str) -> str:
    """Normalize Unicode and whitespace while retaining meaningful punctuation."""

    normalized = unicodedata.normalize("NFKC", value).replace("\u00a0", " ")
    return _WHITESPACE.sub(" ", normalized).strip()


def normalize_manufacturer(value: str) -> str:
    """Conservatively normalize a manufacturer without destroying its branding."""

    return normalize_whitespace(value)


def normalize_part_number(value: str) -> str:
    """Create an exact-match-friendly part number while preserving ``raw``."""

    return normalize_whitespace(value.translate(_DASH_TRANSLATION)).upper()


def normalize_unit(value: str) -> str:
    """Normalize common engineering unit spellings."""

    cleaned = normalize_whitespace(value).replace("μ", "µ")
    return _UNIT_ALIASES.get(cleaned.casefold(), cleaned)


def _element_mapping(value: Any) -> dict[str, Any] | None:
    if isinstance(value, Mapping):
        return dict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        mapped = to_dict()
        return dict(mapped) if isinstance(mapped, Mapping) else None
    return None


def _iter_extracted_elements(value: Any) -> Iterable[dict[str, Any]]:
    """Yield element dictionaries from DocumentExtraction, pages, or flat lists."""

    if value is None or isinstance(value, (str, bytes)):
        return

    direct = _element_mapping(value)
    if direct is not None and "element_id" in direct:
        yield direct
        return

    if not isinstance(value, Mapping):
        elements = getattr(value, "elements", None)
        if elements is not None:
            for item in elements:
                if mapped := _element_mapping(item):
                    yield mapped
            return

    if isinstance(value, Mapping):
        if "element_id" in value:
            yield dict(value)
            return
        if isinstance(value.get("pages"), Iterable):
            for page in value["pages"]:
                yield from _iter_extracted_elements(page)
            return
        if isinstance(value.get("elements"), Iterable):
            for item in value["elements"]:
                yield from _iter_extracted_elements(item)
            return

    if isinstance(value, Iterable):
        for item in value:
            yield from _iter_extracted_elements(item)


def build_evidence_index(extracted_elements: Any) -> dict[str, dict[str, Any]]:
    """Build an element-ID index from extraction dataclasses or serialized data."""

    index: dict[str, dict[str, Any]] = {}
    for element in _iter_extracted_elements(extracted_elements):
        element_id = element.get("element_id")
        if element_id is not None:
            index[str(element_id)] = element
    return index


def _canonical_evidence(
    element: Mapping[str, Any],
    *,
    source_document_id: str,
    source_path: str | None,
) -> Evidence:
    metadata = dict(element.get("metadata") or {})
    if element.get("element_type") is not None:
        metadata.setdefault("element_type", element["element_type"])
    if element.get("type") is not None:
        metadata.setdefault("element_type", element["type"])
    if element.get("sequence_number") is not None:
        metadata.setdefault("sequence_number", element["sequence_number"])
    bbox = element.get("bbox")
    bbox_coordinate_space = metadata.get("bbox_coordinate_space")
    if (
        bbox_coordinate_space in {"pixel", "point", "normalized"}
        and isinstance(bbox, (list, tuple))
        and len(bbox) == 4
    ):
        bbox = {
            "x0": bbox[0],
            "y0": bbox[1],
            "x1": bbox[2],
            "y1": bbox[3],
            "coordinate_space": bbox_coordinate_space,
        }
    return Evidence.model_validate(
        {
            "source_document_id": element.get("document_id")
            or element.get("source_document_id")
            or source_document_id,
            "source_path": element.get("source_path") or source_path,
            "page_number": element.get("page_number"),
            "element_id": element.get("element_id"),
            "bbox": bbox,
            "text": element.get("text"),
            "extraction_method": element.get("extraction_method") or element.get("method"),
            "model": element.get("model"),
            "confidence": element.get("confidence"),
            "metadata": metadata,
        }
    )


def _fallback_evidence_mapping(
    reference: Mapping[str, Any],
    *,
    source_document_id: str,
    source_path: str | None,
) -> dict[str, Any]:
    metadata = dict(reference.get("metadata") or {})
    if reference.get("type") is not None:
        metadata.setdefault("element_type", reference["type"])
    allowed = {
        "source_document_id": reference.get("source_document_id")
        or reference.get("document_id")
        or source_document_id,
        "source_path": reference.get("source_path") or source_path,
        "page_number": reference.get("page_number"),
        "element_id": reference.get("element_id"),
        "bbox": reference.get("bbox"),
        "text": reference.get("text") or reference.get("excerpt"),
        "extraction_method": reference.get("extraction_method") or reference.get("method"),
        "model": reference.get("model"),
        "confidence": reference.get("confidence"),
        "metadata": metadata,
    }
    return allowed


def hydrate_product_evidence(
    product_payload: Mapping[str, Any],
    *,
    extracted_elements: Any,
    source_document_id: str,
    source_path: str | None = None,
) -> EvidenceResolution:
    """Replace compact LLM evidence references with canonical extraction evidence.

    Supported LLM forms are ``"el_..."``, ``{"element_id": "el_..."}``, and
    complete evidence objects.  When source extraction is supplied, unknown IDs
    are errors and claimed page/text differences are reported before canonical
    source values replace the claims.
    """

    evidence_source_supplied = extracted_elements is not None
    index = build_evidence_index(extracted_elements)
    issues: list[ValidationIssue] = []

    for collection_name, limit in (
        ("specifications", MAX_SPECIFICATIONS_PER_PRODUCT),
        ("other_details", MAX_OTHER_DETAILS_PER_PRODUCT),
    ):
        collection = product_payload.get(collection_name)
        if isinstance(collection, list) and len(collection) >= limit:
            issues.append(
                ValidationIssue(
                    code="wire_collection_limit_reached",
                    message=(
                        f"{collection_name} reached its structured-output limit of {limit}; "
                        "review the source for additional facts"
                    ),
                    path=collection_name,
                    severity="warning",
                )
            )

    by_page_text: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for element in index.values():
        try:
            page = int(element.get("page_number"))
        except (TypeError, ValueError):
            continue
        text = str(element.get("text") or "")
        by_page_text.setdefault((page, text), []).append(element)

    def resolve_reference(reference: Any, path: str) -> dict[str, Any] | None:
        claimed: Mapping[str, Any]
        if isinstance(reference, str):
            element_id = reference
            claimed = {"element_id": reference}
        elif isinstance(reference, Mapping):
            claimed = reference
            raw_id = reference.get("element_id")
            element_id = str(raw_id) if raw_id is not None else None
        else:
            issues.append(
                ValidationIssue(
                    code="invalid_evidence_reference",
                    message="evidence must be an element ID or object",
                    path=path,
                    severity="error",
                )
            )
            return None

        element = index.get(element_id) if element_id is not None else None
        if element is None and isinstance(claimed, Mapping):
            try:
                page_key = int(claimed.get("page_number"))
            except (TypeError, ValueError):
                page_key = -1
            text_key = str(claimed.get("text") or claimed.get("excerpt") or "")
            candidates = by_page_text.get((page_key, text_key), [])
            if len(candidates) == 1:
                element = candidates[0]

        if element is not None:
            element_document_id = element.get("document_id") or element.get("source_document_id")
            if element_document_id is not None and str(element_document_id) != source_document_id:
                issues.append(
                    ValidationIssue(
                        code="cross_document_evidence",
                        message=(
                            f"element {element.get('element_id')!r} belongs to "
                            f"{element_document_id!r}, not {source_document_id!r}"
                        ),
                        path=path,
                        severity="error",
                    )
                )
                return None
            for claimed_key, canonical_key in (
                ("page_number", "page_number"),
                ("text", "text"),
                ("excerpt", "text"),
                ("document_id", "document_id"),
                ("source_document_id", "document_id"),
                ("method", "extraction_method"),
                ("extraction_method", "extraction_method"),
            ):
                if claimed.get(claimed_key) is None or element.get(canonical_key) is None:
                    continue
                if str(claimed[claimed_key]) != str(element[canonical_key]):
                    issues.append(
                        ValidationIssue(
                            code="evidence_claim_mismatch",
                            message=(
                                f"claimed {claimed_key} differs from canonical "
                                f"element {element.get('element_id')!r}; canonical value used"
                            ),
                            path=path,
                            severity="warning",
                        )
                    )
            try:
                return _canonical_evidence(
                    element,
                    source_document_id=source_document_id,
                    source_path=source_path,
                ).model_dump(mode="json", exclude_none=True)
            except (TypeError, ValueError) as exc:
                issues.append(
                    ValidationIssue(
                        code="invalid_source_evidence",
                        message=str(exc),
                        path=path,
                        severity="error",
                    )
                )
                return None

        fallback = _fallback_evidence_mapping(
            claimed,
            source_document_id=source_document_id,
            source_path=source_path,
        )
        try:
            resolved = Evidence.model_validate(fallback)
        except (TypeError, ValueError) as exc:
            issues.append(
                ValidationIssue(
                    code="unknown_evidence_reference",
                    message=(
                        f"evidence element {element_id!r} was not found and the supplied "
                        f"object is incomplete: {exc}"
                    ),
                    path=path,
                    severity="error",
                )
            )
            return None

        issues.append(
            ValidationIssue(
                code=(
                    "unknown_evidence_reference"
                    if evidence_source_supplied
                    else "unverified_evidence"
                ),
                message=(
                    f"evidence element {element_id!r} was not found in supplied source extraction"
                    if evidence_source_supplied
                    else "complete evidence object accepted without an extraction source"
                ),
                path=path,
                severity="error" if evidence_source_supplied else "warning",
            )
        )
        return resolved.model_dump(mode="json", exclude_none=True)

    def visit(value: Any, path: str) -> Any:
        if isinstance(value, Mapping):
            output = {
                str(key): visit(item, f"{path}.{key}" if path else str(key))
                for key, item in value.items()
            }
            if "evidence" in value:
                evidence_path = f"{path}.evidence" if path else "evidence"
                raw_evidence = value["evidence"]
                if not isinstance(raw_evidence, list):
                    issues.append(
                        ValidationIssue(
                            code="invalid_evidence_collection",
                            message="evidence must be a list",
                            path=evidence_path,
                            severity="error",
                        )
                    )
                    output["evidence"] = []
                else:
                    resolved_items = [
                        resolve_reference(item, f"{evidence_path}.{index}")
                        for index, item in enumerate(raw_evidence)
                    ]
                    output["evidence"] = [item for item in resolved_items if item is not None]
            return output
        if isinstance(value, list):
            return [visit(item, f"{path}.{index}") for index, item in enumerate(value)]
        return value

    hydrated = visit(dict(product_payload), "")
    guided_other_details = hydrated.get("other_details")
    if isinstance(guided_other_details, list):
        canonical_details: dict[str, Any] = {}
        duplicate_counts: dict[str, int] = {}
        for index, detail in enumerate(guided_other_details):
            path = f"other_details.{index}"
            if not isinstance(detail, Mapping):
                issues.append(
                    ValidationIssue(
                        code="invalid_other_detail",
                        message="other detail must contain name and value objects",
                        path=path,
                        severity="error",
                    )
                )
                continue
            name_field = detail.get("name")
            value_field = detail.get("value")
            if not isinstance(name_field, Mapping) or not isinstance(value_field, Mapping):
                issues.append(
                    ValidationIssue(
                        code="invalid_other_detail",
                        message="other detail name and value must be extracted-value objects",
                        path=path,
                        severity="error",
                    )
                )
                continue
            raw_name = name_field.get("raw")
            normalized_name = name_field.get("normalized")
            detail_name = normalized_name if normalized_name is not None else raw_name
            if not isinstance(detail_name, str) or not detail_name.strip():
                issues.append(
                    ValidationIssue(
                        code="invalid_other_detail_name",
                        message="other detail name is blank",
                        path=f"{path}.name",
                        severity="error",
                    )
                )
                continue
            base_key = normalize_whitespace(detail_name)
            duplicate_counts[base_key] = duplicate_counts.get(base_key, 0) + 1
            occurrence = duplicate_counts[base_key]
            key = base_key if occurrence == 1 else f"{base_key} ({occurrence})"
            if occurrence > 1:
                issues.append(
                    ValidationIssue(
                        code="duplicate_other_detail_name",
                        message=f"duplicate detail name {base_key!r} retained as {key!r}",
                        path=path,
                        severity="warning",
                    )
                )

            merged_evidence: list[Any] = []
            seen_evidence: set[str] = set()
            for candidate in (
                *list(name_field.get("evidence") or []),
                *list(value_field.get("evidence") or []),
            ):
                identity = json.dumps(candidate, ensure_ascii=False, sort_keys=True)
                if identity not in seen_evidence:
                    seen_evidence.add(identity)
                    merged_evidence.append(candidate)
            canonical_value = dict(value_field)
            canonical_value["evidence"] = merged_evidence
            canonical_details[key] = canonical_value
        hydrated["other_details"] = canonical_details
    elif guided_other_details is not None and not isinstance(guided_other_details, Mapping):
        issues.append(
            ValidationIssue(
                code="invalid_other_details_collection",
                message="other_details must be a list of name/value facts",
                path="other_details",
                severity="error",
            )
        )
        hydrated["other_details"] = {}
    hydrated["source_document_id"] = source_document_id
    if source_path is not None:
        hydrated["source_path"] = source_path
    return EvidenceResolution(payload=hydrated, issues=issues)


def prepare_product_payload(
    raw_payload: Any,
    *,
    ordinal: int,
    extracted_elements: Any,
    source_document_id: str,
    source_path: str | None = None,
    strict: bool = False,
    require_evidence: bool = True,
    low_confidence_threshold: float = 0.65,
) -> PreparedProduct:
    """Hydrate, normalize, and validate one untrusted LLM product independently."""

    if not isinstance(raw_payload, Mapping):
        return PreparedProduct(
            ordinal=ordinal,
            raw_payload=raw_payload,
            issues=[
                ValidationIssue(
                    code="invalid_product_payload",
                    message="product entry must be a JSON object",
                    path=f"products.{ordinal}",
                    severity="error",
                )
            ],
        )

    resolution = hydrate_product_evidence(
        raw_payload,
        extracted_elements=extracted_elements,
        source_document_id=source_document_id,
        source_path=source_path,
    )
    hydrated = resolution.payload
    if not hydrated.get("record_id"):
        stable_payload = json.dumps(raw_payload, ensure_ascii=False, sort_keys=True, default=str)
        digest = hashlib.sha256(
            f"{source_document_id}\x00{ordinal}\x00{stable_payload}".encode()
        ).hexdigest()
        hydrated["record_id"] = f"prd_{digest[:32]}"

    report = validate_and_normalize_product(
        hydrated,
        strict=strict,
        require_evidence=require_evidence,
        low_confidence_threshold=low_confidence_threshold,
    )
    issues = [*resolution.issues, *report.issues]
    product = report.product
    if product is not None:
        warnings = [
            f"{issue.code}:{issue.path}:{issue.message}"
            for issue in issues
            if issue.severity != "info"
        ]
        if warnings:
            product = product.model_copy(
                update={
                    "review_status": (
                        "needs_review"
                        if product.review_status == "unreviewed"
                        else product.review_status
                    ),
                    "validation_warnings": warnings,
                }
            )
            product = ProductRecord.model_validate(product.model_dump())
    return PreparedProduct(
        ordinal=ordinal,
        raw_payload=dict(raw_payload),
        hydrated_payload=hydrated,
        product=product,
        issues=issues,
    )


def _normalize_text_value(
    value: TextValue | None,
    normalizer: Any = normalize_whitespace,
    method: str = "unicode_nfkc_whitespace_v1",
) -> TextValue | None:
    if value is None:
        return None
    candidate = value.normalized if value.normalized is not None else value.raw
    if candidate is None:
        return value
    normalized = normalizer(candidate)
    return value.model_copy(
        update={
            "normalized": normalized,
            "normalization_method": value.normalization_method or method,
        }
    )


def _normalize_specification(specification: Specification) -> Specification:
    value = specification.value
    normalized_value = value.normalized
    if isinstance(normalized_value, str):
        normalized_value = normalize_whitespace(normalized_value)
    elif normalized_value is None and value.raw is not None:
        normalized_value = normalize_whitespace(value.raw)
    normalized_field = value.model_copy(
        update={
            "normalized": normalized_value,
            "normalization_method": value.normalization_method or "unicode_nfkc_whitespace_v1",
        }
    )
    return specification.model_copy(
        update={
            "name": _normalize_text_value(specification.name),
            "value": normalized_field,
            "unit": _normalize_text_value(
                specification.unit,
                normalize_unit,
                "engineering_unit_aliases_v1",
            ),
            "category": _normalize_text_value(specification.category),
            "qualifier": _normalize_text_value(specification.qualifier),
        }
    )


def calculate_record_confidence(product: ProductRecord) -> float | None:
    """Calculate a weighted mean without fabricating missing confidences."""

    weighted: list[tuple[float, float]] = []
    for value, weight in (
        (product.manufacturer, 2.0),
        (product.part_name, 2.0),
        (product.part_number, 3.0),
        (product.category, 1.0),
        (product.description, 1.0),
    ):
        if value is not None and value.effective_confidence is not None:
            weighted.append((value.effective_confidence, weight))
    for specification in (*product.specifications, *product.operating_conditions):
        for value in (specification.name, specification.value, specification.unit):
            if value is not None and value.effective_confidence is not None:
                weighted.append((value.effective_confidence, 1.0))
    if not weighted:
        return None
    numerator = sum(confidence * weight for confidence, weight in weighted)
    denominator = sum(weight for _, weight in weighted)
    return round(numerator / denominator, 6)


def normalize_product_record(product: ProductRecord | Mapping[str, Any]) -> ProductRecord:
    """Return a normalized copy and leave all raw strings untouched."""

    parsed = (
        product if isinstance(product, ProductRecord) else ProductRecord.model_validate(product)
    )

    text_collections: dict[str, list[TextValue]] = {}
    for name in (
        "materials",
        "certifications",
        "compatible_parts",
        "aliases_and_cross_references",
    ):
        text_collections[name] = [_normalize_text_value(item) for item in getattr(parsed, name)]

    other_details: dict[str, ExtractedValue[Any]] = {}
    for key, value in parsed.other_details.items():
        normalized = value.normalized
        if isinstance(normalized, str):
            normalized = normalize_whitespace(normalized)
        elif normalized is None and value.raw is not None:
            normalized = normalize_whitespace(value.raw)
        other_details[normalize_whitespace(key)] = value.model_copy(
            update={
                "normalized": normalized,
                "normalization_method": value.normalization_method or "unicode_nfkc_whitespace_v1",
            }
        )

    updates: dict[str, Any] = {
        "manufacturer": _normalize_text_value(
            parsed.manufacturer,
            normalize_manufacturer,
            "manufacturer_whitespace_v1",
        ),
        "part_name": _normalize_text_value(parsed.part_name),
        "part_number": _normalize_text_value(
            parsed.part_number,
            normalize_part_number,
            "part_number_exact_match_v1",
        ),
        "category": _normalize_text_value(parsed.category),
        "description": _normalize_text_value(parsed.description),
        "specifications": [_normalize_specification(item) for item in parsed.specifications],
        "operating_conditions": [
            _normalize_specification(item) for item in parsed.operating_conditions
        ],
        "other_details": other_details,
        **text_collections,
    }
    normalized_product = parsed.model_copy(update=updates)
    if normalized_product.extraction.record_confidence is None:
        confidence = calculate_record_confidence(normalized_product)
        normalized_product = normalized_product.model_copy(
            update={
                "extraction": normalized_product.extraction.model_copy(
                    update={"record_confidence": confidence}
                )
            }
        )
    return ProductRecord.model_validate(normalized_product.model_dump())


def _add_value_issues(
    issues: list[ValidationIssue],
    *,
    path: str,
    value: ExtractedValue[Any],
    product: ProductRecord,
    require_evidence: bool,
    low_confidence_threshold: float,
    missing_evidence_severity: Severity,
) -> None:
    if isinstance(value.raw, str) and not value.raw.strip():
        issues.append(
            ValidationIssue(
                code="blank_raw_value",
                message="raw field content is blank",
                path=path,
                severity="warning",
            )
        )
    if isinstance(value.normalized, str) and not value.normalized.strip():
        issues.append(
            ValidationIssue(
                code="blank_normalized_value",
                message="normalized field content is blank",
                path=path,
                severity="warning",
            )
        )
    if require_evidence and not value.evidence:
        issues.append(
            ValidationIssue(
                code="missing_evidence",
                message="field has no page-level source evidence",
                path=path,
                severity=missing_evidence_severity,
            )
        )
    confidence = value.effective_confidence
    if confidence is not None and confidence < low_confidence_threshold:
        issues.append(
            ValidationIssue(
                code="low_confidence",
                message=(
                    f"field confidence {confidence:.3f} is below {low_confidence_threshold:.3f}"
                ),
                path=path,
                severity="warning",
            )
        )
    for index, evidence in enumerate(value.evidence):
        evidence_path = f"{path}.evidence.{index}"
        if evidence.source_document_id != product.source_document_id:
            issues.append(
                ValidationIssue(
                    code="evidence_document_mismatch",
                    message=(
                        f"evidence references {evidence.source_document_id!r}, not "
                        f"product document {product.source_document_id!r}"
                    ),
                    path=evidence_path,
                    severity="warning",
                )
            )
        if "\ufffd" in evidence.text:
            issues.append(
                ValidationIssue(
                    code="replacement_character",
                    message="evidence contains a Unicode replacement character",
                    path=f"{evidence_path}.text",
                    severity="warning",
                )
            )


def _iter_values(product: ProductRecord):
    for field_name in ("manufacturer", "part_name", "part_number", "category", "description"):
        value = getattr(product, field_name)
        if value is not None:
            yield field_name, value
    for collection_name in (
        "materials",
        "certifications",
        "compatible_parts",
        "aliases_and_cross_references",
    ):
        for index, value in enumerate(getattr(product, collection_name)):
            yield f"{collection_name}.{index}", value
    for collection_name in ("specifications", "operating_conditions"):
        for index, specification in enumerate(getattr(product, collection_name)):
            for component in ("name", "value", "unit", "category", "qualifier"):
                value = getattr(specification, component)
                if value is not None:
                    yield f"{collection_name}.{index}.{component}", value
    for key, value in product.other_details.items():
        yield f"other_details.{key}", value


def validate_product_record(
    product: ProductRecord | Mapping[str, Any],
    *,
    strict: bool = False,
    require_evidence: bool = True,
    low_confidence_threshold: float = 0.65,
) -> ValidationReport:
    """Validate schema and extraction quality without raising on bad records."""

    try:
        parsed = (
            product if isinstance(product, ProductRecord) else ProductRecord.model_validate(product)
        )
    except ValidationError as exc:
        issues = [
            ValidationIssue(
                code="schema_error",
                message=error["msg"],
                path=".".join(str(segment) for segment in error["loc"]),
                severity="error",
            )
            for error in exc.errors()
        ]
        return ValidationReport(valid=False, issues=issues, product=None)

    issues: list[ValidationIssue] = []
    strict_severity: Severity = "error" if strict else "warning"

    if parsed.part_name is None and parsed.part_number is None:
        issues.append(
            ValidationIssue(
                code="missing_product_identity",
                message="at least a part name or part number is required",
                path="",
                severity="error",
            )
        )
    if parsed.manufacturer is None:
        issues.append(
            ValidationIssue(
                code="missing_manufacturer",
                message="manufacturer was not extracted",
                path="manufacturer",
                severity=strict_severity,
            )
        )
    if parsed.part_number is None:
        issues.append(
            ValidationIssue(
                code="missing_part_number",
                message="part number was not extracted",
                path="part_number",
                severity=strict_severity,
            )
        )
    if not parsed.specifications:
        issues.append(
            ValidationIssue(
                code="missing_specifications",
                message="no technical specifications were extracted",
                path="specifications",
                severity=strict_severity,
            )
        )

    if parsed.part_number is not None:
        part_number = parsed.part_number.normalized or parsed.part_number.raw or ""
        if "\n" in part_number or "\r" in part_number:
            issues.append(
                ValidationIssue(
                    code="multiline_part_number",
                    message="part number contains a line break",
                    path="part_number",
                    severity="error",
                )
            )
        if len(part_number) > 160:
            issues.append(
                ValidationIssue(
                    code="suspicious_part_number_length",
                    message="part number is longer than 160 characters",
                    path="part_number",
                    severity="warning",
                )
            )
        if "\ufffd" in part_number:
            issues.append(
                ValidationIssue(
                    code="replacement_character",
                    message="part number contains a Unicode replacement character",
                    path="part_number",
                    severity="error",
                )
            )

    for path, value in _iter_values(parsed):
        _add_value_issues(
            issues,
            path=path,
            value=value,
            product=parsed,
            require_evidence=require_evidence,
            low_confidence_threshold=low_confidence_threshold,
            missing_evidence_severity="error" if strict else "warning",
        )

    seen_specs: dict[tuple[str, str], str] = {}
    for index, specification in enumerate(parsed.specifications):
        name = specification.name.normalized or specification.name.raw or ""
        unit = (
            specification.unit.normalized or specification.unit.raw
            if specification.unit is not None
            else ""
        )
        value = specification.value.normalized
        if value is None:
            value = specification.value.raw
        key = (str(name).casefold(), str(unit).casefold())
        serialized_value = repr(value)
        if key in seen_specs:
            code = (
                "duplicate_specification"
                if seen_specs[key] == serialized_value
                else "conflicting_specification"
            )
            issues.append(
                ValidationIssue(
                    code=code,
                    message=f"specification {name!r} is repeated",
                    path=f"specifications.{index}",
                    severity="warning",
                )
            )
        else:
            seen_specs[key] = serialized_value

    valid = not any(issue.severity == "error" for issue in issues)
    return ValidationReport(valid=valid, issues=issues, product=parsed)


def validate_and_normalize_product(
    product: ProductRecord | Mapping[str, Any],
    **validation_options: Any,
) -> ValidationReport:
    """Normalize a product, then return its extraction-quality report."""

    try:
        normalized = normalize_product_record(product)
    except ValidationError:
        return validate_product_record(product, **validation_options)
    return validate_product_record(normalized, **validation_options)


def require_valid_product(
    product: ProductRecord | Mapping[str, Any],
    *,
    normalize: bool = True,
    **validation_options: Any,
) -> ProductRecord:
    """Return a valid product or raise :class:`ProductValidationError`."""

    candidate = normalize_product_record(product) if normalize else product
    report = validate_product_record(candidate, **validation_options)
    if not report.valid or report.product is None:
        raise ProductValidationError(report)
    return report.product


__all__ = [
    "EvidenceResolution",
    "PreparedProduct",
    "ProductValidationError",
    "Severity",
    "ValidationIssue",
    "ValidationReport",
    "build_evidence_index",
    "calculate_record_confidence",
    "hydrate_product_evidence",
    "normalize_manufacturer",
    "normalize_part_number",
    "normalize_product_record",
    "normalize_unit",
    "normalize_whitespace",
    "prepare_product_payload",
    "require_valid_product",
    "validate_and_normalize_product",
    "validate_product_record",
]
