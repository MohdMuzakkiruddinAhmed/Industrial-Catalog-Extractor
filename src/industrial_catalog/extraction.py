"""Evidence-preserving industrial catalog extraction pipeline.

This module keeps page routing, model invocation, evidence normalization, and
checkpointing separate.  It processes pages in ascending order and creates
content-derived element identifiers so retries and resumed runs are idempotent.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from html import unescape
from pathlib import Path
from typing import Any, Literal, Protocol

from .nvidia_clients import (
    GuidedJsonClient,
    GuidedJsonResult,
    ModelResponse,
    NemotronOCRClient,
    NemotronParseClient,
    NIMRequestError,
    NvidiaClientBundle,
    parse_json_content,
)
from .routing import PageRoute, PageRouter, RoutingDecision

PIPELINE_VERSION = "1.1"


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _stable_hash(*values: Any) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = value if isinstance(value, bytes) else _stable_json(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def sha256_file(path: str | os.PathLike[str], chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """Page-space coordinates in ``left, top, right, bottom`` order."""

    left: float
    top: float
    right: float
    bottom: float

    def __post_init__(self) -> None:
        coordinates = (self.left, self.top, self.right, self.bottom)
        if not all(math.isfinite(value) for value in coordinates):
            raise ValueError("bounding-box coordinates must be finite")
        if self.right < self.left or self.bottom < self.top:
            raise ValueError("bounding-box right/bottom must not precede left/top")

    def to_list(self) -> list[float]:
        return [self.left, self.top, self.right, self.bottom]

    @classmethod
    def from_value(cls, value: Any) -> BoundingBox | None:
        if value is None:
            return None
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            aliases = (
                ("left", "top", "right", "bottom"),
                ("x0", "y0", "x1", "y1"),
                ("x", "y", "width", "height"),
            )
            for names in aliases:
                if all(name in value for name in names):
                    if names == ("x", "y", "width", "height"):
                        left = float(value["x"])
                        top = float(value["y"])
                        return cls(
                            left,
                            top,
                            left + float(value["width"]),
                            top + float(value["height"]),
                        )
                    return cls(*(float(value[name]) for name in names))
            return None
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            if len(value) == 4:
                return cls(*(float(item) for item in value))
        return None


@dataclass(frozen=True, slots=True)
class NativeTextBlock:
    text: str
    bbox: BoundingBox | None = None
    element_type: str = "text"
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PageInput:
    """One page plus cheap native-PDF signals and a lazy image renderer."""

    page_number: int
    native_text: str = ""
    native_blocks: tuple[NativeTextBlock, ...] = ()
    image_bytes: bytes | None = None
    image_loader: Callable[[], bytes] | None = field(default=None, repr=False)
    image_mime_type: str = "image/png"
    width: float | None = None
    height: float | None = None
    image_coverage: float = 0.0
    table_count: int = 0
    is_multi_column: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.page_number < 1:
            raise ValueError("page_number is one-based and must be positive")

    def get_image_bytes(self) -> bytes:
        if self.image_bytes is None:
            if self.image_loader is None:
                raise RuntimeError(
                    f"page {self.page_number} requires an image but has no renderer"
                )
            self.image_bytes = self.image_loader()
        return self.image_bytes


@dataclass(frozen=True, slots=True)
class ExtractedElement:
    """A source-backed text/layout observation, never a synthesized fact."""

    element_id: str
    document_id: str
    page_number: int
    sequence_number: int
    element_type: str
    text: str
    extraction_method: str
    bbox: BoundingBox | None = None
    confidence: float | None = None
    model: str | None = None
    source_path: str | None = None
    raw_payload: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        document_id: str,
        page_number: int,
        sequence_number: int,
        element_type: str,
        text: str,
        extraction_method: str,
        bbox: BoundingBox | None = None,
        confidence: float | None = None,
        model: str | None = None,
        source_path: str | None = None,
        raw_payload: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ExtractedElement:
        bounded_confidence = confidence
        if bounded_confidence is not None:
            bounded_confidence = min(1.0, max(0.0, float(bounded_confidence)))
        identifier = _stable_hash(
            document_id,
            page_number,
            sequence_number,
            element_type,
            text,
            extraction_method,
            bbox.to_list() if bbox else None,
            model,
        )
        return cls(
            element_id=f"el_{identifier[:32]}",
            document_id=document_id,
            page_number=page_number,
            sequence_number=sequence_number,
            element_type=element_type,
            text=text,
            extraction_method=extraction_method,
            bbox=bbox,
            confidence=bounded_confidence,
            model=model,
            source_path=source_path,
            raw_payload=dict(raw_payload or {}),
            metadata=dict(metadata or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "element_id": self.element_id,
            "document_id": self.document_id,
            "page_number": self.page_number,
            "sequence_number": self.sequence_number,
            "element_type": self.element_type,
            "text": self.text,
            "extraction_method": self.extraction_method,
            "bbox": self.bbox.to_list() if self.bbox else None,
            "confidence": self.confidence,
            "model": self.model,
            "source_path": self.source_path,
            "raw_payload": dict(self.raw_payload),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ExtractedElement:
        return cls(
            element_id=str(value["element_id"]),
            document_id=str(value["document_id"]),
            page_number=int(value["page_number"]),
            sequence_number=int(value["sequence_number"]),
            element_type=str(value["element_type"]),
            text=str(value.get("text", "")),
            extraction_method=str(value["extraction_method"]),
            bbox=BoundingBox.from_value(value.get("bbox")),
            confidence=(
                float(value["confidence"]) if value.get("confidence") is not None else None
            ),
            model=str(value["model"]) if value.get("model") is not None else None,
            source_path=(
                str(value["source_path"])
                if value.get("source_path") is not None
                else None
            ),
            raw_payload=dict(value.get("raw_payload", {})),
            metadata=dict(value.get("metadata", {})),
        )


@dataclass(frozen=True, slots=True)
class PageExtraction:
    document_id: str
    page_number: int
    fingerprint: str
    routing: RoutingDecision
    elements: tuple[ExtractedElement, ...]
    model_responses: tuple[Mapping[str, Any], ...] = ()
    restored_from_checkpoint: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "page_number": self.page_number,
            "fingerprint": self.fingerprint,
            "routing": self.routing.to_dict(),
            "elements": [element.to_dict() for element in self.elements],
            "model_responses": [dict(response) for response in self.model_responses],
        }

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any], *, restored_from_checkpoint: bool = False
    ) -> PageExtraction:
        return cls(
            document_id=str(value["document_id"]),
            page_number=int(value["page_number"]),
            fingerprint=str(value["fingerprint"]),
            routing=RoutingDecision.from_dict(value["routing"]),
            elements=tuple(
                ExtractedElement.from_dict(element)
                for element in value.get("elements", ())
            ),
            model_responses=tuple(
                dict(response) for response in value.get("model_responses", ())
            ),
            restored_from_checkpoint=restored_from_checkpoint,
        )


@dataclass(frozen=True, slots=True)
class DocumentExtraction:
    document_id: str
    source_path: str | None
    pages: tuple[PageExtraction, ...]
    pipeline_version: str = PIPELINE_VERSION

    @property
    def elements(self) -> tuple[ExtractedElement, ...]:
        return tuple(element for page in self.pages for element in page.elements)

    @property
    def text(self) -> str:
        return "\n".join(element.text for element in self.elements if element.text)

    @property
    def checkpoint_hits(self) -> int:
        return sum(page.restored_from_checkpoint for page in self.pages)

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "source_path": self.source_path,
            "pipeline_version": self.pipeline_version,
            "pages": [page.to_dict() for page in self.pages],
        }


@dataclass(frozen=True, slots=True)
class ManufacturerCandidate:
    """Deterministic manufacturer text anchored to one extracted element."""

    raw: str
    element_id: str
    page_number: int
    basis: str
    priority: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw": self.raw,
            "element_id": self.element_id,
            "page_number": self.page_number,
            "basis": self.basis,
            "priority": self.priority,
        }


@dataclass(frozen=True, slots=True)
class ProductCardinalityDecision:
    """Whether source rows justify multiple product records on this page chunk."""

    mode: Literal["single_family", "multi_product_rows"]
    reason: str
    identifiers: tuple[str, ...] = ()
    evidence_element_ids: tuple[str, ...] = ()

    @property
    def maximum_products(self) -> int | None:
        return 1 if self.mode == "single_family" else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "reason": self.reason,
            "maximum_products": self.maximum_products,
            "identifiers": list(self.identifiers),
            "evidence_element_ids": list(self.evidence_element_ids),
        }


@dataclass(frozen=True, slots=True)
class StructuredSourceHints:
    manufacturer_candidates: tuple[ManufacturerCandidate, ...]
    cardinality: ProductCardinalityDecision

    def to_dict(self) -> dict[str, Any]:
        return {
            "manufacturer_candidates": [
                candidate.to_dict() for candidate in self.manufacturer_candidates
            ],
            "cardinality": self.cardinality.to_dict(),
        }

    def prompt_text(self) -> str:
        candidate_payload = [
            {
                "raw": candidate.raw,
                "element_id": candidate.element_id,
                "basis": candidate.basis,
            }
            for candidate in self.manufacturer_candidates
        ]
        if candidate_payload:
            manufacturer_rule = (
                "Every emitted product MUST use manufacturer.raw exactly from one "
                "candidate and MUST cite that candidate's element_id. Materials, "
                "finishes, product adjectives, and title words are not manufacturers."
            )
        else:
            manufacturer_rule = (
                "No deterministic manufacturer candidate was found. Do not treat a "
                "material, finish, adjective, title phrase, domain, or person as the "
                "manufacturer. Return products=[] unless explicit manufacturer text in "
                "SOURCE EVIDENCE still satisfies the product eligibility rules."
            )
        if self.cardinality.mode == "single_family":
            cardinality_rule = (
                "No explicit table rows with distinct orderable part/model identifiers "
                "were found. Emit at most ONE product for this entire evidence chunk; "
                "group same-family fragments into it."
            )
        else:
            cardinality_rule = (
                "Distinct orderable part/model rows were found. Emit EXACTLY one product "
                "for EACH identifier in CARDINALITY_DECISION.identifiers, with "
                "part_number.raw copied exactly from that identifier. Cover every listed "
                "identifier once: no missing, duplicate, or additional product records. "
                "The part_number evidence must cite an element_id listed in "
                "CARDINALITY_DECISION.evidence_element_ids. This is a lean identity-only "
                "row pass: emit only manufacturer, part_name, and part_number fields; do "
                "not copy table bodies into specifications, descriptions, or other details."
            )
        return (
            "DETERMINISTIC SOURCE HINTS (computed from SOURCE EVIDENCE):\n"
            f"MANUFACTURER_CANDIDATES={_stable_json(candidate_payload)}\n"
            f"MANUFACTURER_REQUIREMENT={manufacturer_rule}\n"
            f"CARDINALITY_DECISION={_stable_json(self.cardinality.to_dict())}\n"
            f"CARDINALITY_REQUIREMENT={cardinality_rule}"
        )


_LEGAL_SUFFIX = (
    r"(?:L\.?L\.?C\.?|L\.?L\.?P\.?|P\.?L\.?C\.?|Inc(?:orporated)?\.?|"
    r"Ltd\.?|Limited|Corp(?:oration)?\.?|Company|Co\.?|GmbH|S\.?A\.?|"
    r"S\.?p\.?A\.?|B\.?V\.?|N\.?V\.?|P\.?L\.?C\.?|Pty\.?\s+Ltd\.?|"
    r"SAS|SARL|A\.?G\.?|A\.?B\.?|Oy|K\.?K\.?)"
)
_LEGAL_COMPANY_RE = re.compile(
    rf"(?<!\w)(?P<name>[A-Za-z0-9][A-Za-z0-9&'’()./\-]*"
    rf"(?:\s+[A-Za-z0-9&'’()./\-]+){{0,7}}\s*,?\s+{_LEGAL_SUFFIX})(?!\w)",
    re.IGNORECASE,
)
_COPYRIGHT_RE = re.compile(
    r"(?:©|\(c\)|\bcopyright\b)\s*(?:©\s*)?(?:\d{4}(?:\s*[-–]\s*\d{4})?\s*)?"
    r"(?:by\s+)?(?P<name>[^|;\n]{2,120})",
    re.IGNORECASE,
)
_LABELED_MANUFACTURER_RE = re.compile(
    r"\b(?:manufacturer|manufactured\s+by|made\s+by|published\s+by|"
    r"a\s+brand\s+of|brand)\s*[:\-]?\s*(?P<name>[^|;\n]{2,120})",
    re.IGNORECASE,
)
_ATTRIBUTION_PREFIX_RE = re.compile(
    r"^(?:(?:copyright|all\s+rights\s+reserved|manufacturer|manufactured|"
    r"made|published|distributed|contact|visit|for|by)\b[\s:,.©-]*|"
    r"\(c\)[\s:,.©-]*|©[\s:,.©-]*|\d{4}(?:\s*[-–]\s*\d{4})?\s*)+",
    re.IGNORECASE,
)
_ATTRIBUTION_TAIL_RE = re.compile(
    r"\b(?:all\s+rights\s+reserved|www\.|https?://|phone|tel(?:ephone)?|fax)\b.*$",
    re.IGNORECASE,
)
_ORDERABLE_HEADER_RE = re.compile(
    r"\b(?:part\s*(?:number|no\.?|#)|p\s*/\s*n|model\s*(?:number|no\.?|#)?|"
    r"catalog\s*(?:number|no\.?|#)|ordering\s*(?:code|number|no\.?)|"
    r"item\s*(?:number|no\.?|#)|sku)\b",
    re.IGNORECASE,
)
_HTML_ROW_RE = re.compile(r"<tr\b[^>]*>(.*?)</tr>", re.IGNORECASE | re.DOTALL)
_HTML_CELL_RE = re.compile(
    r"<(?:th|td)\b[^>]*>(.*?)</(?:th|td)>", re.IGNORECASE | re.DOTALL
)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_TRADEMARK_MARK_RE = re.compile(r"[®™℠]")
_GENERIC_BRANDED_HEADER_RE = re.compile(
    r"\b(?:catalog(?:ue)?|brochure|datasheet|manual|guide|product(?:s)?|"
    r"specification(?:s)?|technical|overview|selection|stainless|steel|"
    r"alumin(?:um|ium)|bronze|plastic|coupling(?:s)?|hub(?:s)?|bearing(?:s)?|"
    r"sensor(?:s)?|valve(?:s)?|motor(?:s)?|cable(?:s)?|connector(?:s)?|"
    r"fitting(?:s)?)\b",
    re.IGNORECASE,
)
_LATEX_BEGIN_TABULAR_RE = re.compile(
    r"\\+begin\s*\{(?P<environment>tabular\*?)\}", re.IGNORECASE
)
_LATEX_END_TABULAR_RE = re.compile(
    r"\\+end\s*\{tabular\*?\}", re.IGNORECASE
)
_LATEX_MULTICOLUMN_RE = re.compile(
    r"\\+multicolumn\s*\{[^{}]*\}\s*\{[^{}]*\}\s*\{([^{}]*)\}",
    re.IGNORECASE,
)
_LATEX_MULTIROW_RE = re.compile(
    r"\\+multirow\s*\{[^{}]*\}\s*\{[^{}]*\}\s*\{([^{}]*)\}",
    re.IGNORECASE,
)
_LATEX_FRACTION_RE = re.compile(
    r"\\+(?:d?frac)\s*\{([^{}]*)\}\s*\{([^{}]*)\}", re.IGNORECASE
)
_LATEX_FORMATTING_RE = re.compile(
    r"\\+(?:textbf|textit|texttt|textrm|mathrm|mathbf|mathit|emph|mbox|"
    r"makecell|thead)\s*\{([^{}]*)\}",
    re.IGNORECASE,
)
_LATEX_RULE_RE = re.compile(
    r"\\+(?:hline|toprule|midrule|bottomrule|addlinespace)\b|"
    r"\\+(?:cline|cmidrule)\s*\{[^{}]*\}",
    re.IGNORECASE,
)
_LATEX_COMMAND_RE = re.compile(r"\\+[A-Za-z]+\*?")
_SIZE_HEADER_RE = re.compile(
    r"^(?:(?:product|coupling)\s+)?size(?:\s*(?:code|number|no\.?|#))?$",
    re.IGNORECASE,
)
_PHYSICAL_DIMENSION_RE = re.compile(
    r"\d+(?:\.\d+)?(?:mm|cm|m|in|inch|inches|ft|feet)$", re.IGNORECASE
)


def _clean_manufacturer_candidate(value: str) -> str:
    candidate = " ".join(value.split()).strip(" |;:")
    by_matches = list(re.finditer(r"\b(?:by|from)\s+(?=[A-Z0-9])", candidate))
    if by_matches:
        candidate = candidate[by_matches[-1].end() :]
    candidate = _ATTRIBUTION_PREFIX_RE.sub("", candidate).strip(" |;:")
    candidate = _ATTRIBUTION_TAIL_RE.sub("", candidate).strip(" |;:")
    return candidate


def _manufacturer_candidate_is_usable(value: str) -> bool:
    if not 2 <= len(value) <= 120 or not any(character.isalpha() for character in value):
        return False
    lowered = value.casefold().strip(" .")
    if lowered in {
        "stainless steel",
        "steel",
        "aluminum",
        "aluminium",
        "bronze",
        "plastic",
        "manufacturer",
        "company",
    }:
        return False
    return "@" not in value and "www." not in lowered and "http" not in lowered


def _trademark_header_manufacturer(value: str, element_type: str) -> str | None:
    """Return a standalone trademarked brand masthead, never a generic title."""

    if not any(token in element_type.casefold() for token in ("header", "heading")):
        return None
    decoded = unescape(value)
    if not _TRADEMARK_MARK_RE.search(decoded):
        return None
    visible = _HTML_TAG_RE.sub(" ", decoded)
    visible = re.sub(r"^\s{0,3}#{1,6}\s*", "", visible).strip(" `*_")
    if "\n" in visible or any(separator in visible for separator in ("|", ":", ";")):
        return None
    match = re.fullmatch(r"(?P<brand>.+?)\s*[®™℠]\s*", visible)
    if match is None:
        return None
    brand = " ".join(match.group("brand").strip(" `*_.,-–—").split())
    if not 2 <= len(brand) <= 60 or not 1 <= len(brand.split()) <= 4:
        return None
    if not any(character.isalpha() for character in brand):
        return None
    if _GENERIC_BRANDED_HEADER_RE.search(brand):
        return None
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9&'’()./ +\-]*", brand):
        return None
    return brand if _manufacturer_candidate_is_usable(brand) else None


def manufacturer_candidates_from_extraction(
    extraction: DocumentExtraction,
) -> tuple[ManufacturerCandidate, ...]:
    """Derive ranked legal/brand manufacturer hints without an LLM."""

    candidates: list[ManufacturerCandidate] = []

    def add(raw: str, element: ExtractedElement, basis: str, priority: int) -> None:
        cleaned = _clean_manufacturer_candidate(raw)
        if not _manufacturer_candidate_is_usable(cleaned):
            return
        candidates.append(
            ManufacturerCandidate(
                raw=cleaned,
                element_id=element.element_id,
                page_number=element.page_number,
                basis=basis,
                priority=priority,
            )
        )

    for element in extraction.elements:
        text = element.text.strip()
        if not text:
            continue
        for match in _LABELED_MANUFACTURER_RE.finditer(text):
            labeled = match.group("name")
            legal = _LEGAL_COMPANY_RE.search(labeled)
            add(legal.group("name") if legal else labeled, element, "manufacturer_label", 0)
        for match in _COPYRIGHT_RE.finditer(text):
            attributed = match.group("name")
            legal = _LEGAL_COMPANY_RE.search(attributed)
            add(legal.group("name") if legal else attributed, element, "copyright", 1)
        for match in _LEGAL_COMPANY_RE.finditer(text):
            add(match.group("name"), element, "legal_suffix", 2)
        element_type = element.element_type.casefold()
        trademark_header = _trademark_header_manufacturer(text, element_type)
        if trademark_header is not None:
            add(trademark_header, element, "trademark_header", 2)
        if any(token in element_type for token in ("brand", "logo", "company")):
            for line in text.splitlines():
                if 1 <= len(line.split()) <= 8:
                    add(line, element, "brand_element", 3)

    candidates.sort(
        key=lambda item: (
            item.priority,
            item.page_number,
            item.element_id,
            item.raw.casefold(),
        )
    )
    deduplicated: list[ManufacturerCandidate] = []
    seen: set[str] = set()
    for candidate in candidates:
        identity = candidate.raw.casefold()
        if identity in seen:
            continue
        seen.add(identity)
        deduplicated.append(candidate)
    return tuple(deduplicated)


def _read_latex_group(text: str, offset: int) -> tuple[str, int] | None:
    while offset < len(text) and text[offset].isspace():
        offset += 1
    if offset >= len(text) or text[offset] != "{":
        return None
    depth = 1
    start = offset + 1
    cursor = start
    while cursor < len(text):
        character = text[cursor]
        if character == "\\" and cursor + 1 < len(text):
            cursor += 2
            continue
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return text[start:cursor], cursor + 1
        cursor += 1
    return None


def _strip_latex_tabular_commands(text: str) -> str:
    stripped = text
    while match := _LATEX_BEGIN_TABULAR_RE.search(stripped):
        cursor = match.end()
        argument_count = 2 if match.group("environment").endswith("*") else 1
        for _ in range(argument_count):
            group = _read_latex_group(stripped, cursor)
            if group is None:
                break
            _, cursor = group
        stripped = stripped[: match.start()] + stripped[cursor:]
    return _LATEX_END_TABULAR_RE.sub("", stripped)


def _split_latex_rows(text: str) -> list[str]:
    rows: list[str] = []
    current: list[str] = []
    brace_depth = 0
    cursor = 0
    while cursor < len(text):
        character = text[cursor]
        if character == "{":
            brace_depth += 1
        elif character == "}" and brace_depth:
            brace_depth -= 1
        if character == "\\":
            run_end = cursor
            while run_end < len(text) and text[run_end] == "\\":
                run_end += 1
            if brace_depth == 0 and run_end - cursor >= 2:
                rows.append("".join(current))
                current = []
                cursor = run_end
                if cursor < len(text) and text[cursor] == "[":
                    option_end = text.find("]", cursor + 1)
                    if option_end >= 0:
                        cursor = option_end + 1
                continue
            current.append(text[cursor:run_end])
            cursor = run_end
            continue
        current.append(character)
        cursor += 1
    rows.append("".join(current))
    return rows


def _split_latex_cells(row: str) -> list[str]:
    cells: list[str] = []
    current: list[str] = []
    brace_depth = 0
    cursor = 0
    while cursor < len(row):
        character = row[cursor]
        if character == "\\" and cursor + 1 < len(row):
            current.extend((character, row[cursor + 1]))
            cursor += 2
            continue
        if character == "{":
            brace_depth += 1
        elif character == "}" and brace_depth:
            brace_depth -= 1
        if character == "&" and brace_depth == 0:
            cells.append("".join(current))
            current = []
        else:
            current.append(character)
        cursor += 1
    cells.append("".join(current))
    return cells


def _clean_table_cell(value: str) -> str:
    candidate = _HTML_TAG_RE.sub(" ", value)
    candidate = _LATEX_RULE_RE.sub(" ", candidate)
    previous = None
    while candidate != previous:
        previous = candidate
        candidate = _LATEX_FRACTION_RE.sub(r"\1/\2", candidate)
        candidate = _LATEX_FORMATTING_RE.sub(r"\1", candidate)
        candidate = _LATEX_MULTICOLUMN_RE.sub(r"\1", candidate)
        candidate = _LATEX_MULTIROW_RE.sub(r"\1", candidate)
    replacements = {
        r"\&": "&",
        r"\%": "%",
        r"\_": "_",
        r"\#": "#",
        r"\$": "$",
        r"\{": "{",
        r"\}": "}",
    }
    for encoded, decoded in replacements.items():
        candidate = candidate.replace(encoded, decoded)
    candidate = _LATEX_COMMAND_RE.sub(" ", candidate)
    candidate = candidate.replace("~", " ").strip()
    if len(candidate) >= 2 and candidate.startswith("$") and candidate.endswith("$"):
        candidate = candidate[1:-1]
    candidate = candidate.replace("{", "").replace("}", "")
    return " ".join(candidate.split()).strip(" `*|")


def _latex_table_rows(text: str) -> list[list[str]]:
    if not _LATEX_BEGIN_TABULAR_RE.search(text):
        return []
    body = _strip_latex_tabular_commands(text)
    rows: list[list[str]] = []
    for raw_row in _split_latex_rows(body):
        cells = [_clean_table_cell(cell) for cell in _split_latex_cells(raw_row)]
        if any(cells):
            rows.append(cells)
    return rows


def _table_rows(text: str, *, allow_plain_columns: bool) -> list[list[str]]:
    latex_rows = _latex_table_rows(text)
    if latex_rows:
        return latex_rows

    rows: list[list[str]] = []
    for html_row in _HTML_ROW_RE.findall(text):
        cells = [
            " ".join(_HTML_TAG_RE.sub(" ", cell).split())
            for cell in _HTML_CELL_RE.findall(html_row)
        ]
        if len(cells) >= 2:
            rows.append(cells)
    if rows:
        return rows

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if "|" in stripped:
            cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        elif "\t" in stripped:
            cells = [cell.strip() for cell in stripped.split("\t")]
        elif allow_plain_columns:
            cells = [cell.strip() for cell in re.split(r"\s{2,}", stripped)]
        else:
            continue
        if len(cells) >= 2:
            rows.append([_clean_table_cell(cell) for cell in cells])
    return rows


def _valid_orderable_identifier(value: str) -> bool:
    candidate = _clean_table_cell(value)
    if not 1 <= len(candidate) <= 80 or not any(character.isalnum() for character in candidate):
        return False
    if re.fullmatch(r"[-:]+", candidate):
        return False
    return _ORDERABLE_HEADER_RE.search(candidate) is None


def _valid_code_like_identifier(value: str) -> bool:
    candidate = _clean_table_cell(value)
    if not 2 <= len(candidate) <= 80 or " " in candidate:
        return False
    if _PHYSICAL_DIMENSION_RE.fullmatch(candidate):
        return False
    if not any(character.isalpha() for character in candidate):
        return False
    if not any(character.isdigit() for character in candidate):
        return False
    return re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+/#-]*", candidate) is not None


def product_cardinality_from_extraction(
    extraction: DocumentExtraction,
) -> ProductCardinalityDecision:
    """Unlock multi-product mode only for explicit orderable table rows."""

    identifiers: list[str] = []
    evidence_ids: list[str] = []
    for element in extraction.elements:
        element_type = element.element_type.casefold()
        looks_like_table = (
            "table" in element_type
            or "|" in element.text
            or "<tr" in element.text
            or _LATEX_BEGIN_TABULAR_RE.search(element.text) is not None
        )
        if not looks_like_table:
            continue
        rows = _table_rows(element.text, allow_plain_columns="table" in element_type)
        matched_identifier_column = False
        for header_index, header in enumerate(rows):
            cleaned_header = [_clean_table_cell(cell) for cell in header]
            column_rules: list[tuple[int, Callable[[str], bool]]] = [
                (index, _valid_orderable_identifier)
                for index, cell in enumerate(cleaned_header)
                if _ORDERABLE_HEADER_RE.search(cell)
            ]
            column_rules.extend(
                (index, _valid_code_like_identifier)
                for index, cell in enumerate(cleaned_header)
                if _SIZE_HEADER_RE.fullmatch(cell)
                and all(existing_index != index for existing_index, _ in column_rules)
            )
            if not column_rules:
                continue

            for column, validator in column_rules:
                column_identifiers: list[str] = []
                for row in rows[header_index + 1 :]:
                    cleaned_row = [_clean_table_cell(cell) for cell in row]
                    if any(
                        _ORDERABLE_HEADER_RE.search(cell)
                        or _SIZE_HEADER_RE.fullmatch(cell)
                        for cell in cleaned_row
                    ):
                        break
                    if column >= len(row):
                        continue
                    value = _clean_table_cell(row[column])
                    if validator(value):
                        column_identifiers.append(value)
                distinct_column_identifiers = list(
                    dict.fromkeys(
                        value.casefold() for value in column_identifiers
                    )
                )
                if len(distinct_column_identifiers) < 2:
                    continue
                identifiers.extend(column_identifiers)
                evidence_ids.append(element.element_id)
                matched_identifier_column = True
                break
            if matched_identifier_column:
                break

    unique_identifiers: list[str] = []
    seen_identifiers: set[str] = set()
    for identifier in identifiers:
        identity = identifier.casefold()
        if identity in seen_identifiers:
            continue
        seen_identifiers.add(identity)
        unique_identifiers.append(identifier)
    unique_evidence_ids = tuple(dict.fromkeys(evidence_ids))
    if len(unique_identifiers) >= 2:
        return ProductCardinalityDecision(
            mode="multi_product_rows",
            reason="explicit_table_rows_with_distinct_orderable_identifiers",
            identifiers=tuple(unique_identifiers),
            evidence_element_ids=unique_evidence_ids,
        )
    return ProductCardinalityDecision(
        mode="single_family",
        reason="no_explicit_distinct_orderable_part_or_model_rows",
        identifiers=tuple(unique_identifiers),
        evidence_element_ids=unique_evidence_ids,
    )


def derive_structured_source_hints(extraction: DocumentExtraction) -> StructuredSourceHints:
    return StructuredSourceHints(
        manufacturer_candidates=manufacturer_candidates_from_extraction(extraction),
        cardinality=product_cardinality_from_extraction(extraction),
    )


class CheckpointStore(Protocol):
    def load_page(
        self, document_id: str, page_number: int, fingerprint: str
    ) -> PageExtraction | None:
        """Return a matching completed page, or ``None``."""

    def save_page(self, page: PageExtraction) -> None:
        """Atomically persist a completed page."""


class MemoryCheckpointStore:
    """In-memory checkpoint implementation useful for tests and notebooks."""

    def __init__(self) -> None:
        self._pages: dict[tuple[str, int], dict[str, Any]] = {}

    def load_page(
        self, document_id: str, page_number: int, fingerprint: str
    ) -> PageExtraction | None:
        value = self._pages.get((document_id, page_number))
        if value is None or value.get("fingerprint") != fingerprint:
            return None
        restored = PageExtraction.from_dict(value, restored_from_checkpoint=True)
        if restored.document_id != document_id or restored.page_number != page_number:
            return None
        return restored

    def save_page(self, page: PageExtraction) -> None:
        self._pages[(page.document_id, page.page_number)] = page.to_dict()


class JsonPageCheckpointStore:
    """One atomic JSON file per page, suitable for long resumable batch runs."""

    def __init__(self, root_directory: str | os.PathLike[str]) -> None:
        self.root_directory = Path(root_directory)

    def _page_path(self, document_id: str, page_number: int) -> Path:
        safe_document_key = hashlib.sha256(document_id.encode("utf-8")).hexdigest()
        return self.root_directory / safe_document_key / f"{page_number:06d}.json"

    def load_page(
        self, document_id: str, page_number: int, fingerprint: str
    ) -> PageExtraction | None:
        path = self._page_path(document_id, page_number)
        try:
            with path.open("r", encoding="utf-8") as stream:
                value = json.load(stream)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return None
        if value.get("fingerprint") != fingerprint:
            return None
        try:
            restored = PageExtraction.from_dict(value, restored_from_checkpoint=True)
        except (KeyError, TypeError, ValueError):
            return None
        if restored.document_id != document_id or restored.page_number != page_number:
            return None
        return restored

    def save_page(self, page: PageExtraction) -> None:
        path = self._page_path(page.document_id, page.page_number)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.stem}.",
                suffix=".tmp",
                delete=False,
            ) as stream:
                temporary_path = stream.name
                json.dump(
                    page.to_dict(),
                    stream,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, path)
        finally:
            if temporary_path and os.path.exists(temporary_path):
                try:
                    os.unlink(temporary_path)
                except OSError:
                    pass


class PageSource(Protocol):
    def iter_pages(self) -> Iterator[PageInput]:
        """Yield pages in strictly increasing one-based page order."""


class PyMuPDFPageSource:
    """Lazy PDF source using PyMuPDF, imported only when a PDF is processed."""

    def __init__(self, path: str | os.PathLike[str], *, render_dpi: int = 144) -> None:
        self.path = Path(path)
        self.render_dpi = render_dpi
        if render_dpi < 72:
            raise ValueError("render_dpi must be at least 72")

    def iter_pages(self) -> Iterator[PageInput]:
        try:
            import pymupdf as fitz  # type: ignore
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "PyMuPDF is required for extract_document; install package 'PyMuPDF'"
            ) from exc

        with fitz.open(self.path) as document:
            for zero_based_page_number, page in enumerate(document):
                native_text = page.get_text("text", sort=True) or ""
                raw_blocks = page.get_text("blocks", sort=True) or []
                blocks: list[NativeTextBlock] = []
                for raw_block in raw_blocks:
                    if len(raw_block) < 5:
                        continue
                    block_text = str(raw_block[4]).strip()
                    if not block_text:
                        continue
                    block_type = "text"
                    if len(raw_block) > 6 and int(raw_block[6]) != 0:
                        block_type = "image"
                    blocks.append(
                        NativeTextBlock(
                            text=block_text,
                            bbox=BoundingBox.from_value(raw_block[:4]),
                            element_type=block_type,
                            metadata={
                                "native_block_number": (
                                    int(raw_block[5]) if len(raw_block) > 5 else len(blocks)
                                )
                            },
                        )
                    )

                page_width = float(page.rect.width)
                page_height = float(page.rect.height)
                image_coverage = _estimate_image_coverage(
                    raw_blocks, page_width, page_height
                )
                table_count = _estimate_table_count(native_text)
                is_multi_column = _estimate_multi_column(blocks, page_width)
                # Raster block coverage cannot see vector-only artwork.  Keep
                # both source-object counts so the extraction boundary can
                # distinguish a truly empty page from one that merely lacks
                # native text.
                drawing_count = len(page.get_drawings())
                embedded_image_count = len(page.get_images(full=True))
                annotations = page.annots()
                annotation_count = sum(1 for _ in annotations) if annotations else 0
                link_count = len(page.get_links())

                def render(
                    current_page: Any = page,
                    dpi: int = self.render_dpi,
                    pdf_path: Path = self.path,
                    page_index: int = zero_based_page_number,
                ) -> bytes:
                    """Render while streaming, or safely reopen after iterator close."""

                    scale = dpi / 72.0
                    if getattr(current_page, "parent", None) is not None:
                        pixmap = current_page.get_pixmap(
                            matrix=fitz.Matrix(scale, scale), alpha=False
                        )
                        return pixmap.tobytes("png")
                    with fitz.open(pdf_path) as render_document:
                        render_page = render_document.load_page(page_index)
                        pixmap = render_page.get_pixmap(
                            matrix=fitz.Matrix(scale, scale), alpha=False
                        )
                        return pixmap.tobytes("png")

                yield PageInput(
                    page_number=zero_based_page_number + 1,
                    native_text=native_text,
                    native_blocks=tuple(blocks),
                    image_loader=render,
                    width=page_width,
                    height=page_height,
                    image_coverage=image_coverage,
                    table_count=table_count,
                    is_multi_column=is_multi_column,
                    metadata={
                        "source_pdf": str(self.path),
                        "drawing_count": drawing_count,
                        "embedded_image_count": embedded_image_count,
                        "annotation_count": annotation_count,
                        "link_count": link_count,
                        "source_object_inventory_complete": True,
                    },
                )


def _estimate_image_coverage(
    raw_blocks: Sequence[Sequence[Any]], page_width: float, page_height: float
) -> float:
    page_area = max(1.0, page_width * page_height)
    image_area = 0.0
    for block in raw_blocks:
        if len(block) <= 6:
            continue
        try:
            is_image = int(block[6]) != 0
            left, top, right, bottom = (float(item) for item in block[:4])
        except (TypeError, ValueError):
            continue
        if is_image:
            image_area += max(0.0, right - left) * max(0.0, bottom - top)
    return min(1.0, image_area / page_area)


def _estimate_table_count(text: str) -> int:
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) < 3:
        return 0
    tabular_lines = sum(
        bool("\t" in line or re.search(r"\S\s{2,}\S\s{2,}\S", line))
        for line in lines
    )
    return 1 if tabular_lines >= 3 and tabular_lines / len(lines) >= 0.2 else 0


def _estimate_multi_column(blocks: Sequence[NativeTextBlock], page_width: float) -> bool:
    if page_width <= 0 or len(blocks) < 4:
        return False
    left_blocks = [
        block
        for block in blocks
        if block.bbox and block.bbox.right <= page_width * 0.58
    ]
    right_blocks = [
        block
        for block in blocks
        if block.bbox and block.bbox.left >= page_width * 0.42
    ]
    if len(left_blocks) < 2 or len(right_blocks) < 2:
        return False
    for left in left_blocks:
        for right in right_blocks:
            assert left.bbox is not None and right.bbox is not None
            vertical_overlap = min(left.bbox.bottom, right.bbox.bottom) - max(
                left.bbox.top, right.bbox.top
            )
            if vertical_overlap > 0:
                return True
    return False


class ExtractionPipeline:
    """Route and extract pages with optional OCR and guided-JSON parsing."""

    def __init__(
        self,
        *,
        router: PageRouter | None = None,
        parser: NemotronParseClient | None = None,
        ocr: NemotronOCRClient | None = None,
        llm: GuidedJsonClient | None = None,
        pipeline_version: str = PIPELINE_VERSION,
    ) -> None:
        self.router = router or PageRouter()
        self.parser = parser
        self.ocr = ocr
        self.llm = llm
        self.pipeline_version = pipeline_version

    @classmethod
    def from_env(cls, *, router: PageRouter | None = None) -> ExtractionPipeline:
        clients = NvidiaClientBundle.from_env()
        return cls(
            router=router,
            parser=clients.parser,
            ocr=clients.ocr,
            llm=clients.llm,
        )

    def extract_document(
        self,
        path: str | os.PathLike[str],
        *,
        checkpoint: CheckpointStore | None = None,
        on_page_complete: Callable[[PageExtraction], None] | None = None,
        page_source: PageSource | None = None,
    ) -> DocumentExtraction:
        source_path = str(Path(path).resolve())
        document_id = f"sha256:{sha256_file(path)}"
        source = page_source or PyMuPDFPageSource(path)
        return self.extract_pages(
            document_id,
            source.iter_pages(),
            source_path=source_path,
            checkpoint=checkpoint,
            on_page_complete=on_page_complete,
        )

    def extract_pages(
        self,
        document_id: str,
        pages: Iterable[PageInput],
        *,
        source_path: str | None = None,
        checkpoint: CheckpointStore | None = None,
        on_page_complete: Callable[[PageExtraction], None] | None = None,
    ) -> DocumentExtraction:
        completed: list[PageExtraction] = []
        previous_page_number = 0
        for page in pages:
            if page.page_number <= previous_page_number:
                raise ValueError(
                    "pages must be unique and yielded in strictly increasing order"
                )
            previous_page_number = page.page_number
            decision = self.router.decide(
                page.native_text,
                image_coverage=page.image_coverage,
                table_count=page.table_count,
                is_multi_column=page.is_multi_column,
            )
            fingerprint = self._page_fingerprint(document_id, page, decision)
            extracted: PageExtraction | None = None
            if checkpoint is not None:
                extracted = checkpoint.load_page(
                    document_id, page.page_number, fingerprint
                )
            if extracted is None:
                extracted = self.extract_page(
                    document_id,
                    page,
                    decision=decision,
                    fingerprint=fingerprint,
                    source_path=source_path,
                )
                if checkpoint is not None:
                    checkpoint.save_page(extracted)
            completed.append(extracted)
            if on_page_complete is not None:
                on_page_complete(extracted)

        return DocumentExtraction(
            document_id=document_id,
            source_path=source_path,
            pages=tuple(completed),
            pipeline_version=self.pipeline_version,
        )

    def extract_page(
        self,
        document_id: str,
        page: PageInput,
        *,
        decision: RoutingDecision | None = None,
        fingerprint: str | None = None,
        source_path: str | None = None,
    ) -> PageExtraction:
        decision = decision or self.router.decide(
            page.native_text,
            image_coverage=page.image_coverage,
            table_count=page.table_count,
            is_multi_column=page.is_multi_column,
        )
        fingerprint = fingerprint or self._page_fingerprint(
            document_id, page, decision
        )
        elements: list[ExtractedElement] = []
        model_responses: list[Mapping[str, Any]] = []

        if self._is_deterministically_blank(page):
            model_responses.append(
                {
                    "audit_type": "page_extraction",
                    "event": "deterministic_blank_page",
                    "outcome": "empty_page_extraction",
                    "page_number": page.page_number,
                    "routing": decision.to_dict(),
                }
            )
            return PageExtraction(
                document_id=document_id,
                page_number=page.page_number,
                fingerprint=fingerprint,
                routing=decision,
                elements=(),
                model_responses=tuple(model_responses),
            )

        if decision.route is PageRoute.NATIVE_TEXT:
            elements.extend(
                self._native_elements(
                    document_id=document_id,
                    page=page,
                    source_path=source_path,
                    sequence_offset=0,
                )
            )
        else:
            if self.parser is None:
                raise RuntimeError("document VLM route selected but no parser client configured")
            image_bytes = page.get_image_bytes()
            try:
                response = self.parser.parse_page(
                    image_bytes,
                    document_id=document_id,
                    page_number=page.page_number,
                    mime_type=page.image_mime_type,
                )
            except NIMRequestError as exc:
                if not (
                    self._is_unsupported_content_exhaustion(exc)
                    and self._has_meaningful_native_evidence(page)
                ):
                    raise
                error_details = exc.to_dict()
                model_responses.append(
                    {
                        "audit_type": "page_extraction_fallback",
                        "event": "vlm_unsupported_content_native_fallback",
                        "outcome": "fallback_succeeded",
                        "page_number": page.page_number,
                        "source_route": decision.route.value,
                        "fallback_route": PageRoute.NATIVE_TEXT.value,
                        "failure_kind": "unsupported_assistant_content",
                        "ocr_verification_required": decision.requires_ocr_verification,
                        "error_details": error_details,
                    }
                )
                elements.extend(
                    self._native_elements(
                        document_id=document_id,
                        page=page,
                        source_path=source_path,
                        sequence_offset=0,
                        metadata_extra={
                            "vlm_fallback": True,
                            "vlm_failure_kind": "unsupported_assistant_content",
                        },
                    )
                )
            else:
                model_responses.append(response.to_dict())
                elements.extend(
                    _elements_from_model_response(
                        response,
                        document_id=document_id,
                        page_number=page.page_number,
                        extraction_method="nemotron_parse",
                        source_path=source_path,
                        sequence_offset=len(elements),
                    )
                )

        if decision.requires_ocr_verification:
            if self.ocr is None:
                raise RuntimeError("OCR verification selected but no OCR client configured")
            image_bytes = page.get_image_bytes()
            response = self.ocr.ocr_page(
                image_bytes,
                document_id=document_id,
                page_number=page.page_number,
                mime_type=page.image_mime_type,
            )
            model_responses.append(response.to_dict())
            elements.extend(
                _elements_from_model_response(
                    response,
                    document_id=document_id,
                    page_number=page.page_number,
                    extraction_method="nemotron_ocr_verification",
                    source_path=source_path,
                    sequence_offset=len(elements),
                )
            )

        return PageExtraction(
            document_id=document_id,
            page_number=page.page_number,
            fingerprint=fingerprint,
            routing=decision,
            elements=tuple(elements),
            model_responses=tuple(model_responses),
        )

    def parse_structured(
        self,
        extraction: DocumentExtraction,
        *,
        schema: Mapping[str, Any],
        instructions: str | None = None,
        max_tokens: int = 8192,
    ) -> GuidedJsonResult:
        """Parse source evidence into schema-constrained product JSON.

        The LLM receives evidence IDs and page numbers.  Callers should retain
        those IDs in the output schema so every parsed field stays traceable.
        """

        if self.llm is None:
            raise RuntimeError("structured parsing requested but no LLM client configured")
        evidence = [
            {
                "element_id": element.element_id,
                "page_number": element.page_number,
                "type": element.element_type,
                "text": element.text,
                "bbox": element.bbox.to_list() if element.bbox else None,
                "method": element.extraction_method,
                "confidence": element.confidence,
            }
            for element in extraction.elements
        ]
        source_hints = derive_structured_source_hints(extraction)
        user_prompt = (
            PRODUCT_GROUPING_INSTRUCTIONS.strip()
            + "\n\n"
            + source_hints.prompt_text()
        )
        if instructions:
            user_prompt += "\n\nADDITIONAL OUTPUT RULES:\n" + instructions.strip()
        user_prompt += "\n\nSOURCE EVIDENCE (JSON):\n" + _stable_json(evidence)
        return self.llm.generate_json(
            schema=schema,
            # Nemotron Nano's non-reasoning mode requires this exact system text;
            # task instructions belong in the user message.
            system_prompt="detailed thinking off",
            user_prompt=user_prompt,
            source_key=extraction.document_id,
            max_tokens=max_tokens,
        )

    def _native_elements(
        self,
        *,
        document_id: str,
        page: PageInput,
        source_path: str | None,
        sequence_offset: int,
        metadata_extra: Mapping[str, Any] | None = None,
    ) -> list[ExtractedElement]:
        blocks = tuple(block for block in page.native_blocks if block.text.strip())
        if not blocks and page.native_text.strip():
            blocks = (NativeTextBlock(text=page.native_text.strip()),)
        return [
            ExtractedElement.create(
                document_id=document_id,
                page_number=page.page_number,
                sequence_number=sequence_offset + index,
                element_type=block.element_type,
                text=block.text,
                extraction_method="pdf_native_text",
                bbox=block.bbox,
                confidence=1.0,
                source_path=source_path,
                raw_payload={"text": block.text},
                metadata={**dict(block.metadata), **dict(metadata_extra or {})},
            )
            for index, block in enumerate(blocks)
        ]

    @staticmethod
    def _is_unsupported_content_exhaustion(exc: NIMRequestError) -> bool:
        failures = exc.response.get("successful_http_failures")
        return (
            exc.status_code == 200
            and exc.response.get("failure_kind") == "unsupported_assistant_content"
            and isinstance(failures, list)
            and bool(failures)
        )

    def _has_meaningful_native_evidence(self, page: PageInput) -> bool:
        candidates = [page.native_text]
        if page.native_blocks:
            candidates.append(
                "\n".join(block.text for block in page.native_blocks if block.text.strip())
            )
        return any(
            candidate.strip()
            and self.router.decide(
                candidate,
                image_coverage=0.0,
                table_count=0,
                is_multi_column=False,
            ).route
            is PageRoute.NATIVE_TEXT
            for candidate in candidates
        )

    @staticmethod
    def _is_deterministically_blank(page: PageInput) -> bool:
        def verified_zero_count(key: str) -> bool:
            value = page.metadata.get(key)
            return isinstance(value, int) and not isinstance(value, bool) and value == 0

        return (
            page.metadata.get("source_object_inventory_complete") is True
            and not page.native_text.strip()
            and not page.native_blocks
            and page.table_count == 0
            and page.image_coverage == 0.0
            and not page.is_multi_column
            and page.image_bytes is None
            and verified_zero_count("drawing_count")
            and verified_zero_count("embedded_image_count")
            and verified_zero_count("annotation_count")
            and verified_zero_count("link_count")
        )

    def _page_fingerprint(
        self,
        document_id: str,
        page: PageInput,
        decision: RoutingDecision,
    ) -> str:
        parser_config = (
            self.parser.config.safe_dict() if self.parser is not None else None
        )
        ocr_config = self.ocr.config.safe_dict() if self.ocr is not None else None
        image_hash = _stable_hash(page.image_bytes) if page.image_bytes is not None else None
        return _stable_hash(
            self.pipeline_version,
            document_id,
            page.page_number,
            page.native_text,
            [
                {
                    "text": block.text,
                    "bbox": block.bbox.to_list() if block.bbox else None,
                    "type": block.element_type,
                }
                for block in page.native_blocks
            ],
            page.width,
            page.height,
            page.metadata,
            image_hash,
            decision.to_dict(),
            parser_config,
            ocr_config,
        )


PRODUCT_GROUPING_INSTRUCTIONS = """
Treat the complete SOURCE EVIDENCE array as one catalog-page context. Evidence
elements are fragments of that page, not separate products.

PRODUCT ELIGIBILITY RULES:
1. Emit a product only when the evidence explicitly supplies both an exact
   manufacturer name and an exact product or part name. Copy their raw text;
   never infer either identity from a domain, logo image, person, or context.
   For the manufacturer of an otherwise eligible product, prefer an explicit
   legal company or brand line, or a copyright/footer attribution, over a
   possessive marketing phrase. Preserve the selected source spelling. A
   copyright/footer may support that product's manufacturer field, but it must
   never create a separate product record.
2. Never create products from page headers, footers, navigation, contact names,
   addresses, phone numbers, legal notices, slogans, section headings, generic
   body paragraphs, or other fragments that do not identify an orderable product.
3. Group evidence fragments that describe the same product into one product.
   Emit multiple products only when explicit distinct orderable part numbers,
   model identifiers, or clearly separate product rows identify them. Do not
   create one product per evidence element, paragraph, heading, or table cell.
4. If the page cannot satisfy these rules, return exactly {"products":[]}.

FIELD AND EVIDENCE RULES:
5. Preserve exact source strings, especially manufacturer, part name, model, and
   part number. Never guess, complete, normalize, or combine identifier text.
6. Every evidence reference must copy an element_id supplied in SOURCE EVIDENCE
   character-for-character. Reference only elements that directly support that
   specific field; never invent an ID.
7. Emit a specification only when the evidence explicitly states a key/value
   relationship. Keep its name and value short and exact. Do not convert prose,
   headings, contact details, or unrelated numbers into specifications.
8. Omit unsupported optional facts. Output only the JSON object; no prose.
"""


_NEMOTRON_PARSE_ELEMENT_RE = re.compile(
    r"<x_(?P<x0>-?\d+(?:\.\d+)?)><y_(?P<y0>-?\d+(?:\.\d+)?)>"
    r"(?P<text>.*?)"
    r"<x_(?P<x1>-?\d+(?:\.\d+)?)><y_(?P<y1>-?\d+(?:\.\d+)?)>"
    r"<class_(?P<element_type>[^>]+)>",
    re.DOTALL,
)


def _nemotron_parse_tagged_elements(content: str) -> list[dict[str, Any]]:
    """Decode Nemotron Parse's grounded class/bbox/text wire format.

    Parse v1.2 emits normalized ``x/y`` tags around each element followed by a
    semantic class tag. Keeping those blocks separate gives the downstream LLM
    compact evidence IDs and exact page regions rather than one page-sized blob.
    """

    elements: list[dict[str, Any]] = []
    for match in _NEMOTRON_PARSE_ELEMENT_RE.finditer(content):
        text = match.group("text").strip()
        if not text:
            continue
        bbox = [
            float(match.group("x0")),
            float(match.group("y0")),
            float(match.group("x1")),
            float(match.group("y1")),
        ]
        elements.append(
            {
                "type": match.group("element_type").strip(),
                "text": text,
                "bbox": bbox,
                "bbox_coordinate_space": "normalized",
                "raw_tagged_block": match.group(0),
            }
        )
    return elements


def _elements_from_model_response(
    response: ModelResponse,
    *,
    document_id: str,
    page_number: int,
    extraction_method: str,
    source_path: str | None,
    sequence_offset: int,
) -> list[ExtractedElement]:
    content = response.content.strip()
    if not content:
        return []
    try:
        parsed = parse_json_content(content)
    except ValueError:
        tagged_elements = _nemotron_parse_tagged_elements(content)
        parsed = {
            "elements": tagged_elements
            or [{"type": "document", "text": content}]
        }

    raw_elements: Any
    if isinstance(parsed, Mapping):
        raw_elements = parsed.get("elements")
        if raw_elements is None and "text" in parsed:
            raw_elements = [parsed]
    elif isinstance(parsed, list):
        raw_elements = parsed
    else:
        raw_elements = []

    if not isinstance(raw_elements, list):
        raw_elements = []
    elements: list[ExtractedElement] = []
    for raw in raw_elements:
        if isinstance(raw, str):
            raw = {"type": "text", "text": raw}
        if not isinstance(raw, Mapping):
            continue
        text = _model_element_text(raw)
        if not text.strip():
            continue
        bbox = BoundingBox.from_value(
            raw.get("bbox") or raw.get("bounding_box") or raw.get("box")
        )
        confidence = _optional_float(raw.get("confidence"))
        elements.append(
            ExtractedElement.create(
                document_id=document_id,
                page_number=page_number,
                sequence_number=sequence_offset + len(elements),
                element_type=str(raw.get("type") or raw.get("element_type") or "text"),
                text=text,
                extraction_method=extraction_method,
                bbox=bbox,
                confidence=confidence,
                model=response.model,
                source_path=source_path,
                raw_payload=dict(raw),
                metadata={
                    "request_id": response.request_id,
                    "dry_run": response.dry_run,
                    **(
                        {"bbox_coordinate_space": raw["bbox_coordinate_space"]}
                        if raw.get("bbox_coordinate_space")
                        else {}
                    ),
                },
            )
        )
    return elements


def _model_element_text(raw: Mapping[str, Any]) -> str:
    for key in ("text", "content", "markdown", "html", "value"):
        value = raw.get(key)
        if isinstance(value, str):
            return value
        if isinstance(value, (list, dict)):
            return _stable_json(value)
    return ""


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


__all__ = [
    "BoundingBox",
    "CheckpointStore",
    "DocumentExtraction",
    "ExtractedElement",
    "ExtractionPipeline",
    "JsonPageCheckpointStore",
    "ManufacturerCandidate",
    "MemoryCheckpointStore",
    "NativeTextBlock",
    "PIPELINE_VERSION",
    "PageExtraction",
    "PageInput",
    "PageSource",
    "PRODUCT_GROUPING_INSTRUCTIONS",
    "ProductCardinalityDecision",
    "PyMuPDFPageSource",
    "StructuredSourceHints",
    "derive_structured_source_hints",
    "manufacturer_candidates_from_extraction",
    "product_cardinality_from_extraction",
    "sha256_file",
]
