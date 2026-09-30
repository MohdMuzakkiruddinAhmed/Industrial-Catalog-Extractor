"""Bounded, resumable benchmark orchestration for the extraction pipeline."""

from __future__ import annotations

import itertools
import json
import os
import re
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from .checkpoints import Checkpoint
from .checkpoints import CheckpointStore as DocumentCheckpointStore
from .config import EndpointConfig, PipelineConfig, ensure_runtime_directories
from .extraction import (
    DocumentExtraction,
    ExtractionPipeline,
    JsonPageCheckpointStore,
    PyMuPDFPageSource,
    StructuredSourceHints,
    derive_structured_source_hints,
    sha256_file,
)
from .knowledge_base import NvidiaEmbeddingClient, SQLiteKnowledgeBase
from .models import SourceDocument, product_batch_guided_json_schema
from .nvidia_clients import (
    GuidedJsonClient,
    GuidedJsonDecodeError,
    GuidedJsonMode,
    NemotronOCRClient,
    NemotronParseClient,
    NIMEndpointConfig,
    RetryPolicy,
)
from .routing import PageRouter, RoutingConfig
from .storage import SQLiteCatalogStore


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _json_default(value: Any) -> Any:
    if isinstance(value, (Path, datetime)):
        return str(value)
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _explicit_error_details(exc: Exception) -> dict[str, Any] | None:
    """Return only an exception's explicit, strictly JSON-safe audit payload."""

    to_dict = getattr(exc, "to_dict", None)
    if not callable(to_dict):
        return None
    try:
        raw_details = to_dict()
    except Exception:
        return None
    if not isinstance(raw_details, Mapping):
        return None
    try:
        encoded = json.dumps(raw_details, ensure_ascii=False, allow_nan=False)
        decoded = json.loads(encoded)
    except (OverflowError, TypeError, ValueError):
        return None
    return dict(decoded) if isinstance(decoded, Mapping) else None


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_name = stream.name
            json.dump(payload, stream, ensure_ascii=False, indent=2, default=_json_default)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name:
            Path(temporary_name).unlink(missing_ok=True)


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=_json_default,
            )
        )
        stream.write("\n")
        stream.flush()


@dataclass(frozen=True, slots=True)
class BenchmarkLimits:
    max_documents: int | None = None
    max_pages: int | None = None
    max_pages_per_document: int | None = None
    pages_per_parse: int = 1

    def __post_init__(self) -> None:
        for name in ("max_documents", "max_pages", "max_pages_per_document"):
            value = getattr(self, name)
            if value is not None and value < 1:
                raise ValueError(f"{name} must be positive")
        if self.pages_per_parse < 1 or self.pages_per_parse > 100:
            raise ValueError("pages_per_parse must be between 1 and 100")


@dataclass(frozen=True, slots=True)
class BenchmarkOptions:
    config: PipelineConfig
    output_dir: Path
    limits: BenchmarkLimits = BenchmarkLimits()
    include_globs: tuple[str, ...] = ("**/*.pdf",)
    dry_run: bool | None = None
    strict_validation: bool = False
    fail_fast: bool = False
    run_id: str | None = None


@dataclass(frozen=True, slots=True)
class BenchmarkRunResult:
    summary: Mapping[str, Any]
    output_dir: Path

    @property
    def exit_code(self) -> int:
        return 1 if int(self.summary.get("documents_failed", 0)) else 0


def _nim_config(
    endpoint: EndpointConfig | None,
    *,
    default_model: str,
    dry_run: bool,
) -> NIMEndpointConfig | None:
    if endpoint is None:
        if not dry_run:
            return None
        return NIMEndpointConfig(base_url="", model=default_model, dry_run=True)
    return NIMEndpointConfig(
        base_url=endpoint.base_url,
        model=endpoint.model,
        api_key_env=endpoint.api_key_env or "NVIDIA_API_KEY",
        timeout_seconds=endpoint.timeout_seconds,
        retry=RetryPolicy(max_attempts=endpoint.max_retries + 1),
        dry_run=dry_run,
    )


def build_extraction_pipeline(config: PipelineConfig, *, dry_run: bool) -> ExtractionPipeline:
    parse_config = _nim_config(
        config.parse_endpoint,
        default_model="nvidia/NVIDIA-Nemotron-Parse-v1.2",
        dry_run=dry_run,
    )
    ocr_config = _nim_config(
        config.ocr_endpoint,
        default_model="nvidia/nemotron-ocr",
        dry_run=dry_run,
    )
    llm_config = _nim_config(
        config.llm_endpoint,
        default_model="nvidia/Llama-3.1-Nemotron-Nano-8B-v1",
        dry_run=dry_run,
    )
    if parse_config is None or llm_config is None:
        raise ValueError("live benchmark requires parse_endpoint and llm_endpoint")

    # Public/self-hosted vLLM servers expose OpenAI response_format rather than
    # NVIDIA's hosted nvext wrapper.
    llm = GuidedJsonClient(
        llm_config,
        mode=GuidedJsonMode.OPENAI_RESPONSE_FORMAT,
        serialize_nvext_schema=False,
    )
    router = PageRouter(
        RoutingConfig(
            min_native_characters=config.routing.native_min_chars,
            use_vlm_for_tables=config.routing.verify_tables,
            verify_vlm_with_ocr=ocr_config is not None,
            verify_native_identifiers_with_ocr=(
                config.routing.verify_identifier_regions and ocr_config is not None
            ),
        )
    )
    return ExtractionPipeline(
        router=router,
        parser=NemotronParseClient(parse_config),
        ocr=NemotronOCRClient(ocr_config) if ocr_config is not None else None,
        llm=llm,
    )


def product_batch_schema_for_hints(hints: StructuredSourceHints) -> dict[str, Any]:
    """Apply source-backed manufacturer/cardinality constraints to the wire schema."""

    schema = json.loads(json.dumps(product_batch_guided_json_schema()))
    products_schema = schema["properties"]["products"]
    product_schema = products_schema["items"]
    properties = product_schema["properties"]

    if hints.cardinality.mode == "single_family":
        products_schema["maxItems"] = 1
        # A code-like token in prose is not enough to claim an orderable part
        # number. Only explicit part/model table rows unlock this property.
        properties.pop("part_number", None)
    else:
        identifiers = list(hints.cardinality.identifiers)
        identifier_count = len(identifiers)
        products_schema["minItems"] = identifier_count
        products_schema["maxItems"] = identifier_count
        identity_fields = {"manufacturer", "part_name", "part_number"}
        for field_name in tuple(properties):
            if field_name not in identity_fields:
                properties.pop(field_name)
        product_schema["required"] = [
            "manufacturer",
            "part_name",
            "part_number",
        ]
        part_number_schema = properties["part_number"]
        part_number_schema["properties"]["raw"] = {
            "type": "string",
            "enum": identifiers,
        }
        part_number_schema["properties"]["evidence"]["items"]["properties"]["element_id"][
            "enum"
        ] = list(hints.cardinality.evidence_element_ids)

    if hints.manufacturer_candidates:
        manufacturer_schema = properties["manufacturer"]
        raw_values = list(
            dict.fromkeys(candidate.raw for candidate in hints.manufacturer_candidates)
        )
        element_ids = list(
            dict.fromkeys(candidate.element_id for candidate in hints.manufacturer_candidates)
        )
        manufacturer_schema["properties"]["raw"] = {
            "type": "string",
            "enum": raw_values,
        }
        manufacturer_schema["properties"]["evidence"]["items"]["properties"]["element_id"][
            "enum"
        ] = element_ids
    return schema


_IDENTITY_ONLY_PRODUCT_FIELDS = frozenset({"manufacturer", "part_name", "part_number"})
_IDENTITY_ROW_REPAIR_POLICY_VERSION = "identity-row-repair-v2"
_IDENTIFIER_TOKEN_CHARACTER_CLASS = r"A-Za-z0-9._/\-"
_MAX_CONCISE_PRODUCT_TITLE_CHARACTERS = 120
_MAX_CONCISE_PRODUCT_TITLE_WORDS = 12
_MARKDOWN_HEADING_PREFIX_RE = re.compile(r"^\s{0,3}#{1,6}\s+")
_MARKDOWN_HEADING_SUFFIX_RE = re.compile(r"\s+#{1,6}\s*$")
_HTML_HEADING_WRAPPER_RE = re.compile(
    r"^\s*<h[1-6]\b[^>]*>(?P<text>.*?)</h[1-6]>\s*$",
    re.IGNORECASE | re.DOTALL,
)
_HTML_INLINE_WRAPPER_RE = re.compile(
    r"^\s*<(?P<tag>strong|b|em|i)\b[^>]*>"
    r"(?P<text>.*?)</(?P=tag)>\s*$",
    re.IGNORECASE | re.DOTALL,
)
_GENERIC_DOCUMENT_HEADINGS = frozenset(
    {
        "applications",
        "benefits",
        "catalog",
        "contents",
        "data sheet",
        "datasheet",
        "description",
        "features",
        "general information",
        "installation",
        "installation instructions",
        "instructions",
        "manual",
        "ordering information",
        "overview",
        "part numbers",
        "product catalog",
        "product information",
        "product overview",
        "products",
        "specification",
        "specifications",
        "table of contents",
        "technical data",
        "technical datasheet",
    }
)
_GENERIC_DOCUMENT_HEADING_SUFFIXES = (
    " catalog",
    " data sheet",
    " datasheet",
    " installation instructions",
    " ordering information",
    " product catalog",
    " product information",
    " product overview",
    " specifications",
    " technical data",
    " technical datasheet",
)


def _contains_exact_identifier(text: str, identifier: str) -> bool:
    pattern = (
        rf"(?<![{_IDENTIFIER_TOKEN_CHARACTER_CLASS}])"
        rf"{re.escape(identifier)}"
        rf"(?![{_IDENTIFIER_TOKEN_CHARACTER_CLASS}])"
    )
    return re.search(pattern, text) is not None


def _presentation_trimmed_element_text(text: str) -> str:
    """Remove only whole-element presentation wrappers from source text."""

    candidate = text.strip()
    html_heading = _HTML_HEADING_WRAPPER_RE.fullmatch(candidate)
    if html_heading is not None:
        candidate = html_heading.group("text").strip()
    inline_wrapper = _HTML_INLINE_WRAPPER_RE.fullmatch(candidate)
    if inline_wrapper is not None:
        candidate = inline_wrapper.group("text").strip()
    candidate = _MARKDOWN_HEADING_PREFIX_RE.sub("", candidate)
    candidate = _MARKDOWN_HEADING_SUFFIX_RE.sub("", candidate).strip()
    wrappers = (("**", "**"), ("__", "__"), ("*", "*"), ("_", "_"), ("`", "`"))
    changed = True
    while changed:
        changed = False
        for prefix, suffix in wrappers:
            if (
                candidate.startswith(prefix)
                and candidate.endswith(suffix)
                and len(candidate) > len(prefix) + len(suffix)
            ):
                candidate = candidate[len(prefix) : -len(suffix)].strip()
                changed = True
                break
    return candidate


def _is_concise_non_generic_product_title(value: str) -> bool:
    if (
        len(value) < 3
        or len(value) > _MAX_CONCISE_PRODUCT_TITLE_CHARACTERS
        or "\n" in value
        or not any(character.isalpha() for character in value)
        or value.rstrip().endswith((".", ";", "?", "!"))
    ):
        return False
    words = re.findall(r"[A-Za-z0-9]+(?:[._/+&-][A-Za-z0-9]+)*", value)
    if not words or len(words) > _MAX_CONCISE_PRODUCT_TITLE_WORDS:
        return False
    normalized = re.sub(
        r"^[ .:|_-]+|[ .:|_-]+$",
        "",
        " ".join(value.casefold().split()),
    )
    if normalized in _GENERIC_DOCUMENT_HEADINGS:
        return False
    if normalized.startswith(("chapter ", "section ")):
        return False
    return not any(normalized.endswith(suffix) for suffix in _GENERIC_DOCUMENT_HEADING_SUFFIXES)


def _strict_identity_value(
    value: Any,
) -> tuple[str, tuple[str, ...]] | None:
    """Decode the compact raw/evidence wire shape without normalizing it."""

    if not isinstance(value, Mapping) or set(value) != {"raw", "evidence"}:
        return None
    raw = value.get("raw")
    evidence = value.get("evidence")
    if not isinstance(raw, str) or not raw.strip() or not isinstance(evidence, list):
        return None
    element_ids: list[str] = []
    for reference in evidence:
        if not isinstance(reference, Mapping) or set(reference) != {"element_id"}:
            return None
        element_id = reference.get("element_id")
        if not isinstance(element_id, str) or not element_id:
            return None
        element_ids.append(element_id)
    if not element_ids or len(element_ids) != len(set(element_ids)):
        return None
    return raw, tuple(element_ids)


def _identity_row_repair_failure(
    *,
    expected_identifiers: list[str],
    emitted_identifiers: list[str | None],
    reason: str,
) -> dict[str, Any]:
    return {
        "policy_version": _IDENTITY_ROW_REPAIR_POLICY_VERSION,
        "attempted": True,
        "applied": False,
        "expected_identifiers": expected_identifiers,
        "emitted_identifiers": emitted_identifiers,
        "failed_precondition": reason,
    }


def _repair_identity_only_multi_product_rows(
    retained: list[dict[str, Any]],
    *,
    original_product_count: int,
    prior_dropped_count: int,
    hints: StructuredSourceHints,
    extraction: DocumentExtraction | None,
    candidates_by_raw: Mapping[str, set[str]],
) -> tuple[list[dict[str, Any]] | None, dict[str, Any], list[dict[str, Any]]]:
    """Repair only a duplicate/missing identifier permutation proven by source.

    No name or manufacturer is inferred. The only reconstructed values are the
    exact identifiers already derived from explicit source rows, with citations
    to the exact cardinality elements containing them.
    """

    expected_identifiers = list(hints.cardinality.identifiers)
    emitted_identifiers: list[str | None] = []
    for product in retained:
        parsed = _strict_identity_value(product.get("part_number"))
        emitted_identifiers.append(parsed[0] if parsed is not None else None)

    def fail(
        reason: str,
    ) -> tuple[None, dict[str, Any], list[dict[str, Any]]]:
        return (
            None,
            _identity_row_repair_failure(
                expected_identifiers=expected_identifiers,
                emitted_identifiers=emitted_identifiers,
                reason=reason,
            ),
            [],
        )

    if extraction is None:
        return fail("document_extraction_not_available")
    if (
        not expected_identifiers
        or any(not identifier for identifier in expected_identifiers)
        or len(expected_identifiers) != len(set(expected_identifiers))
    ):
        return fail("expected_identifiers_not_nonempty_and_unique")
    cardinality_element_ids = list(hints.cardinality.evidence_element_ids)
    if not cardinality_element_ids or len(cardinality_element_ids) != len(
        set(cardinality_element_ids)
    ):
        return fail("cardinality_evidence_ids_not_nonempty_and_unique")

    elements_by_id: dict[str, Any] = {}
    for element in extraction.elements:
        if element.element_id in elements_by_id:
            return fail("extraction_contains_duplicate_element_ids")
        elements_by_id[element.element_id] = element
    if any(element_id not in elements_by_id for element_id in cardinality_element_ids):
        return fail("cardinality_evidence_element_not_found")

    source_element_by_identifier: dict[str, str] = {}
    for identifier in expected_identifiers:
        matching_element_ids = [
            element_id
            for element_id in cardinality_element_ids
            if _contains_exact_identifier(
                elements_by_id[element_id].text,
                identifier,
            )
        ]
        if not matching_element_ids:
            return fail("expected_identifier_not_exactly_source_backed")
        source_element_by_identifier[identifier] = matching_element_ids[0]

    if prior_dropped_count or original_product_count != len(retained):
        return fail("products_were_rejected_before_identifier_repair")
    if len(retained) != len(expected_identifiers):
        return fail("emitted_product_count_does_not_match_expected_count")
    if any(set(product) != _IDENTITY_ONLY_PRODUCT_FIELDS for product in retained):
        return fail("product_is_not_identity_only")

    parsed_rows: list[
        tuple[
            tuple[str, tuple[str, ...]],
            tuple[str, tuple[str, ...]],
            tuple[str, tuple[str, ...]],
        ]
    ] = []
    for product in retained:
        manufacturer = _strict_identity_value(product.get("manufacturer"))
        part_name = _strict_identity_value(product.get("part_name"))
        part_number = _strict_identity_value(product.get("part_number"))
        if manufacturer is None or part_name is None or part_number is None:
            return fail("identity_field_does_not_match_strict_raw_evidence_shape")
        parsed_rows.append((manufacturer, part_name, part_number))

    emitted_identifiers = [row[2][0] for row in parsed_rows]
    if any(identifier not in expected_identifiers for identifier in emitted_identifiers):
        return fail("emitted_part_number_is_not_expected")

    cardinality_element_id_set = set(cardinality_element_ids)
    for _, _, (part_number_raw, evidence_ids) in parsed_rows:
        if any(element_id not in cardinality_element_id_set for element_id in evidence_ids):
            return fail("part_number_evidence_is_not_from_cardinality_elements")
        if any(
            not _contains_exact_identifier(
                elements_by_id[element_id].text,
                part_number_raw,
            )
            for element_id in evidence_ids
        ):
            return fail("part_number_evidence_does_not_contain_exact_identifier")

    manufacturer_values = {row[0][0] for row in parsed_rows}
    if len(manufacturer_values) != 1:
        return fail("manufacturer_identity_is_not_consistent")
    manufacturer_raw = next(iter(manufacturer_values))
    allowed_manufacturer_ids = candidates_by_raw.get(manufacturer_raw)
    if not allowed_manufacturer_ids:
        return fail("manufacturer_is_not_a_deterministic_source_candidate")
    for (raw, evidence_ids), _, _ in parsed_rows:
        if not set(evidence_ids).intersection(allowed_manufacturer_ids):
            return fail("manufacturer_candidate_evidence_is_not_cited")
        if any(
            element_id not in elements_by_id or raw not in elements_by_id[element_id].text
            for element_id in evidence_ids
        ):
            return fail("manufacturer_identity_is_not_exactly_source_backed")

    for _, (raw, evidence_ids), _ in parsed_rows:
        if any(
            element_id not in elements_by_id or raw not in elements_by_id[element_id].text
            for element_id in evidence_ids
        ):
            return fail("part_name_identity_is_not_exactly_source_backed")

    part_name_values = {row[1][0] for row in parsed_rows}
    part_name_source_element_id: str | None = None
    if len(part_name_values) == 1:
        part_name_raw = next(iter(part_name_values))
        part_name_selection_mode = "identical_across_emitted_rows"
        canonical_part_name = deepcopy(retained[0]["part_name"])
    else:
        concise_candidates: dict[str, set[str]] = {}
        for _, (raw, evidence_ids), _ in parsed_rows:
            for element_id in evidence_ids:
                if _presentation_trimmed_element_text(
                    elements_by_id[element_id].text
                ) == raw and _is_concise_non_generic_product_title(raw):
                    concise_candidates.setdefault(raw, set()).add(element_id)
        if not concise_candidates:
            return fail("no_concise_standalone_part_name_candidate")
        if len(concise_candidates) > 1:
            return fail("multiple_concise_standalone_part_name_candidates")
        part_name_raw, title_element_ids = next(iter(concise_candidates.items()))
        part_name_source_element_id = sorted(title_element_ids)[0]
        part_name_selection_mode = "unique_concise_standalone_source_title"
        canonical_part_name = {
            "raw": part_name_raw,
            "evidence": [{"element_id": part_name_source_element_id}],
        }

    canonical_manufacturer = deepcopy(retained[0]["manufacturer"])
    repaired = [
        {
            "manufacturer": deepcopy(canonical_manufacturer),
            "part_name": deepcopy(canonical_part_name),
            "part_number": {
                "raw": identifier,
                "evidence": [{"element_id": source_element_by_identifier[identifier]}],
            },
        }
        for identifier in expected_identifiers
    ]
    repair_audit = {
        "policy_version": _IDENTITY_ROW_REPAIR_POLICY_VERSION,
        "attempted": True,
        "applied": True,
        "expected_identifiers": expected_identifiers,
        "emitted_identifiers": emitted_identifiers,
        "manufacturer_raw": manufacturer_raw,
        "part_name_raw": part_name_raw,
        "part_name_values_seen": sorted(part_name_values),
        "part_name_selection_mode": part_name_selection_mode,
        "part_name_source_element_id": part_name_source_element_id,
        "source_element_by_identifier": source_element_by_identifier,
    }
    actions: list[dict[str, Any]] = [
        {
            "action": "repaired_multi_product_identifier_set",
            "policy_version": _IDENTITY_ROW_REPAIR_POLICY_VERSION,
            "emitted_identifiers": emitted_identifiers,
            "repaired_identifiers": expected_identifiers,
            "part_name_raw": part_name_raw,
            "part_name_selection_mode": part_name_selection_mode,
            "part_name_source_element_id": part_name_source_element_id,
        }
    ]
    actions.extend(
        {
            "action": "rebuilt_identity_only_product",
            "product_index": index,
            "part_number_raw": identifier,
            "part_number_evidence_element_id": source_element_by_identifier[identifier],
        }
        for index, identifier in enumerate(expected_identifiers)
    )
    return repaired, repair_audit, actions


def enforce_structured_source_hints(
    data: Any,
    hints: StructuredSourceHints,
    *,
    extraction: DocumentExtraction | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fail closed when a model response violates deterministic source hints.

    Guided JSON is the first line of defense, but not every inference server
    enforces every enum or collection bound. The sole repair path reconstructs
    an identity-only row set from exact identifiers already proven by source
    tables; every other unsupported response remains fail closed.
    """

    if not isinstance(data, Mapping):
        raise ValueError("guided ProductBatch must be a JSON object")
    guarded = deepcopy(dict(data))
    products = guarded.get("products")
    if not isinstance(products, list):
        raise ValueError("guided ProductBatch products must be a list")

    candidates_by_raw: dict[str, set[str]] = {}
    for candidate in hints.manufacturer_candidates:
        candidates_by_raw.setdefault(candidate.raw, set()).add(candidate.element_id)

    retained: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    for index, raw_product in enumerate(products):
        if not isinstance(raw_product, Mapping):
            dropped.append(
                {
                    "product_index": index,
                    "reason": "product_is_not_an_object",
                    "product": raw_product,
                }
            )
            continue

        product = deepcopy(dict(raw_product))
        manufacturer = product.get("manufacturer")
        manufacturer_raw = manufacturer.get("raw") if isinstance(manufacturer, Mapping) else None
        if not isinstance(manufacturer_raw, str) or not manufacturer_raw.strip():
            dropped.append(
                {
                    "product_index": index,
                    "reason": "missing_manufacturer_identity",
                    "product": raw_product,
                }
            )
            continue

        if candidates_by_raw:
            allowed_ids = candidates_by_raw.get(manufacturer_raw)
            if allowed_ids is None:
                dropped.append(
                    {
                        "product_index": index,
                        "reason": "manufacturer_not_in_source_candidates",
                        "manufacturer_raw": manufacturer_raw,
                        "product": raw_product,
                    }
                )
                continue
            evidence = manufacturer.get("evidence")
            cited_ids = (
                {
                    item.get("element_id")
                    for item in evidence
                    if isinstance(item, Mapping) and isinstance(item.get("element_id"), str)
                }
                if isinstance(evidence, list)
                else set()
            )
            if not cited_ids.intersection(allowed_ids):
                dropped.append(
                    {
                        "product_index": index,
                        "reason": "manufacturer_candidate_evidence_not_cited",
                        "manufacturer_raw": manufacturer_raw,
                        "required_element_ids": sorted(allowed_ids),
                        "cited_element_ids": sorted(cited_ids),
                        "product": raw_product,
                    }
                )
                continue

        part_name = product.get("part_name")
        part_name_raw = part_name.get("raw") if isinstance(part_name, Mapping) else None
        if not isinstance(part_name_raw, str) or not part_name_raw.strip():
            dropped.append(
                {
                    "product_index": index,
                    "reason": "missing_part_name_identity",
                    "product": raw_product,
                }
            )
            continue

        if hints.cardinality.mode == "single_family" and "part_number" in product:
            removed = product.pop("part_number")
            actions.append(
                {
                    "product_index": index,
                    "action": "removed_unsupported_part_number",
                    "removed_value": removed,
                }
            )
        retained.append(product)

    maximum_products = (
        1 if hints.cardinality.mode == "single_family" else len(hints.cardinality.identifiers)
    )
    cardinality_violation: dict[str, Any] | None = None
    identity_row_repair: dict[str, Any] | None = None
    if hints.cardinality.mode == "multi_product_rows":
        expected_identifiers = list(hints.cardinality.identifiers)
        emitted_identifiers: list[str | None] = []
        for product in retained:
            part_number = product.get("part_number")
            part_number_raw = part_number.get("raw") if isinstance(part_number, Mapping) else None
            emitted_identifiers.append(
                part_number_raw if isinstance(part_number_raw, str) else None
            )
        emitted_strings = [
            identifier for identifier in emitted_identifiers if identifier is not None
        ]
        emitted_counts = Counter(emitted_strings)
        duplicate_identifiers = list(
            dict.fromkeys(
                identifier for identifier in emitted_strings if emitted_counts[identifier] > 1
            )
        )
        missing_identifiers = [
            identifier for identifier in expected_identifiers if identifier not in emitted_counts
        ]
        unexpected_identifiers = list(
            dict.fromkeys(
                identifier
                for identifier in emitted_strings
                if identifier not in expected_identifiers
            )
        )
        missing_part_number_indexes = [
            index for index, identifier in enumerate(emitted_identifiers) if identifier is None
        ]
        complete_identifier_set = (
            not missing_part_number_indexes
            and not duplicate_identifiers
            and not missing_identifiers
            and not unexpected_identifiers
            and len(emitted_strings) == len(expected_identifiers)
        )
        if not complete_identifier_set:
            cardinality_violation = {
                "reason": "multi_product_identifier_set_mismatch",
                "expected_identifiers": expected_identifiers,
                "emitted_identifiers": emitted_identifiers,
                "missing_identifiers": missing_identifiers,
                "duplicate_identifiers": duplicate_identifiers,
                "unexpected_identifiers": unexpected_identifiers,
                "missing_part_number_product_indexes": missing_part_number_indexes,
            }
            repaired, identity_row_repair, repair_actions = (
                _repair_identity_only_multi_product_rows(
                    retained,
                    original_product_count=len(products),
                    prior_dropped_count=len(dropped),
                    hints=hints,
                    extraction=extraction,
                    candidates_by_raw=candidates_by_raw,
                )
            )
            if repaired is not None:
                retained = repaired
                actions.extend(repair_actions)
                cardinality_violation["resolved_by"] = "source_backed_identity_row_repair"
            else:
                for index, product in enumerate(retained):
                    dropped.append(
                        {
                            "product_index": index,
                            "reason": ("incomplete_or_non_unique_multi_product_identifier_set"),
                            "product": product,
                        }
                    )
                retained = []
    elif len(retained) > maximum_products:
        cardinality_violation = {
            "reason": "response_exceeds_source_backed_product_cardinality",
            "maximum_products": maximum_products,
            "retained_before_cardinality_guard": len(retained),
        }
        for index, product in enumerate(retained):
            dropped.append(
                {
                    "product_index": index,
                    "reason": "ambiguous_excess_product_set",
                    "product": product,
                }
            )
        retained = []

    guarded["products"] = retained
    audit = {
        "policy_version": "source-guard-v3",
        "source_hints": hints.to_dict(),
        "candidate_manufacturers": [
            {
                "raw": raw,
                "element_ids": sorted(element_ids),
            }
            for raw, element_ids in candidates_by_raw.items()
        ],
        "maximum_products": maximum_products,
        "original_product_count": len(products),
        "final_product_count": len(retained),
        "dropped_product_count": len(products) - len(retained),
        "dropped_products": dropped,
        "actions": actions,
        "cardinality_violation": cardinality_violation,
        "identity_row_repair": identity_row_repair,
    }
    return guarded, audit


def build_embedding_client(
    config: PipelineConfig,
    *,
    dry_run: bool,
) -> NvidiaEmbeddingClient | None:
    endpoint = config.embedding_endpoint
    if dry_run or endpoint is None:
        return None
    nim_config = NIMEndpointConfig(
        base_url=endpoint.base_url,
        model=endpoint.model,
        endpoint_path="/v1/embeddings",
        api_key_env=endpoint.api_key_env or "NVIDIA_API_KEY",
        timeout_seconds=endpoint.timeout_seconds,
        retry=RetryPolicy(max_attempts=endpoint.max_retries + 1),
    )
    return NvidiaEmbeddingClient(nim_config)


def discover_pdfs(corpus_root: Path, globs: Iterable[str]) -> list[Path]:
    if not corpus_root.exists():
        raise FileNotFoundError(f"corpus root does not exist: {corpus_root}")
    if corpus_root.is_file():
        if corpus_root.suffix.casefold() != ".pdf":
            raise ValueError(f"corpus root file is not a PDF: {corpus_root}")
        return [corpus_root.resolve()]
    paths: dict[str, Path] = {}
    for pattern in globs:
        for path in corpus_root.glob(pattern):
            if path.is_file() and path.suffix.casefold() == ".pdf":
                resolved = path.resolve()
                paths[str(resolved).casefold()] = resolved
    return sorted(paths.values(), key=lambda item: str(item).casefold())


def _chunks(values: tuple[Any, ...], size: int) -> Iterable[tuple[Any, ...]]:
    for offset in range(0, len(values), size):
        yield values[offset : offset + size]


_CERTIFIED_BLANK_AUDIT_IDENTITY = {
    "audit_type": "page_extraction",
    "event": "deterministic_blank_page",
    "outcome": "empty_page_extraction",
}


def _is_certified_blank_page(page: Any) -> bool:
    """Trust only the exact deterministic blank-page audit emitted by extraction."""

    if page.elements or len(page.model_responses) != 1:
        return False
    audit = page.model_responses[0]
    if not isinstance(audit, Mapping):
        return False
    if any(audit.get(key) != value for key, value in _CERTIFIED_BLANK_AUDIT_IDENTITY.items()):
        return False
    try:
        audited_page_number = int(audit.get("page_number"))
    except (TypeError, ValueError):
        return False
    return (
        audited_page_number == page.page_number and audit.get("routing") == page.routing.to_dict()
    )


def _is_certified_blank_chunk(pages: tuple[Any, ...]) -> bool:
    return bool(pages) and all(_is_certified_blank_page(page) for page in pages)


class BenchmarkRunner:
    def __init__(
        self,
        options: BenchmarkOptions,
        *,
        progress: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> None:
        self.options = options
        self.progress = progress or (lambda _: None)
        self.output_dir = options.output_dir.resolve()
        self.dry_run = (
            options.dry_run if options.dry_run is not None else options.config.mode == "dry-run"
        )
        self.run_id = (
            options.run_id
            or os.getenv("BENCHMARK_RUN_ID")
            or (f"{options.config.run_name}-{_utc_now().strftime('%Y%m%dT%H%M%SZ')}")
        )

    def run(self) -> BenchmarkRunResult:
        started = _utc_now()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        ensure_runtime_directories(self.options.config)
        pipeline = build_extraction_pipeline(self.options.config, dry_run=self.dry_run)
        page_checkpoints = JsonPageCheckpointStore(self.output_dir / "checkpoints" / "pages")
        document_checkpoints = DocumentCheckpointStore(self.options.config.paths.checkpoint_db)
        store = SQLiteCatalogStore(
            self.output_dir / "catalog.sqlite3",
            jsonl_path=self.output_dir / "products.jsonl",
        )
        knowledge_base = SQLiteKnowledgeBase(self.output_dir / "knowledge_base.sqlite3")
        embedding_client = build_embedding_client(
            self.options.config,
            dry_run=self.dry_run,
        )
        discovered = discover_pdfs(
            self.options.config.paths.corpus_root, self.options.include_globs
        )
        selected = discovered[: self.options.limits.max_documents]
        counters: Counter[str] = Counter()
        remaining_pages = self.options.limits.max_pages

        for path in selected:
            if remaining_pages is not None and remaining_pages <= 0:
                break
            per_document_cap = self.options.limits.max_pages_per_document
            if remaining_pages is not None:
                per_document_cap = (
                    remaining_pages
                    if per_document_cap is None
                    else min(per_document_cap, remaining_pages)
                )
            document_summary = self._run_document(
                path,
                pipeline=pipeline,
                page_checkpoints=page_checkpoints,
                document_checkpoints=document_checkpoints,
                store=store,
                knowledge_base=knowledge_base,
                embedding_client=embedding_client,
                page_cap=per_document_cap,
            )
            _append_jsonl(self.output_dir / "documents.jsonl", document_summary)
            self.progress(document_summary)
            counters["documents_attempted"] += 1
            counters[f"documents_{document_summary['status']}"] += 1
            for key in (
                "pages",
                "elements",
                "checkpoint_hits",
                "products_parsed",
                "products_guard_rejected",
                "products_stored",
                "products_invalid",
                "knowledge_chunks",
            ):
                counters[key] += int(document_summary.get(key, 0))
            if remaining_pages is not None:
                remaining_pages -= int(document_summary.get("pages", 0))
            if document_summary["status"] == "failed" and self.options.fail_fast:
                break

        finished = _utc_now()
        summary: dict[str, Any] = {
            "schema_version": "1.0",
            "kind": "industrial-catalog-benchmark-summary",
            "run_id": self.run_id,
            "mode": "dry-run" if self.dry_run else "live",
            "started_at": started.isoformat(),
            "finished_at": finished.isoformat(),
            "elapsed_seconds": round((finished - started).total_seconds(), 6),
            "corpus_root": str(self.options.config.paths.corpus_root),
            "output_dir": str(self.output_dir),
            "documents_discovered": len(discovered),
            "documents_selected": len(selected),
            **dict(counters),
            "documents_succeeded": counters["documents_succeeded"],
            "documents_review": counters["documents_review"],
            "documents_failed": counters["documents_failed"],
            "limits": {
                "max_documents": self.options.limits.max_documents,
                "max_pages": self.options.limits.max_pages,
                "max_pages_per_document": self.options.limits.max_pages_per_document,
                "pages_per_parse": self.options.limits.pages_per_parse,
            },
        }
        _write_json_atomic(self.output_dir / "summary.json", summary)
        return BenchmarkRunResult(summary=summary, output_dir=self.output_dir)

    def _run_document(
        self,
        path: Path,
        *,
        pipeline: ExtractionPipeline,
        page_checkpoints: JsonPageCheckpointStore,
        document_checkpoints: DocumentCheckpointStore,
        store: SQLiteCatalogStore,
        knowledge_base: SQLiteKnowledgeBase,
        embedding_client: NvidiaEmbeddingClient | None,
        page_cap: int | None,
    ) -> dict[str, Any]:
        started = time.monotonic()
        source_sha256 = sha256_file(path)
        document_id = f"sha256:{source_sha256}"
        base: dict[str, Any] = {
            "run_id": self.run_id,
            "document_id": document_id,
            "source_path": str(path),
            "source_sha256": source_sha256,
        }
        document_checkpoints.upsert(
            Checkpoint(document_id, str(path), source_sha256, "running", attempt=1)
        )
        try:
            source = PyMuPDFPageSource(path, render_dpi=self.options.config.routing.render_dpi)
            pages: Iterable[Any] = source.iter_pages()
            if page_cap is not None:
                pages = itertools.islice(pages, page_cap)
            extraction = pipeline.extract_pages(
                document_id,
                pages,
                source_path=str(path),
                checkpoint=page_checkpoints,
            )
            artifact_key = source_sha256[:24]
            _write_json_atomic(
                self.output_dir / "extractions" / f"{artifact_key}.json",
                extraction.to_dict(),
            )
            store.upsert_document(
                SourceDocument(
                    source_document_id=document_id,
                    source_path=str(path),
                    sha256=source_sha256,
                    page_count=len(extraction.pages) or None,
                    metadata={"benchmark_page_cap": page_cap},
                )
            )

            # Index exact extracted page evidence independently of product admission.
            # Rejected or unresolved pages therefore remain searchable with page,
            # element, and bounding-box citations while canonical product RAG stays
            # protected by the validation/source-hint gates below.
            page_evidence_chunks = knowledge_base.index_document_extraction(
                extraction,
                embedding_provider=embedding_client,
                embedding_model=(
                    self.options.config.embedding_endpoint.model
                    if embedding_client is not None
                    and self.options.config.embedding_endpoint is not None
                    else None
                ),
            )
            products_parsed = products_stored = products_invalid = 0
            knowledge_chunks = len(page_evidence_chunks)
            products_guard_rejected = 0
            needs_review = False
            for chunk_index, page_chunk in enumerate(
                _chunks(extraction.pages, self.options.limits.pages_per_parse), start=1
            ):
                chunk = DocumentExtraction(
                    document_id=extraction.document_id,
                    source_path=extraction.source_path,
                    pages=page_chunk,
                    pipeline_version=extraction.pipeline_version,
                )
                structured_path = (
                    self.output_dir / "structured" / f"{artifact_key}.chunk-{chunk_index:04d}.json"
                )
                source_hints = derive_structured_source_hints(chunk)
                guided_schema = product_batch_schema_for_hints(source_hints)
                decision_path = (
                    self.output_dir
                    / "decisions"
                    / f"{artifact_key}.chunk-{chunk_index:04d}.hints.json"
                )
                decision_record: dict[str, Any] = {
                    "source_hints": source_hints.to_dict(),
                    "schema_product_minimum": guided_schema["properties"]["products"].get(
                        "minItems", 0
                    ),
                    "schema_product_maximum": guided_schema["properties"]["products"]["maxItems"],
                    "part_number_enabled": "part_number"
                    in guided_schema["properties"]["products"]["items"]["properties"],
                }
                _write_json_atomic(
                    decision_path,
                    decision_record,
                )
                if _is_certified_blank_chunk(page_chunk):
                    blank_pages = [page.page_number for page in page_chunk]
                    skip_reason = "certified_blank_page"
                    skip_record: dict[str, Any] = {
                        "event_schema_version": "1.0",
                        "audit_type": "structured_parse",
                        "event": "structured_parse_skipped",
                        "outcome": "skipped",
                        "structured_parse_skipped_reason": skip_reason,
                        "chunk": chunk_index,
                        "pages": blank_pages,
                        "page_fingerprints": [page.fingerprint for page in page_chunk],
                    }
                    _write_json_atomic(structured_path, skip_record)
                    decision_record.update(
                        {
                            "structured_parse_skipped": True,
                            "structured_parse_skipped_reason": skip_reason,
                            "certified_blank_pages": blank_pages,
                        }
                    )
                    _write_json_atomic(decision_path, decision_record)
                    _append_jsonl(
                        self.output_dir / "chunks.jsonl",
                        {
                            **base,
                            "chunk": chunk_index,
                            "pages": blank_pages,
                            "status": "skipped",
                            "batch_id": None,
                            "batch_persisted": False,
                            "structured_parse_skipped": True,
                            "structured_parse_skipped_reason": skip_reason,
                            "certified_blank_pages": blank_pages,
                            "accepted_count": 0,
                            "rejected_count": 0,
                            "products_parsed": 0,
                            "products_after_source_guard": 0,
                            "products_guard_rejected": 0,
                            "products_stored": 0,
                            "knowledge_chunks": 0,
                        },
                    )
                    continue
                try:
                    parsed = pipeline.parse_structured(
                        chunk,
                        schema=guided_schema,
                        instructions=(
                            "Return exactly one compact JSON ProductBatch object. Apply the "
                            "product eligibility rules to all evidence on this page before "
                            "creating any record. Omit unknown optional fields and do not pad "
                            "the response with whitespace."
                        ),
                    )
                except GuidedJsonDecodeError as error:
                    _write_json_atomic(
                        structured_path.with_suffix(".failure.json"), error.to_dict()
                    )
                    raise
                _write_json_atomic(
                    structured_path,
                    parsed.to_dict(),
                )
                model_products = (
                    parsed.data.get("products", []) if isinstance(parsed.data, Mapping) else []
                )
                if not isinstance(model_products, list):
                    raise ValueError("guided ProductBatch products must be a list")
                products_parsed += len(model_products)
                guarded_data, enforcement = enforce_structured_source_hints(
                    parsed.data,
                    source_hints,
                    extraction=chunk,
                )
                decision_record["enforcement"] = enforcement
                _write_json_atomic(decision_path, decision_record)
                products_guard_rejected += int(enforcement["dropped_product_count"])
                if enforcement["dropped_product_count"]:
                    needs_review = True
                parsed = replace(parsed, data=guarded_data)
                raw_products = guarded_data["products"]
                persistence = store.persist_parsed_batch(
                    parsed,
                    source_document_id=document_id,
                    source_path=str(path),
                    extraction=chunk,
                    pipeline_version=extraction.pipeline_version,
                    strict=self.options.strict_validation,
                    require_evidence=not self.dry_run,
                )
                products_stored += persistence.accepted_count
                products_invalid += persistence.rejected_count
                accepted_products = [
                    prepared.product
                    for prepared in persistence.prepared
                    if prepared.accepted and prepared.product is not None
                ]
                indexed_chunks = knowledge_base.index_products(
                    accepted_products,
                    embedding_provider=embedding_client,
                    embedding_model=(
                        self.options.config.embedding_endpoint.model
                        if embedding_client is not None
                        and self.options.config.embedding_endpoint is not None
                        else None
                    ),
                )
                knowledge_chunks += len(indexed_chunks)
                if persistence.status != "succeeded" or (persistence.issues and not self.dry_run):
                    needs_review = True
                for prepared in persistence.prepared:
                    validation_event: dict[str, Any] = {
                        **base,
                        "chunk": chunk_index,
                        "product_index": prepared.ordinal,
                        "batch_id": persistence.batch.batch_id,
                        "status": "accepted" if prepared.accepted else "invalid",
                        "record_id": (
                            prepared.product.record_id if prepared.product is not None else None
                        ),
                        "issues": [issue.model_dump(mode="json") for issue in prepared.issues],
                    }
                    _append_jsonl(self.output_dir / "validation.jsonl", validation_event)

                _append_jsonl(
                    self.output_dir / "chunks.jsonl",
                    {
                        **base,
                        "chunk": chunk_index,
                        "pages": [page.page_number for page in page_chunk],
                        **persistence.to_dict(),
                        "products_parsed": len(model_products),
                        "products_after_source_guard": len(raw_products),
                        "products_guard_rejected": enforcement["dropped_product_count"],
                        "products_stored": persistence.accepted_count,
                        "knowledge_chunks": len(indexed_chunks),
                    },
                )

            route_counts = Counter(page.routing.route.value for page in extraction.pages)
            status = "review" if needs_review else "succeeded"
            detail = {
                "pages": len(extraction.pages),
                "elements": len(extraction.elements),
                "products_stored": products_stored,
            }
            document_checkpoints.upsert(
                Checkpoint(document_id, str(path), source_sha256, status, 1, detail)
            )
            return {
                **base,
                "status": status,
                "elapsed_seconds": round(time.monotonic() - started, 6),
                "pages": len(extraction.pages),
                "elements": len(extraction.elements),
                "checkpoint_hits": extraction.checkpoint_hits,
                "routes": dict(route_counts),
                "ocr_pages": sum(
                    page.routing.requires_ocr_verification for page in extraction.pages
                ),
                "products_parsed": products_parsed,
                "products_guard_rejected": products_guard_rejected,
                "products_stored": products_stored,
                "products_invalid": products_invalid,
                "knowledge_chunks": knowledge_chunks,
            }
        except Exception as exc:
            failure_detail: dict[str, Any] = {
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            error_details = _explicit_error_details(exc)
            if error_details is not None:
                failure_detail["error_details"] = error_details
            document_checkpoints.upsert(
                Checkpoint(
                    document_id,
                    str(path),
                    source_sha256,
                    "failed",
                    1,
                    failure_detail,
                )
            )
            return {
                **base,
                "status": "failed",
                "elapsed_seconds": round(time.monotonic() - started, 6),
                **failure_detail,
                "pages": 0,
                "elements": 0,
                "checkpoint_hits": 0,
                "products_parsed": 0,
                "products_guard_rejected": 0,
                "products_stored": 0,
                "products_invalid": 0,
                "knowledge_chunks": 0,
            }


def run_benchmark(
    options: BenchmarkOptions,
    *,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
) -> BenchmarkRunResult:
    return BenchmarkRunner(options, progress=progress).run()


__all__ = [
    "BenchmarkLimits",
    "BenchmarkOptions",
    "BenchmarkRunResult",
    "BenchmarkRunner",
    "build_embedding_client",
    "build_extraction_pipeline",
    "discover_pdfs",
    "enforce_structured_source_hints",
    "product_batch_schema_for_hints",
    "product_batch_guided_json_schema",
    "run_benchmark",
]
