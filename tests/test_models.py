from __future__ import annotations

import json
import sqlite3

import pytest
from pydantic import ValidationError

from industrial_catalog.models import (
    MAX_EVIDENCE_REFERENCES_PER_FIELD,
    MAX_OTHER_DETAILS_PER_PRODUCT,
    MAX_PRODUCTS_PER_BATCH,
    MAX_SPECIFICATIONS_PER_PRODUCT,
    BoundingBox,
    Evidence,
    ExtractedValue,
    ProductBatch,
    ProductRecord,
    Specification,
    product_batch_guided_json_schema,
)
from industrial_catalog.nvidia_clients import GuidedJsonResult, ModelResponse
from industrial_catalog.storage import SQLiteCatalogStore, append_jsonl
from industrial_catalog.validation import (
    hydrate_product_evidence,
    normalize_product_record,
    prepare_product_payload,
    validate_product_record,
)


def evidence(text: str, page: int = 7) -> Evidence:
    return Evidence(
        document_id="catalog-sha256",
        source_path="vendor/catalog.pdf",
        page_number=page,
        element_id=f"p{page}-e1",
        bbox=[10, 20, 300, 80],
        excerpt=text,
        extraction_method="nemotron-parse",
        model="nvidia/nemotron-parse-v1.2",
        confidence=0.96,
    )


def sample_product() -> ProductRecord:
    manufacturer_evidence = evidence("  ACME  Corporation ")
    part_evidence = evidence("PN–100")
    temperature_evidence = evidence("Operating Temperature -20 to 80 degrees C")
    return ProductRecord(
        record_id="product-1",
        document_id="catalog-sha256",
        source_path="vendor/catalog.pdf",
        manufacturer=ExtractedValue[str](
            raw="  ACME  Corporation ",
            confidence=0.95,
            evidence=[manufacturer_evidence],
        ),
        part_name=ExtractedValue[str](
            raw="  Industrial   Sensor ",
            evidence=[evidence("Industrial Sensor")],
        ),
        part_number=ExtractedValue[str](
            raw="pn–100",
            evidence=[part_evidence],
        ),
        specifications=[
            Specification(
                specification_id="spec-temperature",
                name=ExtractedValue[str](
                    raw="Operating  Temperature",
                    evidence=[temperature_evidence],
                ),
                value=ExtractedValue(raw="-20 to 80", evidence=[temperature_evidence]),
                unit=ExtractedValue[str](
                    raw="degrees C",
                    evidence=[temperature_evidence],
                ),
            )
        ],
        other_details={"Ingress Rating": ExtractedValue(raw="IP67", evidence=[evidence("IP67")])},
    )


def test_bbox_accepts_sequence_and_rejects_inverted_geometry() -> None:
    bbox = BoundingBox.model_validate([1, 2, 11, 12])
    assert bbox.as_list() == [1.0, 2.0, 11.0, 12.0]

    with pytest.raises(ValidationError):
        BoundingBox(x0=5, y0=0, x1=4, y1=1)

    with pytest.raises(ValidationError):
        BoundingBox(x0=0, y0=0, x1=2, y1=1, coordinate_space="normalized")


def test_evidence_accepts_pipeline_aliases_and_keeps_coordinates() -> None:
    item = evidence("PN-100")
    assert item.source_document_id == "catalog-sha256"
    assert item.text == "PN-100"
    assert item.page_number == 7
    assert item.bbox is not None
    assert item.bbox.x1 == 300


def test_extracted_value_requires_raw_or_normalized_content() -> None:
    with pytest.raises(ValidationError):
        ExtractedValue[str]()

    value = ExtractedValue[str](raw="Source Text", normalized="source text", confidence=0.8)
    assert value.raw == "Source Text"
    assert value.normalized == "source text"
    assert value.effective_confidence == 0.8


def test_normalization_preserves_raw_values_and_provenance() -> None:
    original = sample_product()
    normalized = normalize_product_record(original)

    assert normalized.manufacturer is not None
    assert normalized.manufacturer.raw == "  ACME  Corporation "
    assert normalized.manufacturer.normalized == "ACME Corporation"
    assert normalized.part_number is not None
    assert normalized.part_number.raw == "pn–100"
    assert normalized.part_number.normalized == "PN-100"
    assert normalized.specifications[0].unit is not None
    assert normalized.specifications[0].unit.raw == "degrees C"
    assert normalized.specifications[0].unit.normalized == "°C"
    assert normalized.manufacturer.evidence[0].page_number == 7
    assert normalized.extraction.record_confidence is not None

    # Normalization returns a copy and does not rewrite the parser output.
    assert original.manufacturer.normalized is None
    assert original.part_number.normalized is None


def test_product_round_trip_includes_per_field_evidence() -> None:
    product = normalize_product_record(sample_product())
    payload = product.model_dump_json()
    restored = ProductRecord.model_validate_json(payload)

    assert restored == product
    paths = [path for path, _ in restored.iter_field_evidence()]
    assert "manufacturer" in paths
    assert "part_number" in paths
    assert "specifications.0.value" in paths
    assert "other_details.Ingress Rating" in paths


def test_quality_validation_reports_partial_records_without_crashing() -> None:
    report = validate_product_record(
        {
            "source_document_id": "doc-1",
            "part_name": {"raw": "Valve"},
        }
    )
    assert report.valid
    assert {issue.code for issue in report.issues} >= {
        "missing_manufacturer",
        "missing_part_number",
        "missing_specifications",
        "missing_evidence",
    }

    strict_report = validate_product_record(
        {
            "source_document_id": "doc-1",
            "part_name": {"raw": "Valve"},
        },
        strict=True,
    )
    assert not strict_report.valid

    schema_report = validate_product_record({"part_name": {"raw": "Valve"}})
    assert not schema_report.valid
    assert schema_report.issues[0].code == "schema_error"


def test_sqlite_upsert_projects_fields_evidence_and_jsonl(tmp_path) -> None:
    db_path = tmp_path / "catalog.sqlite3"
    jsonl_path = tmp_path / "events" / "products.jsonl"
    product = normalize_product_record(sample_product())
    store = SQLiteCatalogStore(db_path, jsonl_path=jsonl_path)

    store.save_product(product)
    assert store.count_products() == 1
    assert store.get_product("product-1") == product
    assert store.find_by_part_number("pn-100")[0].record_id == "product-1"
    assert store.find_products(manufacturer="ACME Corporation")[0].record_id == "product-1"
    assert any(path == "specifications.0.name" for path, _ in store.get_evidence("product-1"))

    # Reprocessing the same record replaces relational projections, not duplicates.
    updated = product.model_copy(
        update={
            "part_name": product.part_name.model_copy(
                update={"normalized": "Updated Industrial Sensor"}
            )
        }
    )
    store.save_product(updated)
    assert store.count_products() == 1
    assert store.get_product("product-1").part_name.normalized == "Updated Industrial Sensor"

    with sqlite3.connect(db_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM specifications").fetchone()[0] == 1
        evidence_count = connection.execute("SELECT COUNT(*) FROM evidence").fetchone()[0]
    assert evidence_count == len(list(updated.iter_field_evidence()))

    lines = jsonl_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[-1])["record_id"] == "product-1"


def test_standalone_jsonl_writer_emits_one_record_per_line(tmp_path) -> None:
    path = tmp_path / "products.jsonl"
    products = [normalize_product_record(sample_product())]
    assert append_jsonl(path, products) == 1
    assert ProductRecord.model_validate_json(path.read_text(encoding="utf-8")) == products[0]


def extraction_elements() -> dict:
    return {
        "document_id": "catalog-sha256",
        "pages": [
            {
                "page_number": 7,
                "elements": [
                    {
                        "element_id": "el_identity",
                        "document_id": "catalog-sha256",
                        "page_number": 7,
                        "sequence_number": 0,
                        "element_type": "table",
                        "text": "ACME Industrial Sensor PN-100 Operating Temperature -20 to 80 °C",
                        "extraction_method": "nemotron_parse",
                        "bbox": [10, 20, 500, 100],
                        "confidence": 0.97,
                        "model": "nvidia/nemotron-parse-v1.2",
                        "source_path": "vendor/catalog.pdf",
                    }
                ],
            }
        ],
    }


def compact_product(element_id: str = "el_identity") -> dict:
    ref = {"element_id": element_id}
    return {
        "manufacturer": {"raw": "ACME", "normalized": "ACME", "evidence": [ref]},
        "part_name": {
            "raw": "Industrial Sensor",
            "normalized": "Industrial Sensor",
            "evidence": [ref],
        },
        "part_number": {"raw": "PN-100", "normalized": "PN-100", "evidence": [ref]},
        "specifications": [
            {
                "name": {
                    "raw": "Operating Temperature",
                    "normalized": "Operating Temperature",
                    "evidence": [ref],
                },
                "value": {"raw": "-20 to 80", "normalized": "-20 to 80", "evidence": [ref]},
                "unit": {"raw": "°C", "normalized": "°C", "evidence": [ref]},
            }
        ],
    }


def test_compact_guided_schema_requests_element_ids_not_full_evidence() -> None:
    schema = product_batch_guided_json_schema()
    encoded = json.dumps(schema, separators=(",", ":"))
    assert schema["required"] == ["products"]
    assert '"element_id"' in encoded
    assert '"source_document_id"' not in encoded
    assert '"page_number"' not in encoded
    assert '"normalized"' not in encoded
    assert '"confidence"' not in encoded
    assert '"anyOf"' not in encoded
    assert len(encoded) < 5_000

    products_schema = schema["properties"]["products"]
    product_schema = products_schema["items"]
    assert products_schema["maxItems"] == MAX_PRODUCTS_PER_BATCH
    assert "minItems" not in products_schema  # Empty means identity is unsupported.
    assert product_schema["required"] == [
        "manufacturer",
        "part_name",
        "specifications",
        "other_details",
    ]
    assert "part_number" not in product_schema["required"]
    assert set(product_schema["properties"]) == {
        "manufacturer",
        "part_name",
        "part_number",
        "category",
        "description",
        "specifications",
        "other_details",
    }
    manufacturer_schema = product_schema["properties"]["manufacturer"]
    assert manufacturer_schema["required"] == ["raw", "evidence"]
    evidence_schema = manufacturer_schema["properties"]["evidence"]
    assert evidence_schema["minItems"] == 1
    assert evidence_schema["maxItems"] == MAX_EVIDENCE_REFERENCES_PER_FIELD
    assert evidence_schema["items"]["required"] == ["element_id"]
    assert evidence_schema["items"]["additionalProperties"] is False
    assert (
        product_schema["properties"]["specifications"]["maxItems"] == MAX_SPECIFICATIONS_PER_PRODUCT
    )
    assert (
        product_schema["properties"]["other_details"]["maxItems"] == MAX_OTHER_DETAILS_PER_PRODUCT
    )


def test_hydration_resolves_compact_reference_to_canonical_source() -> None:
    payload = compact_product()
    payload["part_number"]["evidence"] = [
        {"element_id": "el_identity", "text": "LLM-modified text", "page_number": 99}
    ]
    result = hydrate_product_evidence(
        payload,
        extracted_elements=extraction_elements(),
        source_document_id="catalog-sha256",
        source_path="vendor/catalog.pdf",
    )

    assert result.valid
    hydrated = result.payload["part_number"]["evidence"][0]
    assert hydrated["text"].startswith("ACME Industrial Sensor")
    assert hydrated["page_number"] == 7
    assert hydrated["bbox"] == {
        "x0": 10.0,
        "y0": 20.0,
        "x1": 500.0,
        "y1": 100.0,
        "coordinate_space": "pixel",
    }
    assert {issue.code for issue in result.issues} == {"evidence_claim_mismatch"}


def test_minimal_guided_payload_normalizes_and_preserves_other_detail_evidence() -> None:
    ref = {"element_id": "el_identity"}
    raw = {
        "manufacturer": {"raw": " ACME ", "evidence": [ref]},
        "part_name": {"raw": "Industrial Sensor", "evidence": [ref]},
        "part_number": {"raw": "pn–100", "evidence": [ref]},
        "specifications": [
            {
                "name": {"raw": "Operating Temperature", "evidence": [ref]},
                "value": {"raw": "-20 to 80", "evidence": [ref]},
                "unit": {"raw": "degrees C", "evidence": [ref]},
            }
        ],
        "other_details": [
            {
                "name": {"raw": "Ingress Rating", "evidence": [ref]},
                "value": {"raw": "IP67", "evidence": [ref]},
            }
        ],
    }
    prepared = prepare_product_payload(
        raw,
        ordinal=0,
        extracted_elements=extraction_elements(),
        source_document_id="catalog-sha256",
        source_path="vendor/catalog.pdf",
    )

    assert prepared.accepted
    assert prepared.product is not None
    assert prepared.product.manufacturer.normalized == "ACME"
    assert prepared.product.part_number.normalized == "PN-100"
    assert prepared.product.specifications[0].unit.normalized == "°C"
    detail = prepared.product.other_details["Ingress Rating"]
    assert detail.raw == "IP67"
    assert detail.normalized == "IP67"
    assert detail.effective_confidence == 0.97
    assert detail.evidence[0].element_id == "el_identity"


def test_guided_other_detail_duplicate_names_are_retained_for_review() -> None:
    ref = {"element_id": "el_identity"}
    raw = {
        "part_number": {"raw": "PN-100", "evidence": [ref]},
        "specifications": [],
        "other_details": [
            {
                "name": {"raw": "Connection", "evidence": [ref]},
                "value": {"raw": "M12", "evidence": [ref]},
            },
            {
                "name": {"raw": "Connection", "evidence": [ref]},
                "value": {"raw": "4-pin", "evidence": [ref]},
            },
        ],
    }
    prepared = prepare_product_payload(
        raw,
        ordinal=0,
        extracted_elements=extraction_elements(),
        source_document_id="catalog-sha256",
    )

    assert prepared.accepted
    assert prepared.product is not None
    assert set(prepared.product.other_details) == {"Connection", "Connection (2)"}
    assert "duplicate_other_detail_name" in {issue.code for issue in prepared.issues}


def test_wire_collection_limit_is_flagged_for_source_review() -> None:
    raw = compact_product()
    raw["specifications"] = raw["specifications"] * MAX_SPECIFICATIONS_PER_PRODUCT
    result = hydrate_product_evidence(
        raw,
        extracted_elements=extraction_elements(),
        source_document_id="catalog-sha256",
    )

    assert "wire_collection_limit_reached" in {issue.code for issue in result.issues}


def test_wire_product_limit_is_not_silently_accepted(tmp_path) -> None:
    store = SQLiteCatalogStore(tmp_path / "catalog.sqlite3")
    parsed = {"products": ["invalid-product"] * MAX_PRODUCTS_PER_BATCH}
    result = store.persist_parsed_batch(
        parsed,
        source_document_id="catalog-sha256",
        extraction=extraction_elements(),
    )

    assert result.rejected_count == MAX_PRODUCTS_PER_BATCH
    assert "wire_product_limit_reached" in {issue.code for issue in result.issues}


def test_prepare_product_rejects_unknown_evidence_without_losing_payload() -> None:
    raw = compact_product("el_hallucinated")
    prepared = prepare_product_payload(
        raw,
        ordinal=3,
        extracted_elements=extraction_elements(),
        source_document_id="catalog-sha256",
        source_path="vendor/catalog.pdf",
    )

    assert not prepared.accepted
    assert prepared.raw_payload == raw
    assert prepared.product is not None
    assert "unknown_evidence_reference" in {issue.code for issue in prepared.issues}
    assert prepared.product.record_id.startswith("prd_")


def test_explicitly_empty_extraction_rejects_complete_fabricated_evidence() -> None:
    fabricated_evidence = {
        "document_id": "catalog-sha256",
        "source_path": "vendor/catalog.pdf",
        "page_number": 1,
        "element_id": "el_fabricated",
        "text": "ACME Industrial Sensor PN-FAKE",
        "extraction_method": "claimed_model_output",
    }
    raw = {
        "manufacturer": {"raw": "ACME", "evidence": [fabricated_evidence]},
        "part_name": {"raw": "Industrial Sensor", "evidence": [fabricated_evidence]},
        "part_number": {"raw": "PN-FAKE", "evidence": [fabricated_evidence]},
        "specifications": [],
    }

    prepared = prepare_product_payload(
        raw,
        ordinal=0,
        extracted_elements={"document_id": "catalog-sha256", "pages": []},
        source_document_id="catalog-sha256",
        source_path="vendor/catalog.pdf",
    )

    assert not prepared.accepted
    assert prepared.product is not None
    issue_codes = {issue.code for issue in prepared.issues}
    assert "unknown_evidence_reference" in issue_codes
    assert "unverified_evidence" not in issue_codes
    assert all(
        issue.severity == "error"
        for issue in prepared.issues
        if issue.code == "unknown_evidence_reference"
    )

    no_source = prepare_product_payload(
        raw,
        ordinal=0,
        extracted_elements=None,
        source_document_id="catalog-sha256",
        source_path="vendor/catalog.pdf",
    )
    assert no_source.accepted
    no_source_codes = {issue.code for issue in no_source.issues}
    assert "unverified_evidence" in no_source_codes
    assert "unknown_evidence_reference" not in no_source_codes


def test_hydration_rejects_cross_document_evidence() -> None:
    extraction = extraction_elements()
    extraction["pages"][0]["elements"][0]["document_id"] = "other-document"
    result = hydrate_product_evidence(
        compact_product(),
        extracted_elements=extraction,
        source_document_id="catalog-sha256",
    )

    assert not result.valid
    assert "cross_document_evidence" in {issue.code for issue in result.issues}


def test_product_batch_prevents_cross_document_products() -> None:
    product = normalize_product_record(sample_product())
    with pytest.raises(ValidationError):
        ProductBatch(source_document_id="other-document", products=[product])


def test_batch_persistence_keeps_valid_sibling_and_raw_rejection(tmp_path) -> None:
    products_jsonl = tmp_path / "products.jsonl"
    store = SQLiteCatalogStore(tmp_path / "catalog.sqlite3", jsonl_path=products_jsonl)
    parsed = {"products": [compact_product(), compact_product("el_unknown")]}

    result = store.persist_parsed_batch(
        parsed,
        source_document_id="catalog-sha256",
        source_path="vendor/catalog.pdf",
        extraction=extraction_elements(),
        pipeline_version="1.0",
    )

    assert result.status == "partial"
    assert result.accepted_count == 1
    assert result.rejected_count == 1
    assert result.raw_preserved
    assert result.products_jsonl_written
    assert result.batch_jsonl_written
    assert store.count_products() == 1

    audit = store.get_batch_ingestion(result.batch.batch_id)
    assert audit is not None
    assert audit["raw_llm"] == parsed
    assert audit["raw_extraction"]["pages"][0]["elements"][0]["element_id"] == "el_identity"
    rejected = store.get_rejected_products(result.batch.batch_id)
    assert len(rejected) == 1
    assert rejected[0]["raw_product"]["manufacturer"]["evidence"][0]["element_id"] == "el_unknown"

    product_lines = products_jsonl.read_text(encoding="utf-8").splitlines()
    batch_path = tmp_path / "products.batches.jsonl"
    batch_lines = batch_path.read_text(encoding="utf-8").splitlines()
    assert len(product_lines) == 1
    assert len(batch_lines) == 1
    event = json.loads(batch_lines[0])
    assert event["event_schema_version"] == "1.0"
    assert event["status"] == "partial"
    assert event["rejections"][0]["raw_payload"] == parsed["products"][1]

    # Deterministic batch/record IDs make a retry idempotent in SQLite and JSONL.
    retry = store.persist_parsed_batch(
        parsed,
        source_document_id="catalog-sha256",
        source_path="vendor/catalog.pdf",
        extraction=extraction_elements(),
        pipeline_version="1.0",
    )
    assert retry.batch.batch_id == result.batch.batch_id
    assert store.count_products() == 1
    assert len(products_jsonl.read_text(encoding="utf-8").splitlines()) == 1
    assert len(batch_path.read_text(encoding="utf-8").splitlines()) == 1


def test_empty_batches_on_different_pages_have_distinct_stable_ids(tmp_path) -> None:
    store = SQLiteCatalogStore(tmp_path / "catalog.sqlite3")

    def parsed(request_id: str) -> GuidedJsonResult:
        data = {"products": []}
        return GuidedJsonResult(
            data=data,
            response=ModelResponse(
                content=json.dumps(data),
                model="test-model",
                request_id=request_id,
                raw_response={"id": request_id, "data": data},
            ),
            schema_sha256="test-schema-sha256",
        )

    first_extraction = {
        "document_id": "catalog-sha256",
        "pipeline_version": "1.0",
        "pages": [
            {
                "page_number": 1,
                "fingerprint": "page-fingerprint-1",
                "elements": [],
            }
        ],
    }
    second_extraction = {
        "document_id": "catalog-sha256",
        "pipeline_version": "1.0",
        "pages": [
            {
                "page_number": 2,
                "fingerprint": "page-fingerprint-2",
                "elements": [],
            }
        ],
    }

    first = store.persist_parsed_batch(
        parsed("volatile-request-1"),
        source_document_id="catalog-sha256",
        extraction=first_extraction,
        pipeline_version="1.0",
    )
    second = store.persist_parsed_batch(
        parsed("volatile-request-2"),
        source_document_id="catalog-sha256",
        extraction=second_extraction,
        pipeline_version="1.0",
    )
    retry = store.persist_parsed_batch(
        parsed("different-retry-request-id"),
        source_document_id="catalog-sha256",
        extraction=first_extraction,
        pipeline_version="1.0",
    )

    assert first.batch.batch_id != second.batch.batch_id
    assert retry.batch.batch_id == first.batch.batch_id
    assert first.batch.metadata["extraction_context"] == [
        {"page_number": 1, "fingerprint": "page-fingerprint-1"}
    ]
    assert second.batch.metadata["extraction_context"] == [
        {"page_number": 2, "fingerprint": "page-fingerprint-2"}
    ]
    first_audit = store.get_batch_ingestion(first.batch.batch_id)
    second_audit = store.get_batch_ingestion(second.batch.batch_id)
    assert first_audit is not None
    assert second_audit is not None
    assert first_audit["raw_extraction"]["pages"][0]["page_number"] == 1
    assert second_audit["raw_extraction"]["pages"][0]["page_number"] == 2
