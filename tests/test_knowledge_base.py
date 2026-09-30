from __future__ import annotations

import json
import sqlite3

from industrial_catalog.cli import main
from industrial_catalog.extraction import (
    BoundingBox as ExtractionBoundingBox,
)
from industrial_catalog.extraction import (
    DocumentExtraction,
    ExtractedElement,
    PageExtraction,
)
from industrial_catalog.knowledge_base import (
    NvidiaEmbeddingClient,
    SQLiteKnowledgeBase,
    generate_document_chunks,
    generate_page_chunks,
    generate_product_chunks,
)
from industrial_catalog.models import Evidence, ExtractedValue, ProductRecord, Specification
from industrial_catalog.nvidia_clients import NIMEndpointConfig, TransportResponse
from industrial_catalog.routing import PageRouter
from industrial_catalog.validation import normalize_product_record


def source_evidence(text: str, element_id: str = "el_1") -> Evidence:
    return Evidence(
        source_document_id="doc-1",
        source_path="catalogs/acme.pdf",
        page_number=12,
        element_id=element_id,
        bbox=[20, 30, 500, 120],
        text=text,
        extraction_method="nemotron_parse",
        confidence=0.97,
    )


def product() -> ProductRecord:
    identity = source_evidence("ACME TS-100 Temperature Sensor")
    temperature = source_evidence("Operating temperature -20 to 80 °C", element_id="el_temperature")
    return normalize_product_record(
        ProductRecord(
            record_id="product-1",
            source_document_id="doc-1",
            source_path="catalogs/acme.pdf",
            manufacturer=ExtractedValue(raw="ACME", evidence=[identity]),
            part_name=ExtractedValue(raw="Temperature Sensor", evidence=[identity]),
            part_number=ExtractedValue(raw="TS-100", evidence=[identity]),
            specifications=[
                Specification(
                    specification_id="temperature",
                    name=ExtractedValue(raw="Operating Temperature", evidence=[temperature]),
                    value=ExtractedValue(raw="-20 to 80", evidence=[temperature]),
                    unit=ExtractedValue(raw="degrees C", evidence=[temperature]),
                )
            ],
        )
    )


def extracted_page(
    *,
    document_id: str = "raw-doc-1",
    page_number: int = 3,
    texts: tuple[str, ...] = (
        "Lovejoy Stainless Steel Jaw Couplings",
        "The SS-type jaw coupling resists corrosion.",
    ),
) -> PageExtraction:
    elements = tuple(
        ExtractedElement.create(
            document_id=document_id,
            page_number=page_number,
            sequence_number=index,
            element_type="title" if index == 0 else "text",
            text=text,
            extraction_method="nemotron_parse",
            bbox=ExtractionBoundingBox(0.1, 0.2 + index * 0.2, 0.9, 0.3 + index * 0.2),
            confidence=0.95 - index * 0.05,
            model="nvidia/nemotron-parse",
            metadata={"bbox_coordinate_space": "normalized"},
        )
        for index, text in enumerate(texts)
    )
    return PageExtraction(
        document_id=document_id,
        page_number=page_number,
        fingerprint=f"fingerprint-{page_number}",
        routing=PageRouter().decide(""),
        # Deliberately reverse storage order; chunking must use sequence order.
        elements=tuple(reversed(elements)),
    )


def test_chunk_generation_is_deterministic_and_citation_bearing() -> None:
    first = generate_product_chunks(product())
    second = generate_product_chunks(product())

    assert first == second
    assert [chunk.chunk_type for chunk in first] == ["identity", "specification"]
    assert first[0].chunk_id.startswith("kb_")
    assert "Part number: TS-100" in first[0].text
    assert "Operating Temperature = -20 to 80 °C" in first[1].text
    assert {citation.page_number for citation in first[1].citations} == {12}
    assert {citation.element_id for citation in first[1].citations} == {
        "el_1",
        "el_temperature",
    }
    temperature_citation = next(
        citation for citation in first[1].citations if citation.element_id == "el_temperature"
    )
    assert temperature_citation.field_paths == [
        "specifications.0.name",
        "specifications.0.unit",
        "specifications.0.value",
    ]
    assert temperature_citation.bbox is not None
    assert temperature_citation.bbox.as_list() == [20.0, 30.0, 500.0, 120.0]


def test_page_chunk_generation_is_deterministic_and_preserves_exact_citations() -> None:
    page = extracted_page()

    first = generate_page_chunks(page, source_path="catalogs/lovejoy.pdf")
    second = generate_page_chunks(page, source_path="catalogs/lovejoy.pdf")

    assert first == second
    assert [chunk.text for chunk in first] == [
        "Lovejoy Stainless Steel Jaw Couplings",
        "The SS-type jaw coupling resists corrosion.",
    ]
    assert {chunk.chunk_kind for chunk in first} == {"page_evidence"}
    assert {chunk.chunk_type for chunk in first} == {"page_element"}
    assert {chunk.product_record_id for chunk in first} == {None}
    assert {chunk.page_number for chunk in first} == {3}
    citation = first[0].citations[0]
    assert citation.source_document_id == "raw-doc-1"
    assert citation.source_path == "catalogs/lovejoy.pdf"
    assert citation.page_number == 3
    assert citation.element_id == page.elements[1].element_id
    assert citation.excerpt == first[0].text
    assert citation.field_paths == ["pages.3.elements.0.text"]
    assert citation.bbox is not None
    assert citation.bbox.coordinate_space == "normalized"
    assert citation.bbox.as_list() == [0.1, 0.2, 0.9, 0.3]


def test_document_chunk_generation_orders_pages_and_rejects_mismatched_evidence() -> None:
    page_one = extracted_page(page_number=1, texts=("Page one",))
    page_two = extracted_page(page_number=2, texts=("Page two",))
    extraction = DocumentExtraction(
        document_id="raw-doc-1",
        source_path="catalogs/lovejoy.pdf",
        pages=(page_two, page_one),
    )

    chunks = generate_document_chunks(extraction)

    assert [chunk.page_number for chunk in chunks] == [1, 2]
    assert [chunk.text for chunk in chunks] == ["Page one", "Page two"]
    assert all(
        chunk.metadata["pipeline_version"] == extraction.pipeline_version
        for chunk in chunks
    )

    mismatched = DocumentExtraction(
        document_id="another-document",
        source_path=None,
        pages=(page_one,),
    )
    try:
        generate_document_chunks(mismatched)
    except ValueError as exc:
        assert "document_id does not match" in str(exc)
    else:
        raise AssertionError("mismatched page evidence was accepted")


class KeywordEmbedder:
    def embed_documents(self, texts):
        return [[1.0, 0.0] if "Operating Temperature =" in text else [0.0, 1.0] for text in texts]


def test_sqlite_kb_persists_vectors_citations_and_searches(tmp_path) -> None:
    kb = SQLiteKnowledgeBase(tmp_path / "catalog.sqlite3")
    chunks = kb.index_product(
        product(),
        embedding_provider=KeywordEmbedder(),
        embedding_model="nvidia/nemotron-3-embed-1b",
    )

    assert kb.count_chunks() == 2
    restored = kb.get_chunks(product_record_id="product-1")
    assert restored == chunks
    assert restored[0].embedding_model == "nvidia/nemotron-3-embed-1b"
    assert kb.search_text("TS-100")[0].product_record_id == "product-1"

    matches = kb.similarity_search([1.0, 0.0])
    assert matches[0].score == 1.0
    assert matches[0].chunk.chunk_type == "specification"
    assert matches[0].chunk.citations[0].source_path == "catalogs/acme.pdf"


def test_document_index_is_idempotent_and_isolated_from_products(tmp_path) -> None:
    kb = SQLiteKnowledgeBase(tmp_path / "document-knowledge.sqlite3")
    kb.index_product(product())
    page = extracted_page()
    extraction = DocumentExtraction(
        document_id=page.document_id,
        source_path="catalogs/lovejoy.pdf",
        pages=(page,),
    )

    first = kb.index_document_extraction(
        extraction,
        embedding_provider=KeywordEmbedder(),
        embedding_model="nvidia/nemotron-3-embed-1b",
    )
    second = kb.index_document_extraction(
        extraction,
        embedding_provider=KeywordEmbedder(),
        embedding_model="nvidia/nemotron-3-embed-1b",
    )

    assert first == second
    assert kb.count_chunks() == 4
    assert kb.get_chunks(
        chunk_kind="page_evidence", source_document_id="raw-doc-1"
    ) == second
    assert len(kb.get_chunks(product_record_id="product-1")) == 2
    page_match = kb.search_text(
        "SS-type", chunk_kind="page_evidence", limit=1
    )[0]
    assert page_match.chunk_kind == "page_evidence"
    assert page_match.product_record_id is None
    assert page_match.embedding_model == "nvidia/nemotron-3-embed-1b"

    replacement_page = extracted_page(texts=("Replacement page evidence",))
    replacement = DocumentExtraction(
        document_id=replacement_page.document_id,
        source_path="catalogs/lovejoy.pdf",
        pages=(replacement_page,),
    )
    replacement_chunks = kb.index_document_extraction(replacement)

    assert kb.count_chunks() == 3
    assert kb.get_chunks(chunk_kind="page_evidence") == replacement_chunks
    assert kb.search_text("SS-type", chunk_kind="page_evidence") == []
    assert len(kb.get_chunks(product_record_id="product-1")) == 2


class FailOnceEmbedder:
    def __init__(self) -> None:
        self.calls = 0

    def embed_documents(self, texts):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("temporary embedding failure")
        return [[0.25, 0.75] for _ in texts]


def test_page_index_retry_keeps_last_good_state_and_does_not_duplicate(tmp_path) -> None:
    kb = SQLiteKnowledgeBase(tmp_path / "retry.sqlite3")
    original_page = extracted_page(texts=("Original page evidence",))
    original_chunks = kb.index_page_extraction(
        original_page, source_path="catalogs/lovejoy.pdf"
    )
    replacement_page = extracted_page(texts=("Retried page evidence",))
    embedder = FailOnceEmbedder()

    try:
        kb.index_page_extraction(
            replacement_page,
            source_path="catalogs/lovejoy.pdf",
            embedding_provider=embedder,
            embedding_model="test-embedder",
        )
    except RuntimeError as exc:
        assert "temporary embedding failure" in str(exc)
    else:
        raise AssertionError("embedding failure did not propagate")

    assert kb.get_chunks(chunk_kind="page_evidence") == original_chunks

    retried = kb.index_page_extraction(
        replacement_page,
        source_path="catalogs/lovejoy.pdf",
        embedding_provider=embedder,
        embedding_model="test-embedder",
    )
    repeated = kb.index_page_extraction(
        replacement_page,
        source_path="catalogs/lovejoy.pdf",
        embedding_provider=embedder,
        embedding_model="test-embedder",
    )

    assert retried == repeated
    assert kb.count_chunks() == 1
    assert kb.get_chunks(chunk_kind="page_evidence") == retried
    assert retried[0].text == "Retried page evidence"


def test_initialize_migrates_legacy_product_only_knowledge_base(tmp_path) -> None:
    path = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE knowledge_chunks (
                chunk_id TEXT PRIMARY KEY,
                schema_version TEXT NOT NULL,
                content_sha256 TEXT NOT NULL,
                product_record_id TEXT NOT NULL,
                source_document_id TEXT NOT NULL,
                source_path TEXT,
                chunk_type TEXT NOT NULL,
                ordinal INTEGER NOT NULL,
                text TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                embedding_json TEXT,
                embedding_dimensions INTEGER,
                embedding_model TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(product_record_id, chunk_type, ordinal)
            );
            """
        )
        connection.execute(
            """
            INSERT INTO knowledge_chunks(
                chunk_id, schema_version, content_sha256, product_record_id,
                source_document_id, source_path, chunk_type, ordinal, text,
                metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "legacy-chunk",
                "1.0",
                "0" * 64,
                "legacy-product",
                "legacy-document",
                "catalogs/legacy.pdf",
                "identity",
                0,
                "Legacy product evidence",
                "{}",
            ),
        )

    kb = SQLiteKnowledgeBase(path)

    restored = kb.get_chunks(product_record_id="legacy-product")
    assert len(restored) == 1
    assert restored[0].chunk_kind == "product"
    assert restored[0].page_number is None
    with sqlite3.connect(path) as connection:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(knowledge_chunks)")
        }
    assert {"chunk_kind", "page_number"} <= columns


def test_reindex_replaces_stale_chunks_deterministically(tmp_path) -> None:
    kb = SQLiteKnowledgeBase(tmp_path / "knowledge.sqlite3")
    original = product()
    original_chunks = kb.index_product(original)
    specification = original.specifications[0]
    changed_value = specification.value.model_copy(
        update={"raw": "-30 to 90", "normalized": "-30 to 90"}
    )
    changed_specification = specification.model_copy(update={"value": changed_value})
    changed = original.model_copy(update={"specifications": [changed_specification]})

    changed_chunks = kb.index_product(changed)

    assert kb.count_chunks() == 2
    assert original_chunks[0].chunk_id == changed_chunks[0].chunk_id
    assert original_chunks[1].chunk_id != changed_chunks[1].chunk_id
    assert "-30 to 90" in kb.get_chunks(product_record_id="product-1")[1].text


class RecordingEmbeddingTransport:
    def __init__(self) -> None:
        self.payloads = []

    def post_json(self, url, *, headers, payload, timeout_seconds):
        self.payloads.append((url, headers, payload, timeout_seconds))
        return TransportResponse(
            200,
            {
                "model": payload["model"],
                "data": [
                    {"index": index, "embedding": [float(index), 1.0, 0.5]}
                    for index, _ in enumerate(payload["input"])
                ],
            },
        )


def test_nvidia_embedding_client_uses_retrieval_prefixes() -> None:
    transport = RecordingEmbeddingTransport()
    client = NvidiaEmbeddingClient(
        NIMEndpointConfig(
            base_url="http://127.0.0.1:8004",
            endpoint_path="/v1/embeddings",
            model="nvidia/Nemotron-3-Embed-8B-BF16",
        ),
        transport=transport,
    )

    vectors = client.embed_documents(["pump", "valve"])
    query = client.embed_query("find pumps")

    assert vectors == [[0.0, 1.0, 0.5], [1.0, 1.0, 0.5]]
    assert query == [0.0, 1.0, 0.5]
    assert transport.payloads[0][2]["input"] == ["passage: pump", "passage: valve"]
    assert transport.payloads[1][2]["input"] == ["query: find pumps"]


def test_query_cli_returns_citation_bearing_text_results(tmp_path, capsys) -> None:
    path = tmp_path / "knowledge.sqlite3"
    SQLiteKnowledgeBase(path).index_product(product())

    exit_code = main(
        [
            "query",
            "TS-100",
            "--kb",
            str(path),
            "--text-only",
            "--limit",
            "1",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["retrieval_mode"] == "text"
    assert payload["count"] == 1
    assert payload["results"][0]["chunk"]["citations"]
    assert "embedding" not in payload["results"][0]["chunk"]


def test_query_cli_does_not_create_a_missing_database(tmp_path, capsys) -> None:
    path = tmp_path / "missing.sqlite3"

    exit_code = main(["query", "pump", "--kb", str(path), "--text-only"])

    payload = json.loads(capsys.readouterr().err)
    assert exit_code == 2
    assert payload["error_type"] == "ValueError"
    assert "does not exist" in payload["error"]
    assert not path.exists()
