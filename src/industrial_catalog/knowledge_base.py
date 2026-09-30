"""Deterministic, citation-bearing RAG chunks backed by SQLite.

This is the local knowledge-base foundation.  It intentionally keeps chunking
and storage independent from a specific embedding model; an NVIDIA embedding
NIM (or any compatible callable) can be supplied when indexing without changing
the canonical chunk format.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import threading
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import Field, model_validator

from .extraction import DocumentExtraction, ExtractedElement, PageExtraction
from .models import (
    BoundingBox,
    CatalogModel,
    Evidence,
    ExtractedValue,
    JsonValue,
    ProductRecord,
    Specification,
)
from .nvidia_clients import (
    JsonTransport,
    NIMEndpointConfig,
    NIMRequestError,
    UrllibJsonTransport,
)

KB_SCHEMA_VERSION = "1.1"
ChunkKind = Literal["product", "page_evidence"]
ChunkType = Literal[
    "identity",
    "specification",
    "operating_condition",
    "page_element",
]


def _stable_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _display_value(value: ExtractedValue[Any] | None) -> str:
    if value is None:
        return ""
    candidate = value.normalized if value.normalized is not None else value.raw
    if candidate is None:
        return ""
    if isinstance(candidate, (dict, list)):
        return _stable_json(candidate)
    return str(candidate).strip()


class SourceCitation(CatalogModel):
    """Exact source location supporting one or more fields in a RAG chunk."""

    source_document_id: str = Field(min_length=1)
    source_path: str | None = None
    page_number: int = Field(ge=1)
    element_id: str | None = None
    bbox: BoundingBox | None = None
    excerpt: str = Field(min_length=1)
    confidence: float | None = Field(default=None, ge=0, le=1)
    field_paths: list[str] = Field(min_length=1)


class KnowledgeChunk(CatalogModel):
    """A deterministic retrieval unit from a product or raw page evidence."""

    schema_version: str = KB_SCHEMA_VERSION
    chunk_id: str = Field(min_length=1)
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    chunk_kind: ChunkKind = "product"
    product_record_id: str | None = Field(default=None, min_length=1)
    source_document_id: str = Field(min_length=1)
    source_path: str | None = None
    page_number: int | None = Field(default=None, ge=1)
    chunk_type: ChunkType
    ordinal: int = Field(ge=0)
    text: str = Field(min_length=1)
    citations: list[SourceCitation] = Field(default_factory=list)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    embedding: list[float] | None = None
    embedding_model: str | None = None

    @model_validator(mode="after")
    def validate_embedding(self) -> KnowledgeChunk:
        if self.chunk_kind == "product":
            if self.product_record_id is None:
                raise ValueError("product chunks require product_record_id")
            if self.chunk_type == "page_element":
                raise ValueError("product chunks cannot use page_element chunk_type")
        else:
            if self.product_record_id is not None:
                raise ValueError("page-evidence chunks cannot have product_record_id")
            if self.page_number is None:
                raise ValueError("page-evidence chunks require page_number")
            if self.chunk_type != "page_element":
                raise ValueError("page-evidence chunks must use page_element chunk_type")
        if self.embedding is not None:
            if not self.embedding:
                raise ValueError("embedding cannot be empty")
            if not all(math.isfinite(value) for value in self.embedding):
                raise ValueError("embedding values must be finite")
        return self


class KnowledgeSearchResult(CatalogModel):
    chunk: KnowledgeChunk
    score: float


class EmbeddingProvider(Protocol):
    """Protocol implemented by NVIDIA NIM and local embedding adapters."""

    def embed_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """Embed texts in the supplied order."""


class NvidiaEmbeddingClient:
    """OpenAI-compatible NVIDIA embedding adapter with deterministic retries."""

    def __init__(
        self,
        config: NIMEndpointConfig,
        *,
        transport: JsonTransport | None = None,
        batch_size: int = 16,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.config = config
        self.transport = transport or UrllibJsonTransport()
        self.batch_size = batch_size

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            **dict(self.config.extra_headers),
        }
        if api_key := self.config.resolved_api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def _embed(self, texts: Sequence[str], *, prefix: str) -> list[list[float]]:
        if not texts:
            return []
        output: list[list[float]] = []
        for offset in range(0, len(texts), self.batch_size):
            batch = [
                f"{prefix}: {str(text).strip()}"
                for text in texts[offset : offset + self.batch_size]
            ]
            payload = {
                "model": self.config.model,
                "input": batch,
                "encoding_format": "float",
            }
            response = None
            policy = self.config.retry
            for attempt in range(1, policy.max_attempts + 1):
                if attempt > 1:
                    time.sleep(policy.delay_before_attempt(attempt))
                try:
                    response = self.transport.post_json(
                        self.config.url,
                        headers=self._headers(),
                        payload=payload,
                        timeout_seconds=self.config.timeout_seconds,
                    )
                except OSError as exc:
                    if attempt < policy.max_attempts:
                        continue
                    raise NIMRequestError(
                        f"NVIDIA embedding request failed after {attempt} attempts: {exc}",
                        attempts=attempt,
                    ) from exc
                if 200 <= response.status_code < 300:
                    break
                if (
                    response.status_code in policy.retry_status_codes
                    and attempt < policy.max_attempts
                ):
                    continue
                raise NIMRequestError(
                    f"NVIDIA embedding endpoint returned HTTP {response.status_code}",
                    status_code=response.status_code,
                    response=response.body,
                    attempts=attempt,
                )
            if response is None or not 200 <= response.status_code < 300:
                raise NIMRequestError("NVIDIA embedding request did not return a response")
            raw_data = response.body.get("data")
            if not isinstance(raw_data, list):
                raise ValueError("embedding response does not contain a data array")
            ordered = sorted(
                (item for item in raw_data if isinstance(item, Mapping)),
                key=lambda item: int(item.get("index", 0)),
            )
            if len(ordered) != len(batch):
                raise ValueError("embedding response count does not match request count")
            for item in ordered:
                vector = item.get("embedding")
                if not isinstance(vector, list) or not vector:
                    raise ValueError("embedding response contains an invalid vector")
                output.append([float(value) for value in vector])
        return output

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed(texts, prefix="passage")

    def embed_query(self, text: str) -> list[float]:
        vectors = self._embed([text], prefix="query")
        return vectors[0]


def _citation_key(evidence: Evidence) -> str:
    return _stable_json(
        {
            "source_document_id": evidence.source_document_id,
            "source_path": evidence.source_path,
            "page_number": evidence.page_number,
            "element_id": evidence.element_id,
            "bbox": evidence.bbox.model_dump(mode="json") if evidence.bbox else None,
            "text": evidence.text,
        }
    )


def _merge_citations(
    evidence_fields: Iterable[tuple[str, Evidence]],
) -> list[SourceCitation]:
    grouped: dict[str, tuple[Evidence, set[str]]] = {}
    for field_path, evidence in evidence_fields:
        key = _citation_key(evidence)
        if key not in grouped:
            grouped[key] = (evidence, set())
        grouped[key][1].add(field_path)
    output: list[SourceCitation] = []
    for evidence, field_paths in grouped.values():
        output.append(
            SourceCitation(
                source_document_id=evidence.source_document_id,
                source_path=evidence.source_path,
                page_number=evidence.page_number,
                element_id=evidence.element_id,
                bbox=evidence.bbox,
                excerpt=evidence.text,
                confidence=evidence.confidence,
                field_paths=sorted(field_paths),
            )
        )
    return sorted(
        output,
        key=lambda item: (
            item.page_number,
            item.element_id or "",
            item.excerpt,
            item.field_paths,
        ),
    )


def _identity_values(product: ProductRecord) -> list[tuple[str, ExtractedValue[Any]]]:
    return [
        (name, value)
        for name in ("manufacturer", "part_name", "part_number", "category", "description")
        if (value := getattr(product, name)) is not None
    ]


def _identity_label(product: ProductRecord) -> str:
    parts = [
        _display_value(product.manufacturer),
        _display_value(product.part_name),
        _display_value(product.part_number),
    ]
    return " | ".join(part for part in parts if part)


def _make_chunk(
    *,
    product: ProductRecord,
    chunk_type: ChunkType,
    ordinal: int,
    text: str,
    evidence_fields: Iterable[tuple[str, Evidence]],
    metadata: Mapping[str, JsonValue],
) -> KnowledgeChunk:
    citations = _merge_citations(evidence_fields)
    hash_payload = {
        "schema_version": KB_SCHEMA_VERSION,
        "product_record_id": product.record_id,
        "chunk_type": chunk_type,
        "ordinal": ordinal,
        "text": text,
        "citations": [citation.model_dump(mode="json") for citation in citations],
        "metadata": dict(metadata),
    }
    content_sha256 = hashlib.sha256(_stable_json(hash_payload).encode()).hexdigest()
    return KnowledgeChunk(
        chunk_id=f"kb_{content_sha256[:32]}",
        content_sha256=content_sha256,
        product_record_id=product.record_id,
        source_document_id=product.source_document_id,
        source_path=product.source_path,
        chunk_type=chunk_type,
        ordinal=ordinal,
        text=text,
        citations=citations,
        metadata=dict(metadata),
    )


def _specification_chunk(
    product: ProductRecord,
    specification: Specification,
    *,
    collection_name: Literal["specifications", "operating_conditions"],
    ordinal: int,
) -> KnowledgeChunk:
    name = _display_value(specification.name)
    value = _display_value(specification.value)
    unit = _display_value(specification.unit)
    category = _display_value(specification.category)
    qualifier = _display_value(specification.qualifier)
    value_with_unit = " ".join(item for item in (value, unit) if item)
    lines = [f"Product: {_identity_label(product)}"]
    label = "Specification" if collection_name == "specifications" else "Operating condition"
    lines.append(f"{label}: {name} = {value_with_unit}".rstrip(" ="))
    if category:
        lines.append(f"Category: {category}")
    if qualifier:
        lines.append(f"Qualifier: {qualifier}")

    evidence_fields: list[tuple[str, Evidence]] = []
    for field_name, field_value in _identity_values(product):
        evidence_fields.extend((field_name, item) for item in field_value.evidence)
    for field_name in ("name", "value", "unit", "category", "qualifier"):
        field_value = getattr(specification, field_name)
        if field_value is not None:
            path = f"{collection_name}.{ordinal}.{field_name}"
            evidence_fields.extend((path, item) for item in field_value.evidence)

    metadata: dict[str, JsonValue] = {
        "manufacturer": _display_value(product.manufacturer),
        "part_name": _display_value(product.part_name),
        "part_number": _display_value(product.part_number),
        "category": _display_value(product.category),
        "specification_name": name,
        "unit": unit,
        "collection": collection_name,
    }
    return _make_chunk(
        product=product,
        chunk_type=(
            "specification" if collection_name == "specifications" else "operating_condition"
        ),
        ordinal=ordinal,
        text="\n".join(lines),
        evidence_fields=evidence_fields,
        metadata=metadata,
    )


def generate_product_chunks(product: ProductRecord | Mapping[str, Any]) -> list[KnowledgeChunk]:
    """Generate stable identity and one-per-specification retrieval chunks."""

    parsed = (
        product if isinstance(product, ProductRecord) else ProductRecord.model_validate(product)
    )
    identity_values = _identity_values(parsed)
    chunks: list[KnowledgeChunk] = []
    if identity_values:
        labels = {
            "manufacturer": "Manufacturer",
            "part_name": "Part name",
            "part_number": "Part number",
            "category": "Category",
            "description": "Description",
        }
        lines = ["Industrial catalog product"]
        evidence_fields: list[tuple[str, Evidence]] = []
        for field_name, value in identity_values:
            display = _display_value(value)
            if display:
                lines.append(f"{labels[field_name]}: {display}")
            evidence_fields.extend((field_name, item) for item in value.evidence)
        chunks.append(
            _make_chunk(
                product=parsed,
                chunk_type="identity",
                ordinal=0,
                text="\n".join(lines),
                evidence_fields=evidence_fields,
                metadata={
                    "manufacturer": _display_value(parsed.manufacturer),
                    "part_name": _display_value(parsed.part_name),
                    "part_number": _display_value(parsed.part_number),
                    "category": _display_value(parsed.category),
                },
            )
        )
    chunks.extend(
        _specification_chunk(
            parsed,
            specification,
            collection_name="specifications",
            ordinal=index,
        )
        for index, specification in enumerate(parsed.specifications)
    )
    chunks.extend(
        _specification_chunk(
            parsed,
            specification,
            collection_name="operating_conditions",
            ordinal=index,
        )
        for index, specification in enumerate(parsed.operating_conditions)
    )
    return chunks


def _page_element_bbox(element: ExtractedElement) -> BoundingBox | None:
    if element.bbox is None:
        return None
    coordinate_space = element.metadata.get("bbox_coordinate_space", "pixel")
    if coordinate_space not in {"pixel", "point", "normalized"}:
        coordinate_space = "pixel"
    left, top, right, bottom = element.bbox.to_list()
    return BoundingBox(
        x0=left,
        y0=top,
        x1=right,
        y1=bottom,
        coordinate_space=coordinate_space,
    )


def _page_element_chunk(
    page: PageExtraction,
    element: ExtractedElement,
    *,
    ordinal: int,
    source_path: str | None,
    pipeline_version: str | None,
) -> KnowledgeChunk:
    resolved_source_path = element.source_path or source_path
    citation = SourceCitation(
        source_document_id=element.document_id,
        source_path=resolved_source_path,
        page_number=element.page_number,
        element_id=element.element_id,
        bbox=_page_element_bbox(element),
        excerpt=element.text,
        confidence=element.confidence,
        field_paths=[
            f"pages.{page.page_number}.elements.{element.sequence_number}.text"
        ],
    )
    metadata: dict[str, JsonValue] = {
        "page_number": page.page_number,
        "page_fingerprint": page.fingerprint,
        "element_id": element.element_id,
        "element_sequence_number": element.sequence_number,
        "element_type": element.element_type,
        "extraction_method": element.extraction_method,
        "routing_route": page.routing.route.value,
        "routing_reasons": list(page.routing.reasons),
    }
    if element.model is not None:
        metadata["extraction_model"] = element.model
    if pipeline_version is not None:
        metadata["pipeline_version"] = pipeline_version
    hash_payload = {
        "schema_version": KB_SCHEMA_VERSION,
        "chunk_kind": "page_evidence",
        "source_document_id": page.document_id,
        "source_path": resolved_source_path,
        "page_number": page.page_number,
        "chunk_type": "page_element",
        "ordinal": ordinal,
        "text": element.text,
        "citations": [citation.model_dump(mode="json")],
        "metadata": metadata,
    }
    content_sha256 = hashlib.sha256(_stable_json(hash_payload).encode()).hexdigest()
    return KnowledgeChunk(
        chunk_id=f"kb_{content_sha256[:32]}",
        content_sha256=content_sha256,
        chunk_kind="page_evidence",
        product_record_id=None,
        source_document_id=page.document_id,
        source_path=resolved_source_path,
        page_number=page.page_number,
        chunk_type="page_element",
        ordinal=ordinal,
        text=element.text,
        citations=[citation],
        metadata=metadata,
    )


def generate_page_chunks(
    page: PageExtraction,
    *,
    source_path: str | None = None,
    pipeline_version: str | None = None,
) -> list[KnowledgeChunk]:
    """Generate one exact, citation-bearing retrieval chunk per page element.

    Empty elements are omitted. Element ordering is normalized so equivalent
    checkpoint restores and retries produce the same chunk IDs.
    """

    ordered_elements = sorted(
        page.elements,
        key=lambda element: (element.sequence_number, element.element_id),
    )
    chunks: list[KnowledgeChunk] = []
    for element in ordered_elements:
        if element.document_id != page.document_id:
            raise ValueError(
                f"element {element.element_id} document_id does not match its page"
            )
        if element.page_number != page.page_number:
            raise ValueError(
                f"element {element.element_id} page_number does not match its page"
            )
        if not element.text.strip():
            continue
        chunks.append(
            _page_element_chunk(
                page,
                element,
                ordinal=len(chunks),
                source_path=source_path,
                pipeline_version=pipeline_version,
            )
        )
    return chunks


def generate_document_chunks(extraction: DocumentExtraction) -> list[KnowledgeChunk]:
    """Generate deterministic page-evidence chunks for a complete document."""

    pages = sorted(extraction.pages, key=lambda page: page.page_number)
    page_numbers = [page.page_number for page in pages]
    if len(page_numbers) != len(set(page_numbers)):
        raise ValueError("document extraction contains duplicate page numbers")
    chunks: list[KnowledgeChunk] = []
    for page in pages:
        if page.document_id != extraction.document_id:
            raise ValueError(
                f"page {page.page_number} document_id does not match its document"
            )
        chunks.extend(
            generate_page_chunks(
                page,
                source_path=extraction.source_path,
                pipeline_version=extraction.pipeline_version,
            )
        )
    return chunks


def _embed_texts(provider: Any, texts: Sequence[str]) -> list[list[float]]:
    embed_documents = getattr(provider, "embed_documents", None)
    vectors = embed_documents(texts) if callable(embed_documents) else provider(texts)
    vectors = [list(vector) for vector in vectors]
    if len(vectors) != len(texts):
        raise ValueError("embedding provider returned a different number of vectors than texts")
    dimensions: int | None = None
    for vector in vectors:
        if not vector:
            raise ValueError("embedding provider returned an empty vector")
        if dimensions is None:
            dimensions = len(vector)
        elif len(vector) != dimensions:
            raise ValueError("embedding vectors must have consistent dimensions")
        if not all(math.isfinite(float(value)) for value in vector):
            raise ValueError("embedding vectors must contain finite numbers")
    return [[float(value) for value in vector] for vector in vectors]


def _page_storage_owner(source_document_id: str, page_number: int) -> str:
    """Return a private owner key for the legacy non-null product column."""

    digest = hashlib.sha256(
        _stable_json([source_document_id, page_number]).encode()
    ).hexdigest()
    return f"__page_evidence_{digest[:32]}"


class SQLiteKnowledgeBase:
    """SQLite chunk/vector store usable standalone or beside the catalog tables."""

    def __init__(self, db_path: str | Path, *, timeout: float = 30.0):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self._lock = threading.RLock()
        self.initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=self.timeout)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(f"PRAGMA busy_timeout = {int(self.timeout * 1000)}")
        return connection

    def initialize(self) -> None:
        with self._lock, self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS knowledge_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    schema_version TEXT NOT NULL,
                    content_sha256 TEXT NOT NULL,
                    chunk_kind TEXT NOT NULL DEFAULT 'product',
                    product_record_id TEXT NOT NULL,
                    source_document_id TEXT NOT NULL,
                    source_path TEXT,
                    page_number INTEGER,
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

                CREATE TABLE IF NOT EXISTS knowledge_chunk_citations (
                    chunk_id TEXT NOT NULL,
                    ordinal INTEGER NOT NULL,
                    source_document_id TEXT NOT NULL,
                    source_path TEXT,
                    page_number INTEGER NOT NULL,
                    element_id TEXT,
                    bbox_json TEXT,
                    excerpt TEXT NOT NULL,
                    confidence REAL,
                    field_paths_json TEXT NOT NULL,
                    PRIMARY KEY(chunk_id, ordinal),
                    FOREIGN KEY(chunk_id) REFERENCES knowledge_chunks(chunk_id)
                        ON UPDATE CASCADE ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_knowledge_product
                    ON knowledge_chunks(product_record_id);
                CREATE INDEX IF NOT EXISTS idx_knowledge_document
                    ON knowledge_chunks(source_document_id);
                CREATE INDEX IF NOT EXISTS idx_knowledge_type
                    ON knowledge_chunks(chunk_type);
                CREATE INDEX IF NOT EXISTS idx_knowledge_citation_page
                    ON knowledge_chunk_citations(source_document_id, page_number);
                """
            )
            columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(knowledge_chunks)")
            }
            if "chunk_kind" not in columns:
                connection.execute(
                    "ALTER TABLE knowledge_chunks "
                    "ADD COLUMN chunk_kind TEXT NOT NULL DEFAULT 'product'"
                )
            if "page_number" not in columns:
                connection.execute(
                    "ALTER TABLE knowledge_chunks ADD COLUMN page_number INTEGER"
                )
            connection.executescript(
                """
                CREATE INDEX IF NOT EXISTS idx_knowledge_kind
                    ON knowledge_chunks(chunk_kind);
                CREATE INDEX IF NOT EXISTS idx_knowledge_page_scope
                    ON knowledge_chunks(
                        chunk_kind, source_document_id, page_number
                    );
                """
            )

    @staticmethod
    def _chunks_with_embeddings(
        chunks: Sequence[KnowledgeChunk],
        *,
        embedding_provider: EmbeddingProvider | Any | None,
        embedding_model: str | None,
    ) -> list[KnowledgeChunk]:
        output = list(chunks)
        if embedding_provider is None or not output:
            return output
        vectors = _embed_texts(embedding_provider, [chunk.text for chunk in output])
        return [
            chunk.model_copy(
                update={"embedding": vector, "embedding_model": embedding_model}
            )
            for chunk, vector in zip(output, vectors, strict=True)
        ]

    @staticmethod
    def _insert_chunks(
        connection: sqlite3.Connection,
        chunks: Iterable[KnowledgeChunk],
    ) -> None:
        for chunk in chunks:
            storage_owner = chunk.product_record_id
            if storage_owner is None:
                if chunk.page_number is None:
                    raise ValueError("page-evidence chunk is missing page_number")
                storage_owner = _page_storage_owner(
                    chunk.source_document_id, chunk.page_number
                )
            connection.execute(
                """
                INSERT INTO knowledge_chunks(
                    chunk_id, schema_version, content_sha256, chunk_kind,
                    product_record_id, source_document_id, source_path,
                    page_number, chunk_type, ordinal, text, metadata_json,
                    embedding_json, embedding_dimensions, embedding_model
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chunk.chunk_id,
                    chunk.schema_version,
                    chunk.content_sha256,
                    chunk.chunk_kind,
                    storage_owner,
                    chunk.source_document_id,
                    chunk.source_path,
                    chunk.page_number,
                    chunk.chunk_type,
                    chunk.ordinal,
                    chunk.text,
                    _stable_json(chunk.metadata),
                    _stable_json(chunk.embedding)
                    if chunk.embedding is not None
                    else None,
                    len(chunk.embedding) if chunk.embedding is not None else None,
                    chunk.embedding_model,
                ),
            )
            for ordinal, citation in enumerate(chunk.citations):
                connection.execute(
                    """
                    INSERT INTO knowledge_chunk_citations(
                        chunk_id, ordinal, source_document_id, source_path,
                        page_number, element_id, bbox_json, excerpt, confidence,
                        field_paths_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk.chunk_id,
                        ordinal,
                        citation.source_document_id,
                        citation.source_path,
                        citation.page_number,
                        citation.element_id,
                        citation.bbox.model_dump_json(exclude_none=True)
                        if citation.bbox is not None
                        else None,
                        citation.excerpt,
                        citation.confidence,
                        _stable_json(citation.field_paths),
                    ),
                )

    def index_product(
        self,
        product: ProductRecord | Mapping[str, Any],
        *,
        embedding_provider: EmbeddingProvider | Any | None = None,
        embedding_model: str | None = None,
    ) -> list[KnowledgeChunk]:
        """Replace all chunks for one product, removing stale prior chunks."""

        parsed = (
            product if isinstance(product, ProductRecord) else ProductRecord.model_validate(product)
        )
        chunks = self._chunks_with_embeddings(
            generate_product_chunks(parsed),
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
        )
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                DELETE FROM knowledge_chunks
                WHERE chunk_kind = 'product' AND product_record_id = ?
                """,
                (parsed.record_id,),
            )
            self._insert_chunks(connection, chunks)
        return chunks

    def index_products(
        self,
        products: Iterable[ProductRecord | Mapping[str, Any]],
        *,
        embedding_provider: EmbeddingProvider | Any | None = None,
        embedding_model: str | None = None,
    ) -> list[KnowledgeChunk]:
        output: list[KnowledgeChunk] = []
        for product in products:
            output.extend(
                self.index_product(
                    product,
                    embedding_provider=embedding_provider,
                    embedding_model=embedding_model,
                )
            )
        return output

    def index_page_extraction(
        self,
        page: PageExtraction,
        *,
        source_path: str | None = None,
        pipeline_version: str | None = None,
        embedding_provider: EmbeddingProvider | Any | None = None,
        embedding_model: str | None = None,
    ) -> list[KnowledgeChunk]:
        """Atomically replace the raw evidence chunks for one extracted page.

        Embeddings are completed before the transaction starts. A failed model
        request therefore leaves the last successfully indexed page intact and
        makes a caller retry safe.
        """

        chunks = self._chunks_with_embeddings(
            generate_page_chunks(
                page,
                source_path=source_path,
                pipeline_version=pipeline_version,
            ),
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
        )
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                DELETE FROM knowledge_chunks
                WHERE chunk_kind = 'page_evidence'
                  AND source_document_id = ? AND page_number = ?
                """,
                (page.document_id, page.page_number),
            )
            self._insert_chunks(connection, chunks)
        return chunks

    def index_document_extraction(
        self,
        extraction: DocumentExtraction,
        *,
        embedding_provider: EmbeddingProvider | Any | None = None,
        embedding_model: str | None = None,
    ) -> list[KnowledgeChunk]:
        """Atomically replace all raw page evidence for one document."""

        chunks = self._chunks_with_embeddings(
            generate_document_chunks(extraction),
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
        )
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                DELETE FROM knowledge_chunks
                WHERE chunk_kind = 'page_evidence' AND source_document_id = ?
                """,
                (extraction.document_id,),
            )
            self._insert_chunks(connection, chunks)
        return chunks

    def _citations(self, connection: sqlite3.Connection, chunk_id: str) -> list[SourceCitation]:
        rows = connection.execute(
            """
            SELECT * FROM knowledge_chunk_citations
            WHERE chunk_id = ? ORDER BY ordinal
            """,
            (chunk_id,),
        ).fetchall()
        return [
            SourceCitation(
                source_document_id=row["source_document_id"],
                source_path=row["source_path"],
                page_number=row["page_number"],
                element_id=row["element_id"],
                bbox=(
                    BoundingBox.model_validate_json(row["bbox_json"]) if row["bbox_json"] else None
                ),
                excerpt=row["excerpt"],
                confidence=row["confidence"],
                field_paths=json.loads(row["field_paths_json"]),
            )
            for row in rows
        ]

    def _chunk_from_row(self, connection: sqlite3.Connection, row: sqlite3.Row) -> KnowledgeChunk:
        chunk_kind: ChunkKind = row["chunk_kind"]
        return KnowledgeChunk(
            schema_version=row["schema_version"],
            chunk_id=row["chunk_id"],
            content_sha256=row["content_sha256"],
            chunk_kind=chunk_kind,
            product_record_id=(
                row["product_record_id"] if chunk_kind == "product" else None
            ),
            source_document_id=row["source_document_id"],
            source_path=row["source_path"],
            page_number=row["page_number"],
            chunk_type=row["chunk_type"],
            ordinal=row["ordinal"],
            text=row["text"],
            citations=self._citations(connection, row["chunk_id"]),
            metadata=json.loads(row["metadata_json"]),
            embedding=json.loads(row["embedding_json"]) if row["embedding_json"] else None,
            embedding_model=row["embedding_model"],
        )

    def get_chunks(
        self,
        *,
        product_record_id: str | None = None,
        chunk_kind: ChunkKind | None = None,
        source_document_id: str | None = None,
        page_number: int | None = None,
    ) -> list[KnowledgeChunk]:
        query = "SELECT * FROM knowledge_chunks"
        clauses: list[str] = []
        parameters: list[Any] = []
        if product_record_id is not None:
            clauses.extend(["chunk_kind = 'product'", "product_record_id = ?"])
            parameters.append(product_record_id)
        if chunk_kind is not None:
            clauses.append("chunk_kind = ?")
            parameters.append(chunk_kind)
        if source_document_id is not None:
            clauses.append("source_document_id = ?")
            parameters.append(source_document_id)
        if page_number is not None:
            if page_number < 1:
                raise ValueError("page_number must be positive")
            clauses.append("page_number = ?")
            parameters.append(page_number)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += (
            " ORDER BY chunk_kind, source_document_id, page_number, "
            "product_record_id, chunk_type, ordinal"
        )
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
            return [self._chunk_from_row(connection, row) for row in rows]

    def search_text(
        self,
        query: str,
        *,
        limit: int = 20,
        chunk_kind: ChunkKind | None = None,
    ) -> list[KnowledgeChunk]:
        if limit < 1:
            raise ValueError("limit must be positive")
        escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        kind_clause = "" if chunk_kind is None else " AND chunk_kind = ?"
        parameters: list[Any] = [f"%{escaped}%"]
        if chunk_kind is not None:
            parameters.append(chunk_kind)
        parameters.append(limit)
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT * FROM knowledge_chunks
                WHERE text LIKE ? ESCAPE '\\' COLLATE NOCASE
                {kind_clause}
                ORDER BY chunk_kind, source_document_id, page_number,
                         product_record_id, chunk_type, ordinal
                LIMIT ?
                """,
                parameters,
            ).fetchall()
            return [self._chunk_from_row(connection, row) for row in rows]

    def similarity_search(
        self,
        query_embedding: Sequence[float],
        *,
        limit: int = 10,
        chunk_kind: ChunkKind | None = None,
    ) -> list[KnowledgeSearchResult]:
        """Exact cosine search; replaceable later by sqlite-vec or a vector DB."""

        if limit < 1:
            raise ValueError("limit must be positive")
        query_vector = [float(value) for value in query_embedding]
        if not query_vector or not all(math.isfinite(value) for value in query_vector):
            raise ValueError("query embedding must contain finite values")
        query_norm = math.sqrt(sum(value * value for value in query_vector))
        if query_norm == 0:
            raise ValueError("query embedding cannot be the zero vector")

        scored: list[KnowledgeSearchResult] = []
        with self._connect() as connection:
            query = "SELECT * FROM knowledge_chunks WHERE embedding_json IS NOT NULL"
            parameters: tuple[Any, ...] = ()
            if chunk_kind is not None:
                query += " AND chunk_kind = ?"
                parameters = (chunk_kind,)
            rows = connection.execute(query, parameters).fetchall()
            for row in rows:
                vector = [float(value) for value in json.loads(row["embedding_json"])]
                if len(vector) != len(query_vector):
                    continue
                vector_norm = math.sqrt(sum(value * value for value in vector))
                if vector_norm == 0:
                    continue
                score = sum(
                    left * right
                    for left, right in zip(query_vector, vector, strict=True)
                ) / (query_norm * vector_norm)
                scored.append(
                    KnowledgeSearchResult(
                        chunk=self._chunk_from_row(connection, row),
                        score=score,
                    )
                )
        return sorted(scored, key=lambda result: (-result.score, result.chunk.chunk_id))[:limit]

    def count_chunks(self) -> int:
        with self._connect() as connection:
            row = connection.execute("SELECT COUNT(*) AS count FROM knowledge_chunks").fetchone()
        return int(row["count"])

    def delete_product(self, product_record_id: str) -> int:
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """
                DELETE FROM knowledge_chunks
                WHERE chunk_kind = 'product' AND product_record_id = ?
                """,
                (product_record_id,),
            )
        return cursor.rowcount

    def delete_document_evidence(self, source_document_id: str) -> int:
        """Delete only raw page evidence, leaving canonical product chunks."""

        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """
                DELETE FROM knowledge_chunks
                WHERE chunk_kind = 'page_evidence' AND source_document_id = ?
                """,
                (source_document_id,),
            )
        return cursor.rowcount


__all__ = [
    "ChunkKind",
    "EmbeddingProvider",
    "KB_SCHEMA_VERSION",
    "KnowledgeChunk",
    "KnowledgeSearchResult",
    "NvidiaEmbeddingClient",
    "SQLiteKnowledgeBase",
    "SourceCitation",
    "generate_document_chunks",
    "generate_page_chunks",
    "generate_product_chunks",
]
