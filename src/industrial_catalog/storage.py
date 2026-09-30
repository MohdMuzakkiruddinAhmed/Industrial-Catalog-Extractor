"""SQLite and JSONL persistence for canonical industrial catalog records."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import (
    MAX_PRODUCTS_PER_BATCH,
    Evidence,
    ProductBatch,
    ProductRecord,
    SourceDocument,
    Specification,
)
from .validation import PreparedProduct, ValidationIssue, prepare_product_payload

SCHEMA_VERSION = 2


def _coerce_records(
    records: ProductRecord | Mapping[str, Any] | Iterable[ProductRecord | Mapping[str, Any]],
) -> list[ProductRecord]:
    if isinstance(records, ProductRecord):
        return [records]
    if isinstance(records, Mapping):
        return [ProductRecord.model_validate(records)]
    return [
        item if isinstance(item, ProductRecord) else ProductRecord.model_validate(item)
        for item in records
    ]


def append_jsonl(
    path: str | Path,
    records: ProductRecord | Mapping[str, Any] | Iterable[ProductRecord | Mapping[str, Any]],
) -> int:
    """Append canonical records to UTF-8 JSONL and return the number written."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    parsed = _coerce_records(records)
    with target.open("a", encoding="utf-8", newline="\n") as stream:
        for product in parsed:
            stream.write(product.model_dump_json(exclude_none=True))
            stream.write("\n")
        stream.flush()
    return len(parsed)


class JsonlWriter:
    """Small thread-safe writer for append-only extraction output."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock = threading.RLock()

    def append(
        self,
        records: ProductRecord | Mapping[str, Any] | Iterable[ProductRecord | Mapping[str, Any]],
    ) -> int:
        with self._lock:
            return append_jsonl(self.path, records)

    def overwrite(
        self,
        records: ProductRecord | Mapping[str, Any] | Iterable[ProductRecord | Mapping[str, Any]],
    ) -> int:
        parsed = _coerce_records(records)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.tmp")
        with self._lock:
            with temporary.open("w", encoding="utf-8", newline="\n") as stream:
                for product in parsed:
                    stream.write(product.model_dump_json(exclude_none=True))
                    stream.write("\n")
                stream.flush()
            temporary.replace(self.path)
        return len(parsed)


def _to_jsonable(value: Any) -> Any:
    """Best-effort lossless conversion for raw model/extraction journaling."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(item) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _to_jsonable(model_dump(mode="json"))
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _to_jsonable(to_dict())
    return repr(value)


class JsonEventWriter:
    """Thread-safe JSONL writer for versioned batch/rejection events."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock = threading.RLock()

    def append(self, event: Mapping[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(
            _to_jsonable(event),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        with self._lock, self.path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(encoded)
            stream.write("\n")
            stream.flush()


@dataclass(frozen=True, slots=True)
class BatchPersistenceResult:
    """Outcome of independently validating and persisting one parsed batch."""

    batch: ProductBatch
    prepared: tuple[PreparedProduct, ...]
    issues: tuple[ValidationIssue, ...]
    status: str
    raw_preserved: bool
    products_jsonl_written: bool
    batch_jsonl_written: bool

    @property
    def accepted_count(self) -> int:
        return len(self.batch.products)

    @property
    def rejected_count(self) -> int:
        return sum(not item.accepted for item in self.prepared)

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch.batch_id,
            "source_document_id": self.batch.source_document_id,
            "status": self.status,
            "accepted_count": self.accepted_count,
            "rejected_count": self.rejected_count,
            "raw_preserved": self.raw_preserved,
            "products_jsonl_written": self.products_jsonl_written,
            "batch_jsonl_written": self.batch_jsonl_written,
            "issues": [issue.model_dump(mode="json") for issue in self.issues],
        }


def _parsed_batch_data(parsed_output: Any) -> Any:
    """Return GuidedJsonResult.data or a directly supplied JSON-compatible value."""

    return getattr(parsed_output, "data", parsed_output)


def _stable_extraction_context(extraction: Any) -> list[dict[str, Any]]:
    """Return the ordered page scope used to identify a logical parse batch.

    Model response/request identifiers are deliberately excluded: they change
    across equivalent retries. Page numbers and extraction fingerprints are the
    stable boundary between otherwise identical (especially empty) parse results.
    """

    if isinstance(extraction, Mapping):
        pages = extraction.get("pages")
    else:
        pages = getattr(extraction, "pages", None)
    if not isinstance(pages, (list, tuple)):
        return []

    context: list[dict[str, Any]] = []
    for page in pages:
        if isinstance(page, Mapping):
            page_number = page.get("page_number")
            fingerprint = page.get("fingerprint")
        else:
            page_number = getattr(page, "page_number", None)
            fingerprint = getattr(page, "fingerprint", None)
        if page_number is None and fingerprint is None:
            continue
        try:
            normalized_page_number: int | str | None = (
                int(page_number) if page_number is not None else None
            )
        except (TypeError, ValueError):
            normalized_page_number = str(page_number)
        context.append(
            {
                "page_number": normalized_page_number,
                "fingerprint": str(fingerprint) if fingerprint is not None else None,
            }
        )
    return context


def _product_entries(parsed_data: Any) -> tuple[list[Any], list[ValidationIssue]]:
    if isinstance(parsed_data, Mapping):
        if "products" in parsed_data:
            products = parsed_data["products"]
            if isinstance(products, list):
                issues: list[ValidationIssue] = []
                if len(products) >= MAX_PRODUCTS_PER_BATCH:
                    issues.append(
                        ValidationIssue(
                            code="wire_product_limit_reached",
                            message=(
                                "products reached the structured-output limit of "
                                f"{MAX_PRODUCTS_PER_BATCH}; review the source for more rows"
                            ),
                            path="products",
                            severity="warning",
                        )
                    )
                return products, issues
            return [], [
                ValidationIssue(
                    code="invalid_product_batch",
                    message="top-level products must be a JSON array",
                    path="products",
                    severity="error",
                )
            ]
        # Accept a single product for defensive interoperability with early prompts.
        return [parsed_data], [
            ValidationIssue(
                code="single_product_envelope",
                message="top-level products array was absent; treated payload as one product",
                path="",
                severity="warning",
            )
        ]
    if isinstance(parsed_data, list):
        return parsed_data, [
            ValidationIssue(
                code="unversioned_product_array",
                message="bare product array accepted; expected an object with products",
                path="",
                severity="warning",
            )
        ]
    return [], [
        ValidationIssue(
            code="invalid_product_batch",
            message="parsed model output must be an object or array",
            path="",
            severity="error",
        )
    ]


class SQLiteCatalogStore:
    """Canonical SQLite product store with searchable columns and full JSON.

    The complete Pydantic payload is authoritative.  Selected identity columns,
    specifications, and field evidence are projected into relational tables for
    exact lookup, audits, and citation retrieval. Upserts replace projections in
    one transaction, preventing stale evidence after a record is reprocessed.
    """

    def __init__(
        self,
        db_path: str | Path,
        *,
        jsonl_path: str | Path | None = None,
        batch_jsonl_path: str | Path | None = None,
        timeout: float = 30.0,
    ):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.jsonl_writer = JsonlWriter(jsonl_path) if jsonl_path is not None else None
        if batch_jsonl_path is None and jsonl_path is not None:
            product_path = Path(jsonl_path)
            suffix = product_path.suffix or ".jsonl"
            batch_jsonl_path = product_path.with_name(f"{product_path.stem}.batches{suffix}")
        self.batch_jsonl_writer = (
            JsonEventWriter(batch_jsonl_path) if batch_jsonl_path is not None else None
        )
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
            connection.execute("PRAGMA synchronous = NORMAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS schema_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS documents (
                    source_document_id TEXT PRIMARY KEY,
                    source_path TEXT,
                    filename TEXT,
                    sha256 TEXT,
                    page_count INTEGER,
                    manufacturer_hint TEXT,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS products (
                    record_id TEXT PRIMARY KEY,
                    schema_version TEXT NOT NULL,
                    source_document_id TEXT NOT NULL,
                    source_path TEXT,
                    manufacturer_raw TEXT,
                    manufacturer_normalized TEXT COLLATE NOCASE,
                    part_name_raw TEXT,
                    part_name_normalized TEXT COLLATE NOCASE,
                    part_number_raw TEXT,
                    part_number_normalized TEXT COLLATE NOCASE,
                    category_normalized TEXT COLLATE NOCASE,
                    record_confidence REAL,
                    review_status TEXT NOT NULL,
                    extracted_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (source_document_id)
                        REFERENCES documents(source_document_id)
                        ON UPDATE CASCADE ON DELETE RESTRICT
                );

                CREATE TABLE IF NOT EXISTS specifications (
                    product_record_id TEXT NOT NULL,
                    collection_name TEXT NOT NULL,
                    specification_id TEXT NOT NULL,
                    ordinal INTEGER NOT NULL,
                    name_raw TEXT,
                    name_normalized TEXT COLLATE NOCASE,
                    value_raw TEXT,
                    value_normalized_json TEXT,
                    unit_raw TEXT,
                    unit_normalized TEXT COLLATE NOCASE,
                    category_normalized TEXT COLLATE NOCASE,
                    payload_json TEXT NOT NULL,
                    PRIMARY KEY (product_record_id, collection_name, specification_id),
                    FOREIGN KEY (product_record_id)
                        REFERENCES products(record_id)
                        ON UPDATE CASCADE ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS evidence (
                    evidence_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    product_record_id TEXT NOT NULL,
                    field_path TEXT NOT NULL,
                    source_document_id TEXT NOT NULL,
                    source_path TEXT,
                    page_number INTEGER NOT NULL,
                    element_id TEXT,
                    bbox_json TEXT,
                    text TEXT NOT NULL,
                    extraction_method TEXT,
                    model TEXT,
                    confidence REAL,
                    payload_json TEXT NOT NULL,
                    FOREIGN KEY (product_record_id)
                        REFERENCES products(record_id)
                        ON UPDATE CASCADE ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS ingestion_batches (
                    batch_id TEXT PRIMARY KEY,
                    schema_version TEXT NOT NULL,
                    source_document_id TEXT NOT NULL,
                    source_path TEXT,
                    pipeline_version TEXT,
                    parser_schema_sha256 TEXT,
                    status TEXT NOT NULL,
                    accepted_count INTEGER NOT NULL,
                    rejected_count INTEGER NOT NULL,
                    raw_llm_json TEXT NOT NULL,
                    raw_extraction_json TEXT NOT NULL,
                    issues_json TEXT NOT NULL,
                    products_jsonl_written INTEGER NOT NULL DEFAULT 0,
                    batch_jsonl_written INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS batch_products (
                    batch_id TEXT NOT NULL,
                    product_record_id TEXT NOT NULL,
                    ordinal INTEGER NOT NULL,
                    PRIMARY KEY (batch_id, product_record_id),
                    UNIQUE (batch_id, ordinal),
                    FOREIGN KEY (batch_id)
                        REFERENCES ingestion_batches(batch_id)
                        ON UPDATE CASCADE ON DELETE CASCADE,
                    FOREIGN KEY (product_record_id)
                        REFERENCES products(record_id)
                        ON UPDATE CASCADE ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS rejected_products (
                    rejection_id TEXT PRIMARY KEY,
                    batch_id TEXT NOT NULL,
                    ordinal INTEGER NOT NULL,
                    raw_product_json TEXT NOT NULL,
                    hydrated_product_json TEXT,
                    issues_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (batch_id)
                        REFERENCES ingestion_batches(batch_id)
                        ON UPDATE CASCADE ON DELETE CASCADE,
                    UNIQUE (batch_id, ordinal)
                );

                CREATE INDEX IF NOT EXISTS idx_products_part_number
                    ON products(part_number_normalized);
                CREATE INDEX IF NOT EXISTS idx_products_manufacturer
                    ON products(manufacturer_normalized);
                CREATE INDEX IF NOT EXISTS idx_products_part_name
                    ON products(part_name_normalized);
                CREATE INDEX IF NOT EXISTS idx_products_document
                    ON products(source_document_id);
                CREATE INDEX IF NOT EXISTS idx_specifications_name
                    ON specifications(name_normalized);
                CREATE INDEX IF NOT EXISTS idx_evidence_product_field
                    ON evidence(product_record_id, field_path);
                CREATE INDEX IF NOT EXISTS idx_evidence_document_page
                    ON evidence(source_document_id, page_number);
                CREATE INDEX IF NOT EXISTS idx_batches_document
                    ON ingestion_batches(source_document_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_rejections_batch
                    ON rejected_products(batch_id, ordinal);
                """
            )
            connection.execute(
                """
                INSERT INTO schema_metadata(key, value) VALUES ('schema_version', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (str(SCHEMA_VERSION),),
            )

    @staticmethod
    def _raw(value: Any) -> str | None:
        return None if value is None else value.raw

    @staticmethod
    def _normalized_text(value: Any) -> str | None:
        if value is None or value.normalized is None:
            return None
        return str(value.normalized)

    @staticmethod
    def _json(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)

    def upsert_document(self, document: SourceDocument | Mapping[str, Any]) -> SourceDocument:
        parsed = (
            document
            if isinstance(document, SourceDocument)
            else SourceDocument.model_validate(document)
        )
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO documents(
                    source_document_id, source_path, filename, sha256, page_count,
                    manufacturer_hint, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_document_id) DO UPDATE SET
                    source_path = excluded.source_path,
                    filename = excluded.filename,
                    sha256 = excluded.sha256,
                    page_count = excluded.page_count,
                    manufacturer_hint = excluded.manufacturer_hint,
                    metadata_json = excluded.metadata_json,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    parsed.source_document_id,
                    parsed.source_path,
                    parsed.filename,
                    parsed.sha256,
                    parsed.page_count,
                    parsed.manufacturer_hint,
                    self._json(parsed.metadata),
                ),
            )
        return parsed

    def _ensure_product_document(
        self, connection: sqlite3.Connection, product: ProductRecord
    ) -> None:
        filename = Path(product.source_path).name if product.source_path else None
        connection.execute(
            """
            INSERT INTO documents(source_document_id, source_path, filename)
            VALUES (?, ?, ?)
            ON CONFLICT(source_document_id) DO UPDATE SET
                source_path = COALESCE(excluded.source_path, documents.source_path),
                filename = COALESCE(excluded.filename, documents.filename),
                updated_at = CURRENT_TIMESTAMP
            """,
            (product.source_document_id, product.source_path, filename),
        )

    def _insert_specifications(
        self,
        connection: sqlite3.Connection,
        product: ProductRecord,
        collection_name: str,
        specifications: list[Specification],
    ) -> None:
        for ordinal, specification in enumerate(specifications):
            connection.execute(
                """
                INSERT INTO specifications(
                    product_record_id, collection_name, specification_id, ordinal,
                    name_raw, name_normalized, value_raw, value_normalized_json,
                    unit_raw, unit_normalized, category_normalized, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    product.record_id,
                    collection_name,
                    specification.specification_id,
                    ordinal,
                    specification.name.raw,
                    self._normalized_text(specification.name),
                    specification.value.raw,
                    self._json(specification.value.normalized),
                    self._raw(specification.unit),
                    self._normalized_text(specification.unit),
                    self._normalized_text(specification.category),
                    specification.model_dump_json(exclude_none=True),
                ),
            )

    def _insert_evidence(self, connection: sqlite3.Connection, product: ProductRecord) -> None:
        for field_path, evidence in product.iter_field_evidence():
            connection.execute(
                """
                INSERT INTO evidence(
                    product_record_id, field_path, source_document_id, source_path,
                    page_number, element_id, bbox_json, text, extraction_method,
                    model, confidence, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    product.record_id,
                    field_path,
                    evidence.source_document_id,
                    evidence.source_path,
                    evidence.page_number,
                    evidence.element_id,
                    evidence.bbox.model_dump_json(exclude_none=True)
                    if evidence.bbox is not None
                    else None,
                    evidence.text,
                    evidence.extraction_method,
                    evidence.model,
                    evidence.confidence,
                    evidence.model_dump_json(exclude_none=True),
                ),
            )

    def _upsert_product(self, connection: sqlite3.Connection, product: ProductRecord) -> None:
        self._ensure_product_document(connection, product)
        connection.execute(
            """
            INSERT INTO products(
                record_id, schema_version, source_document_id, source_path,
                manufacturer_raw, manufacturer_normalized, part_name_raw,
                part_name_normalized, part_number_raw, part_number_normalized,
                category_normalized, record_confidence, review_status, extracted_at,
                payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(record_id) DO UPDATE SET
                schema_version = excluded.schema_version,
                source_document_id = excluded.source_document_id,
                source_path = excluded.source_path,
                manufacturer_raw = excluded.manufacturer_raw,
                manufacturer_normalized = excluded.manufacturer_normalized,
                part_name_raw = excluded.part_name_raw,
                part_name_normalized = excluded.part_name_normalized,
                part_number_raw = excluded.part_number_raw,
                part_number_normalized = excluded.part_number_normalized,
                category_normalized = excluded.category_normalized,
                record_confidence = excluded.record_confidence,
                review_status = excluded.review_status,
                extracted_at = excluded.extracted_at,
                payload_json = excluded.payload_json,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                product.record_id,
                product.schema_version,
                product.source_document_id,
                product.source_path,
                self._raw(product.manufacturer),
                self._normalized_text(product.manufacturer),
                self._raw(product.part_name),
                self._normalized_text(product.part_name),
                self._raw(product.part_number),
                self._normalized_text(product.part_number),
                self._normalized_text(product.category),
                product.extraction.record_confidence,
                product.review_status,
                product.extraction.extracted_at.isoformat(),
                product.model_dump_json(exclude_none=True),
            ),
        )
        connection.execute(
            "DELETE FROM specifications WHERE product_record_id = ?",
            (product.record_id,),
        )
        connection.execute("DELETE FROM evidence WHERE product_record_id = ?", (product.record_id,))
        self._insert_specifications(connection, product, "specifications", product.specifications)
        self._insert_specifications(
            connection,
            product,
            "operating_conditions",
            product.operating_conditions,
        )
        self._insert_evidence(connection, product)

    def save_product(self, product: ProductRecord | Mapping[str, Any]) -> ProductRecord:
        """Upsert one product and optionally append its canonical JSONL event."""

        parsed = (
            product if isinstance(product, ProductRecord) else ProductRecord.model_validate(product)
        )
        with self._lock, self._connect() as connection:
            self._upsert_product(connection, parsed)
        if self.jsonl_writer is not None:
            self.jsonl_writer.append(parsed)
        return parsed

    upsert_product = save_product

    def save_products(
        self,
        products: Iterable[ProductRecord | Mapping[str, Any]],
    ) -> list[ProductRecord]:
        """Upsert a batch in one SQLite transaction."""

        parsed = _coerce_records(products)
        if not parsed:
            return []
        with self._lock, self._connect() as connection:
            for product in parsed:
                self._upsert_product(connection, product)
        if self.jsonl_writer is not None:
            self.jsonl_writer.append(parsed)
        return parsed

    def persist_parsed_batch(
        self,
        parsed_output: Any,
        *,
        source_document_id: str,
        extraction: Any,
        source_path: str | None = None,
        pipeline_version: str | None = None,
        strict: bool = False,
        require_evidence: bool = True,
        low_confidence_threshold: float = 0.65,
    ) -> BatchPersistenceResult:
        """Safely persist untrusted NVIDIA LLM output and its raw source extraction.

        Products are hydrated and validated independently. Valid siblings commit
        even when another item is malformed; rejected raw payloads, validation
        issues, the complete LLM response, and extraction evidence are stored in
        the same SQLite transaction for later replay.
        """

        parsed_data = _parsed_batch_data(parsed_output)
        raw_llm = _to_jsonable(parsed_output)
        raw_extraction = _to_jsonable(extraction)
        entries, batch_issues = _product_entries(parsed_data)
        if not entries and not batch_issues:
            batch_issues.append(
                ValidationIssue(
                    code="empty_product_batch",
                    message="structured parser returned no products",
                    path="products",
                    severity="warning",
                )
            )

        prepared: list[PreparedProduct] = []
        for ordinal, entry in enumerate(entries):
            try:
                prepared.append(
                    prepare_product_payload(
                        entry,
                        ordinal=ordinal,
                        extracted_elements=extraction,
                        source_document_id=source_document_id,
                        source_path=source_path,
                        strict=strict,
                        require_evidence=require_evidence,
                        low_confidence_threshold=low_confidence_threshold,
                    )
                )
            except Exception as exc:  # preserve unanticipated parser-shape failures
                prepared.append(
                    PreparedProduct(
                        ordinal=ordinal,
                        raw_payload=_to_jsonable(entry),
                        issues=[
                            ValidationIssue(
                                code="product_preparation_failure",
                                message=f"{type(exc).__name__}: {exc}",
                                path=f"products.{ordinal}",
                                severity="error",
                            )
                        ],
                    )
                )

        if not prepared and any(issue.severity == "error" for issue in batch_issues):
            prepared.append(
                PreparedProduct(
                    ordinal=0,
                    raw_payload=parsed_data,
                    issues=list(batch_issues),
                )
            )

        all_issues = [*batch_issues]
        for item in prepared:
            all_issues.extend(item.issues)
        accepted_pairs = [
            (item.ordinal, item.product)
            for item in prepared
            if item.accepted and item.product is not None
        ]
        accepted = [product for _, product in accepted_pairs]
        rejected = [item for item in prepared if not item.accepted]

        inferred_pipeline_version = pipeline_version or getattr(
            extraction, "pipeline_version", None
        )
        parser_schema_sha256 = getattr(parsed_output, "schema_sha256", None)
        response = getattr(parsed_output, "response", None)
        parser_model = getattr(response, "model", None)
        extraction_context = _stable_extraction_context(extraction)
        batch_seed = self._json(
            {
                "source_document_id": source_document_id,
                "pipeline_version": inferred_pipeline_version,
                "parser_schema_sha256": parser_schema_sha256,
                "extraction_context": extraction_context,
                "parsed_data": _to_jsonable(parsed_data),
            }
        )
        batch_id = "batch_" + hashlib.sha256(batch_seed.encode("utf-8")).hexdigest()[:32]
        batch_metadata: dict[str, Any] = {"extraction_context": extraction_context}
        if parser_model:
            batch_metadata["parser_model"] = parser_model
        batch = ProductBatch(
            batch_id=batch_id,
            source_document_id=source_document_id,
            source_path=source_path,
            pipeline_version=inferred_pipeline_version,
            parser_schema_sha256=parser_schema_sha256,
            products=accepted,
            metadata=batch_metadata,
        )
        status = "partial" if accepted and rejected else "failed" if rejected else "succeeded"

        raw_llm_json = self._json(raw_llm)
        raw_extraction_json = self._json(raw_extraction)
        issues_json = self._json([issue.model_dump(mode="json") for issue in all_issues])
        prior_products_written = False
        prior_batch_written = False

        with self._lock, self._connect() as connection:
            prior = connection.execute(
                """
                SELECT products_jsonl_written, batch_jsonl_written
                FROM ingestion_batches WHERE batch_id = ?
                """,
                (batch_id,),
            ).fetchone()
            if prior is not None:
                prior_products_written = bool(prior["products_jsonl_written"])
                prior_batch_written = bool(prior["batch_jsonl_written"])
            connection.execute(
                """
                INSERT INTO ingestion_batches(
                    batch_id, schema_version, source_document_id, source_path,
                    pipeline_version, parser_schema_sha256, status, accepted_count,
                    rejected_count, raw_llm_json, raw_extraction_json, issues_json,
                    products_jsonl_written, batch_jsonl_written
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(batch_id) DO UPDATE SET
                    source_path = excluded.source_path,
                    pipeline_version = excluded.pipeline_version,
                    parser_schema_sha256 = excluded.parser_schema_sha256,
                    status = excluded.status,
                    accepted_count = excluded.accepted_count,
                    rejected_count = excluded.rejected_count,
                    raw_llm_json = excluded.raw_llm_json,
                    raw_extraction_json = excluded.raw_extraction_json,
                    issues_json = excluded.issues_json,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    batch_id,
                    batch.schema_version,
                    source_document_id,
                    source_path,
                    inferred_pipeline_version,
                    parser_schema_sha256,
                    status,
                    len(accepted),
                    len(rejected),
                    raw_llm_json,
                    raw_extraction_json,
                    issues_json,
                    int(prior_products_written),
                    int(prior_batch_written),
                ),
            )
            connection.execute("DELETE FROM batch_products WHERE batch_id = ?", (batch_id,))
            connection.execute("DELETE FROM rejected_products WHERE batch_id = ?", (batch_id,))
            for ordinal, product in accepted_pairs:
                self._upsert_product(connection, product)
                connection.execute(
                    """
                    INSERT INTO batch_products(batch_id, product_record_id, ordinal)
                    VALUES (?, ?, ?)
                    """,
                    (batch_id, product.record_id, ordinal),
                )
            for item in rejected:
                rejection_seed = f"{batch_id}\x00{item.ordinal}".encode()
                rejection_id = "reject_" + hashlib.sha256(rejection_seed).hexdigest()[:32]
                connection.execute(
                    """
                    INSERT INTO rejected_products(
                        rejection_id, batch_id, ordinal, raw_product_json,
                        hydrated_product_json, issues_json
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        rejection_id,
                        batch_id,
                        item.ordinal,
                        self._json(_to_jsonable(item.raw_payload)),
                        self._json(_to_jsonable(item.hydrated_payload))
                        if item.hydrated_payload is not None
                        else None,
                        self._json([issue.model_dump(mode="json") for issue in item.issues]),
                    ),
                )

        products_written = prior_products_written
        batch_written = prior_batch_written
        if self.jsonl_writer is not None and not products_written:
            try:
                if accepted:
                    self.jsonl_writer.append(accepted)
                products_written = True
            except (OSError, TypeError, ValueError) as exc:
                all_issues.append(
                    ValidationIssue(
                        code="products_jsonl_write_failed",
                        message=str(exc),
                        path="",
                        severity="warning",
                    )
                )

        batch_event = {
            "event_schema_version": "1.0",
            "event_type": "industrial_catalog.product_batch_persisted",
            "batch": batch.model_dump(mode="json", exclude_none=True),
            "status": status,
            "rejections": [
                {
                    "ordinal": item.ordinal,
                    "raw_payload": _to_jsonable(item.raw_payload),
                    "hydrated_payload": _to_jsonable(item.hydrated_payload),
                    "issues": [issue.model_dump(mode="json") for issue in item.issues],
                }
                for item in rejected
            ],
            "issues": [issue.model_dump(mode="json") for issue in all_issues],
            "raw_llm_output": raw_llm,
            "raw_extraction": raw_extraction,
        }
        if self.batch_jsonl_writer is not None and not batch_written:
            try:
                self.batch_jsonl_writer.append(batch_event)
                batch_written = True
            except (OSError, TypeError, ValueError) as exc:
                all_issues.append(
                    ValidationIssue(
                        code="batch_jsonl_write_failed",
                        message=str(exc),
                        path="",
                        severity="warning",
                    )
                )

        final_issues_json = self._json([issue.model_dump(mode="json") for issue in all_issues])
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                UPDATE ingestion_batches SET
                    products_jsonl_written = ?,
                    batch_jsonl_written = ?,
                    issues_json = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE batch_id = ?
                """,
                (
                    int(products_written),
                    int(batch_written),
                    final_issues_json,
                    batch_id,
                ),
            )

        return BatchPersistenceResult(
            batch=batch,
            prepared=tuple(prepared),
            issues=tuple(all_issues),
            status=status,
            raw_preserved=True,
            products_jsonl_written=products_written,
            batch_jsonl_written=batch_written,
        )

    def get_batch_ingestion(self, batch_id: str) -> dict[str, Any] | None:
        """Return a stored raw batch/audit record for replay or review."""

        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM ingestion_batches WHERE batch_id = ?", (batch_id,)
            ).fetchone()
        if row is None:
            return None
        output = dict(row)
        for key in ("raw_llm_json", "raw_extraction_json", "issues_json"):
            output[key.removesuffix("_json")] = json.loads(output.pop(key))
        output["products_jsonl_written"] = bool(output["products_jsonl_written"])
        output["batch_jsonl_written"] = bool(output["batch_jsonl_written"])
        return output

    def get_rejected_products(self, batch_id: str) -> list[dict[str, Any]]:
        """Return rejected raw items and validation issues in original order."""

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM rejected_products
                WHERE batch_id = ? ORDER BY ordinal
                """,
                (batch_id,),
            ).fetchall()
        output: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["raw_product"] = json.loads(item.pop("raw_product_json"))
            hydrated = item.pop("hydrated_product_json")
            item["hydrated_product"] = json.loads(hydrated) if hydrated else None
            item["issues"] = json.loads(item.pop("issues_json"))
            output.append(item)
        return output

    def get_product(self, record_id: str) -> ProductRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload_json FROM products WHERE record_id = ?",
                (record_id,),
            ).fetchone()
        return ProductRecord.model_validate_json(row["payload_json"]) if row else None

    def find_by_part_number(self, part_number: str) -> list[ProductRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT payload_json FROM products
                WHERE part_number_normalized = ? COLLATE NOCASE
                   OR part_number_raw = ? COLLATE NOCASE
                ORDER BY updated_at DESC, record_id
                """,
                (part_number, part_number),
            ).fetchall()
        return [ProductRecord.model_validate_json(row["payload_json"]) for row in rows]

    def find_products(
        self,
        *,
        manufacturer: str | None = None,
        part_number: str | None = None,
        part_name_contains: str | None = None,
        source_document_id: str | None = None,
        review_status: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[ProductRecord]:
        if limit < 1 or limit > 10_000:
            raise ValueError("limit must be between 1 and 10000")
        if offset < 0:
            raise ValueError("offset must be non-negative")
        clauses: list[str] = []
        parameters: list[Any] = []
        if manufacturer is not None:
            clauses.append("manufacturer_normalized = ? COLLATE NOCASE")
            parameters.append(manufacturer)
        if part_number is not None:
            clauses.append("part_number_normalized = ? COLLATE NOCASE")
            parameters.append(part_number)
        if part_name_contains is not None:
            clauses.append("part_name_normalized LIKE ? ESCAPE '\\' COLLATE NOCASE")
            escaped = (
                part_name_contains.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            )
            parameters.append(f"%{escaped}%")
        if source_document_id is not None:
            clauses.append("source_document_id = ?")
            parameters.append(source_document_id)
        if review_status is not None:
            clauses.append("review_status = ?")
            parameters.append(review_status)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        parameters.extend((limit, offset))
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT payload_json FROM products
                {where}
                ORDER BY manufacturer_normalized, part_number_normalized, record_id
                LIMIT ? OFFSET ?
                """,
                parameters,
            ).fetchall()
        return [ProductRecord.model_validate_json(row["payload_json"]) for row in rows]

    def iter_products(self, *, batch_size: int = 500) -> Iterator[ProductRecord]:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        with self._connect() as connection:
            cursor = connection.execute("SELECT payload_json FROM products ORDER BY record_id")
            while rows := cursor.fetchmany(batch_size):
                for row in rows:
                    yield ProductRecord.model_validate_json(row["payload_json"])

    def get_evidence(
        self,
        record_id: str,
        *,
        field_path: str | None = None,
    ) -> list[tuple[str, Evidence]]:
        query = "SELECT field_path, payload_json FROM evidence WHERE product_record_id = ?"
        parameters: list[Any] = [record_id]
        if field_path is not None:
            query += " AND field_path = ?"
            parameters.append(field_path)
        query += " ORDER BY page_number, evidence_id"
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [
            (row["field_path"], Evidence.model_validate_json(row["payload_json"])) for row in rows
        ]

    def count_products(self) -> int:
        with self._connect() as connection:
            row = connection.execute("SELECT COUNT(*) AS count FROM products").fetchone()
        return int(row["count"])

    def export_jsonl(self, path: str | Path, *, overwrite: bool = True) -> int:
        writer = JsonlWriter(path)
        records = list(self.iter_products())
        return writer.overwrite(records) if overwrite else writer.append(records)

    def close(self) -> None:
        """Compatibility no-op; this store intentionally uses short-lived connections."""

    def __enter__(self) -> SQLiteCatalogStore:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


CatalogStore = SQLiteCatalogStore


__all__ = [
    "BatchPersistenceResult",
    "CatalogStore",
    "JsonEventWriter",
    "JsonlWriter",
    "SCHEMA_VERSION",
    "SQLiteCatalogStore",
    "append_jsonl",
]
