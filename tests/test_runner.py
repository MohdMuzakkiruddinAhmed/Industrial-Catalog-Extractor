from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pymupdf as fitz
import pytest
import yaml

from industrial_catalog.cli import main
from industrial_catalog.config import PipelineConfig
from industrial_catalog.extraction import (
    DocumentExtraction,
    ExtractionPipeline,
    PageExtraction,
)
from industrial_catalog.nvidia_clients import (
    GuidedJsonClient,
    GuidedJsonDecodeError,
    GuidedJsonMode,
    GuidedJsonResult,
    JsonCandidateFailure,
    ModelResponse,
    NIMRequestError,
)
from industrial_catalog.routing import PageRouter
from industrial_catalog.runner import build_extraction_pipeline, discover_pdfs
from industrial_catalog.storage import SQLiteCatalogStore


def _make_pdf(path: Path, *, pages: int = 1) -> None:
    document = fitz.open()
    for page_number in range(1, pages + 1):
        page = document.new_page()
        page.insert_textbox(
            fitz.Rect(40, 40, 550, 760),
            (
                f"ACME Industrial Controls Product Catalog page {page_number}. "
                "Motor starter part number MS-4400-24V provides three-phase control. "
                "Rated voltage is 24 VDC, contact rating is 10 A, ingress protection "
                "is IP67, and operating temperature is -20 to 60 degrees Celsius. "
                "Use the installation instructions and safety requirements shown here."
            ),
            fontsize=11,
        )
    document.save(path)
    document.close()


def _write_dry_config(tmp_path: Path, corpus: Path) -> Path:
    config_path = tmp_path / "pipeline.yaml"
    payload = {
        "run_name": "test-benchmark",
        "mode": "dry-run",
        "paths": {
            "corpus_root": str(corpus),
            "artifact_root": str(tmp_path / "artifacts"),
            "output_root": str(tmp_path / "extraction"),
            "checkpoint_db": str(tmp_path / "checkpoints" / "documents.sqlite3"),
        },
        "routing": {
            "native_min_chars": 80,
            "native_min_chars_per_square_inch": 2,
            "render_dpi": 96,
            "verify_identifier_regions": True,
            "verify_tables": True,
        },
        "gpu_ids": [1, 3, 4, 5, 6, 7],
    }
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return config_path


def _empty_extraction(
    document_id: str,
    *,
    source_path: str | None,
    certified_blank: bool,
) -> DocumentExtraction:
    routing = PageRouter().decide("")
    audit = (
        {
            "audit_type": "page_extraction",
            "event": "deterministic_blank_page",
            "outcome": "empty_page_extraction",
            "page_number": 1,
            "routing": routing.to_dict(),
        },
    )
    return DocumentExtraction(
        document_id=document_id,
        source_path=source_path,
        pages=(
            PageExtraction(
                document_id=document_id,
                page_number=1,
                fingerprint="empty-page-fingerprint",
                routing=routing,
                elements=(),
                model_responses=audit if certified_blank else (),
            ),
        ),
        pipeline_version="blank-native-fallback-v1",
    )


def test_cli_smoke_run_is_bounded_resumable_and_writes_summaries(tmp_path: Path) -> None:
    corpus = tmp_path / "vendor-corpus"
    corpus.mkdir()
    _make_pdf(corpus / "a.pdf", pages=3)
    _make_pdf(corpus / "b.pdf", pages=1)
    config_path = _write_dry_config(tmp_path, corpus)
    run_dir = tmp_path / "run"

    exit_code = main(
        [
            "benchmark",
            "--config",
            str(config_path),
            "--output-dir",
            str(run_dir),
            "--smoke",
            "--max-pages",
            "1",
            "--run-id",
            "test-run",
        ]
    )

    assert exit_code == 0
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["mode"] == "dry-run"
    assert summary["documents_discovered"] == 2
    assert summary["documents_attempted"] == 1
    assert summary["pages"] == 1
    assert summary["documents_succeeded"] == 1
    assert (run_dir / "documents.jsonl").is_file()
    assert (run_dir / "chunks.jsonl").is_file()
    assert (run_dir / "catalog.sqlite3").is_file()
    assert (run_dir / "knowledge_base.sqlite3").is_file()
    assert len(list((run_dir / "extractions").glob("*.json"))) == 1
    assert len(list((run_dir / "structured").glob("*.json"))) == 1

    second_exit_code = main(
        [
            "benchmark",
            "--config",
            str(config_path),
            "--output-dir",
            str(run_dir),
            "--smoke",
            "--max-pages",
            "1",
            "--run-id",
            "test-run-resumed",
        ]
    )
    resumed = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert second_exit_code == 0
    assert resumed["checkpoint_hits"] == 1


def test_certified_blank_chunk_skips_llm_and_batch_persistence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = tmp_path / "vendor-corpus"
    corpus.mkdir()
    _make_pdf(corpus / "blank.pdf")
    config_path = _write_dry_config(tmp_path, corpus)
    run_dir = tmp_path / "blank-run"

    def extract_blank(self, document_id, pages, *, source_path=None, **kwargs):
        return _empty_extraction(
            document_id,
            source_path=source_path,
            certified_blank=True,
        )

    def unexpected_call(*args, **kwargs):
        raise AssertionError("certified blank chunk reached a model or batch store")

    monkeypatch.setattr(ExtractionPipeline, "extract_pages", extract_blank)
    monkeypatch.setattr(ExtractionPipeline, "parse_structured", unexpected_call)
    monkeypatch.setattr(SQLiteCatalogStore, "persist_parsed_batch", unexpected_call)

    exit_code = main(
        [
            "benchmark",
            "--config",
            str(config_path),
            "--output-dir",
            str(run_dir),
            "--smoke",
            "--max-pages",
            "1",
        ]
    )

    assert exit_code == 0
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["documents_succeeded"] == 1
    assert summary["documents_review"] == 0
    assert summary["products_parsed"] == 0
    assert summary["products_invalid"] == 0

    structured_paths = list((run_dir / "structured").glob("*.json"))
    decision_paths = list((run_dir / "decisions").glob("*.json"))
    assert len(structured_paths) == 1
    assert len(decision_paths) == 1
    structured = json.loads(structured_paths[0].read_text(encoding="utf-8"))
    decision = json.loads(decision_paths[0].read_text(encoding="utf-8"))
    chunk = json.loads((run_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert structured["event"] == "structured_parse_skipped"
    assert structured["structured_parse_skipped_reason"] == "certified_blank_page"
    assert decision["structured_parse_skipped"] is True
    assert decision["structured_parse_skipped_reason"] == "certified_blank_page"
    assert chunk["status"] == "skipped"
    assert chunk["batch_persisted"] is False
    assert chunk["structured_parse_skipped_reason"] == "certified_blank_page"
    assert chunk["products_parsed"] == 0
    with sqlite3.connect(run_dir / "catalog.sqlite3") as connection:
        assert connection.execute("SELECT COUNT(*) FROM ingestion_batches").fetchone()[0] == 0


def test_uncertified_empty_chunk_runs_llm_and_rejects_fabricated_product(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = tmp_path / "vendor-corpus"
    corpus.mkdir()
    _make_pdf(corpus / "empty-but-uncertified.pdf")
    config_path = _write_dry_config(tmp_path, corpus)
    run_dir = tmp_path / "uncertified-empty-run"
    calls = {"llm": 0, "persist": 0}

    def extract_empty(self, document_id, pages, *, source_path=None, **kwargs):
        return _empty_extraction(
            document_id,
            source_path=source_path,
            certified_blank=False,
        )

    fabricated_evidence = {
        "document_id": "placeholder-replaced-by-runner",
        "page_number": 1,
        "element_id": "el_fabricated",
        "text": "ACME Fabricated Sensor",
        "extraction_method": "claimed_model_output",
    }

    def parse_fabricated(self, extraction, **kwargs):
        calls["llm"] += 1
        evidence = {**fabricated_evidence, "document_id": extraction.document_id}
        data = {
            "products": [
                {
                    "manufacturer": {"raw": "ACME", "evidence": [evidence]},
                    "part_name": {
                        "raw": "Fabricated Sensor",
                        "evidence": [evidence],
                    },
                    "part_number": {"raw": "PN-FAKE", "evidence": [evidence]},
                    "specifications": [],
                }
            ]
        }
        response = ModelResponse(
            content=json.dumps(data),
            model="test-guided-json",
            request_id="volatile-request-id",
            raw_response={"id": "volatile-request-id", "data": data},
        )
        return GuidedJsonResult(
            data=data,
            response=response,
            schema_sha256="test-schema-sha256",
        )

    original_persist = SQLiteCatalogStore.persist_parsed_batch

    def track_persistence(self, *args, **kwargs):
        calls["persist"] += 1
        return original_persist(self, *args, **kwargs)

    monkeypatch.setattr(ExtractionPipeline, "extract_pages", extract_empty)
    monkeypatch.setattr(ExtractionPipeline, "parse_structured", parse_fabricated)
    monkeypatch.setattr(SQLiteCatalogStore, "persist_parsed_batch", track_persistence)

    exit_code = main(
        [
            "benchmark",
            "--config",
            str(config_path),
            "--output-dir",
            str(run_dir),
            "--smoke",
            "--max-pages",
            "1",
        ]
    )

    assert exit_code == 0
    assert calls == {"llm": 1, "persist": 1}
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["documents_review"] == 1
    assert summary["products_parsed"] == 1
    assert summary["products_stored"] == 0
    assert summary["products_invalid"] == 1
    structured_path = next((run_dir / "structured").glob("*.json"))
    structured = json.loads(structured_path.read_text(encoding="utf-8"))
    assert "structured_parse_skipped_reason" not in structured
    chunk = json.loads((run_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert "structured_parse_skipped" not in chunk
    assert chunk["status"] == "failed"
    assert chunk["rejected_count"] == 1
    with sqlite3.connect(run_dir / "catalog.sqlite3") as connection:
        assert connection.execute("SELECT COUNT(*) FROM products").fetchone()[0] == 0
        issues_json = connection.execute("SELECT issues_json FROM ingestion_batches").fetchone()[0]
    assert "unknown_evidence_reference" in {issue["code"] for issue in json.loads(issues_json)}


def test_live_pipeline_uses_openai_response_format(tmp_path: Path) -> None:
    raw = {
        "mode": "benchmark",
        "paths": {
            "corpus_root": str(tmp_path),
            "artifact_root": str(tmp_path / "artifacts"),
            "output_root": str(tmp_path / "output"),
            "checkpoint_db": str(tmp_path / "checkpoints.sqlite3"),
        },
        "parse_endpoint": {"base_url": "http://parse", "model": "parse"},
        "ocr_endpoint": {"base_url": "http://ocr", "model": "ocr"},
        "llm_endpoint": {"base_url": "http://llm", "model": "llm"},
        "gpu_ids": [1, 3, 4, 5, 6, 7],
    }
    pipeline = build_extraction_pipeline(PipelineConfig.model_validate(raw), dry_run=False)

    assert pipeline.llm is not None
    assert pipeline.llm.mode is GuidedJsonMode.OPENAI_RESPONSE_FORMAT
    assert pipeline.llm.config.dry_run is False


def test_discovery_is_recursive_deduplicated_and_sorted(tmp_path: Path) -> None:
    nested = tmp_path / "vendor"
    nested.mkdir()
    _make_pdf(tmp_path / "Z.pdf")
    _make_pdf(nested / "a.PDF")
    (tmp_path / "ignore.txt").write_text("not a catalog", encoding="utf-8")

    discovered = discover_pdfs(tmp_path, ("**/*.pdf", "**/*.PDF"))

    assert [path.name for path in discovered] == ["a.PDF", "Z.pdf"]


def test_runner_persists_raw_structured_decode_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = tmp_path / "vendor-corpus"
    corpus.mkdir()
    _make_pdf(corpus / "catalog.pdf")
    config_path = _write_dry_config(tmp_path, corpus)
    run_dir = tmp_path / "failed-run"
    raw = {
        "id": "truncated-response",
        "model": "nemotron-test",
        "choices": [
            {
                "message": {"content": '{"products": ['},
                "finish_reason": "length",
            }
        ],
    }
    attempt = ModelResponse(
        content='{"products": [',
        model="nemotron-test",
        request_id="truncated-response",
        raw_response=raw,
        finish_reason="length",
    )
    failure = JsonCandidateFailure(
        source="choices[0].message.content",
        kind="truncated",
        message="provider marked response incomplete",
        finish_reason="length",
        content_length=14,
        trimmed_content_length=14,
    )

    def fail_generation(self, **kwargs):
        raise GuidedJsonDecodeError((attempt,), ((failure,),))

    monkeypatch.setattr(GuidedJsonClient, "generate_json", fail_generation)

    exit_code = main(
        [
            "benchmark",
            "--config",
            str(config_path),
            "--output-dir",
            str(run_dir),
            "--smoke",
            "--max-pages",
            "1",
        ]
    )

    assert exit_code == 1
    failure_paths = list((run_dir / "structured").glob("*.failure.json"))
    assert len(failure_paths) == 1
    persisted = json.loads(failure_paths[0].read_text(encoding="utf-8"))
    assert persisted["attempts"][0]["raw_response"]["id"] == "truncated-response"
    assert persisted["failures"][0][0]["kind"] == "truncated"

    document_events = [
        json.loads(line)
        for line in (run_dir / "documents.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(document_events) == 1
    error_details = document_events[0]["error_details"]
    assert error_details["error_type"] == "GuidedJsonDecodeError"
    assert error_details["attempts"][0]["raw_response"]["id"] == ("truncated-response")
    assert error_details["failures"][0][0]["kind"] == "truncated"

    with sqlite3.connect(tmp_path / "checkpoints" / "documents.sqlite3") as connection:
        row = connection.execute("SELECT detail_json FROM checkpoints").fetchone()
    assert row is not None
    checkpoint_detail = json.loads(row[0])
    assert checkpoint_detail["error_details"] == error_details


def test_runner_preserves_explicit_nim_raw_response_details(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = tmp_path / "vendor-corpus"
    corpus.mkdir()
    _make_pdf(corpus / "catalog.pdf")
    config_path = _write_dry_config(tmp_path, corpus)
    run_dir = tmp_path / "nim-failed-run"
    raw_response = {
        "id": "parse-empty-8",
        "model": "parse-test",
        "choices": [],
    }
    nim_error = NIMRequestError(
        "NIM returned successful HTTP responses without supported assistant content",
        status_code=200,
        response={
            "failure_kind": "unsupported_assistant_content",
            "successful_http_failures": [
                {
                    "attempt": 8,
                    "status_code": 200,
                    "raw_response": raw_response,
                }
            ],
            "last_raw_response": raw_response,
        },
        attempts=8,
    )

    def fail_extraction(self, *args, **kwargs):
        raise nim_error

    monkeypatch.setattr(ExtractionPipeline, "extract_pages", fail_extraction)

    exit_code = main(
        [
            "benchmark",
            "--config",
            str(config_path),
            "--output-dir",
            str(run_dir),
            "--smoke",
            "--max-pages",
            "1",
        ]
    )

    assert exit_code == 1
    document_event = json.loads(
        (run_dir / "documents.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    details = document_event["error_details"]
    assert details["error_type"] == "NIMRequestError"
    assert details["attempts"] == 8
    assert details["response"]["last_raw_response"] == raw_response

    with sqlite3.connect(tmp_path / "checkpoints" / "documents.sqlite3") as connection:
        row = connection.execute("SELECT detail_json FROM checkpoints").fetchone()
    assert row is not None
    assert json.loads(row[0])["error_details"] == details
