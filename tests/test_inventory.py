from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "inventory_corpus.py"
SPEC = importlib.util.spec_from_file_location("inventory_corpus", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
inventory_corpus = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(inventory_corpus)


def test_inventory_is_stable_and_finds_duplicates(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    nested = corpus / "Vendor B"
    nested.mkdir(parents=True)
    (corpus / "part-a.pdf").write_bytes(b"same catalog bytes")
    (nested / "PART-A.PDF").write_bytes(b"same catalog bytes")
    (nested / "part-b.jpg").write_bytes(b"image bytes")
    (corpus / "notes.txt").write_text("not a catalog", encoding="utf-8")

    result = inventory_corpus.build_inventory(corpus, count_pdf_pages=False)

    assert result["status"] == "complete"
    assert result["metrics"] == {
        "file_count": 3,
        "pdf_count": 2,
        "total_bytes": 47,
        "total_pdf_pages": None,
        "pdf_page_counts_unavailable": 0,
        "unique_content_count": 2,
        "duplicate_group_count": 1,
        "duplicate_file_count": 1,
        "ignored_extension_count": 1,
        "skipped_symlink_count": 0,
        "error_count": 0,
    }
    assert [item["relative_path"] for item in result["documents"]] == [
        "part-a.pdf",
        "Vendor B/PART-A.PDF",
        "Vendor B/part-b.jpg",
    ]
    assert result["duplicate_groups"][0]["relative_paths"] == [
        "Vendor B/PART-A.PDF",
        "part-a.pdf",
    ]


def test_atomic_output_and_output_exclusion(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "catalog.pdf").write_bytes(b"catalog")
    output = corpus / "inventory.json"

    exit_code = inventory_corpus.main(
        [
            str(corpus),
            "--output",
            str(output),
            "--extensions",
            "pdf,json",
            "--no-page-count",
        ]
    )
    first = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert first["metrics"]["file_count"] == 1

    exit_code = inventory_corpus.main(
        [
            str(corpus),
            "--output",
            str(output),
            "--extensions",
            "pdf,json",
            "--no-page-count",
        ]
    )
    second = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert second["metrics"]["file_count"] == 1
    assert second["documents"][0]["relative_path"] == "catalog.pdf"


def test_rejects_filesystem_root() -> None:
    with pytest.raises(ValueError, match="filesystem root"):
        inventory_corpus.build_inventory(Path(Path.cwd().anchor))


def test_extension_normalization() -> None:
    assert inventory_corpus.normalize_extensions(["PDF,jPg", ".TIFF"]) == {
        ".pdf",
        ".jpg",
        ".tiff",
    }
    with pytest.raises(ValueError, match="at least one"):
        inventory_corpus.normalize_extensions([" , "])
