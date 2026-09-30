#!/usr/bin/env python3
"""Create a deterministic, evidence-friendly inventory of a catalog corpus.

The script is deliberately read-only with respect to the corpus. It does not
follow symlinks, and it writes the resulting JSON manifest atomically.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_SOURCE = Path("/data/industrial-data-corps/corpus/raw")
DEFAULT_OUTPUT = Path("/data/industrial-data-corps/corpus/manifests/inventory.json")
DEFAULT_EXTENSIONS = frozenset({".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff"})
CHUNK_BYTES = 8 * 1024 * 1024


def utc_now() -> str:
    """Return a machine-readable UTC timestamp."""

    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def normalize_extensions(values: Iterable[str]) -> frozenset[str]:
    """Normalize CLI extension values to lowercase, dot-prefixed suffixes."""

    normalized: set[str] = set()
    for raw_value in values:
        for value in raw_value.split(","):
            value = value.strip().lower()
            if not value:
                continue
            normalized.add(value if value.startswith(".") else f".{value}")
    if not normalized:
        raise ValueError("at least one file extension is required")
    return frozenset(normalized)


def sha256_file(path: Path) -> str:
    """Hash a file without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pdf_page_count(path: Path) -> tuple[int | None, str]:
    """Count PDF pages using an available parser without making it mandatory."""

    try:
        import pymupdf as fitz  # type: ignore[import-not-found]
    except ImportError:
        try:
            from pypdf import PdfReader  # type: ignore[import-not-found]
        except ImportError:
            return None, "unavailable"
        try:
            return len(PdfReader(str(path), strict=False).pages), "pypdf"
        except Exception as exc:  # a damaged PDF should not abort the whole inventory
            return None, f"error:{type(exc).__name__}"

    try:
        with fitz.open(path) as document:
            return document.page_count, "pymupdf"
    except Exception as exc:  # a damaged PDF should not abort the whole inventory
        return None, f"error:{type(exc).__name__}"


def _catalog_files(
    root: Path,
    extensions: frozenset[str],
    excluded_path: Path | None,
) -> tuple[list[Path], int, int]:
    """Return stable file paths plus ignored-file and skipped-symlink counts."""

    selected: list[Path] = []
    ignored_extensions = 0
    skipped_symlinks = 0

    for current, directories, files in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        safe_directories: list[str] = []
        for directory in sorted(directories):
            candidate = current_path / directory
            if candidate.is_symlink():
                skipped_symlinks += 1
            else:
                safe_directories.append(directory)
        directories[:] = safe_directories

        for filename in sorted(files):
            candidate = current_path / filename
            if candidate.is_symlink():
                skipped_symlinks += 1
                continue
            if excluded_path is not None and candidate.resolve() == excluded_path:
                continue
            if candidate.suffix.lower() not in extensions:
                ignored_extensions += 1
                continue
            selected.append(candidate)

    selected.sort(key=lambda path: path.relative_to(root).as_posix().casefold())
    return selected, ignored_extensions, skipped_symlinks


def build_inventory(
    source: str | Path,
    *,
    extensions: Iterable[str] = DEFAULT_EXTENSIONS,
    hash_files: bool = True,
    count_pdf_pages: bool = True,
    excluded_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build and return an inventory without changing ``source``.

    File-level read errors are included in the manifest. A source that does not
    exist, is not a directory, or resolves to a filesystem root is rejected.
    """

    root = Path(source).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(f"corpus source is not a directory: {root}")
    if root == Path(root.anchor):
        raise ValueError("refusing to inventory an entire filesystem root")

    normalized_extensions = normalize_extensions(extensions)
    excluded = Path(excluded_path).expanduser().resolve() if excluded_path else None
    files, ignored_extensions, skipped_symlinks = _catalog_files(
        root, normalized_extensions, excluded
    )

    documents: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    content_paths: dict[str, list[str]] = defaultdict(list)
    total_bytes = 0
    total_pdf_pages = 0
    pdf_page_counts_unavailable = 0

    for path in files:
        relative_path = path.relative_to(root).as_posix()
        try:
            stat_before = path.stat()
            digest = sha256_file(path) if hash_files else None
            stat_after = path.stat()
            if (stat_before.st_size, stat_before.st_mtime_ns) != (
                stat_after.st_size,
                stat_after.st_mtime_ns,
            ):
                raise RuntimeError("file changed while it was being inventoried")

            extension = path.suffix.lower()
            page_count: int | None = None
            page_count_method: str | None = None
            if extension == ".pdf" and count_pdf_pages:
                page_count, page_count_method = _pdf_page_count(path)
                if page_count is None:
                    pdf_page_counts_unavailable += 1
                else:
                    total_pdf_pages += page_count

            document_id_basis = f"{relative_path}\0{digest or stat_after.st_size}"
            record: dict[str, Any] = {
                "document_id": hashlib.sha256(document_id_basis.encode("utf-8")).hexdigest(),
                "relative_path": relative_path,
                "extension": extension,
                "bytes": stat_after.st_size,
                "modified_utc": datetime.fromtimestamp(stat_after.st_mtime, tz=UTC)
                .isoformat(timespec="seconds")
                .replace("+00:00", "Z"),
                "sha256": digest,
                "page_count": page_count,
                "page_count_method": page_count_method,
            }
            documents.append(record)
            total_bytes += stat_after.st_size
            if digest:
                content_paths[digest].append(relative_path)
        except (OSError, RuntimeError) as exc:
            errors.append(
                {
                    "relative_path": relative_path,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )

    duplicate_groups = [
        {"sha256": digest, "copies": len(paths), "relative_paths": sorted(paths)}
        for digest, paths in sorted(content_paths.items())
        if len(paths) > 1
    ]
    duplicate_file_count = sum(group["copies"] - 1 for group in duplicate_groups)
    unique_content_count = len(content_paths) if hash_files else None
    status = "complete" if not errors else "partial"

    return {
        "schema_version": "1.0",
        "kind": "industrial-catalog-corpus-inventory",
        "status": status,
        "generated_at": utc_now(),
        "source_root": str(root),
        "settings": {
            "extensions": sorted(normalized_extensions),
            "hash_algorithm": "sha256" if hash_files else None,
            "count_pdf_pages": count_pdf_pages,
            "follow_symlinks": False,
        },
        "metrics": {
            "file_count": len(documents),
            "pdf_count": sum(doc["extension"] == ".pdf" for doc in documents),
            "total_bytes": total_bytes,
            "total_pdf_pages": total_pdf_pages if count_pdf_pages else None,
            "pdf_page_counts_unavailable": pdf_page_counts_unavailable,
            "unique_content_count": unique_content_count,
            "duplicate_group_count": len(duplicate_groups),
            "duplicate_file_count": duplicate_file_count,
            "ignored_extension_count": ignored_extensions,
            "skipped_symlink_count": skipped_symlinks,
            "error_count": len(errors),
        },
        "documents": documents,
        "duplicate_groups": duplicate_groups,
        "errors": errors,
    }


def write_json_atomic(payload: dict[str, Any], destination: Path) -> None:
    """Write JSON to ``destination`` with an atomic replace."""

    destination = destination.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            Path(temporary_name).unlink(missing_ok=True)
        finally:
            raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="?", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=sorted(DEFAULT_EXTENSIONS),
        metavar="EXT",
        help="catalog suffixes, as a space- or comma-separated list",
    )
    parser.add_argument("--no-hash", action="store_true", help="skip SHA-256 and deduplication")
    parser.add_argument("--no-page-count", action="store_true", help="skip PDF page counts")
    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        help="exit 2 when any catalog file could not be read",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = args.output.expanduser().resolve()
    try:
        inventory = build_inventory(
            args.source,
            extensions=args.extensions,
            hash_files=not args.no_hash,
            count_pdf_pages=not args.no_page_count,
            excluded_path=output,
        )
        write_json_atomic(inventory, output)
    except (OSError, ValueError) as exc:
        print(f"inventory failed: {exc}", file=sys.stderr)
        return 1

    summary = {
        "status": inventory["status"],
        "output": str(output),
        "metrics": inventory["metrics"],
    }
    print(json.dumps(summary, sort_keys=True))
    if args.fail_on_error and inventory["metrics"]["error_count"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
