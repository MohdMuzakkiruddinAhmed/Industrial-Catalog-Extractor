"""Command-line interface for industrial catalog extraction benchmarks."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from . import __version__
from .config import PipelineConfig
from .knowledge_base import NvidiaEmbeddingClient, SQLiteKnowledgeBase
from .nvidia_clients import NIMEndpointConfig
from .runner import BenchmarkLimits, BenchmarkOptions, run_benchmark


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="industrial-catalog",
        description="Evidence-preserving NVIDIA industrial catalog extraction",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subcommands = parser.add_subparsers(dest="command", required=True)
    benchmark = subcommands.add_parser(
        "benchmark", help="run a bounded sequential extraction benchmark"
    )
    benchmark.add_argument("--config", required=True, type=Path)
    benchmark.add_argument("--output-dir", required=True, type=Path)
    benchmark.add_argument(
        "--include-glob",
        action="append",
        dest="include_globs",
        help="corpus-root-relative PDF glob; repeatable (default: **/*.pdf)",
    )
    benchmark.add_argument("--max-documents", type=_positive_integer)
    benchmark.add_argument("--max-pages", type=_positive_integer)
    benchmark.add_argument("--max-pages-per-document", type=_positive_integer)
    benchmark.add_argument("--pages-per-parse", type=_positive_integer, default=1)
    benchmark.add_argument(
        "--smoke",
        action="store_true",
        help="default to one document and at most two pages unless limits are supplied",
    )
    mode = benchmark.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run", action="store_true", help="force deterministic no-network model clients"
    )
    mode.add_argument("--live", action="store_true", help="force configured live endpoints")
    benchmark.add_argument("--strict-validation", action="store_true")
    benchmark.add_argument("--fail-fast", action="store_true")
    benchmark.add_argument("--run-id")
    query = subcommands.add_parser(
        "query", help="search a generated citation-bearing RAG knowledge base"
    )
    query.add_argument("query_text")
    query.add_argument("--kb", required=True, type=Path)
    query.add_argument("--limit", type=_positive_integer, default=5)
    query.add_argument(
        "--embedding-url",
        default="http://127.0.0.1:8004",
        help="Nemotron embedding server base URL",
    )
    query.add_argument(
        "--model",
        default="nvidia/Nemotron-3-Embed-8B-BF16",
    )
    query.add_argument(
        "--text-only",
        action="store_true",
        help="use deterministic SQLite text search without an embedding request",
    )
    return parser


def _load_config(path: Path, *, force_mode: str | None) -> PipelineConfig:
    with path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ValueError("pipeline config must contain a YAML object")
    if force_mode is not None:
        raw["mode"] = force_mode
    return PipelineConfig.model_validate(raw)


def _emit(payload: Mapping[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str), flush=True)


def _benchmark(args: argparse.Namespace) -> int:
    force_mode = "dry-run" if args.dry_run else "benchmark" if args.live else None
    config = _load_config(args.config, force_mode=force_mode)
    max_documents = args.max_documents
    max_pages = args.max_pages
    max_pages_per_document = args.max_pages_per_document
    if args.smoke:
        max_documents = max_documents or 1
        max_pages = max_pages or 2
        max_pages_per_document = max_pages_per_document or 2
    dry_run_override = True if args.dry_run else False if args.live else None
    options = BenchmarkOptions(
        config=config,
        output_dir=args.output_dir,
        limits=BenchmarkLimits(
            max_documents=max_documents,
            max_pages=max_pages,
            max_pages_per_document=max_pages_per_document,
            pages_per_parse=args.pages_per_parse,
        ),
        include_globs=tuple(args.include_globs or ("**/*.pdf",)),
        dry_run=dry_run_override,
        strict_validation=args.strict_validation,
        fail_fast=args.fail_fast,
        run_id=args.run_id,
    )
    result = run_benchmark(
        options,
        progress=lambda payload: _emit({"kind": "document-summary", **payload}),
    )
    _emit(result.summary)
    return result.exit_code


def _query(args: argparse.Namespace) -> int:
    if not args.kb.is_file():
        raise ValueError(f"knowledge-base database does not exist: {args.kb}")
    knowledge_base = SQLiteKnowledgeBase(args.kb)
    if args.text_only:
        chunks = knowledge_base.search_text(args.query_text, limit=args.limit)
        results = [
            {
                "score": None,
                "chunk": chunk.model_dump(mode="json", exclude={"embedding"}),
            }
            for chunk in chunks
        ]
        retrieval_mode = "text"
    else:
        client = NvidiaEmbeddingClient(
            NIMEndpointConfig(
                base_url=args.embedding_url,
                endpoint_path="/v1/embeddings",
                model=args.model,
                timeout_seconds=240,
            )
        )
        query_vector = client.embed_query(args.query_text)
        matches = knowledge_base.similarity_search(query_vector, limit=args.limit)
        results = [
            {
                "score": match.score,
                # Stored 4,096-dimensional vectors are an implementation detail;
                # omitting them keeps CLI/API-style query results small and useful.
                "chunk": match.chunk.model_dump(mode="json", exclude={"embedding"}),
            }
            for match in matches
        ]
        retrieval_mode = "nvidia-vector"
    _emit(
        {
            "kind": "industrial-catalog-rag-results",
            "retrieval_mode": retrieval_mode,
            "query": args.query_text,
            "count": len(results),
            "results": results,
        }
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "benchmark":
            return _benchmark(args)
        if args.command == "query":
            return _query(args)
    except (OSError, RuntimeError, ValueError, ValidationError) as exc:
        print(
            json.dumps(
                {
                    "kind": "industrial-catalog-cli-error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        return 2
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":  # pragma: no cover - exercised through python -m
    raise SystemExit(main())


__all__ = ["build_parser", "main"]
