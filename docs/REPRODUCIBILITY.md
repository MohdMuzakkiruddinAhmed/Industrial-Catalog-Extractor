# Reproducibility guide

## Reproduction levels

| Level | Requires | What it validates |
| --- | --- | --- |
| Offline software | Python 3.12 | Package, routing, evidence contracts, storage, RAG, CLI, tests |
| Live model | NVIDIA GPU endpoints and authorized PDFs | End-to-end model integration on a new corpus |
| Exact experiment | Original corpus, immutable manifests, pinned weights/runtime, run artifacts | The reported experiment itself |

The first level is public and deterministic. The second is reproducible with
user-supplied inputs but may vary with hardware and model revision. The third cannot
be reproduced from this repository alone because the source catalog corpus and model
weights are not redistributed.

## Clean-clone verification

Use Python 3.12:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --require-hashes --requirement requirements-dev.lock
python -m pip install --no-deps --editable .

python -m ruff check src tests scripts examples
python -m pytest
python -m build
bash -n scripts/*.sh
```

Then run the deterministic pipeline:

```bash
python examples/create_sample_catalog.py \
  --output .demo/corpus/sample-catalog.pdf

industrial-catalog benchmark \
  --config configs/pipeline.dry-run.yaml \
  --output-dir .demo/run \
  --dry-run \
  --smoke

industrial-catalog query "FlexCouple" \
  --kb .demo/run/knowledge_base.sqlite3 \
  --text-only
```

Expected structural outcome:

- one document attempted;
- two pages completed;
- no failed document;
- byte-stable synthetic input and deterministic model-stub/evidence content;
- structurally equivalent run artifacts after excluding run IDs, timestamps,
  elapsed durations, and database creation-time fields;
- at least one text-search result carrying a source citation.

The complete run directory is not byte-identical because operational metadata is
time-dependent. Exact content identities can also change when contract versions
intentionally change. The tests are the authoritative release-level expectations.

## Live endpoint reproduction

Record all of the following before inference:

```text
repository commit
pipeline version
configuration SHA-256
prompt and guided-schema SHA-256
corpus manifest SHA-256
model identifiers and immutable revisions
serving engine and version
Python, CUDA, driver, PyTorch, xgrammar, and GPU model
endpoint generation parameters
embedding model, dimension, normalization, and prefixes
```

Do not rely on a mutable model branch such as `main` for a published experiment.
Pin an upstream revision or local weight checksum and store it in the run record.

## Corpus manifest requirements

An exact campaign manifest should contain, for each document:

- corpus-relative POSIX path;
- SHA-256;
- byte size;
- total page count;
- inclusion profile and any known exclusion reason.

Reject paths that escape the corpus root, symlinks unless explicitly allowed,
unexpected hashes, and unexpected page counts. Canonically serialize and hash the
manifest. Store the hash in every work-unit status and release summary.

## Reported validation evidence

The bounded production run recorded:

```text
run_id: 20260807T064527Z-971750
documents_attempted: 20
documents_review: 18
documents_failed: 2
pages: 105
elements: 1622
products_parsed: 73
products_guard_rejected: 41
products_stored: 31
products_invalid: 1
knowledge_chunks: 1655
elapsed_seconds: 828.999771
```

The hardened replay recorded:

```text
run_id: 20260807T073659Z-1185130
documents_review: 2
documents_failed: 0
pages: 24
elements: 531
ingestion_batches: 23
products_stored: 0
knowledge_chunks: 531
citations: 531
elapsed_seconds: 85.468255
```

These summaries are documented operational evidence. The corresponding source PDFs
and remote SQLite databases are not distributed.

## Release gates

Before publishing a source release:

1. Run tests, Ruff, package build, and shell syntax checks from a clean environment.
2. Install the built wheel into a second clean environment and run CLI help/version.
3. Run the synthetic dry-run and citation-bearing text query.
4. Confirm SQLite `integrity_check` returns `ok` and `foreign_key_check` is empty.
5. Scan staged files and Git history for secrets and private paths.
6. Confirm catalogs, raw responses, model caches, databases, and transfer archives
   are absent from the source archive.
7. Extract DOCX XML and PDF text and scan them for personal paths or secrets.
8. Record artifact SHA-256 checksums and software dependency versions.
9. Confirm the report distinguishes bounded validation from full-corpus completion.

## Evaluation protocol for future research

A labeled evaluation set should report:

- page-route precision by document type;
- element text accuracy and reading-order accuracy;
- manufacturer, part-name, part-number, and specification exact/normalized scores;
- evidence-reference validity and human-rated citation support;
- source-guard true rejection and false rejection rates;
- blank-page and native-fallback precision;
- product-tier and raw-tier retrieval recall at K, nDCG, and citation correctness;
- latency distributions, GPU utilization, retry rate, and failure classification.

Publish denominators, confidence intervals where appropriate, failure examples, and
the sampling protocol. Avoid presenting stored-product count as an accuracy metric.
