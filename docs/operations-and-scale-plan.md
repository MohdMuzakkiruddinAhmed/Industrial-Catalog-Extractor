# Industrial Catalog Extractor: Operations and Scale Plan

## Purpose and current status

This runbook covers transfer verification, NVIDIA service operation, bounded extraction,
monitoring, failure audit, and the work required before literal full-corpus execution.

The current pipeline has been exercised successfully enough to validate the architecture and
measure throughput, but it has **not** completed either corpus. In particular, production
validation run `20260807T064527Z-971750` processed 20 documents and 105 pages. It is a
validation batch, not a full-corpus result.

The currently configured extraction profile targets vendor product catalogs. The broader
reference library contains many manuals, data dictionaries, and reports that do not have
manufacturer, part name, or part number fields. Operate it as a separate campaign with its own
manifest and quality profile, even if both campaigns are later merged into one retrieval release.

## Verified environment and corpus

### GPU and model-service assignment

The server has eight NVIDIA A100 80 GB GPUs, numbered 0 through 7. Physical GPUs 0 and 2 are
reserved for pre-existing workloads and must not be used, reset, or stopped by this project.

| Service | Model | Physical GPU | Port | vLLM sequence limit |
| --- | --- | ---: | ---: | ---: |
| Document VLM | `nvidia/NVIDIA-Nemotron-Parse-v1.2` | 1 | 8001 | 8 |
| Structured LLM | `nvidia/Llama-3.1-Nemotron-Nano-8B-v1` | 3 | 8003 | 16 |
| Embeddings | `nvidia/Nemotron-3-Embed-8B-BF16` | 5 | 8004 | 16 |

The LLM service uses vLLM 0.20.0 with the `xgrammar` structured-output backend and
`disable_any_whitespace=true`. This prevents the grammar from exhausting the completion budget
on optional JSON whitespace. GPUs 4, 6, and 7 remain unassigned by the current service launcher;
do not assign them implicitly without a measured capacity plan.

### Corpus inventory

| Corpus | PDFs | Pages | Intended profile |
| --- | ---: | ---: | --- |
| Vendor product catalogs | 110 | 6,069 | Product/specification extraction |
| Industrial reference library | 110 | 11,768 | Reference-document extraction; separate quality policy |
| Combined transfer | 220 | 17,837 | Two pinned campaigns, optionally merged later |

The vendor corpus includes documents as large as 1,152 pages. Nineteen vendor PDFs exceed 50
pages and contain 5,247 pages. A run capped at 50 pages per document would cover only 1,772 of
the 6,069 vendor pages, or 29.2%. Consequently, increasing only the document limit cannot produce
literal corpus completion.

## Transfer and inventory verification

The canonical transfer archive is `industrial_catalogs_canonical_20260806.tar`, 918,048,768
bytes. Its verified SHA-256 is:

```text
26d2a4d92be06cfce8cb2af2210c7081548c8ff1463a49c03e86c56fb56b9a85
```

Verify the archive before extraction. This command fails closed if the server copy differs:

```bash
# Set this to the server path used for the uploaded archive.
archive_path=/replace/with/uploaded/industrial_catalogs_canonical_20260806.tar
expected_sha256=26d2a4d92be06cfce8cb2af2210c7081548c8ff1463a49c03e86c56fb56b9a85
actual_sha256="$(sha256sum "${archive_path}" | awk '{print $1}')"
test "${actual_sha256}" = "${expected_sha256}"
```

Do not extract over a populated corpus directory. Extract into a new staging directory, inventory
it, and promote it only after the checks pass. From the project root:

```bash
data_root=/data/industrial-data-corps
extractor_python="${data_root}/venvs/extractor-py312/bin/python"
manifest_root="${data_root}/corpus/manifests"

"${extractor_python}" scripts/inventory_corpus.py \
  "${data_root}/corpus/raw/vendor_product_catalogs" \
  --extensions pdf \
  --output "${manifest_root}/vendor.inventory.json" \
  --fail-on-error

"${extractor_python}" scripts/inventory_corpus.py \
  "${data_root}/corpus/raw/library" \
  --extensions pdf \
  --output "${manifest_root}/library.inventory.json" \
  --fail-on-error

jq -e \
  '.status == "complete" and .metrics.pdf_count == 110 and
   .metrics.total_pdf_pages == 6069 and .metrics.error_count == 0' \
  "${manifest_root}/vendor.inventory.json"

jq -e \
  '.status == "complete" and .metrics.pdf_count == 110 and
   .metrics.total_pdf_pages == 11768 and .metrics.error_count == 0' \
  "${manifest_root}/library.inventory.json"
```

Retain the two inventory files unchanged. Their canonical JSON hashes must become campaign input
identities for the future sharded runner. A later file modification must create a new manifest and
campaign, not silently alter an existing campaign.

## Service startup and health checks

Bootstrap the extraction environment once, then safely start or adopt the three services:

```bash
bash scripts/bootstrap_remote.sh
bash scripts/start_model_services.sh --with-embedding
```

The launcher verifies the PID, model, port, and physical GPU before reusing a live service. It
refuses unapproved service GPUs. Confirm both health and advertised model identity:

```bash
curl -fsS http://127.0.0.1:8001/health
curl -fsS http://127.0.0.1:8003/health
curl -fsS http://127.0.0.1:8004/health

curl -fsS http://127.0.0.1:8001/v1/models | jq -r '.data[].id'
curl -fsS http://127.0.0.1:8003/v1/models | jq -r '.data[].id'
curl -fsS http://127.0.0.1:8004/v1/models | jq -r '.data[].id'

nvidia-smi -i 1,3,5 \
  --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu \
  --format=csv,noheader,nounits
```

Use the scoped stop script when shutdown is intended:

```bash
bash scripts/stop_model_services.sh
```

Do not use `pkill`, `killall`, an NVIDIA GPU reset, or any command that targets unrelated GPU
processes.

## Current bounded-run safety envelope

`scripts/run_benchmark.sh` is deliberately a bounded validation wrapper. Its defaults are 20
documents, 12 pages per document, at most 240 candidate pages, and a two-hour wall timeout. It
enforces these limits:

- 1 through 100 documents;
- 1 through 50 pages per document;
- at most 500 candidate pages per invocation;
- a wall timeout from 60 through 86,400 seconds;
- physical GPU visibility limited to IDs 1, 3, 4, 5, 6, and 7;
- a unique run directory under `/data/industrial-data-corps/benchmarks`.

Keep `pages_per_parse=1` for production validation. Set wrapper bounds through environment
variables; forwarded `--max-*` arguments are rejected intentionally:

```bash
INDUSTRIAL_BENCHMARK_MAX_DOCUMENTS=20 \
INDUSTRIAL_BENCHMARK_MAX_PAGES_PER_DOCUMENT=12 \
INDUSTRIAL_BENCHMARK_TIMEOUT_SECONDS=7200 \
bash scripts/run_benchmark.sh --live --pages-per-parse 1
```

For an exact one-document retry or investigation, use a corpus-root-relative PDF path and a new
run directory produced by the wrapper:

```bash
INDUSTRIAL_BENCHMARK_MAX_DOCUMENTS=1 \
INDUSTRIAL_BENCHMARK_MAX_PAGES_PER_DOCUMENT=12 \
INDUSTRIAL_BENCHMARK_TIMEOUT_SECONDS=1800 \
bash scripts/run_benchmark.sh \
  --live --fail-fast --pages-per-parse 1 \
  --include-glob 'relative/path/to/catalog.pdf'
```

Never start two processes with the same output directory. SQLite can serialize many operations,
but the JSONL writers use process-local locks and extraction/decision filenames can overwrite one
another. Concurrent bounded runs must also use disjoint exact document selections. Broad globs
plus independent `max-documents` values can overlap silently because discovery is sorted and each
runner starts from the first matching path.

The configured `page_workers`, `document_workers`, and `parser_workers` are not currently consumed
by the runner. Changing those YAML values does not introduce parallelism.

## Verified validation runs

### Production validation batch

Run `20260807T064527Z-971750` produced the following verified result:

| Metric | Value |
| --- | ---: |
| Documents attempted | 20 |
| Documents accepted with warnings (`review`) | 18 |
| Documents failed | 2 |
| Pages counted in successful/review documents | 105 |
| Extracted elements | 1,622 |
| Products parsed | 73 |
| Products rejected by deterministic source guard | 41 |
| Products stored | 31 |
| Products invalid after validation | 1 |
| Knowledge-base chunks | 1,655 |
| Elapsed time | 828.999771 seconds |

The two failed documents were:

- Lovejoy complete catalog;
- Belden Copper Structured Cabling Design Guide.

Both failed on unsupported assistant content. Their final bounded replay result is pending at the
time of the original validation run.

The hardened replay `20260807T073659Z-1185130` subsequently resolved both failures. It completed
24 pages in 85.468255 seconds with two `review` documents, zero failed documents, 531 extracted
elements, and 531 citation-bearing page-evidence chunks. The one product candidate was rejected by
the deterministic source guard, so zero unsupported products entered the canonical catalog.

- Lovejoy page 6 exhausted eight successful-but-empty Parse responses, then used 25 quality-approved
  native text blocks while retaining the original VLM route and complete failure/fallback audit.
- Belden page 2 was certified blank from its zero source-object inventory, produced no evidence
  element, skipped the structured-product LLM, and created no ingestion batch.
- The replay contains 23 ingestion batches for 24 page chunks, exactly reflecting the one certified
  blank skip. Both SQLite databases passed integrity and foreign-key checks; the knowledge base has
  531 chunks and 531 citations.
- A live NVIDIA-vector query for the Lovejoy European-headquarters history returned grounded page
  evidence, including the page-6 native fallback with its element citation.

The 18 `review` documents are not execution failures. Their products were accepted and stored with
quality warnings. `documents_review` and `documents_failed` must remain separate in reporting.

### Throughput interpretation

The validation batch observed 7.90 seconds per counted page, 7.6 pages per minute, or about 456
pages per hour. This is useful for capacity planning, but it includes failed-document time while
the failed documents contribute zero pages to the runner summary. It is therefore not a clean
page-only benchmark and must not be presented as a completion result or an SLA.

At the observed rate, rough single-worker baselines are:

| Scope | Pages | Baseline | Operational planning range |
| --- | ---: | ---: | ---: |
| Vendor corpus | 6,069 | 13.3 hours | 16.6 to 20 hours |
| Reference library | 11,768 | 25.8 hours | Profile separately before committing |
| Combined corpus | 17,837 | 39.1 hours | 49 to 59 hours before measured concurrency gains |

Library routing and document density differ from the vendor sample. The library projection is only
arithmetic, not a validated forecast.

## Live monitoring and post-run checks

Run a one-shot infrastructure snapshot, including the embedding endpoint:

```bash
data_root=/data/industrial-data-corps
extractor_python="${data_root}/venvs/extractor-py312/bin/python"

"${extractor_python}" scripts/monitor_pipeline.py \
  --gpu-ids 1,3,5 \
  --data-root "${data_root}" \
  --jobs-root "${data_root}/benchmarks" \
  --output "${data_root}/status/manual-monitor.json" \
  --history "${data_root}/status/manual-monitor.jsonl" \
  --with-embedding-health \
  --once
```

The monitor counts only actual run/job status payloads. `review` is exposed separately and is
non-degrading. A historical `failed` status remains degrading and must not be hidden by a later
healthy endpoint check.

During a wrapper run:

```bash
run_dir=/data/industrial-data-corps/benchmarks/REPLACE_WITH_RUN_ID

tail -F "${run_dir}/benchmark.log" "${run_dir}/monitor.log"
jq '{status, observed_at, watch, errors, metrics}' "${run_dir}/monitor.latest.json"
```

After completion, reconcile the runner summary with the canonical databases:

```bash
jq '{run_id, elapsed_seconds, documents_attempted, documents_succeeded,
     documents_review, documents_failed, pages, elements, products_parsed,
     products_guard_rejected, products_stored, products_invalid, knowledge_chunks}' \
  "${run_dir}/summary.json"

sqlite3 "${run_dir}/catalog.sqlite3" \
  'PRAGMA integrity_check; PRAGMA foreign_key_check; SELECT COUNT(*) FROM products;'

sqlite3 "${run_dir}/knowledge_base.sqlite3" \
  'PRAGMA integrity_check; PRAGMA foreign_key_check; SELECT COUNT(*) FROM knowledge_chunks;'
```

Stop or pause new work if any of these occur:

- a model endpoint becomes unhealthy;
- repeated invalid/truncated JSON or unsupported assistant content;
- HTTP retries, timeouts, or server errors increase across consecutive units;
- GPU memory approaches exhaustion, an OOM occurs, or temperature exceeds site policy;
- a reserved GPU appears in project process ownership;
- SQLite reports locking/integrity errors or the data volume reaches 90% use;
- a known product page produces no accepted product without an auditable review reason.

## Failure audit and bounded replay

Structured decode failures are preserved under `structured/*.failure.json`, including bounded
attempts, raw provider responses, finish reasons, and failure classifications. Inspect them without
discarding the original run:

```bash
find "${run_dir}/structured" -type f -name '*.failure.json' -print

failure_path="${run_dir}/structured/REPLACE_WITH_FAILURE_FILE.failure.json"
jq '{error_type, error,
     attempts: [.attempts[] | {model, request_id, finish_reason, content_source}],
     failures}' "${failure_path}"

jq -r 'select(.status == "failed") |
       [.source_path, .error_type, .error] | @tsv' \
  "${run_dir}/documents.jsonl"
```

Successful and partially accepted structured batches retain raw LLM output and raw extraction in
`catalog.sqlite3`. List their audit identities before selecting one for deeper review:

```bash
sqlite3 -header -column "${run_dir}/catalog.sqlite3" \
  'SELECT batch_id, source_document_id, status, accepted_count, rejected_count
   FROM ingestion_batches ORDER BY created_at, batch_id;'
```

There is no general-purpose replay command that reuses a stored raw response today. A same-output
rerun is not a clean replay: page checkpoints may be reused, but LLM and embedding work runs again,
JSONL events can be appended again, and the summary reflects only the latest invocation. Until the
campaign runner described below exists, replay one exact document through a new bounded run and
retain both run directories for comparison.

## Why literal full-corpus execution must wait

The current runner discovers files from globs, selects the first bounded set, and only supports a
maximum-page prefix for each document. It has no manifest selection, page start/range, shard ID, or
completed-work-unit ledger. Page checkpoints live inside each run directory, while the shared
document checkpoint database is keyed only by document SHA and is not consulted to skip completed
documents.

Splitting one PDF into page windows also conflicts with current naming and KB behavior:

- extraction artifacts use only a document SHA prefix;
- structured and decision chunks restart numbering for every document invocation;
- document checkpoints cannot represent more than one range of the same PDF;
- document-level page-evidence indexing replaces all existing page evidence for that document;
- concurrent JSONL writes are not protected across processes.

These are correctness constraints, not merely performance limitations. Do not label a sequence of
prefix-capped benchmark runs as full-corpus extraction.

## Required manifest-pinned sharding design

Implement and test the following before processing every page.

### 1. Immutable campaign manifest

Build one campaign manifest for the vendor corpus and a second for the library. Each document entry
must contain its corpus-relative path, SHA-256, byte size, and total page count. Canonically hash the
manifest and store that hash in every work-unit status, output summary, and merged release.

Reject a campaign before inference if a path escapes the corpus root, is a symlink, has a different
hash, or has a different page count. Duplicate content hashes require an explicit keep/deduplicate
policy; they must not be silently assigned twice.

### 2. Stable page-window work units

Split each PDF into non-overlapping, one-based inclusive windows no larger than 50 pages. Use a
stable identity such as `sha-prefix-p000001-000050`. Persist the page assignments in the manifest;
do not regenerate them during a retry.

Include the page range in extraction, structured, decision, checkpoint, and status filenames. A
work unit must write to an isolated attempt directory. Keep extraction page checkpoints in a stable
unit checkpoint directory outside attempt outputs so an interrupted attempt can reuse completed VLM
work without mutating a prior completed attempt.

An atomic completion marker must include:

- work-unit ID and page range;
- source and campaign hashes;
- pipeline/config/prompt versions;
- exact model names;
- attempt ID and timestamps;
- runner summary and output hashes.

Treat `review` as completed with warnings. Retry failed or incomplete units; preserve failed attempt
artifacts for audit.

### 3. External shard workers

Preassign work units to balanced shards by page count and persist the assignment. Approximately 24
vendor shards, averaging about 253 pages, provide a practical first layout. Fifty-page windows
produce 206 vendor work units. The library produces 302 such units; after its separate pilot, about
48 similarly sized shards are a reasonable starting layout. Start with two external workers, one
exclusive shard per process, each using unique output and checkpoint state. This is safer and
smaller than adding threads inside the current runner.

The endpoints support concurrent requests, but linear speedup is not guaranteed because workers
share one GPU-backed service for each stage. Promote from two to three workers only after a
two-worker validation demonstrates better aggregate throughput without queue inflation, JSON
failures, timeouts, OOMs, or excessive p95 latency.

### 4. Deterministic merge

Merge only work units whose completion markers match the campaign hash and expected configuration.
Build a new temporary final directory rather than mutating the last published release.

For the catalog database:

1. create canonical document rows from the manifest;
2. import products, specifications, evidence, ingestion batches, batch-product links, and rejected
   products in foreign-key order;
3. accept a duplicate primary key only when the canonical payload is identical;
4. fail closed on a conflicting payload;
5. regenerate `products.jsonl` from the merged database in deterministic record-ID order.

For the knowledge base, copy stored vectors rather than re-embedding. Import knowledge chunks before
citations, deduplicate identical chunk IDs, and fail on conflicting content. Preserve unit and shard
artifacts as the audit source even after publication.

Before atomic publication, require:

- every manifest page assigned exactly once, with no gaps or overlaps;
- no pending, running, or unresolved failed work units;
- review units counted separately;
- all unit source/config/campaign hashes matching;
- SQLite `integrity_check` equal to `ok` and empty `foreign_key_check` output;
- final document count matching the campaign manifest;
- catalog and KB counts reconciled against completed units;
- one embedding model and dimension across the release;
- a generated release summary and content checksums.

## Completion definitions

A **validation batch** proves that a bounded selection can traverse extraction, structured parsing,
validation, persistence, embedding, and monitoring. It may contain reviews or failures and may be
used to tune capacity. Run `20260807T064527Z-971750` is in this category.

A **completed campaign** means every page in one immutable corpus manifest has a terminal completed
or review work unit, no unresolved failure, and a verified merged release.

A **completed combined corpus** means both the 6,069-page vendor campaign and the 11,768-page library
campaign satisfy their separate completion gates, followed by a conflict-checked combined retrieval
release. No current run meets either full-campaign definition.
