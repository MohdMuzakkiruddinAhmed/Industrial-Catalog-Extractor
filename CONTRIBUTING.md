# Contributing

Thank you for helping improve the Industrial Catalog Extractor.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --require-hashes --requirement requirements-dev.lock
python -m pip install --no-deps --editable .
pre-commit install  # optional
```

On PowerShell, activate with `.\.venv\Scripts\Activate.ps1`.

## Before opening a pull request

```bash
python -m ruff check src tests scripts examples
python -m pytest
python -m build
bash -n scripts/*.sh
```

Add or update tests for behavior changes. Keep model-dependent tests injectable and
deterministic; CI must not require model weights, GPUs, secrets, or network access.

## Evidence and safety invariants

Changes must preserve these properties unless the pull request explicitly proposes
and justifies a contract version change:

- exact source evidence is retained separately from normalized values;
- unsupported or cross-document evidence references fail closed;
- invalid siblings do not erase independently valid records;
- model responses and deterministic decisions remain auditable;
- retries are bounded and idempotent;
- raw page evidence and canonical product RAG tiers remain distinguishable;
- catalog PDFs, credentials, run databases, and generated outputs are not committed.

## Pull request description

Explain the problem, design, tests, compatibility impact, and any schema, prompt,
model, or checkpoint version change. Include small synthetic fixtures rather than
copyrighted catalog pages whenever possible.

## Research contributions

For benchmark claims, provide the corpus inclusion policy, manifest hash, page and
document counts, model revisions, configuration hash, raw aggregate counters, and
limitations. Do not publish source PDFs or extracted content without permission.

By contributing, you agree that your contribution is licensed under Apache-2.0.
