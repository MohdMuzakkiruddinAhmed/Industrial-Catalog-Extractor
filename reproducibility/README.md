# Public research evidence

This directory provides machine-readable, public-safe evidence for the measurements
reported in the architecture report.

- `aggregate-results.json` records the bounded production-validation and hardened
  replay counts.
- `environment.json` records the observed NVIDIA server profile and explicitly marks
  unrecorded immutable runtime/model fields as `null`.

The JSON files were transcribed from verified private operational artifacts. Raw
catalogs, model responses, extracted text, databases, credentials, private paths,
and model weights are not redistributed. The aggregate files therefore support
claim inspection and schema-level reuse, but they are not sufficient to reproduce
the exact private-corpus experiment.

For an exact future experiment, record the immutable fields listed in
`docs/REPRODUCIBILITY.md`, publish a redistribution-cleared manifest, and attach
signed run summaries and checksums.
