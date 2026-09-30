# Synthetic dry-run example

The generator creates a fictional two-page industrial catalog. It contains no
third-party product content and is safe to regenerate in tests or publications.

From the repository root:

```bash
python examples/create_sample_catalog.py \
  --output .demo/corpus/sample-catalog.pdf

industrial-catalog benchmark \
  --config configs/pipeline.dry-run.yaml \
  --output-dir .demo/run \
  --dry-run \
  --smoke
```

The dry-run result proves orchestration and data-contract behavior only. It does
not evaluate NVIDIA model accuracy because model inference is replaced with stable
offline responses.
