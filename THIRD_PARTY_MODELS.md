# Third-party models and services

This repository contains integration code, prompts, schemas, and deployment
scripts. It does not contain or redistribute model weights.

## NVIDIA model cards

| Pipeline role | Upstream model | Terms shown by upstream |
| --- | --- | --- |
| Document VLM | [NVIDIA Nemotron Parse v1.2](https://huggingface.co/nvidia/NVIDIA-Nemotron-Parse-v1.2) | NVIDIA Open Model License |
| Structured LLM | [Llama 3.1 Nemotron Nano 8B v1](https://huggingface.co/nvidia/Llama-3.1-Nemotron-Nano-8B-v1) | NVIDIA Open Model License and applicable Llama terms |
| Embeddings | [Nemotron 3 Embed 8B BF16](https://huggingface.co/nvidia/Nemotron-3-Embed-8B-BF16) | OpenMDW License Agreement 1.1 |

The upstream model card is the authority for the model version, license,
acceptable use, hardware support, dependencies, and deployment instructions.
Those terms can change independently of this repository. Pin a model revision in
published experiments and archive the applicable terms with the release record.

## License boundary

Apache-2.0 covers only original repository code and documentation. It does not
relicense:

- NVIDIA or Meta model weights and tokenizer assets;
- vLLM, PyMuPDF, Pydantic, HTTPX, or other dependencies;
- NVIDIA, Nemotron, or other trademarks;
- catalog PDFs or any extracted copyrighted content;
- data supplied by a user or organization.

Review all upstream terms before commercial use, redistribution, or publication.

## Reproducible model identity

For each experiment, record:

1. Exact model identifier and immutable revision or weight checksum.
2. Serving engine and version.
3. CUDA, driver, PyTorch, and GPU model.
4. Endpoint decoding and structured-output configuration.
5. Prompt, guided-schema, and pipeline versions.
6. Embedding model, dimension, normalization, and query/passage prefixes.

The current research profile uses `passage: ` for indexed text and `query: ` for
queries with the Nemotron embedding model.
