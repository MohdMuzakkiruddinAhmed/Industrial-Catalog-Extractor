# Third-party notices

This file is informational and is not legal advice. Dependency licenses and model
terms remain authoritative at their upstream sources.

## Runtime libraries

The package declares these direct runtime dependencies:

| Dependency | Purpose | Upstream license information |
| --- | --- | --- |
| [PyMuPDF](https://pymupdf.readthedocs.io/en/latest/about.html) | PDF text, objects, rendering, and page geometry | Dual licensed under GNU AGPL and a commercial license from Artifex; evaluate obligations for distribution and network services |
| [Pydantic](https://github.com/pydantic/pydantic) | Typed data contracts and validation | MIT |
| [PyYAML](https://github.com/yaml/pyyaml) | Configuration loading | MIT |

PyMuPDF's license deserves special attention. The Apache-2.0 license in this
repository covers original pipeline code only; it does not replace PyMuPDF's terms.
If the AGPL is unsuitable for an intended distribution or hosted service, consult
qualified counsel and Artifex about a commercial PyMuPDF/MuPDF license.

The development and report extras install additional packages. Inspect the resolved
environment and preserve its license metadata when distributing a bundled product.

## Model-serving stack

The reference deployment uses vLLM and a CUDA/PyTorch environment that is managed
separately from the extractor package. Its transitive licenses, NVIDIA container
terms, CUDA terms, and driver terms are not reproduced here.

## NVIDIA models

See [THIRD_PARTY_MODELS.md](THIRD_PARTY_MODELS.md) for the three model cards and
their separate license boundary. Model weights are not distributed with this
repository.

## Data

Source catalog PDFs and source-derived outputs are not third-party software and are
not covered by the repository code license. See
[DATA_AVAILABILITY.md](DATA_AVAILABILITY.md).
