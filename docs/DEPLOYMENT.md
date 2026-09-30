# Deployment guide

## Library deployment

The Python package runs on any operating system supported by Python 3.12 and
PyMuPDF. It calls model services through OpenAI-compatible HTTP endpoints, so the
extractor and GPUs may run on different hosts when network and data policy allow it.

Install the extractor:

```bash
python -m pip install .
industrial-catalog --help
```

The included Dockerfile packages only the extractor client. It does not contain
CUDA, vLLM, model weights, or source data.

## Model-service deployment

The reference service scripts are Linux-specific and assume:

- Ubuntu 24.04;
- Python 3.12;
- a CUDA/NVIDIA driver stack compatible with the chosen vLLM release;
- authorized access to the upstream model repositories;
- a writable data root below `/data`;
- a separately prepared model-serving environment with vLLM 0.20.0.

The launcher deliberately refuses to modify unknown GPU processes. Review and adapt
the GPU assignments before use on another server.

## Reference eight-A100 profile

| Model | Physical GPU | Port |
| --- | ---: | ---: |
| Nemotron Parse v1.2 | 1 | 8001 |
| Llama 3.1 Nemotron Nano 8B v1 | 3 | 8003 |
| Nemotron 3 Embed 8B BF16 | 5 | 8004 |

GPUs 0 and 2 were reserved by the validation site. GPUs 4, 6, and 7 were left
unassigned. Those constraints are encoded in the reference scripts to protect that
server and are not general recommendations.

## Network boundary

The reference endpoints bind to loopback. If an endpoint is made remote:

- use TLS and authentication;
- restrict ingress and egress;
- confirm that rendered source pages may cross the boundary;
- set API keys through environment variables or a secret manager;
- avoid including secrets in serialized endpoint metadata;
- retain raw requests and responses only as long as required.

## Health verification

```bash
curl -fsS http://127.0.0.1:8001/health
curl -fsS http://127.0.0.1:8003/health
curl -fsS http://127.0.0.1:8004/health

curl -fsS http://127.0.0.1:8001/v1/models
curl -fsS http://127.0.0.1:8003/v1/models
curl -fsS http://127.0.0.1:8004/v1/models
```

Verify the advertised model identities, GPU ownership, memory, utilization, and
temperature before running a live campaign.

## Important reproducibility gap

The current launcher verifies vLLM 0.20.0 but does not create the model-serving
environment or pin immutable model revisions. Treat it as the validated site
launcher, not a universal one-command model installer. A public experiment should
record a container digest or locked environment and exact model revisions.
