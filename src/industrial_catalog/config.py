"""Typed configuration for the extraction pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class EndpointConfig(BaseModel):
    """An OpenAI-compatible or NVIDIA NIM inference endpoint."""

    model_config = ConfigDict(extra="forbid")

    base_url: str
    model: str
    api_key_env: str | None = None
    timeout_seconds: float = Field(default=180.0, gt=0)
    max_retries: int = Field(default=3, ge=0, le=10)


class RoutingConfig(BaseModel):
    """Rules used to select native extraction, OCR, or the document VLM."""

    model_config = ConfigDict(extra="forbid")

    native_min_chars: int = Field(default=80, ge=0)
    native_min_chars_per_square_inch: float = Field(default=2.0, ge=0)
    render_dpi: int = Field(default=200, ge=96, le=400)
    verify_identifier_regions: bool = True
    verify_tables: bool = True


class PipelinePaths(BaseModel):
    """All persistent paths, kept under the dedicated data volume."""

    model_config = ConfigDict(extra="forbid")

    corpus_root: Path
    artifact_root: Path
    output_root: Path
    checkpoint_db: Path


class PipelineConfig(BaseModel):
    """Top-level pipeline configuration."""

    model_config = ConfigDict(extra="forbid")

    run_name: str = "industrial-catalog-extraction"
    mode: Literal["dry-run", "benchmark", "production"] = "dry-run"
    paths: PipelinePaths
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    parse_endpoint: EndpointConfig | None = None
    ocr_endpoint: EndpointConfig | None = None
    llm_endpoint: EndpointConfig | None = None
    embedding_endpoint: EndpointConfig | None = None
    page_workers: int = Field(default=3, ge=1, le=32)
    document_workers: int = Field(default=2, ge=1, le=16)
    parser_workers: int = Field(default=1, ge=1, le=8)
    gpu_ids: tuple[int, ...] = (1, 3, 4, 5, 6, 7)

    @model_validator(mode="after")
    def require_endpoints_for_live_runs(self) -> PipelineConfig:
        if self.mode != "dry-run" and (self.parse_endpoint is None or self.llm_endpoint is None):
            raise ValueError(
                "benchmark and production modes require parse_endpoint and llm_endpoint"
            )
        if any(gpu in {0, 2} for gpu in self.gpu_ids):
            raise ValueError("GPUs 0 and 2 are reserved for existing workloads")
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> PipelineConfig:
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.model_validate(yaml.safe_load(handle))


def ensure_runtime_directories(config: PipelineConfig) -> None:
    """Create only the explicitly configured runtime directories."""

    for path in (
        config.paths.corpus_root,
        config.paths.artifact_root,
        config.paths.output_root,
        config.paths.checkpoint_db.parent,
    ):
        path.mkdir(parents=True, exist_ok=True)
