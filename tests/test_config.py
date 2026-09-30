from pathlib import Path

import pytest
from pydantic import ValidationError

from industrial_catalog.config import PipelineConfig


def base_config(tmp_path: Path) -> dict:
    return {
        "mode": "dry-run",
        "paths": {
            "corpus_root": tmp_path / "corpus",
            "artifact_root": tmp_path / "artifacts",
            "output_root": tmp_path / "output",
            "checkpoint_db": tmp_path / "runtime" / "checkpoints.sqlite3",
        },
    }


def test_dry_run_config_has_safe_gpu_defaults(tmp_path: Path) -> None:
    config = PipelineConfig.model_validate(base_config(tmp_path))
    assert set(config.gpu_ids).isdisjoint({0, 2})


def test_reserved_gpu_is_rejected(tmp_path: Path) -> None:
    raw = base_config(tmp_path)
    raw["gpu_ids"] = [0, 1]
    with pytest.raises(ValidationError, match="reserved"):
        PipelineConfig.model_validate(raw)


def test_live_mode_requires_model_endpoints(tmp_path: Path) -> None:
    raw = base_config(tmp_path)
    raw["mode"] = "benchmark"
    with pytest.raises(ValidationError, match="parse_endpoint"):
        PipelineConfig.model_validate(raw)
