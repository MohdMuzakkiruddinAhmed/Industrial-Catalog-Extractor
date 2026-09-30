from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from industrial_catalog.config import PipelineConfig

PROJECT_ROOT = Path(__file__).parents[1]
MONITOR_SCRIPT = PROJECT_ROOT / "scripts" / "monitor_pipeline.py"
SPEC = importlib.util.spec_from_file_location("monitor_pipeline", MONITOR_SCRIPT)
assert SPEC is not None and SPEC.loader is not None
monitor_pipeline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(monitor_pipeline)


def test_example_config_never_assigns_reserved_gpus() -> None:
    config = PipelineConfig.from_yaml(PROJECT_ROOT / "configs" / "pipeline.example.yaml")
    assert config.mode == "benchmark"
    assert config.gpu_ids == (1, 3, 4, 5, 6, 7)
    assert set(config.gpu_ids).isdisjoint({0, 2})
    assert config.routing.verify_identifier_regions is False
    assert config.routing.verify_tables is True
    assert config.ocr_endpoint is None
    assert config.parse_endpoint is not None
    assert config.parse_endpoint.model == "nvidia/NVIDIA-Nemotron-Parse-v1.2"
    assert config.parse_endpoint.max_retries == 7
    assert config.llm_endpoint is not None
    assert config.llm_endpoint.model == "nvidia/Llama-3.1-Nemotron-Nano-8B-v1"
    assert config.llm_endpoint.max_retries == 3
    assert config.embedding_endpoint is not None
    assert config.embedding_endpoint.max_retries == 3


@pytest.mark.parametrize("gpu_ids", ["0", "2", "1,2", "0,3", "7,8"])
def test_monitor_rejects_reserved_or_unlisted_gpu_ids(gpu_ids: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="not allowed"):
        monitor_pipeline.parse_gpu_ids(gpu_ids)


def test_monitor_queries_only_requested_physical_gpus(monkeypatch: pytest.MonkeyPatch) -> None:
    seen_command: list[str] = []

    def fake_run(command: list[str], **_kwargs: object) -> SimpleNamespace:
        seen_command.extend(command)
        rows = [
            "1, GPU-one, NVIDIA A100-SXM4-80GB, 10, 100, 81920, 35, 80",
            "3, GPU-three, NVIDIA A100-SXM4-80GB, 20, 200, 81920, 36, 90",
        ]
        return SimpleNamespace(stdout="\n".join(rows), stderr="")

    monkeypatch.setattr(monitor_pipeline.subprocess, "run", fake_run)
    metrics, errors = monitor_pipeline.collect_gpu_metrics((1, 3))

    assert errors == []
    assert seen_command[seen_command.index("-i") + 1] == "1,3"
    assert [item["physical_index"] for item in metrics] == [1, 3]


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_monitor_counts_only_run_or_job_status_payloads(tmp_path: Path) -> None:
    run_dir = tmp_path / "benchmarks" / "review-run"
    _write_json(
        run_dir / "run_status.json",
        {
            "kind": "industrial-catalog-benchmark-run",
            "run_id": "review-run",
            "status": "review",
        },
    )
    _write_json(
        run_dir / "summary.json",
        {
            "kind": "industrial-catalog-benchmark-summary",
            "run_id": "review-run",
            "status": "failed",
        },
    )
    _write_json(run_dir / "decisions" / "chunk.hints.json", {"status": "failed"})
    _write_json(run_dir / "extractions" / "document.json", {"state": "error"})

    metrics = monitor_pipeline.collect_job_metrics(tmp_path, 100)

    assert metrics["json_files_scanned"] == 4
    assert metrics["status_payloads_counted"] == 1
    assert metrics["ignored_json_payloads"] == 3
    assert metrics["parse_error_count"] == 0
    assert metrics["states"] == {
        "queued": 0,
        "running": 0,
        "succeeded": 0,
        "review": 1,
        "failed": 0,
        "skipped": 0,
        "unknown": 0,
    }


def test_monitor_review_is_non_degrading_and_historic_failure_still_degrades(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = tmp_path / "data"
    jobs_root = data_root / "benchmarks"
    data_root.mkdir()
    _write_json(
        jobs_root / "review-run" / "run_status.json",
        {
            "kind": "industrial-catalog-benchmark-run",
            "run_id": "review-run",
            "status": "needs_review",
        },
    )
    monkeypatch.setattr(monitor_pipeline, "collect_gpu_metrics", lambda _gpu_ids: ([], []))
    monkeypatch.setattr(
        monitor_pipeline,
        "collect_disk_metrics",
        lambda _path: ({"used_percent": 10.0}, None),
    )

    review_snapshot = monitor_pipeline.snapshot(
        gpu_ids=(1,),
        data_root=data_root,
        jobs_root=jobs_root,
        max_job_files=100,
        watch_pid=None,
        check_model_services=False,
    )

    assert review_snapshot["status"] == "healthy"
    assert review_snapshot["metrics"]["jobs"]["states"]["review"] == 1
    assert review_snapshot["errors"] == []

    _write_json(
        jobs_root / "historic-failure" / "run_status.json",
        {
            "kind": "industrial-catalog-benchmark-run",
            "run_id": "historic-failure",
            "status": "failed",
        },
    )
    failed_snapshot = monitor_pipeline.snapshot(
        gpu_ids=(1,),
        data_root=data_root,
        jobs_root=jobs_root,
        max_job_files=100,
        watch_pid=None,
        check_model_services=False,
    )

    assert failed_snapshot["status"] == "degraded"
    assert failed_snapshot["metrics"]["jobs"]["states"]["review"] == 1
    assert failed_snapshot["metrics"]["jobs"]["states"]["failed"] == 1
    assert "1 pipeline job(s) report a failed state" in failed_snapshot["errors"]


def test_remote_scripts_encode_permission_boundary_and_benchmark_limits() -> None:
    bootstrap = (PROJECT_ROOT / "scripts" / "bootstrap_remote.sh").read_text(encoding="utf-8")
    runner = (PROJECT_ROOT / "scripts" / "run_benchmark.sh").read_text(encoding="utf-8")

    assert '[[ ! -w "${DATA_ROOT}" ]]' in bootstrap
    assert "! -w /data" not in bootstrap
    assert '--max-documents "${MAX_DOCUMENTS}"' in runner
    assert '--max-pages "$((MAX_DOCUMENTS * MAX_PAGES_PER_DOCUMENT))"' in runner
    assert '--max-pages-per-document "${MAX_PAGES_PER_DOCUMENT}"' in runner
    assert "timeout --foreground --signal=TERM --kill-after=60s" in runner
    assert 'CUDA_VISIBLE_DEVICES="${GPU_IDS}"' in runner
    assert "openai_response_format" in runner


def test_model_service_scripts_are_narrowly_scoped() -> None:
    starter = (PROJECT_ROOT / "scripts" / "start_model_services.sh").read_text(encoding="utf-8")
    stopper = (PROJECT_ROOT / "scripts" / "stop_model_services.sh").read_text(encoding="utf-8")

    assert "nvidia/NVIDIA-Nemotron-Parse-v1.2" in starter
    assert "nvidia/Llama-3.1-Nemotron-Nano-8B-v1" in starter
    assert "nvidia/Nemotron-3-Embed-8B-BF16" in starter
    assert "start_service parse 1 8001" in starter
    assert "start_service llm 3 8003" in starter
    assert "start_service embedding 5 8004" in starter
    assert "--with-embedding" in starter
    assert "venvs/model-serve-py312" in starter
    assert "vllm==0.20.0" in starter
    assert "status/nemotron-parse.pid" in starter
    assert "status/nemotron-llm.pid" in starter
    assert "status/nemotron-embed.pid" in starter
    assert "pid_matches_service" in stopper
    assert "kill -TERM" in stopper
    assert "kill -KILL" in stopper
    for broad_action in ("pkill", "killall", "--gpu-reset"):
        assert broad_action not in starter
        assert broad_action not in stopper
