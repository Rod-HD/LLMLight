"""Tests for API replay cache."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.training.api_replay_cache import (
    APIDecisionRecord,
    APIReplayCache,
    CacheMismatch,
    CacheMiss,
    ManifestChecksumError,
    stable_hash_json,
    stable_hash_text,
)
from src.training.multi_backend_api_client import APIBackend, TokenUsageLog


def _record(**overrides) -> APIDecisionRecord:
    values = {
        "run_fingerprint": "run-a",
        "dataset": "jinan_1",
        "phase": 1,
        "mode": "demo",
        "method": "gpt4o_groq",
        "backend": APIBackend.GROQ,
        "model": "llama-3.3-70b-versatile",
        "run_id": 0,
        "seed": 42,
        "timestep": 30,
        "intersection_id": "intersection_1_1",
        "state_hash": stable_hash_json({"queue": 2}),
        "prompt_hash": stable_hash_text("prompt"),
        "prompt": "prompt",
        "raw_response": "<signal>ETWT</signal>",
        "parsed_phase": "ETWT",
        "fallback_used": False,
        "error_type": None,
        "input_tokens": 10,
        "output_tokens": 4,
        "request_started_at": "2026-05-22T10:00:00Z",
        "response_received_at": "2026-05-22T10:00:01Z",
        "latency_ms": 1000,
    }
    values.update(overrides)
    return APIDecisionRecord(**values)


def _manifest(cache: APIReplayCache, jsonl: Path):
    return cache.build_manifest(
        dataset="jinan_1",
        method="gpt4o_groq",
        phase=1,
        mode="demo",
        backend=APIBackend.GROQ,
        model="llama-3.3-70b-versatile",
        run_id=0,
        seed=42,
        run_fingerprint="run-a",
        roadnet_path="roadnet.json",
        roadnet_sha256="r" * 64,
        flow_path="flow.json",
        flow_sha256="f" * 64,
        simulation_config_sha256="c" * 64,
        prompt_template_version="v1",
        code_commit=None,
        replay_file="results/replays/demo.txt",
        metrics_file="results/metrics/demo.json",
        llm_log_dir="results/logs/llm_prompts",
        token_usage=TokenUsageLog(
            total_input_tokens=10,
            total_output_tokens=4,
            total_requests=1,
            backend=APIBackend.GROQ,
        ),
    )


def test_append_writes_jsonl_and_loads_by_manifest(tmp_path):
    cache = APIReplayCache(tmp_path)
    record = _record()

    jsonl = cache.append(record)
    manifest_path = cache.write_manifest(_manifest(cache, jsonl))

    loaded_cache = APIReplayCache(tmp_path)
    manifest = loaded_cache.load_manifest(manifest_path)
    found = loaded_cache.get_decision(
        timestep=record.timestep,
        intersection_id=record.intersection_id,
        state_hash=record.state_hash,
        prompt_hash=record.prompt_hash,
    )

    assert manifest.run_fingerprint == "run-a"
    assert found == record
    assert json.loads(jsonl.read_text(encoding="utf-8").splitlines()[0])[
        "backend"
    ] == "groq"


def test_get_decision_raises_cache_miss(tmp_path):
    cache = APIReplayCache(tmp_path)
    record = _record()
    jsonl = cache.append(record)
    cache.write_manifest(_manifest(cache, jsonl))

    with pytest.raises(CacheMiss, match="No cached API decision"):
        cache.get_decision(
            timestep=999,
            intersection_id=record.intersection_id,
            state_hash=record.state_hash,
            prompt_hash=record.prompt_hash,
        )


def test_get_decision_raises_cache_mismatch(tmp_path):
    cache = APIReplayCache(tmp_path)
    record = _record()
    jsonl = cache.append(record)
    cache.write_manifest(_manifest(cache, jsonl))

    with pytest.raises(CacheMismatch, match="hash mismatch"):
        cache.get_decision(
            timestep=record.timestep,
            intersection_id=record.intersection_id,
            state_hash="bad-state",
            prompt_hash=record.prompt_hash,
        )


def test_manifest_checksum_mismatch_is_rejected(tmp_path):
    cache = APIReplayCache(tmp_path)
    record = _record()
    jsonl = cache.append(record)
    manifest_path = cache.write_manifest(_manifest(cache, jsonl))
    with jsonl.open("a", encoding="utf-8") as handle:
        handle.write("\n")
        handle.write(json.dumps(_record(timestep=31).to_dict()))
        handle.write("\n")

    with pytest.raises(ManifestChecksumError, match="checksum mismatch"):
        APIReplayCache(tmp_path).load_manifest(manifest_path)


def test_verify_decisions_writes_mismatch_log(tmp_path):
    cache = APIReplayCache(tmp_path)
    record = _record()
    jsonl = cache.append(record)
    manifest_path = cache.write_manifest(_manifest(cache, jsonl))
    cache.load_manifest(manifest_path)

    mismatch_log = tmp_path / "replay_mismatch.log"
    ok = cache.verify_decisions(
        [_record(parsed_phase="NLSL")],
        mismatch_log,
    )

    assert not ok
    assert "Phase mismatch" in mismatch_log.read_text(encoding="utf-8")


def test_stable_hash_helpers_are_deterministic():
    assert stable_hash_text("abc") == stable_hash_text("abc")
    assert stable_hash_json({"b": 2, "a": 1}) == stable_hash_json(
        {"a": 1, "b": 2}
    )
