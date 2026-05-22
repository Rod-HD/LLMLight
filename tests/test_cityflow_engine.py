"""Unit tests for ``src.cityflow_engine``.

Validates:
    - Requirement 1.3 (CityFlow simulator wrapper init from config + seed)
    - Requirement 2.1 / 2.4 (next_step + set_phase API)
    - Requirement 2.6 / 2.7 (vehicle conservation tracking)
    - Requirement 10.1-10.5 (timing logic: green 30s + yellow 3s + all-red 2s
      on phase change; only green 30s on phase hold)
    - Requirement 14.11 (save_replay config mutation, replay file path)

Tests use a mock ``engine_factory`` to avoid the real CityFlow binding
(which is only built in WSL2 venv via setup_env.sh — not available on
the Windows dev box).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cityflow_engine import CityFlowEngine  # noqa: E402


# =========================================================================
# Mock CityFlow engine
# =========================================================================


class FakeCityFlowEngine:
    """Mock that mimics the subset of ``cityflow.Engine`` API we depend on.

    Tracks: number of next_step() calls, set_tl_phase() calls (with args),
    and exposes a configurable script of vehicle id sets per timestep so
    tests can validate spawned/completed counters.
    """

    def __init__(
        self,
        config_path: str,
        thread_num: int = 1,
        seed: int | None = None,
        *,
        vehicle_script: list[set[str]] | None = None,
        lane_counts: dict[str, int] | None = None,
    ) -> None:
        self.config_path = config_path
        self.thread_num = thread_num
        self.seed = seed
        self.next_step_calls = 0
        self.set_tl_phase_calls: list[tuple[str, int]] = []

        # Script of vehicle ids per timestep. Index 0 is t=0 (pre any
        # next_step). If exhausted, the final entry is repeated.
        self._vehicle_script: list[set[str]] = (
            vehicle_script if vehicle_script is not None else [set()]
        )
        self._lane_counts = lane_counts or {}

    # -- API mirroring cityflow.Engine ------------------------------------

    def next_step(self) -> None:
        self.next_step_calls += 1

    def set_tl_phase(self, intersection_id: str, phase_index: int) -> None:
        self.set_tl_phase_calls.append((intersection_id, phase_index))

    def get_lane_vehicle_count(self) -> dict[str, int]:
        return dict(self._lane_counts)

    def get_vehicles(self, include_waiting: bool = True) -> list[str]:
        # next_step_calls = number of advances done; index by that.
        idx = min(self.next_step_calls, len(self._vehicle_script) - 1)
        return list(self._vehicle_script[idx])


# =========================================================================
# Helpers
# =========================================================================


def _write_config(tmp_path: Path, name: str = "config.json", **overrides) -> Path:
    """Write a minimal CityFlow config JSON for tests."""
    cfg = {
        "interval": 1.0,
        "seed": 0,
        "dir": str(tmp_path) + "/",
        "roadnetFile": "roadnet.json",
        "flowFile": "flow.json",
        "rlTrafficLight": True,
        "saveReplay": False,
        "roadnetLogFile": "frontend/web/roadnetLogFile.json",
        "replayLogFile": "frontend/web/replayLogFile.txt",
        "laneChange": False,
    }
    cfg.update(overrides)
    p = tmp_path / name
    p.write_text(json.dumps(cfg), encoding="utf-8")
    return p


def _factory_with_script(
    vehicle_script: list[set[str]] | None = None,
    lane_counts: dict[str, int] | None = None,
):
    """Return an ``engine_factory`` callable that captures the FakeCityFlowEngine
    instance and binds a vehicle script."""
    captured: dict[str, FakeCityFlowEngine] = {}

    def factory(*args, **kwargs):
        # Drop unsupported kwarg the wrapper might pass (``seed``).
        # FakeCityFlowEngine accepts seed, so we let it through.
        eng = FakeCityFlowEngine(
            *args,
            **kwargs,
            vehicle_script=vehicle_script,
            lane_counts=lane_counts,
        )
        captured["engine"] = eng
        return eng

    factory.captured = captured  # type: ignore[attr-defined]
    return factory


# =========================================================================
# Initialization & error handling
# =========================================================================


def test_init_with_missing_config_raises_file_not_found(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.json"
    with pytest.raises(FileNotFoundError, match="config_path not found"):
        CityFlowEngine(str(missing), seed=42, engine_factory=_factory_with_script())


def test_init_with_directory_path_raises_file_not_found(tmp_path: Path) -> None:
    # tmp_path is a directory, not a file.
    with pytest.raises(FileNotFoundError, match="not a file"):
        CityFlowEngine(str(tmp_path), seed=42, engine_factory=_factory_with_script())


def test_init_with_invalid_json_raises_json_decode_error(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        CityFlowEngine(str(bad), seed=42, engine_factory=_factory_with_script())


def test_init_with_non_object_json_raises_value_error(tmp_path: Path) -> None:
    bad = tmp_path / "list.json"
    bad.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ValueError, match="top-level config JSON must be an object"):
        CityFlowEngine(str(bad), seed=42, engine_factory=_factory_with_script())


def test_init_with_invalid_seed_type_raises(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    with pytest.raises(ValueError, match="seed must be int"):
        CityFlowEngine(
            str(cfg),
            seed="42",  # type: ignore[arg-type]
            engine_factory=_factory_with_script(),
        )


def test_init_with_invalid_timing_raises(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    with pytest.raises(ValueError, match="green_duration must be > 0"):
        CityFlowEngine(
            str(cfg),
            seed=42,
            green_duration=0,
            engine_factory=_factory_with_script(),
        )


def test_init_succeeds_with_valid_config(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    factory = _factory_with_script()
    engine = CityFlowEngine(str(cfg), seed=42, engine_factory=factory)
    assert engine.seed == 42
    assert engine.green_duration == 30
    assert engine.yellow_duration == 3
    assert engine.all_red_duration == 2
    assert engine.save_replay is False
    assert engine.replay_file is None
    fake = factory.captured["engine"]
    assert fake.thread_num == 1


def test_init_falls_back_when_engine_factory_rejects_seed_kwarg(
    tmp_path: Path,
) -> None:
    """Real cityflow.Engine doesn't accept a ``seed`` kwarg; wrapper must
    fall back to positional-only call."""
    cfg = _write_config(tmp_path)

    calls: list[tuple[tuple, dict]] = []

    class StrictFake(FakeCityFlowEngine):
        def __init__(self, config_path: str, thread_num: int = 1) -> None:  # type: ignore[override]
            super().__init__(config_path, thread_num=thread_num)

    def factory(*args, **kwargs):
        calls.append((args, kwargs))
        if "seed" in kwargs:
            raise TypeError("StrictFake does not accept 'seed'")
        return StrictFake(*args, **kwargs)

    engine = CityFlowEngine(str(cfg), seed=99, engine_factory=factory)
    assert engine.seed == 99
    # Two attempts: first with seed kwarg, second without.
    assert len(calls) == 2
    assert "seed" in calls[0][1]
    assert "seed" not in calls[1][1]


def test_init_without_engine_factory_and_without_binding_raises_import_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _write_config(tmp_path)
    # Force the module-level _cityflow to None, regardless of host env.
    import src.cityflow_engine as ce

    monkeypatch.setattr(ce, "_cityflow", None)
    with pytest.raises(ImportError, match="cityflow Python binding is not available"):
        CityFlowEngine(str(cfg), seed=42)


# =========================================================================
# save_replay config mutation
# =========================================================================


def test_save_replay_requires_dataset_method_run_id(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    with pytest.raises(ValueError, match="save_replay=True requires"):
        CityFlowEngine(
            str(cfg),
            seed=42,
            save_replay=True,
            engine_factory=_factory_with_script(),
        )


def test_save_replay_partial_metadata_raises(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    with pytest.raises(ValueError, match=r"save_replay=True requires"):
        CityFlowEngine(
            str(cfg),
            seed=42,
            save_replay=True,
            dataset="jinan_1",
            method="lightgpt_hf",
            # run_id missing
            engine_factory=_factory_with_script(),
        )


def test_save_replay_writes_modified_config_and_creates_replay_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Run inside tmp_path so ``results/replays/`` is created there
    # (CityFlowEngine resolves _REPLAY_DIR relative to CWD).
    monkeypatch.chdir(tmp_path)

    cfg = _write_config(tmp_path)
    factory = _factory_with_script()
    engine = CityFlowEngine(
        str(cfg),
        seed=42,
        save_replay=True,
        dataset="jinan_1",
        method="lightgpt_hf",
        run_id=2,
        engine_factory=factory,
    )

    replay_dir = tmp_path / "results" / "replays"
    assert replay_dir.is_dir()

    expected_replay = replay_dir / "jinan_1_lightgpt_hf_run2.txt"
    assert engine.save_replay is True
    assert engine.replay_file == str(expected_replay)

    # The fake engine should have been instantiated with the COPY config
    # (not the original).
    fake = factory.captured["engine"]
    assert fake.config_path != str(cfg.resolve())
    assert fake.config_path.endswith("jinan_1_lightgpt_hf_run2_config.json")

    # The copy must contain saveReplay=true and the replay log paths.
    copy_path = Path(fake.config_path)
    copy_data = json.loads(copy_path.read_text(encoding="utf-8"))
    assert copy_data["saveReplay"] is True
    assert copy_data["replayLogFile"] == str(expected_replay)
    assert copy_data["roadnetLogFile"].endswith(
        "jinan_1_lightgpt_hf_run2_roadnet.json"
    )


def test_save_replay_default_false_does_not_create_replay_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    cfg = _write_config(tmp_path)
    engine = CityFlowEngine(
        str(cfg), seed=42, engine_factory=_factory_with_script()
    )
    assert engine.save_replay is False
    assert engine.replay_file is None
    assert not (tmp_path / "results" / "replays").exists()


# =========================================================================
# Phase timing logic
# =========================================================================


def test_set_phase_change_advances_yellow_plus_all_red_plus_green(
    tmp_path: Path,
) -> None:
    cfg = _write_config(tmp_path)
    factory = _factory_with_script()
    engine = CityFlowEngine(str(cfg), seed=42, engine_factory=factory)
    fake = factory.captured["engine"]

    engine.set_phase("intersection_1_1", phase_index=0)

    # First set_phase is a "change" (no previous phase). 3 + 2 + 30 = 35.
    assert fake.next_step_calls == 35
    assert fake.set_tl_phase_calls == [("intersection_1_1", 0)]


def test_set_phase_hold_advances_only_green(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    factory = _factory_with_script()
    engine = CityFlowEngine(str(cfg), seed=42, engine_factory=factory)
    fake = factory.captured["engine"]

    # First call: change to phase 0 → 35 timesteps.
    engine.set_phase("inter_A", phase_index=0)
    assert fake.next_step_calls == 35
    assert fake.set_tl_phase_calls == [("inter_A", 0)]

    # Second call with same phase: hold → only 30 timesteps; no extra
    # set_tl_phase call.
    engine.set_phase("inter_A", phase_index=0)
    assert fake.next_step_calls == 35 + 30
    assert fake.set_tl_phase_calls == [("inter_A", 0)]


def test_set_phase_change_after_hold(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    factory = _factory_with_script()
    engine = CityFlowEngine(str(cfg), seed=42, engine_factory=factory)
    fake = factory.captured["engine"]

    engine.set_phase("inter_A", phase_index=0)  # change → 35
    engine.set_phase("inter_A", phase_index=0)  # hold → 30
    engine.set_phase("inter_A", phase_index=2)  # change → 35

    assert fake.next_step_calls == 35 + 30 + 35
    assert fake.set_tl_phase_calls == [
        ("inter_A", 0),
        ("inter_A", 2),
    ]


def test_set_phase_per_intersection_state_is_independent(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    factory = _factory_with_script()
    engine = CityFlowEngine(str(cfg), seed=42, engine_factory=factory)
    fake = factory.captured["engine"]

    engine.set_phase("inter_A", phase_index=0)  # change → 35
    engine.set_phase("inter_B", phase_index=0)  # change (different inter) → 35
    engine.set_phase("inter_A", phase_index=0)  # hold → 30
    engine.set_phase("inter_B", phase_index=0)  # hold → 30

    assert fake.next_step_calls == 35 + 35 + 30 + 30
    assert fake.set_tl_phase_calls == [
        ("inter_A", 0),
        ("inter_B", 0),
    ]


def test_set_phase_rejects_invalid_phase_index(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    engine = CityFlowEngine(str(cfg), seed=42, engine_factory=_factory_with_script())
    with pytest.raises(ValueError, match="phase_index must be int"):
        engine.set_phase("inter_A", phase_index="0")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="phase_index must be >= 0"):
        engine.set_phase("inter_A", phase_index=-1)
    with pytest.raises(ValueError, match="intersection_id must be a non-empty str"):
        engine.set_phase("", phase_index=0)


def test_set_phase_with_custom_timings(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    factory = _factory_with_script()
    engine = CityFlowEngine(
        str(cfg),
        seed=42,
        green_duration=10,
        yellow_duration=2,
        all_red_duration=1,
        engine_factory=factory,
    )
    fake = factory.captured["engine"]

    engine.set_phase("inter_A", phase_index=0)  # change → 2 + 1 + 10 = 13
    engine.set_phase("inter_A", phase_index=0)  # hold → 10
    assert fake.next_step_calls == 13 + 10


# =========================================================================
# Vehicle counters
# =========================================================================


def test_vehicle_counts_initial_state_empty(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    engine = CityFlowEngine(
        str(cfg),
        seed=42,
        engine_factory=_factory_with_script(vehicle_script=[set()]),
    )
    assert engine.get_vehicle_count() == 0
    assert engine.get_vehicles_spawned_total() == 0
    assert engine.get_vehicles_completed_total() == 0


def test_vehicle_counts_increase_on_spawn(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    # t=0: empty; t=1: one vehicle; t=2: two vehicles.
    script = [set(), {"v1"}, {"v1", "v2"}]
    engine = CityFlowEngine(
        str(cfg),
        seed=42,
        engine_factory=_factory_with_script(vehicle_script=script),
    )

    engine.next_step()  # t=1
    assert engine.get_vehicle_count() == 1
    assert engine.get_vehicles_spawned_total() == 1
    assert engine.get_vehicles_completed_total() == 0

    engine.next_step()  # t=2
    assert engine.get_vehicle_count() == 2
    assert engine.get_vehicles_spawned_total() == 2
    assert engine.get_vehicles_completed_total() == 0


def test_vehicle_counts_track_completion(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    # t=0: empty; t=1: {v1}; t=2: {v1, v2}; t=3: {v2} (v1 left); t=4: {} (v2 left).
    script = [set(), {"v1"}, {"v1", "v2"}, {"v2"}, set()]
    engine = CityFlowEngine(
        str(cfg),
        seed=42,
        engine_factory=_factory_with_script(vehicle_script=script),
    )

    engine.next_step()  # t=1
    engine.next_step()  # t=2
    engine.next_step()  # t=3
    assert engine.get_vehicle_count() == 1
    assert engine.get_vehicles_spawned_total() == 2
    assert engine.get_vehicles_completed_total() == 1

    engine.next_step()  # t=4
    assert engine.get_vehicle_count() == 0
    assert engine.get_vehicles_spawned_total() == 2
    assert engine.get_vehicles_completed_total() == 2


def test_vehicle_conservation_invariant_holds(tmp_path: Path) -> None:
    """Property 5 invariant: current = spawned - completed at every timestep."""
    cfg = _write_config(tmp_path)
    script = [
        set(),
        {"v1"},
        {"v1", "v2"},
        {"v1", "v2", "v3"},
        {"v2", "v3"},
        {"v3", "v4"},
        {"v4"},
        set(),
    ]
    engine = CityFlowEngine(
        str(cfg),
        seed=42,
        engine_factory=_factory_with_script(vehicle_script=script),
    )
    for _ in range(len(script) - 1):
        engine.next_step()
        assert (
            engine.get_vehicle_count()
            == engine.get_vehicles_spawned_total()
            - engine.get_vehicles_completed_total()
        )


def test_get_lane_vehicle_count_returns_dict_with_int_values(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    factory = _factory_with_script(
        lane_counts={"road_0_lane_0": 3, "road_0_lane_1": 0},
    )
    engine = CityFlowEngine(str(cfg), seed=42, engine_factory=factory)
    counts = engine.get_lane_vehicle_count()
    assert counts == {"road_0_lane_0": 3, "road_0_lane_1": 0}
    # Defensive copy: mutating returned dict should not affect engine.
    counts["road_0_lane_0"] = 999
    assert engine.get_lane_vehicle_count()["road_0_lane_0"] == 3


# =========================================================================
# next_step refreshes tracking
# =========================================================================


def test_next_step_advances_underlying_engine_and_refreshes_tracking(
    tmp_path: Path,
) -> None:
    cfg = _write_config(tmp_path)
    factory = _factory_with_script(vehicle_script=[set(), {"v1"}, {"v1", "v2"}])
    engine = CityFlowEngine(str(cfg), seed=42, engine_factory=factory)
    fake = factory.captured["engine"]

    engine.next_step()
    assert fake.next_step_calls == 1
    assert engine.get_vehicle_count() == 1
    engine.next_step()
    assert fake.next_step_calls == 2
    assert engine.get_vehicle_count() == 2
