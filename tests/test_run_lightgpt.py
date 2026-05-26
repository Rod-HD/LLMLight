"""Unit tests for ``scripts/run_lightgpt.py`` (Task 13.1).

Validates:
    - Requirement 1.5 (runner orchestration)
    - Requirement 5.1-5.12 (LightGPT inference, two variants)
    - Requirement 10.1-10.6 (simulation config consistency)
    - Requirement 13.1-13.7 (PhaseApprovalGate + phase label)
    - Requirement 14.6 / 14.10 / 14.11 (replay file + LLM logs for UI)

Strategy:

* ``CityFlowEngine`` and ``LightGPTInference`` factories are injected via
  module-level monkeypatch on ``run_method_runs`` so the tests never load
  the native cityflow binding or transformers.
* ``PreflightChecker.run_all`` is patched to a no-op (the runner exposes
  ``--skip-preflight`` for the same reason but we exercise the patched
  path explicitly).
* ``PhaseApprovalGate`` is exercised with the real implementation; tests
  inject input via ``LLMLIGHT_AUTO_APPROVE=yes`` for Phase 3.
* The ``LLMTSCS_DIR`` env points at a temp dir that contains realistic
  ``data/<dataset>/`` layout with a parseable roadnet stub (so
  ``PhaseIndexMapper`` resolves at least one intersection).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_lightgpt  # noqa: E402
from scripts._runner_utils import DATASET_FILES  # noqa: E402


# ---------------------------------------------------------------------------
# Roadnet stub used by PhaseIndexMapper
# ---------------------------------------------------------------------------


def _stub_roadnet() -> dict:
    """Build a minimal CityFlow roadnet JSON with one intersection that
    has all 4 LLMLight phases (ETWT/NTST/ELWL/NLSL) so PhaseIndexMapper
    resolves them deterministically.

    Layout: 4 incoming roads (E/W/N/S) each with one straight + one left
    movement. Phase 0 = ETWT, 1 = NTST, 2 = ELWL, 3 = NLSL.
    """
    # Helper: build a road approaching the intersection from a compass direction.
    def road(rid: str, dx: int, dy: int) -> dict:
        # ``points[0]`` is far end, ``points[-1]`` is at the intersection.
        return {
            "id": rid,
            "points": [
                {"x": dx, "y": dy},  # far point (compass direction)
                {"x": 0, "y": 0},  # intersection center
            ],
            "lanes": [{"width": 4.0, "maxSpeed": 16.0}],
        }

    # 4 inbound roads, one per direction.
    roads = [
        road("road_E", 100, 0),  # approaches from East (dx > 0 → "E")
        road("road_W", -100, 0),  # West
        road("road_N", 0, 100),  # North
        road("road_S", 0, -100),  # South
    ]

    # 8 movements: 4 straight + 4 left (turn_right would be ignored).
    road_links = [
        {"startRoad": "road_E", "endRoad": "road_W", "type": "go_straight"},
        {"startRoad": "road_W", "endRoad": "road_E", "type": "go_straight"},
        {"startRoad": "road_N", "endRoad": "road_S", "type": "go_straight"},
        {"startRoad": "road_S", "endRoad": "road_N", "type": "go_straight"},
        {"startRoad": "road_E", "endRoad": "road_N", "type": "turn_left"},
        {"startRoad": "road_W", "endRoad": "road_S", "type": "turn_left"},
        {"startRoad": "road_N", "endRoad": "road_E", "type": "turn_left"},
        {"startRoad": "road_S", "endRoad": "road_W", "type": "turn_left"},
    ]

    light_phases = [
        {"availableRoadLinks": [0, 1]},  # ETWT — E/W straight
        {"availableRoadLinks": [2, 3]},  # NTST — N/S straight
        {"availableRoadLinks": [4, 5]},  # ELWL — E/W left
        {"availableRoadLinks": [6, 7]},  # NLSL — N/S left
    ]

    return {
        "intersections": [
            {
                "id": "intersection_1",
                "virtual": False,
                "trafficLight": {"lightphases": light_phases},
                "roadLinks": road_links,
            },
        ],
        "roads": roads,
    }


# ---------------------------------------------------------------------------
# Fakes for CityFlowEngine + LightGPTInference
# ---------------------------------------------------------------------------


class FakeEngine:
    """Replicates the surface of :class:`CityFlowEngine` used by the runner.

    Records ``set_phase`` calls so tests can assert decisions occurred.
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.config_path = kwargs["config_path"]
        self.seed = kwargs["seed"]
        self.save_replay = kwargs.get("save_replay", False)
        self.dataset = kwargs.get("dataset")
        self.method = kwargs.get("method")
        self.run_id = kwargs.get("run_id")
        self.replay_file = (
            f"results/replays/{self.dataset}_{self.method}_run{self.run_id}.txt"
            if self.save_replay
            else None
        )
        self.green_duration = kwargs.get("green_duration", 30)
        self.yellow_duration = kwargs.get("yellow_duration", 3)
        self.all_red_duration = kwargs.get("all_red_duration", 2)
        self.set_phase_calls: list[tuple[str, int]] = []
        self._step = 0

    def get_lane_vehicle_count(self) -> dict[str, int]:
        return {"lane_a": 0, "lane_b": 0}

    def set_phase(self, intersection_id: str, phase_index: int) -> None:
        self.set_phase_calls.append((intersection_id, phase_index))
        # Pretend each set_phase advances ~35s of sim time.
        self._step += 35

    def next_step(self) -> None:
        self._step += 1

    def get_average_travel_time(self) -> float:
        # Returning >0 lets MetricsEvaluator not warn about "no completions".
        return 42.0


class FakeAgent:
    """Fakes :class:`LightGPTInference` returning deterministic responses."""

    LOAD_CALLS: list["FakeAgent"] = []

    def __init__(self, *, variant: str, cache_dir: str, hf_token, device: str):
        self.variant = variant
        self.cache_dir = cache_dir
        self.hf_token = hf_token
        self.device = device
        self.loaded = False

    def load_model(self):
        self.loaded = True
        FakeAgent.LOAD_CALLS.append(self)

    def generate(self, prompt: str) -> str:
        # Always return ETWT so ResponseParser succeeds without warning.
        return "<signal>ETWT</signal>"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Isolate per-test env: clean PROJECT_DIR and disable auto-approve
    leakage so each test starts from a known state."""
    monkeypatch.delenv("LLMLIGHT_AUTO_APPROVE", raising=False)
    monkeypatch.delenv("DEFAULT_RUN_MODE", raising=False)
    monkeypatch.setenv("RANDOM_SEED", "42")
    monkeypatch.setenv("PROJECT_DIR", str(tmp_path))
    yield


@pytest.fixture
def fake_llmtscs_dir(tmp_path: Path) -> Path:
    """Build an LLMTSCS-shaped tree with stub roadnet + flow files for
    Jinan 1 (default dataset under test)."""
    root = tmp_path / "LLMTSCS"
    for subdir, roadnet, flow in DATASET_FILES.values():
        full = root / "data" / subdir
        full.mkdir(parents=True, exist_ok=True)
        (full / roadnet).write_text(json.dumps(_stub_roadnet()), encoding="utf-8")
        # Flow file content doesn't matter — runner only opens the config
        # JSON we generate, never reads flow.
        (full / flow).write_text("[]", encoding="utf-8")
    return root


@pytest.fixture
def sim_config_path(tmp_path: Path) -> Path:
    """Tiny config/simulation.json sufficient for SimulationConfig.from_json."""
    cfg = {
        "green_duration": 30,
        "yellow_duration": 3,
        "all_red_duration": 2,
        "total_timesteps_demo": 250,
        "total_timesteps_full": 3600,
        "phase_set": ["ETWT", "NTST", "ELWL", "NLSL"],
        "datasets": {
            "jinan_1": {"roadnet_file": "anon_3_4_jinan_real"},
            "hangzhou_1": {"roadnet_file": "anon_4_4_hangzhou_real"},
        },
        "phase3_datasets": {
            "newyork_1": {"roadnet_file": "anon_28_7_newyork_real_double"},
        },
    }
    p = tmp_path / "simulation.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    return p


@pytest.fixture
def cwd_in_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Change to ``tmp_path`` so results/replays + results/metrics +
    results/logs/llm_prompts are written under the temp dir, not the
    real workspace."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def patched_factories(monkeypatch: pytest.MonkeyPatch):
    """Patch run_method_runs's default factories to use the FakeEngine and
    FakeAgent. Returns a dict of (cityflow, inference) for further inspection.

    Also patches PreflightChecker.run_all to a no-op so the GPU check
    doesn't run on Windows dev box.
    """
    FakeAgent.LOAD_CALLS = []

    def fake_run_all(self, project_dir):
        return None

    monkeypatch.setattr(
        run_lightgpt.PreflightChecker, "run_all", fake_run_all
    )
    return {"engine": FakeEngine, "agent": FakeAgent}


# ---------------------------------------------------------------------------
# parse_args
# ---------------------------------------------------------------------------


class TestParseArgs:
    def test_dataset_required(self):
        with pytest.raises(SystemExit):
            run_lightgpt.parse_args([])

    def test_minimal(self):
        ns = run_lightgpt.parse_args(["--dataset", "jinan_1"])
        assert ns.dataset == "jinan_1"
        assert ns.num_runs == 3
        assert ns.phase == 1
        assert ns.save_replay == "auto"
        assert ns.method is None  # phase-dependent default in select_methods

    def test_invalid_dataset(self):
        with pytest.raises(SystemExit):
            run_lightgpt.parse_args(["--dataset", "boston_1"])

    def test_invalid_phase(self):
        with pytest.raises(SystemExit):
            run_lightgpt.parse_args(["--dataset", "jinan_1", "--phase", "4"])

    def test_invalid_save_replay_value(self):
        with pytest.raises(SystemExit):
            run_lightgpt.parse_args(
                ["--dataset", "jinan_1", "--save-replay", "maybe"]
            )

    def test_num_runs_must_be_positive(self):
        with pytest.raises(SystemExit):
            run_lightgpt.parse_args(
                ["--dataset", "jinan_1", "--num-runs", "0"]
            )

    def test_method_choices(self):
        for m in ("lightgpt_hf", "lightgpt_mine", "qwen2_0_5b_base", "both"):
            ns = run_lightgpt.parse_args(
                ["--dataset", "jinan_1", "--method", m]
            )
            assert ns.method == m


# ---------------------------------------------------------------------------
# select_methods
# ---------------------------------------------------------------------------


class TestSelectMethods:
    def test_phase1_default_is_lightgpt_hf(self):
        assert run_lightgpt.select_methods(
            None, phase=1, finetuned_exists=True
        ) == ["lightgpt_hf"]

    def test_phase2_default_is_both(self):
        assert run_lightgpt.select_methods(
            None, phase=2, finetuned_exists=True
        ) == ["lightgpt_hf", "lightgpt_mine"]

    def test_phase3_default_is_both(self):
        assert run_lightgpt.select_methods(
            None, phase=3, finetuned_exists=True
        ) == ["lightgpt_hf", "lightgpt_mine"]

    def test_phase1_mine_missing_skips_with_warning(self, caplog):
        with caplog.at_level("WARNING"):
            res = run_lightgpt.select_methods(
                "both", phase=1, finetuned_exists=False
            )
        assert res == ["lightgpt_hf"]
        assert any(
            "lightgpt_mine chưa được train" in rec.message
            for rec in caplog.records
        )

    def test_phase2_mine_missing_raises(self):
        with pytest.raises(FileNotFoundError, match="qwen2_finetuned"):
            run_lightgpt.select_methods(
                "both", phase=2, finetuned_exists=False
            )

    def test_phase3_mine_missing_raises(self):
        with pytest.raises(FileNotFoundError, match="qwen2_finetuned"):
            run_lightgpt.select_methods(
                "lightgpt_mine", phase=3, finetuned_exists=False
            )

    def test_phase1_hf_only_unaffected_by_missing_mine(self):
        assert run_lightgpt.select_methods(
            "lightgpt_hf", phase=1, finetuned_exists=False
        ) == ["lightgpt_hf"]

    def test_invalid_phase(self):
        with pytest.raises(ValueError, match="phase must be one of"):
            run_lightgpt.select_methods("both", phase=4, finetuned_exists=True)


# ---------------------------------------------------------------------------
# main() — happy path
# ---------------------------------------------------------------------------


class TestMainPhase1Happy:
    def test_runs_phase1_demo_default_method(
        self,
        cwd_in_tmp: Path,
        fake_llmtscs_dir: Path,
        sim_config_path: Path,
        patched_factories,
        monkeypatch: pytest.MonkeyPatch,
    ):
        # Patch run_method_runs to use FakeEngine + FakeAgent.
        original = run_lightgpt.run_method_runs

        def patched_runner(**kwargs):
            return original(
                **kwargs,
                cityflow_engine_factory=patched_factories["engine"],
                inference_factory=patched_factories["agent"],
            )

        monkeypatch.setattr(run_lightgpt, "run_method_runs", patched_runner)

        rc = run_lightgpt.main(
            [
                "--dataset", "jinan_1",
                "--phase", "1",
                "--num-runs", "1",
                "--mode", "demo",
                "--simulation-config", str(sim_config_path),
                "--llmtscs-dir", str(fake_llmtscs_dir),
                "--skip-preflight",
            ]
        )
        assert rc == 0

        # ExperimentResult JSON should be written.
        metrics_dir = cwd_in_tmp / "results" / "metrics"
        assert metrics_dir.exists()
        files = list(metrics_dir.glob("lightgpt_hf_jinan_1_phase1_run0.json"))
        assert len(files) == 1, f"Expected 1 result file, got {files}"
        data = json.loads(files[0].read_text(encoding="utf-8"))
        assert data["method"] == "lightgpt_hf"
        assert data["dataset"] == "jinan_1"
        assert data["phase_label"] == "Phase1"
        assert data["seed"] == 42  # base 42 + run_id 0
        assert data["replay_file"] is not None  # Phase 1 auto → on

        # FakeAgent.load_model called once.
        assert len(FakeAgent.LOAD_CALLS) == 1
        assert FakeAgent.LOAD_CALLS[0].variant == "lightgpt_hf"

        # LLM prompt log file written.
        log_dir = cwd_in_tmp / "results" / "logs" / "llm_prompts"
        assert log_dir.exists()
        log_files = list(log_dir.glob("jinan_1_lightgpt_hf_0_t*.txt"))
        assert len(log_files) >= 1
        log_content = log_files[0].read_text(encoding="utf-8")
        assert "<signal>ETWT</signal>" in log_content
        assert "PROMPT" in log_content
        assert "RESPONSE" in log_content


# ---------------------------------------------------------------------------
# main() — phase gating
# ---------------------------------------------------------------------------


class TestMainPhaseGating:
    def test_phase3_without_approval_aborts(
        self,
        cwd_in_tmp: Path,
        fake_llmtscs_dir: Path,
        sim_config_path: Path,
        patched_factories,
        monkeypatch: pytest.MonkeyPatch,
    ):
        # Create Phase 2 prerequisite files so check_prerequisite passes.
        metrics_dir = cwd_in_tmp / "results" / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        (metrics_dir / "comparison_jinan_1_phase2.md").write_text("ok")
        (metrics_dir / "comparison_hangzhou_1_phase2.md").write_text("ok")

        # User declines approval.
        monkeypatch.delenv("LLMLIGHT_AUTO_APPROVE", raising=False)
        monkeypatch.setattr("builtins.input", lambda _prompt="": "no")

        rc = run_lightgpt.main(
            [
                "--dataset", "newyork_1",
                "--phase", "3",
                "--num-runs", "1",
                "--mode", "full",
                "--simulation-config", str(sim_config_path),
                "--llmtscs-dir", str(fake_llmtscs_dir),
                "--skip-preflight",
            ]
        )
        assert rc == 3

    def test_phase3_with_auto_approve(
        self,
        cwd_in_tmp: Path,
        fake_llmtscs_dir: Path,
        sim_config_path: Path,
        patched_factories,
        monkeypatch: pytest.MonkeyPatch,
    ):
        # Create Phase 2 prerequisite files.
        metrics_dir = cwd_in_tmp / "results" / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        (metrics_dir / "comparison_jinan_1_phase2.md").write_text("ok")
        (metrics_dir / "comparison_hangzhou_1_phase2.md").write_text("ok")

        # Create models/qwen2_finetuned so lightgpt_mine doesn't fail.
        (cwd_in_tmp / "models" / "qwen2_finetuned").mkdir(parents=True)

        monkeypatch.setenv("LLMLIGHT_AUTO_APPROVE", "yes")

        original = run_lightgpt.run_method_runs

        def patched_runner(**kwargs):
            return original(
                **kwargs,
                cityflow_engine_factory=patched_factories["engine"],
                inference_factory=patched_factories["agent"],
            )

        monkeypatch.setattr(run_lightgpt, "run_method_runs", patched_runner)

        rc = run_lightgpt.main(
            [
                "--dataset", "newyork_1",
                "--phase", "3",
                "--method", "lightgpt_hf",  # avoid lightgpt_mine here
                "--num-runs", "1",
                "--mode", "full",
                "--simulation-config", str(sim_config_path),
                "--llmtscs-dir", str(fake_llmtscs_dir),
                "--skip-preflight",
            ]
        )
        assert rc == 0
        files = list(
            (cwd_in_tmp / "results" / "metrics").glob(
                "lightgpt_hf_newyork_1_phase3_run0.json"
            )
        )
        assert len(files) == 1

    def test_phase3_rejects_jinan(
        self,
        cwd_in_tmp: Path,
        fake_llmtscs_dir: Path,
        sim_config_path: Path,
        patched_factories,
        monkeypatch: pytest.MonkeyPatch,
    ):
        rc = run_lightgpt.main(
            [
                "--dataset", "jinan_1",
                "--phase", "3",
                "--num-runs", "1",
                "--simulation-config", str(sim_config_path),
                "--llmtscs-dir", str(fake_llmtscs_dir),
                "--skip-preflight",
            ]
        )
        assert rc == 2  # PhaseApprovalGate rejects

    def test_phase1_rejects_newyork(
        self,
        cwd_in_tmp: Path,
        fake_llmtscs_dir: Path,
        sim_config_path: Path,
        patched_factories,
        monkeypatch: pytest.MonkeyPatch,
    ):
        rc = run_lightgpt.main(
            [
                "--dataset", "newyork_1",
                "--phase", "1",
                "--num-runs", "1",
                "--simulation-config", str(sim_config_path),
                "--llmtscs-dir", str(fake_llmtscs_dir),
                "--skip-preflight",
            ]
        )
        assert rc == 2


# ---------------------------------------------------------------------------
# main() — lightgpt_mine handling
# ---------------------------------------------------------------------------


class TestMainLightgptMine:
    def test_phase2_mine_missing_fails(
        self,
        cwd_in_tmp: Path,
        fake_llmtscs_dir: Path,
        sim_config_path: Path,
        patched_factories,
        monkeypatch: pytest.MonkeyPatch,
    ):
        # Provide Phase 1 prerequisites so Phase 2 check_prerequisite passes.
        metrics_dir = cwd_in_tmp / "results" / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        (metrics_dir / "comparison_jinan_1_phase1.md").write_text("ok")
        (metrics_dir / "comparison_hangzhou_1_phase1.md").write_text("ok")

        rc = run_lightgpt.main(
            [
                "--dataset", "jinan_1",
                "--phase", "2",
                "--method", "lightgpt_mine",
                "--num-runs", "1",
                "--mode", "full",
                "--simulation-config", str(sim_config_path),
                "--llmtscs-dir", str(fake_llmtscs_dir),
                "--skip-preflight",
            ]
        )
        assert rc == 7  # FileNotFoundError exit code

    def test_phase1_mine_missing_skips(
        self,
        cwd_in_tmp: Path,
        fake_llmtscs_dir: Path,
        sim_config_path: Path,
        patched_factories,
        monkeypatch: pytest.MonkeyPatch,
        caplog,
    ):
        original = run_lightgpt.run_method_runs

        def patched_runner(**kwargs):
            return original(
                **kwargs,
                cityflow_engine_factory=patched_factories["engine"],
                inference_factory=patched_factories["agent"],
            )

        monkeypatch.setattr(run_lightgpt, "run_method_runs", patched_runner)

        with caplog.at_level("WARNING"):
            rc = run_lightgpt.main(
                [
                    "--dataset", "jinan_1",
                    "--phase", "1",
                    "--method", "both",
                    "--num-runs", "1",
                    "--simulation-config", str(sim_config_path),
                    "--llmtscs-dir", str(fake_llmtscs_dir),
                    "--skip-preflight",
                ]
            )
        assert rc == 0

        # Only lightgpt_hf result file should exist.
        metrics_dir = cwd_in_tmp / "results" / "metrics"
        files = sorted(p.name for p in metrics_dir.glob("*.json"))
        assert files == ["lightgpt_hf_jinan_1_phase1_run0.json"]
        assert any(
            "lightgpt_mine chưa được train" in rec.message
            for rec in caplog.records
        )


# ---------------------------------------------------------------------------
# main() — save-replay handling
# ---------------------------------------------------------------------------


class TestMainSaveReplay:
    def test_off_explicit(
        self,
        cwd_in_tmp: Path,
        fake_llmtscs_dir: Path,
        sim_config_path: Path,
        patched_factories,
        monkeypatch: pytest.MonkeyPatch,
    ):
        original = run_lightgpt.run_method_runs

        def patched_runner(**kwargs):
            return original(
                **kwargs,
                cityflow_engine_factory=patched_factories["engine"],
                inference_factory=patched_factories["agent"],
            )

        monkeypatch.setattr(run_lightgpt, "run_method_runs", patched_runner)

        rc = run_lightgpt.main(
            [
                "--dataset", "jinan_1",
                "--phase", "1",
                "--num-runs", "1",
                "--save-replay", "off",
                "--simulation-config", str(sim_config_path),
                "--llmtscs-dir", str(fake_llmtscs_dir),
                "--skip-preflight",
            ]
        )
        assert rc == 0
        files = list(
            (cwd_in_tmp / "results" / "metrics").glob("*.json")
        )
        assert len(files) == 1
        data = json.loads(files[0].read_text(encoding="utf-8"))
        assert data["replay_file"] is None
