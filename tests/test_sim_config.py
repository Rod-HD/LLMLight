"""Unit tests for ``src.sim_config``.

Validates:
    - Requirement 5.9, 5.10 (lightgpt_hf / lightgpt_mine method enum)
    - Requirement 10.1-10.8 (simulation timing + phase set + total_timesteps
      validation; from_json + parameter mismatch detection)
    - Requirement 11.5 (Unicode path support — exercised via from_json)
    - Requirement 12.6 (MetricsResult bound shape, >= 0)
    - Requirement 13.6 (phase_label Literal["Phase1","Phase2","Phase3"])
    - Requirement 14.5, 14.11 (replay_file field, str | None)

Tests use plain pytest. No third-party deps; sim_config.py keeps stdlib-only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Make ``src`` importable when running ``pytest`` from project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.sim_config import (  # noqa: E402
    VALID_METHODS,
    VALID_PHASES,
    VALID_TOTAL_TIMESTEPS,
    ExperimentResult,
    IntersectionState,
    MetricsResult,
    SimulationConfig,
    TokenUsageLog,
)


# =========================================================================
# IntersectionState
# =========================================================================


class TestIntersectionState:
    def test_valid_state(self):
        state = IntersectionState(
            intersection_id="i_1",
            lane_vehicle_count={"lane_a": 3, "lane_b": 0},
            current_phase="ETWT",
            current_phase_time=12,
        )
        assert state.intersection_id == "i_1"
        assert state.current_phase == "ETWT"

    def test_high_lane_count_allowed_no_upper_bound(self):
        # Per spec: "KHÔNG có giới hạn trên cố định" — must accept large ints.
        state = IntersectionState(
            intersection_id="i_1",
            lane_vehicle_count={"lane_a": 5000},
            current_phase="ETWT",
            current_phase_time=0,
        )
        assert state.lane_vehicle_count["lane_a"] == 5000

    def test_negative_lane_count_rejected(self):
        with pytest.raises(ValueError, match=">= 0"):
            IntersectionState(
                intersection_id="i_1",
                lane_vehicle_count={"lane_a": -1},
                current_phase="ETWT",
                current_phase_time=0,
            )

    def test_invalid_phase_rejected(self):
        with pytest.raises(ValueError, match="current_phase"):
            IntersectionState(
                intersection_id="i_1",
                lane_vehicle_count={"lane_a": 0},
                current_phase="INVALID",
                current_phase_time=0,
            )

    def test_lowercase_phase_rejected(self):
        with pytest.raises(ValueError, match="current_phase"):
            IntersectionState(
                intersection_id="i_1",
                lane_vehicle_count={"lane_a": 0},
                current_phase="etwt",
                current_phase_time=0,
            )

    def test_negative_phase_time_rejected(self):
        with pytest.raises(ValueError, match="current_phase_time"):
            IntersectionState(
                intersection_id="i_1",
                lane_vehicle_count={"lane_a": 0},
                current_phase="ETWT",
                current_phase_time=-1,
            )

    def test_lane_count_must_be_int_not_float(self):
        with pytest.raises(ValueError, match="must be int"):
            IntersectionState(
                intersection_id="i_1",
                lane_vehicle_count={"lane_a": 1.5},  # type: ignore[dict-item]
                current_phase="ETWT",
                current_phase_time=0,
            )

    def test_lane_count_bool_rejected(self):
        # bool is subclass of int — must be rejected explicitly.
        with pytest.raises(ValueError, match="must be int"):
            IntersectionState(
                intersection_id="i_1",
                lane_vehicle_count={"lane_a": True},  # type: ignore[dict-item]
                current_phase="ETWT",
                current_phase_time=0,
            )

    def test_empty_intersection_id_rejected(self):
        with pytest.raises(ValueError, match="intersection_id"):
            IntersectionState(
                intersection_id="",
                lane_vehicle_count={},
                current_phase="ETWT",
                current_phase_time=0,
            )

    def test_lane_vehicle_count_not_dict_rejected(self):
        with pytest.raises(ValueError, match="must be a dict"):
            IntersectionState(
                intersection_id="i_1",
                lane_vehicle_count=[("lane_a", 1)],  # type: ignore[arg-type]
                current_phase="ETWT",
                current_phase_time=0,
            )


# =========================================================================
# SimulationConfig — direct construction
# =========================================================================


class TestSimulationConfigConstruction:
    def _valid_kwargs(self, **overrides):
        kwargs = dict(
            roadnet_file="anon_3_4_jinan_real",
            flow_file="anon_3_4_jinan_real",
            green_duration=30,
            yellow_duration=3,
            all_red_duration=2,
            total_timesteps=3600,
            phase_set=VALID_PHASES,
        )
        kwargs.update(overrides)
        return kwargs

    def test_valid_full_mode(self):
        config = SimulationConfig(**self._valid_kwargs())
        assert config.total_timesteps == 3600
        assert config.phase_set == VALID_PHASES

    def test_valid_demo_mode(self):
        config = SimulationConfig(**self._valid_kwargs(total_timesteps=250))
        assert config.total_timesteps == 250

    @pytest.mark.parametrize("bad_total", [0, 1, 100, 251, 3599, 3601, 7200])
    def test_invalid_total_timesteps_rejected(self, bad_total):
        with pytest.raises(ValueError, match="total_timesteps"):
            SimulationConfig(**self._valid_kwargs(total_timesteps=bad_total))

    def test_zero_green_duration_rejected(self):
        with pytest.raises(ValueError, match="green_duration"):
            SimulationConfig(**self._valid_kwargs(green_duration=0))

    def test_negative_green_duration_rejected(self):
        with pytest.raises(ValueError, match="green_duration"):
            SimulationConfig(**self._valid_kwargs(green_duration=-1))

    def test_zero_yellow_duration_rejected(self):
        with pytest.raises(ValueError, match="yellow_duration"):
            SimulationConfig(**self._valid_kwargs(yellow_duration=0))

    def test_zero_all_red_duration_rejected(self):
        with pytest.raises(ValueError, match="all_red_duration"):
            SimulationConfig(**self._valid_kwargs(all_red_duration=0))

    def test_phase_set_with_invalid_phase_rejected(self):
        with pytest.raises(ValueError, match="phase_set"):
            SimulationConfig(
                **self._valid_kwargs(phase_set=("ETWT", "INVALID"))
            )

    def test_phase_set_with_duplicates_rejected(self):
        with pytest.raises(ValueError, match="duplicates"):
            SimulationConfig(
                **self._valid_kwargs(phase_set=("ETWT", "ETWT"))
            )

    def test_empty_phase_set_rejected(self):
        with pytest.raises(ValueError, match="phase_set"):
            SimulationConfig(**self._valid_kwargs(phase_set=()))

    def test_phase_set_list_normalized_to_tuple(self):
        config = SimulationConfig(
            **self._valid_kwargs(phase_set=["ETWT", "NTST"])
        )
        assert isinstance(config.phase_set, tuple)
        assert config.phase_set == ("ETWT", "NTST")

    def test_empty_roadnet_file_rejected(self):
        with pytest.raises(ValueError, match="roadnet_file"):
            SimulationConfig(**self._valid_kwargs(roadnet_file=""))

    def test_bool_total_timesteps_rejected(self):
        # bool is subclass of int — guard against accidental bool.
        with pytest.raises(ValueError, match="total_timesteps"):
            SimulationConfig(**self._valid_kwargs(total_timesteps=True))


# =========================================================================
# SimulationConfig.from_json
# =========================================================================


def _write_synthetic_config(tmp_path: Path, **overrides) -> Path:
    """Write a small synthetic ``simulation.json``."""
    payload = {
        "green_duration": 30,
        "yellow_duration": 3,
        "all_red_duration": 2,
        "total_timesteps_demo": 250,
        "total_timesteps_full": 3600,
        "phase_set": ["ETWT", "NTST", "ELWL", "NLSL"],
        "datasets": {
            "_comment": "Phase 1/2 datasets",
            "jinan_1": {"roadnet_file": "anon_3_4_jinan_real"},
            "hangzhou_1": {"roadnet_file": "anon_4_4_hangzhou_real"},
        },
        "phase3_datasets": {
            "newyork_1": {"roadnet_file": "anon_28_7_newyork_real"},
        },
    }
    payload.update(overrides)
    cfg = tmp_path / "simulation.json"
    cfg.write_text(json.dumps(payload), encoding="utf-8")
    return cfg


class TestSimulationConfigFromJson:
    def test_loads_jinan_demo(self, tmp_path: Path):
        cfg = _write_synthetic_config(tmp_path)
        config = SimulationConfig.from_json(cfg, dataset="jinan_1", mode="demo")
        assert config.roadnet_file == "anon_3_4_jinan_real"
        assert config.flow_file == "anon_3_4_jinan_real"
        assert config.total_timesteps == 250
        assert config.green_duration == 30

    def test_loads_jinan_full(self, tmp_path: Path):
        cfg = _write_synthetic_config(tmp_path)
        config = SimulationConfig.from_json(cfg, dataset="jinan_1", mode="full")
        assert config.total_timesteps == 3600

    def test_loads_phase3_dataset(self, tmp_path: Path):
        cfg = _write_synthetic_config(tmp_path)
        config = SimulationConfig.from_json(
            cfg, dataset="newyork_1", mode="full"
        )
        assert config.roadnet_file == "anon_28_7_newyork_real"
        assert config.total_timesteps == 3600

    def test_unknown_dataset_lists_available(self, tmp_path: Path):
        cfg = _write_synthetic_config(tmp_path)
        with pytest.raises(ValueError) as excinfo:
            SimulationConfig.from_json(cfg, dataset="seoul_1", mode="full")
        msg = str(excinfo.value)
        assert "seoul_1" in msg
        assert "jinan_1" in msg  # available dataset listed
        assert "newyork_1" in msg

    def test_invalid_mode_rejected(self, tmp_path: Path):
        cfg = _write_synthetic_config(tmp_path)
        with pytest.raises(ValueError, match="mode"):
            SimulationConfig.from_json(
                cfg, dataset="jinan_1", mode="turbo"  # type: ignore[arg-type]
            )

    def test_missing_file_raises_filenotfound(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            SimulationConfig.from_json(
                tmp_path / "missing.json", dataset="jinan_1", mode="full"
            )

    def test_malformed_json_raises_valueerror(self, tmp_path: Path):
        cfg = tmp_path / "broken.json"
        cfg.write_text("{ not valid json", encoding="utf-8")
        with pytest.raises(ValueError, match="parse JSON"):
            SimulationConfig.from_json(cfg, dataset="jinan_1", mode="full")

    def test_dataset_missing_roadnet_file(self, tmp_path: Path):
        cfg = _write_synthetic_config(
            tmp_path,
            datasets={"bad_1": {"_comment": "no roadnet_file"}},
        )
        with pytest.raises(ValueError, match="roadnet_file"):
            SimulationConfig.from_json(cfg, dataset="bad_1", mode="full")

    def test_unicode_path_supported(self, tmp_path: Path):
        # Requirement 11.5: paths with Vietnamese + spaces must work.
        unicode_dir = tmp_path / "Trí tuệ nhân tạo" / "Đồ án"
        unicode_dir.mkdir(parents=True)
        cfg = _write_synthetic_config(unicode_dir)
        config = SimulationConfig.from_json(
            cfg, dataset="jinan_1", mode="full"
        )
        assert config.roadnet_file == "anon_3_4_jinan_real"

    def test_real_config_file_loads(self):
        """Smoke-load the real ``config/simulation.json`` if it exists."""
        real_cfg = PROJECT_ROOT / "config" / "simulation.json"
        if not real_cfg.exists():
            pytest.skip("config/simulation.json not present")
        # Phase 1/2
        c1 = SimulationConfig.from_json(real_cfg, dataset="jinan_1", mode="demo")
        assert c1.total_timesteps == 250
        c2 = SimulationConfig.from_json(real_cfg, dataset="hangzhou_1", mode="full")
        assert c2.total_timesteps == 3600
        # Phase 3
        c3 = SimulationConfig.from_json(real_cfg, dataset="newyork_1", mode="full")
        assert c3.total_timesteps == 3600


# =========================================================================
# SimulationConfig.compare — Requirement 10 AC 7-8
# =========================================================================


def _make_config(**overrides) -> SimulationConfig:
    base = dict(
        roadnet_file="anon_3_4_jinan_real",
        flow_file="anon_3_4_jinan_real",
        green_duration=30,
        yellow_duration=3,
        all_red_duration=2,
        total_timesteps=3600,
        phase_set=VALID_PHASES,
    )
    base.update(overrides)
    return SimulationConfig(**base)


class TestSimulationConfigCompare:
    def test_compare_same_passes(self):
        a = _make_config()
        b = _make_config()
        a.compare(b)  # no raise

    def test_compare_different_total_timesteps_raises(self):
        a = _make_config(total_timesteps=3600)
        b = _make_config(total_timesteps=250)
        with pytest.raises(ValueError) as excinfo:
            a.compare(b)
        msg = str(excinfo.value)
        assert "total_timesteps" in msg
        assert "3600" in msg and "250" in msg

    def test_compare_different_green_duration_raises(self):
        a = _make_config()
        b = _make_config(green_duration=40)
        with pytest.raises(ValueError, match="green_duration"):
            a.compare(b)

    def test_compare_lists_all_mismatches(self):
        a = _make_config()
        b = _make_config(green_duration=40, yellow_duration=5)
        with pytest.raises(ValueError) as excinfo:
            a.compare(b)
        msg = str(excinfo.value)
        assert "green_duration" in msg
        assert "yellow_duration" in msg

    def test_compare_default_does_not_check_roadnet_file(self):
        # Two runners on different datasets but same timing should pass.
        a = _make_config(roadnet_file="anon_3_4_jinan_real")
        b = _make_config(roadnet_file="anon_4_4_hangzhou_real")
        a.compare(b)  # no raise

    def test_compare_explicit_fields_includes_roadnet(self):
        a = _make_config(roadnet_file="anon_3_4_jinan_real")
        b = _make_config(roadnet_file="anon_4_4_hangzhou_real")
        with pytest.raises(ValueError, match="roadnet_file"):
            a.compare(b, fields=["roadnet_file"])

    def test_compare_unknown_field_raises(self):
        a = _make_config()
        b = _make_config()
        with pytest.raises(ValueError, match="unknown field"):
            a.compare(b, fields=["nonexistent_field"])

    def test_compare_other_must_be_simulationconfig(self):
        a = _make_config()
        with pytest.raises(ValueError, match="SimulationConfig"):
            a.compare("not a config")  # type: ignore[arg-type]


# =========================================================================
# MetricsResult
# =========================================================================


class TestMetricsResult:
    def test_valid_metrics(self):
        m = MetricsResult(att=120.5, aql=2.3, awt=80.1)
        assert m.att == 120.5

    def test_int_input_normalized_to_float(self):
        m = MetricsResult(att=100, aql=2, awt=50)
        assert isinstance(m.att, float)
        assert m.att == 100.0

    @pytest.mark.parametrize("field_name", ["att", "aql", "awt"])
    def test_negative_metrics_rejected(self, field_name):
        kwargs = {"att": 1.0, "aql": 1.0, "awt": 1.0}
        kwargs[field_name] = -0.1
        with pytest.raises(ValueError, match=field_name):
            MetricsResult(**kwargs)

    def test_zero_allowed(self):
        m = MetricsResult(att=0, aql=0, awt=0)
        assert m.att == 0.0

    def test_non_numeric_rejected(self):
        with pytest.raises(ValueError, match="att"):
            MetricsResult(att="100", aql=1.0, awt=1.0)  # type: ignore[arg-type]

    def test_bool_rejected(self):
        with pytest.raises(ValueError, match="att"):
            MetricsResult(att=True, aql=1.0, awt=1.0)  # type: ignore[arg-type]


# =========================================================================
# TokenUsageLog
# =========================================================================


class TestTokenUsageLog:
    def test_default_none_backend(self):
        log = TokenUsageLog()
        assert log.backend == "none"
        assert log.total_input_tokens == 0

    def test_groq_backend(self):
        log = TokenUsageLog(
            backend="groq",
            total_input_tokens=100,
            total_output_tokens=50,
            total_requests=2,
        )
        assert log.backend == "groq"

    def test_invalid_backend_rejected(self):
        with pytest.raises(ValueError, match="backend"):
            TokenUsageLog(backend="anthropic")  # type: ignore[arg-type]

    def test_negative_token_count_rejected(self):
        with pytest.raises(ValueError, match="total_input_tokens"):
            TokenUsageLog(total_input_tokens=-1)


# =========================================================================
# ExperimentResult
# =========================================================================


def _make_metrics() -> MetricsResult:
    return MetricsResult(att=120.0, aql=2.0, awt=80.0)


class TestExperimentResult:
    def test_lightgpt_hf_method(self):
        r = ExperimentResult(
            method="lightgpt_hf",
            dataset="jinan_1",
            run_id=0,
            seed=42,
            metrics=_make_metrics(),
            token_usage=None,
            duration_seconds=300.5,
            timestamp="2025-01-01T00:00:00Z",
            phase_label="Phase1",
            replay_file=None,
        )
        assert r.method == "lightgpt_hf"
        assert r.replay_file is None

    def test_lightgpt_mine_method(self):
        # Requirement 5 AC 9: lightgpt_mine is a valid method enum value.
        r = ExperimentResult(
            method="lightgpt_mine",
            dataset="hangzhou_1",
            run_id=1,
            seed=43,
            metrics=_make_metrics(),
            token_usage=None,
            duration_seconds=300.0,
            timestamp="2025-01-01T00:00:00Z",
            phase_label="Phase2",
            replay_file=None,
        )
        assert r.method == "lightgpt_mine"

    def test_old_lightgpt_llama3_value_rejected(self):
        # Requirement 5 AC 9: lightgpt_llama3 was REPLACED by lightgpt_hf /
        # lightgpt_mine. Must be rejected.
        with pytest.raises(ValueError, match="method"):
            ExperimentResult(
                method="lightgpt_llama3",
                dataset="jinan_1",
                run_id=0,
                seed=42,
                metrics=_make_metrics(),
                token_usage=None,
                duration_seconds=1.0,
                timestamp="2025-01-01T00:00:00Z",
                phase_label="Phase1",
            )

    def test_phase3_label_accepted(self):
        # Requirement 13 AC 6: Phase3 is now a valid label (was Phase1/Phase2).
        r = ExperimentResult(
            method="advanced_colight",
            dataset="newyork_1",
            run_id=0,
            seed=42,
            metrics=_make_metrics(),
            token_usage=None,
            duration_seconds=10.0,
            timestamp="2025-01-01T00:00:00Z",
            phase_label="Phase3",
        )
        assert r.phase_label == "Phase3"

    def test_invalid_phase_label_rejected(self):
        with pytest.raises(ValueError, match="phase_label"):
            ExperimentResult(
                method="advanced_colight",
                dataset="jinan_1",
                run_id=0,
                seed=42,
                metrics=_make_metrics(),
                token_usage=None,
                duration_seconds=10.0,
                timestamp="2025-01-01T00:00:00Z",
                phase_label="Phase4",  # type: ignore[arg-type]
            )

    def test_replay_file_str_accepted(self):
        # Requirement 14 AC 5/11: replay_file is the path CityFlow wrote
        # when save_replay=True. UI Demo (Task 18) reads this field.
        r = ExperimentResult(
            method="lightgpt_hf",
            dataset="jinan_1",
            run_id=0,
            seed=42,
            metrics=_make_metrics(),
            token_usage=None,
            duration_seconds=10.0,
            timestamp="2025-01-01T00:00:00Z",
            phase_label="Phase1",
            replay_file="results/replays/jinan_1_lightgpt_hf_0.txt",
        )
        assert r.replay_file == "results/replays/jinan_1_lightgpt_hf_0.txt"

    def test_replay_file_empty_str_rejected(self):
        with pytest.raises(ValueError, match="replay_file"):
            ExperimentResult(
                method="lightgpt_hf",
                dataset="jinan_1",
                run_id=0,
                seed=42,
                metrics=_make_metrics(),
                token_usage=None,
                duration_seconds=10.0,
                timestamp="2025-01-01T00:00:00Z",
                phase_label="Phase1",
                replay_file="",
            )

    def test_invalid_dataset_rejected(self):
        with pytest.raises(ValueError, match="dataset"):
            ExperimentResult(
                method="lightgpt_hf",
                dataset="seoul_1",
                run_id=0,
                seed=42,
                metrics=_make_metrics(),
                token_usage=None,
                duration_seconds=10.0,
                timestamp="2025-01-01T00:00:00Z",
                phase_label="Phase1",
            )

    def test_negative_run_id_rejected(self):
        with pytest.raises(ValueError, match="run_id"):
            ExperimentResult(
                method="lightgpt_hf",
                dataset="jinan_1",
                run_id=-1,
                seed=42,
                metrics=_make_metrics(),
                token_usage=None,
                duration_seconds=10.0,
                timestamp="2025-01-01T00:00:00Z",
                phase_label="Phase1",
            )

    def test_negative_duration_rejected(self):
        with pytest.raises(ValueError, match="duration_seconds"):
            ExperimentResult(
                method="lightgpt_hf",
                dataset="jinan_1",
                run_id=0,
                seed=42,
                metrics=_make_metrics(),
                token_usage=None,
                duration_seconds=-1.0,
                timestamp="2025-01-01T00:00:00Z",
                phase_label="Phase1",
            )

    def test_metrics_must_be_metricsresult(self):
        with pytest.raises(ValueError, match="metrics"):
            ExperimentResult(
                method="lightgpt_hf",
                dataset="jinan_1",
                run_id=0,
                seed=42,
                metrics={"att": 1.0, "aql": 1.0, "awt": 1.0},  # type: ignore[arg-type]
                token_usage=None,
                duration_seconds=10.0,
                timestamp="2025-01-01T00:00:00Z",
                phase_label="Phase1",
            )

    def test_token_usage_for_api_method(self):
        log = TokenUsageLog(backend="groq", total_input_tokens=1000)
        r = ExperimentResult(
            method="gpt4o_groq",
            dataset="jinan_1",
            run_id=0,
            seed=42,
            metrics=_make_metrics(),
            token_usage=log,
            duration_seconds=10.0,
            timestamp="2025-01-01T00:00:00Z",
            phase_label="Phase1",
        )
        assert r.token_usage is not None
        assert r.token_usage.backend == "groq"

    def test_all_methods_accepted(self):
        # Sanity check: every value in VALID_METHODS constructs successfully.
        for method in VALID_METHODS:
            ExperimentResult(
                method=method,
                dataset="jinan_1",
                run_id=0,
                seed=42,
                metrics=_make_metrics(),
                token_usage=None,
                duration_seconds=1.0,
                timestamp="2025-01-01T00:00:00Z",
                phase_label="Phase1",
            )


# =========================================================================
# Module-level constants
# =========================================================================


class TestModuleConstants:
    def test_valid_phases_set(self):
        assert set(VALID_PHASES) == {"ETWT", "NTST", "ELWL", "NLSL"}

    def test_valid_total_timesteps_set(self):
        assert set(VALID_TOTAL_TIMESTEPS) == {250, 3600}

    def test_valid_methods_includes_lightgpt_hf_and_mine(self):
        assert "lightgpt_hf" in VALID_METHODS
        assert "lightgpt_mine" in VALID_METHODS
        assert "qwen2_0_5b_base" in VALID_METHODS
        # Old value removed.
        assert "lightgpt_llama3" not in VALID_METHODS
