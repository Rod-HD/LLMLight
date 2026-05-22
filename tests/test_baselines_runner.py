"""Unit tests for ``src.baselines_runner``.

Validates:
    - Requirement 8.1 (Maxpressure runner wraps run_maxpressure.py)
    - Requirement 8.2 (Advanced-MaxPressure separate baseline)
    - Requirement 8.3 (Advanced-CoLight train+evaluate, convergence detection)
    - Requirement 8.4-8.5 (same simulation config delegated to LLMTSCS scripts)
    - Requirement 8.6 (results in ATT/AQL/AWT)
    - Requirement 8.7 (ConvergenceError on stagnant loss with episode info)
    - Requirement 8.8 (3 runs default per method)
    - Requirement 8.9 (seed = base_seed + run_id, SeedManager applied)
    - Requirement 8.10 (3 runs across all phases)

Tests use a stubbed ``subprocess_runner`` so we never spawn real LLMTSCS
process. Real subprocess invocation is exercised in Task 12 integration
checkpoint.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines_runner import (  # noqa: E402
    BaselineResult,
    ConvergenceError,
    METHOD_ADV_COLIGHT,
    METHOD_ADV_MAXPRESSURE,
    METHOD_MAXPRESSURE,
    RLBaselinesRunner,
    VALID_DATASETS,
    VALID_METHODS,
)
from src.seed_manager import SeedManager  # noqa: E402


# =========================================================================
# Helpers
# =========================================================================


def _completed(
    *,
    stdout: str = "",
    stderr: str = "",
    returncode: int = 0,
    args: list[str] | None = None,
) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(
        args=args or ["python", "stub.py"],
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


def _success_stdout(
    att: float = 250.5, aql: float = 12.34, awt: float = 80.7
) -> str:
    """Build a stdout that mimics LLMTSCS oneline.py final results print."""
    return (
        "Some prologue ...\n"
        "round 0 stuff ...\n"
        "Training time:  42.0\n"
        f"{{'test_reward_over': -123.45, "
        f"'test_avg_queue_len_over': {aql}, "
        f"'test_queuing_vehicle_num_over': 999, "
        f"'test_avg_waiting_time_over': {awt}, "
        f"'test_avg_travel_time_over': {att}}}\n"
        "Training time:  43.0\n"
    )


def _colight_stdout_converging(
    att: float = 200.0, aql: float = 5.0, awt: float = 30.0
) -> str:
    """Build stdout with strictly-decreasing loss across 100 rounds."""
    rounds = "\n".join(
        f"round={i} loss={(100 - i * 0.5):.4f}" for i in range(100)
    )
    final = (
        f"{{'test_reward_over': -1.0, "
        f"'test_avg_queue_len_over': {aql}, "
        f"'test_queuing_vehicle_num_over': 1, "
        f"'test_avg_waiting_time_over': {awt}, "
        f"'test_avg_travel_time_over': {att}}}\n"
    )
    return rounds + "\n" + final


def _colight_stdout_stagnant(
    att: float = 200.0, aql: float = 5.0, awt: float = 30.0
) -> str:
    """Build stdout where loss stops decreasing at round 5 for 25 rounds."""
    lines = []
    for i in range(5):
        lines.append(f"round={i} loss={(50.0 - i):.4f}")
    # 25 rounds of equal loss → triggers ConvergenceError (patience=20).
    for i in range(5, 30):
        lines.append(f"round={i} loss=45.0000")
    final = (
        f"{{'test_reward_over': -1.0, "
        f"'test_avg_queue_len_over': {aql}, "
        f"'test_queuing_vehicle_num_over': 1, "
        f"'test_avg_waiting_time_over': {awt}, "
        f"'test_avg_travel_time_over': {att}}}"
    )
    return "\n".join(lines) + "\n" + final + "\n"


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def fake_llmtscs(tmp_path: Path) -> Path:
    """Minimal LLMTSCS dir layout (just the script files exist)."""
    d = tmp_path / "LLMTSCS"
    d.mkdir()
    (d / "run_maxpressure.py").write_text("# stub", encoding="utf-8")
    (d / "run_advanced_maxpressure.py").write_text("# stub", encoding="utf-8")
    (d / "run_advanced_colight.py").write_text("# stub", encoding="utf-8")
    return d


@pytest.fixture
def sim_config(tmp_path: Path) -> Path:
    p = tmp_path / "simulation.json"
    p.write_text("{}", encoding="utf-8")
    return p


@pytest.fixture
def make_runner(fake_llmtscs: Path, sim_config: Path):
    """Factory creating an RLBaselinesRunner with an injected stub runner."""

    def _factory(
        stub_runner=None,
        base_seed: int = 42,
        seed_manager: SeedManager | None = None,
    ) -> tuple[RLBaselinesRunner, list[dict[str, Any]]]:
        calls: list[dict[str, Any]] = []

        def default_runner(cmd, **kwargs):
            calls.append({"cmd": list(cmd), **kwargs})
            return _completed(stdout=_success_stdout(), args=list(cmd))

        if stub_runner is not None:
            real_stub = stub_runner

            def recording_runner(cmd, **kwargs):
                calls.append({"cmd": list(cmd), **kwargs})
                return real_stub(cmd, **kwargs)

            wrapped = recording_runner
        else:
            wrapped = default_runner

        runner = RLBaselinesRunner(
            sim_config_path=str(sim_config),
            base_seed=base_seed,
            llmtscs_dir=str(fake_llmtscs),
            subprocess_runner=wrapped,
            seed_manager=seed_manager,
        )
        return runner, calls

    return _factory


# =========================================================================
# BaselineResult dataclass
# =========================================================================


class TestBaselineResult:
    def test_valid(self):
        r = BaselineResult(
            method="maxpressure",
            dataset="jinan_1",
            att=120.5,
            aql=4.2,
            awt=33.7,
            run_id=0,
            seed=42,
        )
        assert r.method == "maxpressure"
        assert r.dataset == "jinan_1"
        assert r.att == 120.5
        assert r.aql == 4.2
        assert r.awt == 33.7
        assert r.run_id == 0
        assert r.seed == 42

    def test_method_must_be_in_valid_set(self):
        with pytest.raises(ValueError, match="method must be one of"):
            BaselineResult(
                method="not_a_method",
                dataset="jinan_1",
                att=1.0,
                aql=1.0,
                awt=1.0,
                run_id=0,
                seed=1,
            )

    def test_dataset_must_be_in_valid_set(self):
        with pytest.raises(ValueError, match="dataset must be one of"):
            BaselineResult(
                method="maxpressure",
                dataset="invalid",
                att=1.0,
                aql=1.0,
                awt=1.0,
                run_id=0,
                seed=1,
            )

    @pytest.mark.parametrize("metric", ["att", "aql", "awt"])
    def test_metrics_must_be_non_negative(self, metric: str):
        kwargs = dict(
            method="maxpressure",
            dataset="jinan_1",
            att=1.0,
            aql=1.0,
            awt=1.0,
            run_id=0,
            seed=1,
        )
        kwargs[metric] = -0.1
        with pytest.raises(ValueError, match="must be >= 0"):
            BaselineResult(**kwargs)

    def test_run_id_must_be_non_negative_int(self):
        with pytest.raises(ValueError, match="run_id must be >= 0"):
            BaselineResult(
                method="maxpressure",
                dataset="jinan_1",
                att=1.0,
                aql=1.0,
                awt=1.0,
                run_id=-1,
                seed=0,
            )

    def test_metrics_normalized_to_float(self):
        r = BaselineResult(
            method="maxpressure",
            dataset="jinan_1",
            att=10,  # int input
            aql=2,
            awt=5,
            run_id=0,
            seed=0,
        )
        assert isinstance(r.att, float)
        assert isinstance(r.aql, float)
        assert isinstance(r.awt, float)


# =========================================================================
# RLBaselinesRunner.__init__
# =========================================================================


class TestInit:
    def test_validates_sim_config_exists(
        self, fake_llmtscs: Path, tmp_path: Path
    ):
        missing = tmp_path / "does_not_exist.json"
        with pytest.raises(FileNotFoundError, match="sim_config_path not found"):
            RLBaselinesRunner(
                sim_config_path=str(missing),
                llmtscs_dir=str(fake_llmtscs),
            )

    def test_validates_llmtscs_dir_exists(
        self, sim_config: Path, tmp_path: Path
    ):
        missing = tmp_path / "no_such_dir"
        with pytest.raises(FileNotFoundError, match="llmtscs_dir not found"):
            RLBaselinesRunner(
                sim_config_path=str(sim_config),
                llmtscs_dir=str(missing),
            )

    def test_validates_base_seed_is_int(
        self, sim_config: Path, fake_llmtscs: Path
    ):
        with pytest.raises(ValueError, match="base_seed must be int"):
            RLBaselinesRunner(
                sim_config_path=str(sim_config),
                base_seed="42",  # type: ignore[arg-type]
                llmtscs_dir=str(fake_llmtscs),
            )

    def test_uses_env_var_when_llmtscs_dir_omitted(
        self,
        sim_config: Path,
        fake_llmtscs: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.setenv("LLMTSCS_DIR", str(fake_llmtscs))
        runner = RLBaselinesRunner(sim_config_path=str(sim_config))
        assert Path(runner.llmtscs_dir) == fake_llmtscs.resolve()

    def test_raises_when_no_llmtscs_dir_and_no_env(
        self, sim_config: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.delenv("LLMTSCS_DIR", raising=False)
        with pytest.raises(ValueError, match="llmtscs_dir not provided"):
            RLBaselinesRunner(sim_config_path=str(sim_config))


# =========================================================================
# Subprocess command construction
# =========================================================================


class TestBuildCmd:
    @pytest.mark.parametrize(
        "method,dataset,expected_script,expected_dataset_arg,expected_traffic_file",
        [
            (
                METHOD_MAXPRESSURE,
                "jinan_1",
                "run_maxpressure.py",
                "jinan",
                "anon_3_4_jinan_real.json",
            ),
            (
                METHOD_MAXPRESSURE,
                "hangzhou_1",
                "run_maxpressure.py",
                "hangzhou",
                "anon_4_4_hangzhou_real.json",
            ),
            (
                METHOD_ADV_MAXPRESSURE,
                "jinan_1",
                "run_advanced_maxpressure.py",
                "jinan",
                "anon_3_4_jinan_real.json",
            ),
            (
                METHOD_ADV_COLIGHT,
                "newyork_1",
                "run_advanced_colight.py",
                "newyork_28x7",
                "anon_28_7_newyork_real_double.json",
            ),
        ],
    )
    def test_dataset_path_mapping_correct(
        self,
        make_runner,
        method,
        dataset,
        expected_script,
        expected_dataset_arg,
        expected_traffic_file,
    ):
        runner, _ = make_runner()
        cmd = runner._build_subprocess_cmd(method, dataset)
        assert expected_script in cmd
        assert "--dataset" in cmd
        idx = cmd.index("--dataset")
        assert cmd[idx + 1] == expected_dataset_arg
        assert "--traffic_file" in cmd
        tf_idx = cmd.index("--traffic_file")
        assert cmd[tf_idx + 1] == expected_traffic_file


# =========================================================================
# run_maxpressure
# =========================================================================


class TestRunMaxpressure:
    def test_returns_three_results_with_seeds(self, make_runner):
        runner, calls = make_runner(base_seed=42)
        results = runner.run_maxpressure(dataset="jinan_1", num_runs=3)

        assert len(results) == 3
        assert [r.run_id for r in results] == [0, 1, 2]
        assert [r.seed for r in results] == [42, 43, 44]
        for r in results:
            assert r.method == METHOD_MAXPRESSURE
            assert r.dataset == "jinan_1"
            assert r.att == pytest.approx(250.5)
            assert r.aql == pytest.approx(12.34)
            assert r.awt == pytest.approx(80.7)

    def test_invokes_correct_script(self, make_runner):
        runner, calls = make_runner()
        runner.run_maxpressure(dataset="jinan_1", num_runs=2)

        assert len(calls) == 2
        for call in calls:
            cmd = call["cmd"]
            assert "run_maxpressure.py" in cmd
            assert "run_advanced_maxpressure.py" not in cmd
            assert "run_advanced_colight.py" not in cmd

    def test_passes_random_seed_via_env_per_run(self, make_runner):
        runner, calls = make_runner(base_seed=100)
        runner.run_maxpressure(dataset="hangzhou_1", num_runs=3)

        seeds_seen = [int(call["env"]["RANDOM_SEED"]) for call in calls]
        assert seeds_seen == [100, 101, 102]

    def test_seed_manager_apply_called_before_each_run(self, make_runner):
        """SeedManager.apply must be invoked per run with seed_for_run(run_id)."""
        sm = SeedManager(base_seed=10)
        applied: list[int] = []

        original_apply = sm.apply

        def spy_apply(seed: int) -> None:
            applied.append(seed)
            return original_apply(seed)

        sm.apply = spy_apply  # type: ignore[method-assign]

        runner, _ = make_runner(seed_manager=sm)
        runner.run_maxpressure(dataset="jinan_1", num_runs=3)

        assert applied == [10, 11, 12]

    def test_default_num_runs_is_three(self, make_runner):
        runner, _ = make_runner()
        results = runner.run_maxpressure(dataset="jinan_1")
        assert len(results) == 3

    def test_raises_on_invalid_dataset(self, make_runner):
        runner, _ = make_runner()
        with pytest.raises(ValueError, match="dataset must be one of"):
            runner.run_maxpressure(dataset="bogus", num_runs=1)

    def test_raises_on_subprocess_failure(self, make_runner):
        def failing_runner(cmd, **kwargs):
            return _completed(
                stdout="oops",
                stderr="boom",
                returncode=2,
                args=list(cmd),
            )

        runner, _ = make_runner(stub_runner=failing_runner)
        with pytest.raises(RuntimeError, match="returncode=2"):
            runner.run_maxpressure(dataset="jinan_1", num_runs=1)

    def test_raises_when_metrics_dict_missing(self, make_runner):
        def no_metrics_runner(cmd, **kwargs):
            return _completed(
                stdout="just some text without any metrics dict\n",
                args=list(cmd),
            )

        runner, _ = make_runner(stub_runner=no_metrics_runner)
        with pytest.raises(ValueError, match="no metrics dict"):
            runner.run_maxpressure(dataset="jinan_1", num_runs=1)


# =========================================================================
# run_advanced_maxpressure
# =========================================================================


class TestRunAdvancedMaxpressure:
    def test_invokes_advanced_maxpressure_script_distinct_from_maxpressure(
        self, make_runner
    ):
        runner, calls = make_runner()
        runner.run_advanced_maxpressure(dataset="jinan_1", num_runs=2)

        assert len(calls) == 2
        for call in calls:
            cmd = call["cmd"]
            assert "run_advanced_maxpressure.py" in cmd
            assert "run_maxpressure.py" not in cmd

    def test_results_have_distinct_method_label(self, make_runner):
        runner, _ = make_runner()
        results = runner.run_advanced_maxpressure(
            dataset="jinan_1", num_runs=2
        )
        for r in results:
            assert r.method == METHOD_ADV_MAXPRESSURE
            assert r.method != METHOD_MAXPRESSURE

    def test_seed_strategy_consistent(self, make_runner):
        runner, calls = make_runner(base_seed=7)
        results = runner.run_advanced_maxpressure(
            dataset="hangzhou_1", num_runs=3
        )

        assert [r.seed for r in results] == [7, 8, 9]
        seeds_via_env = [int(call["env"]["RANDOM_SEED"]) for call in calls]
        assert seeds_via_env == [7, 8, 9]


# =========================================================================
# run_advanced_colight
# =========================================================================


class TestRunAdvancedColight:
    def test_invokes_advanced_colight_script(self, make_runner):
        def runner_fn(cmd, **kwargs):
            return _completed(
                stdout=_colight_stdout_converging(), args=list(cmd)
            )

        runner, _ = make_runner(stub_runner=runner_fn)
        results = runner.run_advanced_colight(
            dataset="jinan_1", train_episodes=100, num_runs=1
        )
        assert len(results) == 1
        assert results[0].method == METHOD_ADV_COLIGHT

    def test_returns_results_when_loss_decreases(self, make_runner):
        def runner_fn(cmd, **kwargs):
            return _completed(
                stdout=_colight_stdout_converging(att=199.9, aql=4.4, awt=22.2),
                args=list(cmd),
            )

        runner, _ = make_runner(stub_runner=runner_fn)
        results = runner.run_advanced_colight(
            dataset="hangzhou_1", num_runs=2
        )

        assert [r.run_id for r in results] == [0, 1]
        for r in results:
            assert r.att == pytest.approx(199.9)
            assert r.aql == pytest.approx(4.4)
            assert r.awt == pytest.approx(22.2)

    def test_raises_convergence_error_when_loss_stagnates(self, make_runner):
        def stagnant_runner(cmd, **kwargs):
            return _completed(
                stdout=_colight_stdout_stagnant(), args=list(cmd)
            )

        runner, _ = make_runner(stub_runner=stagnant_runner)
        with pytest.raises(ConvergenceError) as excinfo:
            runner.run_advanced_colight(
                dataset="jinan_1", train_episodes=100, num_runs=1
            )

        msg = str(excinfo.value)
        # Requirement 8.7: message must include dataset + episode number.
        assert "jinan_1" in msg
        assert "episode" in msg.lower()
        assert "20" in msg  # patience size
        # Some episode index should appear (one of the stagnation rounds).
        assert any(f"round{r}=" in msg for r in range(5, 30))

    def test_raises_on_invalid_train_episodes(self, make_runner):
        runner, _ = make_runner()
        with pytest.raises(ValueError, match="train_episodes must be > 0"):
            runner.run_advanced_colight(dataset="jinan_1", train_episodes=0)

    def test_train_episodes_must_be_int(self, make_runner):
        runner, _ = make_runner()
        with pytest.raises(ValueError, match="train_episodes must be int"):
            runner.run_advanced_colight(
                dataset="jinan_1",
                train_episodes=100.5,  # type: ignore[arg-type]
            )

    def test_seed_strategy_consistent(self, make_runner):
        def runner_fn(cmd, **kwargs):
            return _completed(
                stdout=_colight_stdout_converging(), args=list(cmd)
            )

        runner, calls = make_runner(stub_runner=runner_fn, base_seed=200)
        results = runner.run_advanced_colight(
            dataset="newyork_1", num_runs=3
        )
        assert [r.seed for r in results] == [200, 201, 202]
        seeds_via_env = [int(call["env"]["RANDOM_SEED"]) for call in calls]
        assert seeds_via_env == [200, 201, 202]


# =========================================================================
# Cross-method invariants
# =========================================================================


class TestCrossMethodInvariants:
    def test_three_methods_invoke_distinct_scripts(self, make_runner):
        # Use different stub for colight to avoid convergence parsing issues.
        def runner_fn(cmd, **kwargs):
            if "run_advanced_colight.py" in cmd:
                return _completed(
                    stdout=_colight_stdout_converging(), args=list(cmd)
                )
            return _completed(stdout=_success_stdout(), args=list(cmd))

        runner, calls = make_runner(stub_runner=runner_fn)

        runner.run_maxpressure(dataset="jinan_1", num_runs=1)
        runner.run_advanced_maxpressure(dataset="jinan_1", num_runs=1)
        runner.run_advanced_colight(dataset="jinan_1", num_runs=1)

        scripts_invoked = []
        for call in calls:
            for arg in call["cmd"]:
                if arg.startswith("run_") and arg.endswith(".py"):
                    scripts_invoked.append(arg)

        assert scripts_invoked == [
            "run_maxpressure.py",
            "run_advanced_maxpressure.py",
            "run_advanced_colight.py",
        ]

    def test_subprocess_runs_in_llmtscs_dir(
        self, make_runner, fake_llmtscs: Path
    ):
        runner, calls = make_runner()
        runner.run_maxpressure(dataset="jinan_1", num_runs=1)
        assert calls[0]["cwd"] == str(fake_llmtscs.resolve())

    def test_subprocess_capture_stdout_and_stderr(self, make_runner):
        runner, calls = make_runner()
        runner.run_maxpressure(dataset="jinan_1", num_runs=1)
        call = calls[0]
        assert call["stdout"] == subprocess.PIPE
        assert call["stderr"] == subprocess.PIPE
        assert call["text"] is True

    def test_module_exports_three_method_constants(self):
        assert METHOD_MAXPRESSURE != METHOD_ADV_MAXPRESSURE
        assert METHOD_ADV_MAXPRESSURE != METHOD_ADV_COLIGHT
        assert METHOD_MAXPRESSURE != METHOD_ADV_COLIGHT
        assert METHOD_MAXPRESSURE in VALID_METHODS
        assert METHOD_ADV_MAXPRESSURE in VALID_METHODS
        assert METHOD_ADV_COLIGHT in VALID_METHODS

    def test_valid_datasets_contains_three(self):
        assert VALID_DATASETS == frozenset(
            ["jinan_1", "hangzhou_1", "newyork_1"]
        )
