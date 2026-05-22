"""RL Baselines Runner (Component 8).

Wrap LLMTSCS baseline scripts:

* ``LLMTSCS/run_maxpressure.py`` — rule-based, no training.
* ``LLMTSCS/run_advanced_maxpressure.py`` — rule-based, separate baseline.
* ``LLMTSCS/run_advanced_colight.py`` — RL, trains 100 episodes then
  evaluates on the final episode. Critic này được Task 10.3
  (CGPRDataCollector) tái sử dụng làm critic.

Mỗi method chạy ``num_runs`` lần với seed = ``base_seed + run_id``
(Requirement 8.9). :class:`SeedManager` được ``apply`` trước mỗi run để
mọi nguồn ngẫu nhiên trong subprocess (qua ``RANDOM_SEED`` env var) và
process cha đều được seed đồng bộ.

Wrapper sử dụng :func:`subprocess.run` cho cách ly tiến trình (LLMTSCS
scripts dùng ``multiprocessing.Process`` + ``wandb`` — chạy trong cùng
interpreter sẽ rò rỉ state). Các runner đọc ``LLMTSCS_DIR`` từ env,
build command line tương ứng dataset, và parse stdout cuối cùng để trích
ATT/AQL/AWT.

Format stdout của LLMTSCS oneline + pipeline trainers cuối cùng in một
``dict`` Python literal::

    {'test_reward_over': ..., 'test_avg_queue_len_over': X, ...,
     'test_avg_waiting_time_over': Y, 'test_avg_travel_time_over': Z}

Wrapper trích dòng này bằng regex và :func:`ast.literal_eval`. Mapping
metric:

* ``test_avg_travel_time_over`` → ATT (giây)
* ``test_avg_queue_len_over``  → AQL (xe/lane)
* ``test_avg_waiting_time_over`` → AWT (giây)

Đối với Advanced-CoLight, wrapper cũng quét stdout tìm chuỗi
``round=N loss=X`` per-episode. Nếu phát hiện ≥ 20 episodes liên tiếp
mà ``loss`` không giảm (so với episode trước đó), raise
:class:`ConvergenceError` chỉ rõ ``dataset`` + episode number.

Unit tests inject ``subprocess_runner`` (callable mimicking
``subprocess.run``) để KHÔNG thực sự fork process. Tests integration
(Task 12 checkpoint) sẽ gọi subprocess thật trên Jinan 1 / Hangzhou 1.

_Requirements_: 8.1-8.10
"""

from __future__ import annotations

import ast
import logging
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

# Allow ``src`` package import when running unit tests from project root.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.seed_manager import SeedManager  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Method id used by Maxpressure runner (rule-based).
METHOD_MAXPRESSURE: str = "maxpressure"
#: Method id used by Advanced-MaxPressure runner (rule-based).
METHOD_ADV_MAXPRESSURE: str = "advanced_maxpressure"
#: Method id used by Advanced-CoLight runner (RL training).
METHOD_ADV_COLIGHT: str = "advanced_colight"

VALID_METHODS: frozenset[str] = frozenset(
    [METHOD_MAXPRESSURE, METHOD_ADV_MAXPRESSURE, METHOD_ADV_COLIGHT]
)

VALID_DATASETS: frozenset[str] = frozenset(
    ["jinan_1", "hangzhou_1", "newyork_1"]
)

#: Mapping ``dataset_id → (LLMTSCS --dataset cli value, traffic_file)``.
#: KHÔNG hard-code roadnet path; LLMTSCS scripts tự build path từ
#: ``--dataset`` + ``--traffic_file``.
#:
#: NewYork 1 dùng ``--dataset newyork_28x7`` + ``anon_28_7_newyork_real_double.json``
#: vì repo LLMTSCS chỉ có ``_double`` và ``_triple`` (không có ``_real``
#: đơn giản như Jinan/Hangzhou).
_DATASET_MAP: dict[str, tuple[str, str]] = {
    "jinan_1": ("jinan", "anon_3_4_jinan_real.json"),
    "hangzhou_1": ("hangzhou", "anon_4_4_hangzhou_real.json"),
    "newyork_1": ("newyork_28x7", "anon_28_7_newyork_real_double.json"),
}

#: Mapping ``method → script filename`` trong ``LLMTSCS/``.
_METHOD_SCRIPT_MAP: dict[str, str] = {
    METHOD_MAXPRESSURE: "run_maxpressure.py",
    METHOD_ADV_MAXPRESSURE: "run_advanced_maxpressure.py",
    METHOD_ADV_COLIGHT: "run_advanced_colight.py",
}

#: Number of consecutive non-decreasing-loss episodes triggering
#: :class:`ConvergenceError` (Requirement 8.7).
_CONVERGENCE_PATIENCE: int = 20

#: Default subprocess timeout per run (seconds). Set generous to allow
#: 100-episode Advanced-CoLight training.
_DEFAULT_TIMEOUT_SECONDS: int = 7200  # 2h

#: Regex extracting Python dict literal from a single stdout line.
_RESULTS_LINE_RE = re.compile(
    r"\{[^{}]*test_avg_travel_time_over[^{}]*\}"
)

#: Regex matching ``round=<int> loss=<float>`` (case-insensitive).
_ROUND_LOSS_RE = re.compile(
    r"(?:round|episode)[\s:=]*?(\d+).*?loss[\s:=]+(-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ConvergenceError(Exception):
    """Advanced-CoLight failed to converge.

    Raised when ``loss`` does not decrease for :data:`_CONVERGENCE_PATIENCE`
    consecutive episodes (default 20). The exception message includes
    dataset id, the run_id, the episode where stagnation was detected, and
    the loss values for the patience window.
    """


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class BaselineResult:
    """Output of a single baseline run.

    Attributes:
        method: One of :data:`VALID_METHODS`.
        dataset: One of :data:`VALID_DATASETS`.
        att: Average Travel Time (giây), >= 0.
        aql: Average Queue Length (xe/lane), >= 0.
        awt: Average Waiting Time (giây), >= 0.
        run_id: Run index (0-based).
        seed: Effective seed (= ``base_seed + run_id``).
    """

    method: str
    dataset: str
    att: float
    aql: float
    awt: float
    run_id: int
    seed: int

    def __post_init__(self) -> None:
        if self.method not in VALID_METHODS:
            raise ValueError(
                "BaselineResult.method must be one of "
                f"{sorted(VALID_METHODS)}; got {self.method!r}"
            )
        if self.dataset not in VALID_DATASETS:
            raise ValueError(
                "BaselineResult.dataset must be one of "
                f"{sorted(VALID_DATASETS)}; got {self.dataset!r}"
            )
        for fname in ("att", "aql", "awt"):
            value = getattr(self, fname)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    f"BaselineResult.{fname} must be a number; "
                    f"got {type(value).__name__} ({value!r})"
                )
            value = float(value)
            if value < 0:
                raise ValueError(
                    f"BaselineResult.{fname} must be >= 0; got {value}"
                )
            setattr(self, fname, value)
        if isinstance(self.run_id, bool) or not isinstance(self.run_id, int):
            raise ValueError(
                "BaselineResult.run_id must be int; "
                f"got {type(self.run_id).__name__} ({self.run_id!r})"
            )
        if self.run_id < 0:
            raise ValueError(
                f"BaselineResult.run_id must be >= 0; got {self.run_id}"
            )
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError(
                "BaselineResult.seed must be int; "
                f"got {type(self.seed).__name__} ({self.seed!r})"
            )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


#: Type alias for a callable mimicking ``subprocess.run``. Tests inject a
#: stub; production uses :func:`subprocess.run` directly.
SubprocessRunner = Callable[..., subprocess.CompletedProcess]


class RLBaselinesRunner:
    """Run RL baselines via subprocess invocation of LLMTSCS scripts.

    Same simulation config (green 30s, yellow 3s, all-red 2s) is enforced
    by LLMTSCS's hard-coded timing in those scripts; this wrapper does NOT
    pass timing CLI flags (the scripts don't expose them). Cross-method
    consistency is therefore guaranteed by upstream LLMTSCS design.

    :class:`SeedManager` is applied before each subprocess invocation, and
    ``RANDOM_SEED`` is forwarded as an env var to the child so the LLMTSCS
    code (and any libraries it imports) sees the same seed.
    """

    LLMTSCS_DIR_KEY: str = "LLMTSCS_DIR"

    def __init__(
        self,
        sim_config_path: str,
        base_seed: int = 42,
        *,
        llmtscs_dir: str | os.PathLike[str] | None = None,
        subprocess_runner: SubprocessRunner | None = None,
        seed_manager: SeedManager | None = None,
        timeout_seconds: int = _DEFAULT_TIMEOUT_SECONDS,
        python_executable: str | None = None,
    ) -> None:
        """Khởi tạo runner.

        Args:
            sim_config_path: Path tới ``config/simulation.json`` (validated
                for existence; LLMTSCS scripts have their own config).
            base_seed: Base seed; mỗi run dùng ``base_seed + run_id``.
            llmtscs_dir: Override path tới ``LLMTSCS/`` repo. Nếu ``None``,
                đọc từ env ``LLMTSCS_DIR``.
            subprocess_runner: Callable mimicking :func:`subprocess.run`.
                Production: ``None`` → :func:`subprocess.run`.
                Tests: pass a mock that returns a fake
                :class:`subprocess.CompletedProcess`.
            seed_manager: Optional :class:`SeedManager`. Nếu ``None``,
                tạo mới với ``base_seed=base_seed``.
            timeout_seconds: Per-run timeout passed to subprocess.
            python_executable: Override of the Python interpreter used to
                run LLMTSCS scripts. Defaults to :data:`sys.executable`.

        Raises:
            FileNotFoundError: ``sim_config_path`` không tồn tại, hoặc
                ``llmtscs_dir`` không tồn tại / không phải directory.
            ValueError: ``base_seed`` không phải int, hoặc
                ``llmtscs_dir`` không được cung cấp (cả param lẫn env).
        """
        if not isinstance(sim_config_path, (str, os.PathLike)):
            raise ValueError(
                "RLBaselinesRunner: sim_config_path must be a path-like; "
                f"got {type(sim_config_path).__name__}"
            )
        sim_path = Path(sim_config_path)
        if not sim_path.exists() or not sim_path.is_file():
            raise FileNotFoundError(
                f"RLBaselinesRunner: sim_config_path not found: {sim_path}"
            )

        if isinstance(base_seed, bool) or not isinstance(base_seed, int):
            raise ValueError(
                "RLBaselinesRunner: base_seed must be int; "
                f"got {type(base_seed).__name__} ({base_seed!r})"
            )

        if llmtscs_dir is None:
            env_dir = os.environ.get(self.LLMTSCS_DIR_KEY)
            if not env_dir:
                raise ValueError(
                    "RLBaselinesRunner: llmtscs_dir not provided and "
                    f"env {self.LLMTSCS_DIR_KEY!r} is unset. Set in .env "
                    "or pass llmtscs_dir explicitly."
                )
            llmtscs_dir = env_dir
        llmtscs_path = Path(llmtscs_dir)
        if not llmtscs_path.exists() or not llmtscs_path.is_dir():
            raise FileNotFoundError(
                "RLBaselinesRunner: llmtscs_dir not found or not a "
                f"directory: {llmtscs_path}"
            )

        if isinstance(timeout_seconds, bool) or not isinstance(
            timeout_seconds, int
        ):
            raise ValueError(
                "RLBaselinesRunner: timeout_seconds must be int; "
                f"got {type(timeout_seconds).__name__} ({timeout_seconds!r})"
            )
        if timeout_seconds <= 0:
            raise ValueError(
                "RLBaselinesRunner: timeout_seconds must be > 0; "
                f"got {timeout_seconds}"
            )

        self.sim_config_path: str = str(sim_path.resolve())
        self.base_seed: int = int(base_seed)
        self.llmtscs_dir: str = str(llmtscs_path.resolve())
        self.timeout_seconds: int = int(timeout_seconds)
        self._subprocess_runner: SubprocessRunner = (
            subprocess_runner if subprocess_runner is not None else subprocess.run
        )
        self._seed_manager: SeedManager = (
            seed_manager if seed_manager is not None else SeedManager(base_seed)
        )
        self._python_executable: str = (
            python_executable if python_executable is not None else sys.executable
        )

    # ------------------------------------------------------------------ API --

    def run_maxpressure(
        self, dataset: str, num_runs: int = 3
    ) -> list[BaselineResult]:
        """Run Maxpressure baseline (rule-based).

        Wraps ``LLMTSCS/run_maxpressure.py``. Each run uses seed
        ``base_seed + run_id``. Returns one :class:`BaselineResult` per run.
        """
        return self._run_method(
            method=METHOD_MAXPRESSURE,
            dataset=dataset,
            num_runs=num_runs,
            check_convergence=False,
        )

    def run_advanced_maxpressure(
        self, dataset: str, num_runs: int = 3
    ) -> list[BaselineResult]:
        """Run Advanced-MaxPressure baseline (rule-based, distinct from Maxpressure).

        Wraps ``LLMTSCS/run_advanced_maxpressure.py``.
        """
        return self._run_method(
            method=METHOD_ADV_MAXPRESSURE,
            dataset=dataset,
            num_runs=num_runs,
            check_convergence=False,
        )

    def run_advanced_colight(
        self,
        dataset: str,
        train_episodes: int = 100,
        num_runs: int = 3,
    ) -> list[BaselineResult]:
        """Train + evaluate Advanced-CoLight (RL critic).

        Wraps ``LLMTSCS/run_advanced_colight.py`` — trains
        ``train_episodes`` episodes then evaluates the final episode.
        Detects convergence failure: if ``loss`` does not decrease for
        :data:`_CONVERGENCE_PATIENCE` (default 20) consecutive episodes,
        raise :class:`ConvergenceError` (Requirement 8.7).

        Args:
            dataset: One of :data:`VALID_DATASETS`.
            train_episodes: Number of training rounds. Default 100.
            num_runs: Number of evaluation runs. Default 3.

        Raises:
            ConvergenceError: If loss stagnates per the patience criterion.
        """
        if isinstance(train_episodes, bool) or not isinstance(
            train_episodes, int
        ):
            raise ValueError(
                "RLBaselinesRunner.run_advanced_colight: train_episodes "
                f"must be int; got {type(train_episodes).__name__} "
                f"({train_episodes!r})"
            )
        if train_episodes <= 0:
            raise ValueError(
                "RLBaselinesRunner.run_advanced_colight: train_episodes "
                f"must be > 0; got {train_episodes}"
            )

        return self._run_method(
            method=METHOD_ADV_COLIGHT,
            dataset=dataset,
            num_runs=num_runs,
            check_convergence=True,
            train_episodes=train_episodes,
        )

    # -------------------------------------------------------------- Internal --

    def _run_method(
        self,
        *,
        method: str,
        dataset: str,
        num_runs: int,
        check_convergence: bool,
        train_episodes: int | None = None,
    ) -> list[BaselineResult]:
        """Driver loop common to all three methods."""
        if method not in VALID_METHODS:
            raise ValueError(
                "RLBaselinesRunner: method must be one of "
                f"{sorted(VALID_METHODS)}; got {method!r}"
            )
        if dataset not in VALID_DATASETS:
            raise ValueError(
                "RLBaselinesRunner: dataset must be one of "
                f"{sorted(VALID_DATASETS)}; got {dataset!r}"
            )
        if isinstance(num_runs, bool) or not isinstance(num_runs, int):
            raise ValueError(
                "RLBaselinesRunner: num_runs must be int; "
                f"got {type(num_runs).__name__} ({num_runs!r})"
            )
        if num_runs <= 0:
            raise ValueError(
                f"RLBaselinesRunner: num_runs must be > 0; got {num_runs}"
            )

        results: list[BaselineResult] = []
        for run_id in range(num_runs):
            seed = self._seed_manager.seed_for_run(run_id)
            # Apply seed in parent process for any RNG used while preparing
            # CLI args / subprocess invocation; subprocess child inherits
            # via RANDOM_SEED env var.
            self._seed_manager.apply(seed)

            cmd = self._build_subprocess_cmd(method, dataset)
            env = self._build_subprocess_env(seed)

            logger.info(
                "RLBaselinesRunner: launching method=%s dataset=%s "
                "run_id=%d seed=%d cmd=%s",
                method,
                dataset,
                run_id,
                seed,
                " ".join(cmd),
            )

            completed = self._subprocess_runner(
                cmd,
                cwd=self.llmtscs_dir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )

            stdout = completed.stdout or ""
            stderr = completed.stderr or ""

            if completed.returncode != 0:
                raise RuntimeError(
                    f"RLBaselinesRunner: subprocess for method={method} "
                    f"dataset={dataset} run_id={run_id} exited with "
                    f"returncode={completed.returncode}.\n"
                    f"--- stderr (last 1000 chars) ---\n{stderr[-1000:]}"
                )

            if check_convergence:
                self._check_convergence(
                    stdout=stdout, dataset=dataset, run_id=run_id
                )

            result = self._parse_output(
                stdout=stdout,
                method=method,
                dataset=dataset,
                run_id=run_id,
                seed=seed,
            )
            results.append(result)

        return results

    def _build_subprocess_cmd(
        self, method: str, dataset: str
    ) -> list[str]:
        """Build the LLMTSCS subprocess command line."""
        if method not in _METHOD_SCRIPT_MAP:
            raise ValueError(
                f"RLBaselinesRunner: unknown method {method!r}"
            )
        script = _METHOD_SCRIPT_MAP[method]
        if dataset not in _DATASET_MAP:
            raise ValueError(
                f"RLBaselinesRunner: unknown dataset {dataset!r}"
            )
        cli_dataset, cli_traffic_file = _DATASET_MAP[dataset]

        cmd: list[str] = [
            self._python_executable,
            script,
            "--dataset",
            cli_dataset,
            "--traffic_file",
            cli_traffic_file,
        ]
        return cmd

    def _build_subprocess_env(self, seed: int) -> dict[str, str]:
        """Forward parent env + override ``RANDOM_SEED`` for the child."""
        env = dict(os.environ)
        env["RANDOM_SEED"] = str(int(seed))
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("WANDB_MODE", "offline")
        return env

    @staticmethod
    def _parse_output(
        stdout: str,
        method: str,
        dataset: str,
        run_id: int,
        seed: int,
    ) -> BaselineResult:
        """Extract metrics dict from ``stdout`` and build a BaselineResult.

        The LLMTSCS scripts (oneline.py / pipeline.py) print the final
        metrics dict using Python ``print(results)``. We scan stdout for
        ALL matches and pick the LAST occurrence (final episode result).

        Raises:
            ValueError: If no parseable metrics dict is found.
        """
        if not isinstance(stdout, str):
            raise ValueError(
                "RLBaselinesRunner._parse_output: stdout must be str; "
                f"got {type(stdout).__name__}"
            )

        matches = list(_RESULTS_LINE_RE.finditer(stdout))
        if not matches:
            raise ValueError(
                "RLBaselinesRunner._parse_output: no metrics dict found in "
                f"stdout for method={method} dataset={dataset} "
                f"run_id={run_id}. Last 500 chars:\n"
                f"{stdout[-500:]}"
            )

        raw_dict = matches[-1].group(0)
        try:
            parsed = ast.literal_eval(raw_dict)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(
                "RLBaselinesRunner._parse_output: failed to parse metrics "
                f"dict literal {raw_dict!r}: {exc}"
            ) from exc

        if not isinstance(parsed, dict):
            raise ValueError(
                "RLBaselinesRunner._parse_output: parsed metrics must be a "
                f"dict; got {type(parsed).__name__}"
            )

        try:
            att = float(parsed["test_avg_travel_time_over"])
            aql = float(parsed["test_avg_queue_len_over"])
            awt = float(parsed["test_avg_waiting_time_over"])
        except KeyError as exc:
            raise ValueError(
                "RLBaselinesRunner._parse_output: metrics dict missing key "
                f"{exc.args[0]!r}; got keys {sorted(parsed.keys())}"
            ) from exc
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "RLBaselinesRunner._parse_output: metrics value not "
                f"numeric: {exc}"
            ) from exc

        # Defensive: clamp tiny negative floats from numerical noise.
        att = max(0.0, att)
        aql = max(0.0, aql)
        awt = max(0.0, awt)

        return BaselineResult(
            method=method,
            dataset=dataset,
            att=att,
            aql=aql,
            awt=awt,
            run_id=run_id,
            seed=seed,
        )

    @staticmethod
    def _extract_loss_series(stdout: str) -> list[tuple[int, float]]:
        """Parse ``(round_idx, loss)`` pairs from stdout.

        Matches both ``round=N loss=X`` and ``Round N ... loss X`` patterns
        (case-insensitive). Returns the pairs in the order they appear.
        """
        pairs: list[tuple[int, float]] = []
        for match in _ROUND_LOSS_RE.finditer(stdout):
            try:
                round_idx = int(match.group(1))
                loss = float(match.group(2))
            except (TypeError, ValueError):  # pragma: no cover
                continue
            pairs.append((round_idx, loss))
        return pairs

    def _check_convergence(
        self,
        *,
        stdout: str,
        dataset: str,
        run_id: int,
        patience: int = _CONVERGENCE_PATIENCE,
    ) -> None:
        """Verify Advanced-CoLight loss decreases.

        Convergence failure = ``patience`` consecutive rounds where
        ``loss[i] >= loss[i-1]``.

        Edge case: if NO loss lines are present in stdout, this method
        does NOT raise — convergence is unknown rather than failed.

        Raises:
            ConvergenceError: If a non-decreasing window of length
                ``patience`` is detected.
        """
        pairs = self._extract_loss_series(stdout)
        if len(pairs) < patience + 1:
            if not pairs:
                logger.warning(
                    "RLBaselinesRunner._check_convergence: no loss data "
                    "in stdout for dataset=%s run_id=%d; cannot verify "
                    "convergence.",
                    dataset,
                    run_id,
                )
            return

        # Sort by round index in case stdout order is permuted.
        sorted_pairs = sorted(pairs, key=lambda p: p[0])

        consecutive_non_decreasing = 0
        last_loss: float | None = None
        stagnation_window: list[tuple[int, float]] = []

        for round_idx, loss in sorted_pairs:
            if last_loss is not None and loss >= last_loss:
                consecutive_non_decreasing += 1
                stagnation_window.append((round_idx, loss))
                if consecutive_non_decreasing >= patience:
                    losses_str = ", ".join(
                        f"round{r}={l:.6g}" for r, l in stagnation_window
                    )
                    raise ConvergenceError(
                        "Advanced-CoLight failed to converge on dataset="
                        f"{dataset!r} (run_id={run_id}): loss did not "
                        f"decrease for {patience} consecutive episodes. "
                        f"Stagnation detected at episode {round_idx}. "
                        f"Loss series: [{losses_str}]"
                    )
            else:
                consecutive_non_decreasing = 0
                stagnation_window = []
            last_loss = loss


__all__ = [
    "BaselineResult",
    "ConvergenceError",
    "RLBaselinesRunner",
    "VALID_METHODS",
    "VALID_DATASETS",
    "METHOD_MAXPRESSURE",
    "METHOD_ADV_MAXPRESSURE",
    "METHOD_ADV_COLIGHT",
]
