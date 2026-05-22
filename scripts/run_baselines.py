"""scripts/run_baselines.py — main runner for RL baselines (Task 13.3).

Wires SeedManager → PhaseApprovalGate → PreflightChecker → SimulationConfig
→ :class:`RLBaselinesRunner` → :class:`MetricsEvaluator`.

Three baselines are supported (Requirement 8):

* ``maxpressure`` — rule-based.
* ``advanced_maxpressure`` — rule-based, distinct from maxpressure.
* ``colight`` (alias for ``advanced_colight``) — RL critic, trains 100
  episodes then evaluates the final episode.

``--method all`` runs all three sequentially (no parallelism — each
LLMTSCS subprocess uses multiprocessing internally and would race on
GPU memory if launched concurrently).

Run-counts: 3 runs per method per dataset in EVERY phase (Requirement
8.10), regardless of phase. ``--num-runs`` overrides for ad-hoc testing.

The runner does NOT call any LLM, so the LLM prompt log helper from the
shared utilities is intentionally unused here.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Iterable

# Ensure src + scripts importable when invoked as ``python scripts/run_baselines.py``.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts._runner_utils import (  # noqa: E402
    VALID_DATASETS,
    VALID_MODES,
    VALID_PHASES,
    VALID_SAVE_REPLAY,
    resolve_mode,
    resolve_save_replay,
    setup_basic_logging,
    utc_iso_timestamp,
    write_experiment_result,
)
from src.phase_approval_gate import PhaseApprovalGate  # noqa: E402
from src.preflight_checker import PreflightChecker  # noqa: E402
from src.seed_manager import SeedManager  # noqa: E402
from src.sim_config import (  # noqa: E402
    ExperimentResult,
    MetricsResult,
    SimulationConfig,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Mapping ``--method`` choice → :class:`ExperimentResult.method` value.
#: ``colight`` is a user-facing alias of ``advanced_colight`` (matches the
#: design.md table column header). Both resolve to the same underlying
#: runner method.
METHOD_ALIAS_MAP: dict[str, str] = {
    "maxpressure": "maxpressure",
    "advanced_maxpressure": "advanced_maxpressure",
    "colight": "advanced_colight",
    "advanced_colight": "advanced_colight",
}

METHOD_CHOICES: tuple[str, ...] = (
    "maxpressure",
    "advanced_maxpressure",
    "colight",
    "all",
)

#: Order in which ``--method all`` expands. Sequential by design (LLMTSCS
#: subprocesses are CPU-heavy + use multiprocessing).
ALL_METHODS_ORDER: tuple[str, ...] = (
    "maxpressure",
    "advanced_maxpressure",
    "advanced_colight",
)


# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        prog="run_baselines",
        description=(
            "Run RL baselines (Maxpressure / Advanced-MaxPressure / "
            "Advanced-CoLight) over CityFlow simulation."
        ),
    )

    parser.add_argument(
        "--dataset",
        choices=VALID_DATASETS,
        required=True,
        help="Dataset id (Phase 1/2: jinan_1/hangzhou_1; Phase 3: newyork_1).",
    )
    parser.add_argument(
        "--method",
        choices=METHOD_CHOICES,
        default="all",
        help=(
            "Baseline method to run. 'all' = "
            "[maxpressure, advanced_maxpressure, advanced_colight] "
            "sequentially."
        ),
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=3,
        help="Number of runs per method (default 3 — Requirement 8.10).",
    )
    parser.add_argument(
        "--mode",
        choices=VALID_MODES,
        default=None,
        help=(
            "Override DEFAULT_RUN_MODE. Phase 1 default 'demo'; "
            "Phase 2/3 default 'full'."
        ),
    )
    parser.add_argument(
        "--phase",
        type=int,
        choices=VALID_PHASES,
        default=1,
        help="Phase to execute (default 1). Phase 3 needs manual approval.",
    )
    parser.add_argument(
        "--save-replay",
        choices=VALID_SAVE_REPLAY,
        default="auto",
        help=(
            "Replay file policy. 'auto' = on for Phase 1 / off for "
            "Phase 2-3 unless overridden via .streamlit_pref.json."
        ),
    )

    parser.add_argument(
        "--simulation-config",
        default="config/simulation.json",
        help="Path to config/simulation.json.",
    )
    parser.add_argument(
        "--llmtscs-dir",
        default=None,
        help="Override LLMTSCS_DIR (defaults to env var).",
    )
    parser.add_argument(
        "--project-dir",
        default=None,
        help="Project root for PreflightChecker (default $PROJECT_DIR or CWD).",
    )
    parser.add_argument(
        "--train-episodes",
        type=int,
        default=100,
        help="Advanced-CoLight train episodes (default 100).",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip PreflightChecker (intended for unit tests / WSL2 reruns).",
    )

    args = parser.parse_args(argv)

    if args.num_runs <= 0:
        parser.error("--num-runs must be > 0")
    if args.train_episodes <= 0:
        parser.error("--train-episodes must be > 0")

    return args


# ---------------------------------------------------------------------------
# Method expansion
# ---------------------------------------------------------------------------


def expand_methods(method_arg: str) -> list[str]:
    """Expand ``--method`` choice into a concrete list of method names.

    * ``all`` → :data:`ALL_METHODS_ORDER`.
    * ``colight`` / ``advanced_colight`` → ``["advanced_colight"]``.
    * Otherwise → ``[method_arg]``.

    Args:
        method_arg: One of :data:`METHOD_CHOICES`.

    Returns:
        Ordered list of canonical method names usable with
        :class:`RLBaselinesRunner`.

    Raises:
        ValueError: If ``method_arg`` is unknown.
    """
    if method_arg == "all":
        return list(ALL_METHODS_ORDER)
    canonical = METHOD_ALIAS_MAP.get(method_arg)
    if canonical is None:
        raise ValueError(
            f"expand_methods: unknown method {method_arg!r}; "
            f"choose from {METHOD_CHOICES}"
        )
    return [canonical]


# ---------------------------------------------------------------------------
# Per-method dispatch
# ---------------------------------------------------------------------------


def _dispatch_runner(
    *, method: str, runner, dataset: str, num_runs: int, train_episodes: int
):
    """Call the right ``RLBaselinesRunner.run_*`` for the given method."""
    if method == "maxpressure":
        return runner.run_maxpressure(dataset=dataset, num_runs=num_runs)
    if method == "advanced_maxpressure":
        return runner.run_advanced_maxpressure(
            dataset=dataset, num_runs=num_runs
        )
    if method == "advanced_colight":
        return runner.run_advanced_colight(
            dataset=dataset,
            train_episodes=train_episodes,
            num_runs=num_runs,
        )
    raise ValueError(f"_dispatch_runner: unknown method {method!r}")


# ---------------------------------------------------------------------------
# Result conversion
# ---------------------------------------------------------------------------


def baseline_results_to_experiment_results(
    *,
    baseline_results: Iterable,
    phase_label: str,
    timestamp: str,
    duration_per_run: float = 0.0,
    replay_file: str | None = None,
) -> list[ExperimentResult]:
    """Convert ``BaselineResult`` instances to ``ExperimentResult`` for
    canonical ``results/metrics/`` storage.

    Note that :class:`RLBaselinesRunner` does not currently expose
    per-run wall-clock duration, so ``duration_per_run`` is supplied by
    the caller (typically the wall-clock measured around the whole
    method dispatch divided by ``num_runs``).
    """
    out: list[ExperimentResult] = []
    for br in baseline_results:
        metrics = MetricsResult(att=br.att, aql=br.aql, awt=br.awt)
        out.append(
            ExperimentResult(
                method=br.method,
                dataset=br.dataset,
                run_id=br.run_id,
                seed=br.seed,
                metrics=metrics,
                token_usage=None,
                duration_seconds=float(duration_per_run),
                timestamp=timestamp,
                phase_label=phase_label,
                replay_file=replay_file,
            )
        )
    return out


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


def main(
    argv: list[str] | None = None,
    *,
    rl_baselines_runner_factory=None,
) -> int:
    """Entry point. Returns process exit code (0 = success).

    Args:
        argv: CLI args (default ``sys.argv[1:]``).
        rl_baselines_runner_factory: Optional override for
            :class:`RLBaselinesRunner`. Tests inject a fake to skip
            subprocess spawning. Default lazy-imports the real wrapper.
    """
    setup_basic_logging()

    args = parse_args(argv)
    mode = resolve_mode(args.mode, args.phase)
    save_replay = resolve_save_replay(args.save_replay, args.phase)
    # ``save_replay`` is informational only for baselines (LLMTSCS scripts
    # have their own replay machinery). Logged for traceability.
    logger.info(
        "run_baselines: resolved save_replay=%s (informational; baselines "
        "rely on LLMTSCS native replay flag).",
        save_replay,
    )

    # ---- 1) SeedManager ------------------------------------------------
    seed_manager = SeedManager()
    seed_manager.apply(seed_manager.seed_for_run(0))

    # ---- 2) PhaseApprovalGate -----------------------------------------
    gate = PhaseApprovalGate()
    try:
        gate.validate_phase(args.phase, args.dataset, mode)
        gate.check_prerequisite(args.phase)
    except (ValueError, RuntimeError) as exc:
        logger.error("PhaseApprovalGate rejected request: %s", exc)
        return 2

    if args.phase == 3:
        approved = gate.request_manual_approval(
            phase=3,
            estimated_cost_usd=None,
            estimated_time_hours=None,
        )
        if not approved:
            logger.error("Phase 3 manual approval denied; aborting.")
            return 3

    phase_label = gate.phase_label(args.phase)

    # ---- 3) PreflightChecker ------------------------------------------
    if not args.skip_preflight:
        project_dir = (
            args.project_dir
            or os.environ.get("PROJECT_DIR")
            or str(PROJECT_ROOT)
        )
        try:
            PreflightChecker().run_all(project_dir)
        except (RuntimeError, OSError) as exc:
            logger.error("PreflightChecker failed: %s", exc)
            return 4

    # ---- 4) SimulationConfig -------------------------------------------
    try:
        sim_config = SimulationConfig.from_json(
            args.simulation_config, dataset=args.dataset, mode=mode
        )
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Failed to load simulation config: %s", exc)
        return 5
    # sim_config is loaded for parameter consistency check (Requirement 10
    # AC 7-8). LLMTSCS scripts hard-code the same timing internally, so the
    # check is informational at this layer.
    logger.info(
        "run_baselines: sim_config loaded: green=%d yellow=%d all_red=%d "
        "total_timesteps=%d",
        sim_config.green_duration,
        sim_config.yellow_duration,
        sim_config.all_red_duration,
        sim_config.total_timesteps,
    )

    # ---- 5) Resolve LLMTSCS dir ----------------------------------------
    llmtscs_dir = args.llmtscs_dir or os.environ.get("LLMTSCS_DIR")
    if not llmtscs_dir:
        logger.error(
            "LLMTSCS_DIR not set. Pass --llmtscs-dir or set env var in .env."
        )
        return 6
    if not Path(llmtscs_dir).is_dir():
        logger.error(
            "LLMTSCS_DIR %r does not point to an existing directory.",
            llmtscs_dir,
        )
        return 6

    # ---- 6) Build RL runner -------------------------------------------
    if rl_baselines_runner_factory is None:
        from src.baselines_runner import RLBaselinesRunner as _RLR  # noqa

        rl_baselines_runner_factory = _RLR

    runner = rl_baselines_runner_factory(
        sim_config_path=args.simulation_config,
        base_seed=seed_manager.base_seed,
        llmtscs_dir=llmtscs_dir,
        seed_manager=seed_manager,
    )

    # ---- 7) Run methods ------------------------------------------------
    methods = expand_methods(args.method)

    written_paths: list[str] = []
    for method_name in methods:
        logger.info(
            "run_baselines: starting method=%s dataset=%s num_runs=%d "
            "phase=%s",
            method_name,
            args.dataset,
            args.num_runs,
            phase_label,
        )
        start = time.time()
        try:
            baseline_results = _dispatch_runner(
                method=method_name,
                runner=runner,
                dataset=args.dataset,
                num_runs=args.num_runs,
                train_episodes=args.train_episodes,
            )
        except Exception as exc:  # noqa: BLE001
            # Convergence failure or subprocess error → log and abort the
            # remaining methods (don't waste time if Adv-CoLight fails).
            logger.error(
                "run_baselines: method=%s failed: %s; aborting remaining "
                "methods.",
                method_name,
                exc,
            )
            return 9

        duration = time.time() - start
        per_run = duration / max(1, len(baseline_results))
        timestamp = utc_iso_timestamp()
        experiment_results = baseline_results_to_experiment_results(
            baseline_results=baseline_results,
            phase_label=phase_label,
            timestamp=timestamp,
            duration_per_run=per_run,
            replay_file=None,
        )
        for er in experiment_results:
            written_paths.append(write_experiment_result(er))

        logger.info(
            "run_baselines: completed method=%s runs=%d duration=%.1fs",
            method_name,
            len(baseline_results),
            duration,
        )

    logger.info(
        "run_baselines: completed %d run(s) across %d method(s); files "
        "written to results/metrics/.",
        len(written_paths),
        len(methods),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
