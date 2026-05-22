"""scripts/run_lightgpt.py — main runner for LightGPT local inference (Task 13.1).

Wires together SeedManager → PhaseApprovalGate → PreflightChecker →
SimulationConfig → PhaseIndexMapper → CityFlowEngine → ObservationParser →
LightGPTInference → ResponseParser into a single CLI-driven driver loop
that runs ``--num-runs`` simulations on a chosen dataset / phase / mode.

Per-run output:

* :class:`src.sim_config.ExperimentResult` JSON written to
  ``results/metrics/<method>_<dataset>_phase<N>_run<M>.json`` with
  ``phase_label``, ``replay_file``, ``method``.
* Per-decision LLM prompt + raw response logged to
  ``results/logs/llm_prompts/<dataset>_<method>_<run_id>_t<timestep>.txt``
  (consumed by UI Demo "LLM Inspection" panel — Task 18.2).

Two LightGPT 0.5B variants (``lightgpt_hf`` from HuggingFace,
``lightgpt_mine`` from ``models/qwen2_finetuned/``) can be selected via
``--method``. The default depends on phase: Phase 1 → ``lightgpt_hf``
only; Phase 2 / Phase 3 → ``both`` (sequential, never parallel — peak
~7.5 GB VRAM each on RTX 4060 8 GB).

Skip / fail rules for ``lightgpt_mine`` (Requirement 5 AC 11–12):

* Phase 1: model missing → SKIP + warning, KHÔNG fail toàn runner.
* Phase 2 / 3: model missing → FAIL with explicit instruction to run
  ``scripts/run_training.py`` first.

Run-counts (consistent with Requirement 5 AC 8 / Phase plan):

* LightGPT local: 3 runs in every phase. ``--num-runs`` defaults to 3
  but can be overridden for ad-hoc testing.

CLI examples::

    # Phase 1 demo on Jinan 1 (3 runs, lightgpt_hf only by default)
    python scripts/run_lightgpt.py --dataset jinan_1 --phase 1

    # Phase 2 full on Hangzhou 1 (both variants, 3 runs each → 6 total)
    python scripts/run_lightgpt.py --dataset hangzhou_1 --phase 2 \\
        --method both --mode full

    # Single ad-hoc run for debugging, force replay file off
    python scripts/run_lightgpt.py --dataset jinan_1 --num-runs 1 \\
        --save-replay off

_Requirements_: 1.5, 5.1-5.12, 10.1-10.6, 13.1-13.7, 14.6, 14.10, 14.11
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Ensure src + scripts importable when invoked as ``python scripts/run_lightgpt.py``.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts._runner_utils import (  # noqa: E402
    DATASET_FILES,
    VALID_DATASETS,
    VALID_MODES,
    VALID_PHASES,
    VALID_SAVE_REPLAY,
    build_cityflow_config_path,
    log_llm_prompt,
    resolve_mode,
    resolve_save_replay,
    setup_basic_logging,
    utc_iso_timestamp,
    write_experiment_result,
)
from src.metrics_evaluator import MetricsEvaluator  # noqa: E402
from src.observation_parser import ObservationParser  # noqa: E402
from src.phase_approval_gate import PhaseApprovalGate  # noqa: E402
from src.phase_index_mapper import PhaseIndexMapper  # noqa: E402
from src.preflight_checker import PreflightChecker  # noqa: E402
from src.response_parser import ResponseParser  # noqa: E402
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

#: Subset of methods this runner produces (excludes baselines / GPT-4o).
LIGHTGPT_METHODS: tuple[str, ...] = ("lightgpt_hf", "lightgpt_mine")

#: ``--method`` accepts these literal values from the user.
METHOD_CHOICES: tuple[str, ...] = ("lightgpt_hf", "lightgpt_mine", "both")

#: Path to ``models/qwen2_finetuned/`` (Requirement 5 AC 11-12).
SELF_FINETUNED_PATH: str = "models/qwen2_finetuned/"


# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        ``Namespace`` with attributes:
            ``dataset``, ``num_runs``, ``mode``, ``phase``, ``save_replay``,
            ``method``, ``simulation_config``, ``llmtscs_dir``,
            ``project_dir``, ``hf_cache``, ``hf_token``, ``skip_preflight``.
    """
    parser = argparse.ArgumentParser(
        prog="run_lightgpt",
        description=(
            "Run LightGPT local inference (NF4 4-bit) over CityFlow "
            "simulation. Supports two 0.5B variants in parallel "
            "(lightgpt_hf vs lightgpt_mine)."
        ),
    )

    parser.add_argument(
        "--dataset",
        choices=VALID_DATASETS,
        required=True,
        help="Dataset id (Phase 1/2: jinan_1/hangzhou_1; Phase 3: newyork_1).",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=3,
        help="Number of runs per variant (default 3 — Requirement 5 AC 8).",
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
        "--method",
        choices=METHOD_CHOICES,
        default=None,
        help=(
            "Variant to run. Default depends on phase: Phase 1 → "
            "'lightgpt_hf'; Phase 2/3 → 'both' (sequential)."
        ),
    )

    parser.add_argument(
        "--simulation-config",
        default="config/simulation.json",
        help="Path to config/simulation.json (Requirement 10).",
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
        "--hf-cache",
        default=None,
        help="HuggingFace cache dir (default $HF_HOME or ./models/hf_cache).",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip PreflightChecker (intended for unit tests / WSL2 reruns).",
    )

    args = parser.parse_args(argv)

    if args.num_runs <= 0:
        parser.error("--num-runs must be > 0")

    return args


# ---------------------------------------------------------------------------
# Method selection
# ---------------------------------------------------------------------------


def select_methods(
    method_arg: str | None, phase: int, *, finetuned_exists: bool
) -> list[str]:
    """Decide which variant(s) to run for this invocation.

    Implements Requirement 5 AC 9-12:

    * Default by phase: Phase 1 → ``lightgpt_hf``; Phase 2/3 → ``both``.
    * ``both`` expands to ``[lightgpt_hf, lightgpt_mine]`` (sequential).
    * Phase 1 + ``lightgpt_mine`` missing → drop ``lightgpt_mine`` with
      a warning. Other variants continue.
    * Phase 2/3 + ``lightgpt_mine`` missing → raise ``FileNotFoundError``
      with explicit instruction to run training first.

    Args:
        method_arg: Raw value from ``--method`` (``None`` if user omitted).
        phase: 1, 2, or 3.
        finetuned_exists: Whether ``models/qwen2_finetuned/`` exists.

    Returns:
        Ordered list of variant names. ``len`` ≥ 1 if any variant is
        runnable, else ``[]`` (caller logs and exits cleanly).

    Raises:
        FileNotFoundError: Phase 2/3 + ``lightgpt_mine`` requested but
            missing.
    """
    if phase not in VALID_PHASES:
        raise ValueError(
            f"select_methods: phase must be one of {VALID_PHASES}; "
            f"got {phase!r}"
        )

    if method_arg is None:
        method_arg = "lightgpt_hf" if phase == 1 else "both"

    if method_arg == "both":
        requested = ["lightgpt_hf", "lightgpt_mine"]
    else:
        requested = [method_arg]

    if "lightgpt_mine" in requested and not finetuned_exists:
        if phase == 1:
            logger.warning(
                "lightgpt_mine chưa được train, skip ở Phase 1. "
                "Chạy `scripts/run_training.py` trước nếu muốn so sánh "
                "với lightgpt_hf."
            )
            requested = [m for m in requested if m != "lightgpt_mine"]
        else:
            raise FileNotFoundError(
                "Model qwen2_finetuned chưa được train. "
                f"Chạy `scripts/run_training.py --phase {phase}` "
                "trước khi chạy run_lightgpt với --method lightgpt_mine "
                "ở Phase 2/3."
            )

    return requested


# ---------------------------------------------------------------------------
# Single simulation
# ---------------------------------------------------------------------------


def run_simulation(
    *,
    engine,
    obs_parser: ObservationParser,
    response_parser: ResponseParser,
    phase_mapper: PhaseIndexMapper,
    agent,
    sim_config: SimulationConfig,
    dataset: str,
    method: str,
    run_id: int,
) -> tuple[MetricsResult, int]:
    """Run a single simulation and return ``(metrics, decisions)``.

    Loop structure:

    1. While simulated time < ``sim_config.total_timesteps``:
        a. For each intersection: read state, build prompt, call agent,
           parse response, set phase. Log prompt + response.
        b. ``set_phase`` advances simulation by ~35s (change) or 30s (hold).
    2. After loop: collect ATT/AQL/AWT via ``MetricsEvaluator.evaluate``
       (uses engine accessor for ATT and per-step lane queue history).

    Returns:
        ``(MetricsResult, total_decisions)``.
    """
    intersections = phase_mapper.all_intersections()
    if not intersections:
        raise RuntimeError(
            f"run_simulation: no controllable intersections for {dataset!r}"
        )

    total_steps = sim_config.total_timesteps
    decision_cycle_change = (
        sim_config.green_duration
        + sim_config.yellow_duration
        + sim_config.all_red_duration
    )
    # Track per-step lane queues for AQL.
    lane_queues_per_step: list[dict[str, int]] = []
    elapsed = 0
    decisions = 0
    # Per-intersection cumulative phase time (resets on phase change).
    phase_time: dict[str, int] = {i: 0 for i in intersections}
    current_phase: dict[str, str] = {i: "ETWT" for i in intersections}

    while elapsed < total_steps:
        # One decision pass across all intersections.
        for intersection_id in intersections:
            try:
                lane_counts = engine.get_lane_vehicle_count()
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "run_simulation: get_lane_vehicle_count failed at t=%d "
                    "intersection=%s: %s; using empty dict for prompt.",
                    elapsed,
                    intersection_id,
                    exc,
                )
                lane_counts = {}

            state = {
                "lane_vehicle_count": lane_counts,
                "current_phase": current_phase[intersection_id],
                "current_phase_time": phase_time[intersection_id],
            }
            prompt = obs_parser.parse(state)
            raw_response = agent.generate(prompt)
            parsed_phase = response_parser.parse(raw_response)

            # Persist prompt for UI Demo (Task 18.2).
            try:
                log_llm_prompt(
                    dataset=dataset,
                    method=method,
                    run_id=run_id,
                    timestep=elapsed,
                    intersection_id=intersection_id,
                    prompt=prompt,
                    raw_response=raw_response,
                    parsed_phase=parsed_phase,
                )
            except OSError as exc:
                logger.warning(
                    "run_simulation: failed to log LLM prompt: %s", exc
                )

            phase_idx = phase_mapper.get_index(intersection_id, parsed_phase)
            previous = current_phase[intersection_id]
            engine.set_phase(intersection_id, phase_idx)
            decisions += 1

            if parsed_phase != previous:
                current_phase[intersection_id] = parsed_phase
                phase_time[intersection_id] = sim_config.green_duration
            else:
                phase_time[intersection_id] += sim_config.green_duration

        # Sample lane queues once per decision cycle (sufficient for AQL).
        try:
            lane_queues_per_step.append(dict(engine.get_lane_vehicle_count()))
        except Exception:  # noqa: BLE001
            lane_queues_per_step.append({})

        elapsed += decision_cycle_change

    metrics = MetricsEvaluator(total_timesteps=total_steps).evaluate(
        engine=engine, lane_queues_per_step=lane_queues_per_step
    )
    return metrics, decisions


# ---------------------------------------------------------------------------
# Per-method orchestration
# ---------------------------------------------------------------------------


def run_method_runs(
    *,
    method: str,
    args: argparse.Namespace,
    sim_config: SimulationConfig,
    save_replay: bool,
    llmtscs_dir: str,
    hf_cache: str,
    hf_token: str | None,
    phase_label: str,
    seed_manager: SeedManager,
    cityflow_engine_factory=None,
    inference_factory=None,
) -> list[str]:
    """Run all ``args.num_runs`` runs for one variant.

    Args:
        method: ``lightgpt_hf`` or ``lightgpt_mine``.
        args: Parsed CLI namespace.
        sim_config: Loaded :class:`SimulationConfig`.
        save_replay: Already-resolved boolean flag.
        llmtscs_dir: Path to LLMTSCS repo.
        hf_cache: Cache dir for HF downloads.
        hf_token: ``HF_TOKEN`` for HF download (only used by ``lightgpt_hf``).
        phase_label: ``"PhaseN"`` from gate.
        seed_manager: Already-instantiated :class:`SeedManager`.
        cityflow_engine_factory: Optional override for :class:`CityFlowEngine`
            (tests inject a fake to skip native cityflow). Default lazy-imports
            the real wrapper.
        inference_factory: Optional override for :class:`LightGPTInference`
            (tests inject a fake to skip transformers). Default lazy-imports
            the real engine.

    Returns:
        List of absolute paths of written ``ExperimentResult`` JSON files.
    """
    # Lazy import so that test runs that mock these modules never trigger
    # heavy imports on Windows dev box.
    if cityflow_engine_factory is None:
        from src.cityflow_engine import CityFlowEngine as _CFE  # noqa: WPS433

        cityflow_engine_factory = _CFE
    if inference_factory is None:
        from src.lightgpt_inference import LightGPTInference as _LGI  # noqa

        inference_factory = _LGI

    written_paths: list[str] = []

    obs_parser = ObservationParser()
    response_parser = ResponseParser()

    for run_id in range(args.num_runs):
        seed = seed_manager.seed_for_run(run_id)
        seed_manager.apply(seed)

        cityflow_config_path = build_cityflow_config_path(
            args.dataset, llmtscs_dir=llmtscs_dir, seed=seed
        )

        logger.info(
            "run_lightgpt: starting method=%s dataset=%s run_id=%d seed=%d "
            "phase=%s save_replay=%s",
            method,
            args.dataset,
            run_id,
            seed,
            phase_label,
            save_replay,
        )

        # ---- Resolve roadnet path for PhaseIndexMapper ------------------
        subdir, roadnet_name, _flow = DATASET_FILES[args.dataset]
        roadnet_path = (
            Path(llmtscs_dir).resolve() / "data" / subdir / roadnet_name
        )
        phase_mapper = PhaseIndexMapper(str(roadnet_path))

        # ---- Initialize CityFlow engine ---------------------------------
        engine = cityflow_engine_factory(
            config_path=cityflow_config_path,
            seed=seed,
            save_replay=save_replay,
            dataset=args.dataset,
            method=method,
            run_id=run_id,
            green_duration=sim_config.green_duration,
            yellow_duration=sim_config.yellow_duration,
            all_red_duration=sim_config.all_red_duration,
        )

        # ---- Initialize LightGPT agent ----------------------------------
        agent = inference_factory(
            variant=method,
            cache_dir=hf_cache,
            hf_token=hf_token,
            device="cuda:0",
        )
        agent.load_model()

        start_time = time.time()
        metrics, decisions = run_simulation(
            engine=engine,
            obs_parser=obs_parser,
            response_parser=response_parser,
            phase_mapper=phase_mapper,
            agent=agent,
            sim_config=sim_config,
            dataset=args.dataset,
            method=method,
            run_id=run_id,
        )
        duration = time.time() - start_time
        logger.info(
            "run_lightgpt: completed method=%s run_id=%d decisions=%d "
            "ATT=%.2f AQL=%.2f AWT=%.2f duration=%.1fs",
            method,
            run_id,
            decisions,
            metrics.att,
            metrics.aql,
            metrics.awt,
            duration,
        )

        replay_file = getattr(engine, "replay_file", None)
        result = ExperimentResult(
            method=method,
            dataset=args.dataset,
            run_id=run_id,
            seed=seed,
            metrics=metrics,
            token_usage=None,  # local inference — no API token tracking
            duration_seconds=duration,
            timestamp=utc_iso_timestamp(),
            phase_label=phase_label,
            replay_file=replay_file,
        )
        written_paths.append(write_experiment_result(result))

    return written_paths


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code (0 = success)."""
    setup_basic_logging()

    args = parse_args(argv)
    mode = resolve_mode(args.mode, args.phase)
    save_replay = resolve_save_replay(args.save_replay, args.phase)

    # ---- 1) SeedManager (apply ngay) -----------------------------------
    seed_manager = SeedManager()
    seed_manager.apply(seed_manager.seed_for_run(0))  # bootstrap

    # ---- 2) PhaseApprovalGate ------------------------------------------
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

    # ---- 3) PreflightChecker (defensive) -------------------------------
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

    # ---- 5) Resolve LLMTSCS dir + HF cache + token ---------------------
    llmtscs_dir = args.llmtscs_dir or os.environ.get("LLMTSCS_DIR")
    if not llmtscs_dir:
        logger.error(
            "LLMTSCS_DIR not set. Pass --llmtscs-dir or set the env var "
            "in .env."
        )
        return 6
    if not Path(llmtscs_dir).is_dir():
        logger.error(
            "LLMTSCS_DIR %r does not point to an existing directory.",
            llmtscs_dir,
        )
        return 6

    hf_cache = (
        args.hf_cache
        or os.environ.get("HF_HOME")
        or str(PROJECT_ROOT / "models" / "hf_cache")
    )
    Path(hf_cache).mkdir(parents=True, exist_ok=True)
    hf_token = os.environ.get("HF_TOKEN") or None

    # ---- 6) Resolve method list ----------------------------------------
    finetuned_exists = Path(SELF_FINETUNED_PATH).is_dir()
    try:
        methods = select_methods(
            args.method, args.phase, finetuned_exists=finetuned_exists
        )
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 7

    if not methods:
        logger.warning(
            "No runnable method after filtering; nothing to do (dataset=%s "
            "phase=%d). Exiting cleanly.",
            args.dataset,
            args.phase,
        )
        return 0

    # ---- 7) Run each method sequentially -------------------------------
    all_paths: list[str] = []
    for method in methods:
        paths = run_method_runs(
            method=method,
            args=args,
            sim_config=sim_config,
            save_replay=save_replay,
            llmtscs_dir=llmtscs_dir,
            hf_cache=hf_cache,
            hf_token=hf_token,
            phase_label=phase_label,
            seed_manager=seed_manager,
        )
        all_paths.extend(paths)

    logger.info(
        "run_lightgpt: completed %d run(s) across %d method(s); files "
        "written to results/metrics/.",
        len(all_paths),
        len(methods),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
