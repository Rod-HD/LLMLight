"""scripts/run_gpt4o.py — main runner for LLM remote inference (Task 13.2).

Wires SeedManager → PhaseApprovalGate → PreflightChecker → SimulationConfig
→ PhaseIndexMapper → CityFlowEngine → ObservationParser →
:class:`MultiBackendAPIClient` → ResponseParser into a single CLI driver.

Backend is auto-selected from the environment with priority
``OPENAI > GROQ > PUTER`` (Component 6). The runner labels the produced
:class:`ExperimentResult.method` accordingly:

* ``OPENAI``  → ``gpt4o_openai``
* ``GROQ``    → ``gpt4o_groq``
* ``PUTER``   → ``gpt4o_puter``

Run-counts per phase (Requirement 7 AC 12):

* Phase 1 → 1 run (cost-saving inside Puter free quota).
* Phase 2 → 3 runs (full mean ± std).
* Phase 3 → 3 runs (full mean ± std).

``--num-runs`` overrides the default if supplied (use sparingly).

Mode-block (Requirement 7 AC 4): ``--mode full`` + Puter combination is
rejected at client construction time. Phase 2 / Phase 3 default mode is
``full``, so Puter is unusable there unless the user explicitly passes
``--mode demo``.

Per-run, the runner logs total tokens + backend after the run completes
(Requirement 7 AC 11).
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Ensure src + scripts importable when invoked as ``python scripts/run_gpt4o.py``.
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
    experiment_result_filename,
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
    APIBackend,
    ExperimentResult,
    MetricsResult,
    SimulationConfig,
    TokenUsageLog,
)
from src.training.api_replay_cache import (  # noqa: E402
    APIDecisionRecord,
    APIReplayCache,
    APIReplayCacheError,
    stable_hash_json,
    stable_hash_text,
)
from src.training.multi_backend_api_client import (  # noqa: E402
    TokenUsageLog as APITokenUsageLog,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Mapping ``APIBackend value → ExperimentResult.method`` (Requirement 7).
BACKEND_TO_METHOD: dict[str, str] = {
    "puter": "gpt4o_puter",
    "groq": "gpt4o_groq",
    "openai": "gpt4o_openai",
    "codexhub": "gpt4o_codexhub",
    "ckey": "gpt4o_codexhub",  # ckey.vn is OpenAI-compatible; reuse codexhub method
}


#: Default thread-pool size for parallel intra-cycle API calls. Override via
#: env ``LLMLIGHT_API_PARALLEL`` (1 = old serial behaviour).
DEFAULT_API_PARALLEL: int = 12


def _api_parallel_workers() -> int:
    try:
        n = int(os.environ.get("LLMLIGHT_API_PARALLEL", DEFAULT_API_PARALLEL))
    except (TypeError, ValueError):
        n = DEFAULT_API_PARALLEL
    return max(1, n)


def num_runs_for_phase(phase: int) -> int:
    """Return default ``num_runs`` for the given phase (Requirement 7 AC 12)."""
    if phase == 1:
        return 1
    if phase in (2, 3):
        return 3
    raise ValueError(
        f"num_runs_for_phase: phase must be in {VALID_PHASES}; got {phase!r}"
    )


# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments.

    ``--num-runs`` defaults to ``None`` so we can apply the phase-aware
    default (Phase 1 = 1, Phase 2/3 = 3) AFTER the phase is known.
    """
    parser = argparse.ArgumentParser(
        prog="run_gpt4o",
        description=(
            "Run GPT-4o (or Llama-3.3 on Groq) over CityFlow simulation. "
            "Backend auto-selected from env vars with priority "
            "OPENAI > GROQ > PUTER."
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
        default=None,
        help=(
            "Number of runs. Default: Phase 1 = 1, Phase 2/3 = 3 "
            "(Requirement 7 AC 12)."
        ),
    )
    parser.add_argument(
        "--start-run-id",
        type=int,
        default=0,
        help=(
            "First run_id to execute (default 0). Set to 1 to run a second "
            "replicate without overwriting the existing run_0 cache."
        ),
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
        "--env-file",
        default=".env",
        help="Path to .env file consumed by MultiBackendAPIClient.",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip PreflightChecker (intended for unit tests / WSL2 reruns).",
    )
    parser.add_argument(
        "--reuse-api-cache",
        default=None,
        help=(
            "Path to an API cache manifest. When set, replay cached "
            "decisions and do not initialise the API client."
        ),
    )
    parser.add_argument(
        "--allow-cache-miss-api-call",
        action="store_true",
        help=(
            "Only with --reuse-api-cache: allow fallback API calls on cache "
            "miss/hash mismatch. Default is to abort before spending quota."
        ),
    )

    args = parser.parse_args(argv)

    if args.num_runs is not None and args.num_runs <= 0:
        parser.error("--num-runs must be > 0")

    return args


# ---------------------------------------------------------------------------
# Single simulation
# ---------------------------------------------------------------------------


def run_simulation(
    *,
    engine,
    obs_parser: ObservationParser,
    response_parser: ResponseParser,
    phase_mapper: PhaseIndexMapper,
    api_client,
    api_cache: APIReplayCache | None = None,
    run_fingerprint: str | None = None,
    sim_config: SimulationConfig,
    dataset: str,
    method: str,
    run_id: int,
    phase: int,
    seed: int = 42,
    mode: str = "demo",
    model: str = "",
    backend: object = "none",
    allow_cache_miss_api_call: bool = False,
) -> tuple[MetricsResult, int]:
    """Drive simulation through ``api_client`` until ``total_timesteps``.

    Mirrors :func:`scripts.run_lightgpt.run_simulation` but calls
    :meth:`MultiBackendAPIClient.chat_completion` instead of
    ``LightGPTInference.generate``. Empty content from the API → falls
    back to ETWT via :class:`ResponseParser`.
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
    lane_queues_per_step: list[dict[str, int]] = []
    # Cumulative wait time per vehicle. We increment by ``decision_cycle_change``
    # seconds each cycle for every vehicle whose speed < SPEED_THRESHOLD at the
    # cycle boundary. This is a coarse proxy (one sample per decision cycle,
    # not per simulation step) but matches the granularity of the rest of the
    # runner, and the resulting AWT will populate the previously-zero
    # ``vehicle_wait_times`` channel in MetricsEvaluator.
    vehicle_wait_seconds: dict[str, float] = {}
    elapsed = 0
    decisions = 0
    phase_time: dict[str, int] = {i: 0 for i in intersections}
    current_phase: dict[str, str] = {i: "ETWT" for i in intersections}

    # Concurrency for intra-cycle API calls. CityFlow engine itself is NOT
    # thread-safe (state read + set_phase remain serial); only the LLM calls
    # are parallelised. Default = number of intersections (one worker per
    # call) so a full cycle finishes in ~one round-trip latency.
    # The MultiBackendAPIClient is not strictly thread-safe but the shared
    # state (counters, key rotation index, inter-request delay) is benign
    # under contention — torn updates only affect logs/metrics, not correctness.
    parallel_workers = max(1, min(_api_parallel_workers(), len(intersections)))
    cache_lock = threading.Lock()

    def _fetch_decision(
        intersection_id: str,
        prompt: str,
        state_hash: str,
        prompt_hash: str,
    ) -> dict:
        """Resolve a single decision (cache hit or fresh API call).

        Safe to call from worker threads — the API client and cache append
        are serialised by the locks above. Returns a dict with keys
        ``parsed_phase, raw_response, latency_ms, started_at, received_at,
        input_tokens, output_tokens, fallback_used, error_type``.
        """
        started_at = utc_iso_timestamp()
        try:
            if api_cache is not None:
                with cache_lock:
                    record = api_cache.get_decision(
                        elapsed,
                        intersection_id,
                        state_hash,
                        prompt_hash,
                    )
                return {
                    "raw_response": record.raw_response,
                    "parsed_phase": record.parsed_phase,
                    "received_at": record.response_received_at,
                    "latency_ms": record.latency_ms,
                    "started_at": started_at,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "fallback_used": False,
                    "error_type": None,
                }
            raise APIReplayCacheError("fresh API run")
        except APIReplayCacheError:
            if api_client is None:
                raise RuntimeError(
                    "run_simulation: api_client is None but cache "
                    f"miss for t={elapsed} intersection={intersection_id!r}"
                )
            if api_cache is not None and not allow_cache_miss_api_call:
                raise
            request_start = time.monotonic()
            api_response = api_client.chat_completion(prompt)
            latency_ms = max(0, int((time.monotonic() - request_start) * 1000))
            raw_response = api_response.content
            parsed_phase = response_parser.parse(raw_response)
            received_at = utc_iso_timestamp()
            input_tokens = int(api_response.input_tokens)
            output_tokens = int(api_response.output_tokens)
            fallback_used = raw_response == ""
            error_type = "empty_response" if fallback_used else None

            if api_cache is not None and run_fingerprint is not None:
                rec = APIDecisionRecord(
                    run_fingerprint=run_fingerprint,
                    dataset=dataset,
                    phase=phase,
                    mode=mode,
                    method=method,
                    backend=backend,
                    model=model,
                    run_id=run_id,
                    seed=seed,
                    timestep=elapsed,
                    intersection_id=intersection_id,
                    state_hash=state_hash,
                    prompt_hash=prompt_hash,
                    prompt=prompt,
                    raw_response=raw_response,
                    parsed_phase=parsed_phase,
                    fallback_used=fallback_used,
                    error_type=error_type,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    request_started_at=started_at,
                    response_received_at=received_at,
                    latency_ms=latency_ms,
                )
                with cache_lock:
                    api_cache.append(rec)
            return {
                "raw_response": raw_response,
                "parsed_phase": parsed_phase,
                "received_at": received_at,
                "latency_ms": latency_ms,
                "started_at": started_at,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "fallback_used": fallback_used,
                "error_type": error_type,
            }

    executor = (
        ThreadPoolExecutor(max_workers=parallel_workers)
        if parallel_workers > 1
        else None
    )
    try:
        while elapsed < total_steps:
            # Phase 1: build prompts for every intersection (serial — engine
            # is not thread-safe). lane_counts is shared across all of them
            # within a single cycle (one snapshot per cycle).
            try:
                lane_counts = engine.get_lane_vehicle_count()
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "run_simulation: get_lane_vehicle_count failed at t=%d: "
                    "%s; using empty dict.",
                    elapsed,
                    exc,
                )
                lane_counts = {}

            prompts: dict[str, tuple[dict, str, str, str]] = {}
            for intersection_id in intersections:
                state = {
                    "lane_vehicle_count": lane_counts,
                    "current_phase": current_phase[intersection_id],
                    "current_phase_time": phase_time[intersection_id],
                }
                prompt = obs_parser.parse(state)
                state_hash = stable_hash_json(state)
                prompt_hash = stable_hash_text(prompt)
                prompts[intersection_id] = (state, prompt, state_hash, prompt_hash)

            # Phase 2: resolve every decision (parallel API calls when needed).
            results: dict[str, dict] = {}
            if executor is None:
                for intersection_id in intersections:
                    _, prompt, sh, ph = prompts[intersection_id]
                    results[intersection_id] = _fetch_decision(
                        intersection_id, prompt, sh, ph,
                    )
            else:
                future_to_id = {
                    executor.submit(
                        _fetch_decision,
                        iid,
                        prompts[iid][1],
                        prompts[iid][2],
                        prompts[iid][3],
                    ): iid
                    for iid in intersections
                }
                for fut in future_to_id:
                    iid = future_to_id[fut]
                    results[iid] = fut.result()

            # Phase 3: log + apply phase (serial — engine.set_phase not
            # thread-safe; preserve original intersection order so behaviour
            # matches the serial loop).
            for intersection_id in intersections:
                _state, prompt, _sh, _ph = prompts[intersection_id]
                outcome = results[intersection_id]
                raw_response = outcome["raw_response"]
                parsed_phase = outcome["parsed_phase"]
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

            try:
                lane_queues_per_step.append(dict(engine.get_lane_vehicle_count()))
            except Exception:  # noqa: BLE001
                lane_queues_per_step.append({})

            # Sample per-vehicle speeds at the cycle boundary and accumulate
            # cumulative wait time. SPEED_THRESHOLD matches MetricsEvaluator.
            raw_engine = getattr(engine, "_engine", engine)
            try:
                speeds = raw_engine.get_vehicle_speed()
            except Exception:  # noqa: BLE001
                speeds = {}
            if speeds:
                for vid, spd in speeds.items():
                    if spd < MetricsEvaluator.SPEED_THRESHOLD:
                        vehicle_wait_seconds[str(vid)] = (
                            vehicle_wait_seconds.get(str(vid), 0.0)
                            + float(decision_cycle_change)
                        )
                    else:
                        vehicle_wait_seconds.setdefault(str(vid), 0.0)

            elapsed += decision_cycle_change

            if elapsed % (decision_cycle_change * 5) == 0:
                logger.info(
                    "run_simulation: progress dataset=%s run_id=%d t=%d/%d "
                    "decisions=%d",
                    dataset, run_id, elapsed, total_steps, decisions,
                )
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    metrics = MetricsEvaluator(total_timesteps=total_steps).evaluate(
        engine=engine,
        lane_queues_per_step=lane_queues_per_step,
        vehicle_wait_times=list(vehicle_wait_seconds.values()),
    )
    return metrics, decisions


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_run_fingerprint(
    *,
    dataset: str,
    phase: int,
    mode: str,
    method: str,
    run_id: int,
    seed: int,
    roadnet_sha256: str,
    flow_sha256: str,
    simulation_config_sha256: str,
) -> str:
    """Build a stable run fingerprint for API cache manifests."""
    return stable_hash_json(
        {
            "dataset": dataset,
            "phase": phase,
            "mode": mode,
            "method": method,
            "run_id": run_id,
            "seed": seed,
            "roadnet_sha256": roadnet_sha256,
            "flow_sha256": flow_sha256,
            "simulation_config_sha256": simulation_config_sha256,
        }
    )


# ---------------------------------------------------------------------------
# Backend label conversion
# ---------------------------------------------------------------------------


def _backend_value(backend: object) -> str:
    """Coerce a backend marker (Enum or str) to its lowercase string value."""
    val = getattr(backend, "value", backend)
    return str(val).lower()


def _method_for_backend(backend: object) -> str:
    val = _backend_value(backend)
    method = BACKEND_TO_METHOD.get(val)
    if method is None:
        raise ValueError(
            f"_method_for_backend: backend {backend!r} (value={val!r}) is "
            f"not in {sorted(BACKEND_TO_METHOD.keys())}"
        )
    return method


def _backend_for_token_usage(backend: object) -> APIBackend:
    """Cast backend marker to one of the literals accepted by
    :class:`TokenUsageLog`."""
    val = _backend_value(backend)
    if val not in ("puter", "groq", "openai", "codexhub", "ckey"):
        raise ValueError(
            f"_backend_for_token_usage: invalid backend {backend!r}"
        )
    return val  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


def main(
    argv: list[str] | None = None,
    *,
    cityflow_engine_factory=None,
    api_client_factory=None,
) -> int:
    """Entry point. Returns a process exit code (0 = success).

    Args:
        argv: CLI arguments (default ``sys.argv[1:]``).
        cityflow_engine_factory: Optional override (tests inject a fake to
            skip the native cityflow binding). Default lazy-imports
            :class:`CityFlowEngine`.
        api_client_factory: Optional override (tests inject a fake to skip
            HTTP layer). Default lazy-imports
            :class:`MultiBackendAPIClient`.
    """
    setup_basic_logging()

    args = parse_args(argv)
    mode = resolve_mode(args.mode, args.phase)
    save_replay = resolve_save_replay(args.save_replay, args.phase)

    # ---- 1) SeedManager (apply ngay) -----------------------------------
    seed_manager = SeedManager()
    seed_manager.apply(seed_manager.seed_for_run(0))

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

    # ---- 6) Build API client or load replay cache ----------------------
    reuse_cache: APIReplayCache | None = None
    reuse_manifest = None
    if args.reuse_api_cache:
        try:
            reuse_cache = APIReplayCache()
            reuse_manifest = reuse_cache.load_manifest(args.reuse_api_cache)
        except (OSError, ValueError, APIReplayCacheError) as exc:
            logger.error("Failed to load API replay cache: %s", exc)
            return 8

    if api_client_factory is None and (
        reuse_manifest is None or args.allow_cache_miss_api_call
    ):
        from src.training.multi_backend_api_client import (  # noqa: WPS433
            MultiBackendAPIClient as _MBC,
        )

        api_client_factory = _MBC

    api_client = None
    if reuse_manifest is None or args.allow_cache_miss_api_call:
        try:
            api_client = api_client_factory(env_path=args.env_file, mode=mode)
        except ValueError as exc:
            logger.error("Failed to initialise API client: %s", exc)
            return 8

    if reuse_manifest is not None:
        method = reuse_manifest.method
        backend = reuse_manifest.backend
        model = reuse_manifest.model
    else:
        method = _method_for_backend(api_client.backend)
        backend = api_client.backend
        model = api_client.model

    # ---- 7) Resolve num_runs (phase-aware default) ---------------------
    num_runs = args.num_runs if args.num_runs is not None else num_runs_for_phase(args.phase)
    start = max(0, int(args.start_run_id))
    run_ids = (
        [reuse_manifest.run_id]
        if reuse_manifest is not None
        else list(range(start, start + num_runs))
    )

    # ---- 8) Lazy-import CityFlow engine factory ------------------------
    if cityflow_engine_factory is None:
        from src.cityflow_engine import CityFlowEngine as _CFE  # noqa: WPS433

        cityflow_engine_factory = _CFE

    obs_parser = ObservationParser()
    response_parser = ResponseParser()

    # Resolve roadnet path for PhaseIndexMapper.
    subdir, roadnet_name, flow_name = DATASET_FILES[args.dataset]
    data_dir = Path(llmtscs_dir).resolve() / "data" / subdir
    roadnet_path = data_dir / roadnet_name
    flow_path = data_dir / flow_name
    roadnet_sha256 = _sha256_file(roadnet_path)
    flow_sha256 = _sha256_file(flow_path)
    simulation_config_sha256 = _sha256_file(Path(args.simulation_config))

    # ---- 9) Run loop ----------------------------------------------------
    written_paths: list[str] = []
    for run_id in run_ids:
        seed = seed_manager.seed_for_run(run_id)
        seed_manager.apply(seed)
        run_fingerprint = (
            reuse_manifest.run_fingerprint
            if reuse_manifest is not None
            else build_run_fingerprint(
                dataset=args.dataset,
                phase=args.phase,
                mode=mode,
                method=method,
                run_id=run_id,
                seed=seed,
                roadnet_sha256=roadnet_sha256,
                flow_sha256=flow_sha256,
                simulation_config_sha256=simulation_config_sha256,
            )
        )
        run_cache = reuse_cache if reuse_cache is not None else APIReplayCache()
        # Resume support: load any existing JSONL records for this run so we
        # don't re-spend API quota on previously-cached decisions.
        if reuse_cache is None:
            try:
                loaded = run_cache.load_existing(
                    dataset=args.dataset,
                    method=method,
                    phase=args.phase,
                    run_id=run_id,
                )
                if loaded:
                    logger.info(
                        "run_gpt4o: resuming run_id=%d with %d cached decisions.",
                        run_id,
                        loaded,
                    )
            except (OSError, ValueError, APIReplayCacheError) as exc:
                logger.warning(
                    "run_gpt4o: could not load existing cache for run_id=%d: %s",
                    run_id,
                    exc,
                )

        cityflow_config_path = build_cityflow_config_path(
            args.dataset, llmtscs_dir=llmtscs_dir, seed=seed
        )

        logger.info(
            "run_gpt4o: starting method=%s dataset=%s run_id=%d seed=%d "
            "phase=%s save_replay=%s mode=%s",
            method,
            args.dataset,
            run_id,
            seed,
            phase_label,
            save_replay,
            mode,
        )

        phase_mapper = PhaseIndexMapper(str(roadnet_path))

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

        start_time = time.time()
        metrics, decisions = run_simulation(
            engine=engine,
            obs_parser=obs_parser,
            response_parser=response_parser,
            phase_mapper=phase_mapper,
            api_client=api_client,
            sim_config=sim_config,
            dataset=args.dataset,
            method=method,
            run_id=run_id,
            phase=args.phase,
            seed=seed,
            mode=mode,
            model=model,
            backend=backend,
            api_cache=run_cache,
            run_fingerprint=run_fingerprint,
            allow_cache_miss_api_call=(
                args.allow_cache_miss_api_call or reuse_manifest is None
            ),
        )
        duration = time.time() - start_time

        # Move CityFlow replay artefacts out of the ASCII staging dir.
        try:
            moved = engine.finalize_replay() if hasattr(engine, "finalize_replay") else None
            if moved:
                logger.info("run_gpt4o: replay file finalized at %s", moved)
        except Exception as exc:  # noqa: BLE001
            logger.warning("run_gpt4o: finalize_replay failed: %s", exc)

        usage = (
            reuse_manifest.token_usage
            if api_client is None
            else api_client.get_usage_log()
        )
        # When a fresh run resumed from a partially-filled cache, the live
        # client usage only counts the NEW calls. The on-disk JSONL is the
        # source of truth — re-aggregate across every cached decision so the
        # manifest reflects the full run cost.
        if reuse_manifest is None:
            cached_in, cached_out, cached_count = run_cache.aggregate_token_usage()
            usage = APITokenUsageLog(
                total_input_tokens=cached_in,
                total_output_tokens=cached_out,
                total_requests=cached_count,
                backend=backend,
            )
        total_tokens = usage.total_input_tokens + usage.total_output_tokens
        logger.info(
            "run_gpt4o: completed method=%s run_id=%d decisions=%d "
            "ATT=%.2f AQL=%.2f AWT=%.2f tokens=%d backend=%s duration=%.1fs",
            method,
            run_id,
            decisions,
            metrics.att,
            metrics.aql,
            metrics.awt,
            total_tokens,
            _backend_value(backend),
            duration,
        )

        token_usage = TokenUsageLog(
            backend=_backend_for_token_usage(backend),
            total_input_tokens=int(usage.total_input_tokens),
            total_output_tokens=int(usage.total_output_tokens),
            total_requests=int(usage.total_requests),
        )
        replay_file = getattr(engine, "replay_file", None)
        result = ExperimentResult(
            method=method,
            dataset=args.dataset,
            run_id=run_id,
            seed=seed,
            metrics=metrics,
            token_usage=token_usage,
            duration_seconds=duration,
            timestamp=utc_iso_timestamp(),
            phase_label=phase_label,
            replay_file=replay_file,
            api_cache_manifest=None,
            run_fingerprint=run_fingerprint,
        )
        metrics_file = (
            Path("results/metrics") / experiment_result_filename(result)
        )
        manifest_path = None
        if reuse_manifest is not None:
            manifest_path = args.reuse_api_cache
        else:
            api_usage = APITokenUsageLog(
                total_input_tokens=int(usage.total_input_tokens),
                total_output_tokens=int(usage.total_output_tokens),
                total_requests=int(usage.total_requests),
                backend=backend,
            )
            manifest = run_cache.build_manifest(
                dataset=args.dataset,
                method=method,
                phase=args.phase,
                mode=mode,
                backend=backend,
                model=model,
                run_id=run_id,
                seed=seed,
                run_fingerprint=run_fingerprint,
                roadnet_path=str(roadnet_path),
                roadnet_sha256=roadnet_sha256,
                flow_path=str(flow_path),
                flow_sha256=flow_sha256,
                simulation_config_sha256=simulation_config_sha256,
                prompt_template_version="observation-parser-v1",
                code_commit=None,
                replay_file=replay_file,
                metrics_file=str(metrics_file),
                llm_log_dir="results/logs/llm_prompts",
                token_usage=api_usage,
            )
            manifest_path = str(run_cache.write_manifest(manifest))

        result.api_cache_manifest = manifest_path
        written_paths.append(write_experiment_result(result))

    logger.info(
        "run_gpt4o: completed %d run(s); files written to results/metrics/.",
        len(written_paths),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
