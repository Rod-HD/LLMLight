"""Shared helpers for ``scripts/run_lightgpt.py``, ``scripts/run_gpt4o.py``,
and ``scripts/run_baselines.py``.

Centralizes:

* ``resolve_mode``   — CLI > ``DEFAULT_RUN_MODE`` env > phase default.
* ``resolve_save_replay`` — ``auto|on|off`` × phase × ``.streamlit_pref.json``.
* ``build_cityflow_config_path`` — write a temp CityFlow config JSON pointing
  at the LLMTSCS dataset directory and return its absolute path.
* ``DATASET_FILES`` — mapping ``dataset_id → (subdir, roadnet_file, flow_file)``
  resolved per the workspace steering rule (LLMTSCS data layout).
* ``log_llm_prompt`` — write per-decision prompt + raw response to
  ``results/logs/llm_prompts/<dataset>_<method>_<run_id>_t<timestep>.txt``
  (consumed by UI Demo Task 18.2).
* ``write_experiment_result`` — serialize ``ExperimentResult`` to
  ``results/metrics/<method>_<dataset>_phase<N>_run<M>.json``.
* ``utc_iso_timestamp`` — produce ISO 8601 UTC timestamp string.

These helpers contain ZERO heavy dependencies (no torch / no requests / no
cityflow). They import only from stdlib + ``src.sim_config``. This keeps
test surface lean and runner files readable.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.sim_config import ExperimentResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset → LLMTSCS file layout
# ---------------------------------------------------------------------------

#: Layout under ``LLMTSCS/data/``:
#:   ``{subdir}/{roadnet_file}`` and ``{subdir}/{flow_file}``.
#:
#: New York 1 only has ``_double`` / ``_triple`` flow files (no plain
#: ``_real``) per workspace steering rules.
DATASET_FILES: dict[str, tuple[str, str, str]] = {
    "jinan_1": ("Jinan/3_4", "roadnet_3_4.json", "anon_3_4_jinan_real.json"),
    "hangzhou_1": (
        "Hangzhou/4_4",
        "roadnet_4_4.json",
        "anon_4_4_hangzhou_real.json",
    ),
    "newyork_1": (
        "NewYork/28_7",
        "roadnet_28_7.json",
        "anon_28_7_newyork_real_double.json",
    ),
}


VALID_DATASETS: tuple[str, ...] = tuple(DATASET_FILES.keys())
VALID_PHASES: tuple[int, ...] = (1, 2, 3)
VALID_MODES: tuple[str, ...] = ("demo", "full")
VALID_SAVE_REPLAY: tuple[str, ...] = ("auto", "on", "off")


# ---------------------------------------------------------------------------
# Mode resolution
# ---------------------------------------------------------------------------


def resolve_mode(cli_mode: str | None, phase: int) -> str:
    """Resolve simulation mode (``demo`` / ``full``).

    Priority: CLI flag > ``DEFAULT_RUN_MODE`` env > phase default
    (Phase 1 → ``demo``; Phase 2/3 → ``full``).

    Args:
        cli_mode: Value of ``--mode`` from argparse (``None`` if user did not
            pass the flag).
        phase: 1, 2, or 3.

    Returns:
        ``"demo"`` or ``"full"``.

    Raises:
        ValueError: If neither CLI nor env produces a valid value AND phase
            is not 1/2/3.
    """
    if cli_mode is not None:
        cli_mode = cli_mode.strip().lower()
        if cli_mode not in VALID_MODES:
            raise ValueError(
                f"resolve_mode: cli_mode must be one of {VALID_MODES}; "
                f"got {cli_mode!r}"
            )
        return cli_mode

    env_mode = (os.environ.get("DEFAULT_RUN_MODE") or "").strip().lower()
    if env_mode in VALID_MODES:
        return env_mode

    if phase == 1:
        return "demo"
    if phase in (2, 3):
        return "full"
    raise ValueError(
        f"resolve_mode: invalid phase {phase!r}; must be one of {VALID_PHASES}"
    )


# ---------------------------------------------------------------------------
# Save-replay resolution
# ---------------------------------------------------------------------------


def resolve_save_replay(
    value: str,
    phase: int,
    *,
    pref_file: str | os.PathLike[str] | None = None,
) -> bool:
    """Resolve ``--save-replay`` (one of ``auto``, ``on``, ``off``).

    Logic:
        * ``on``  → ``True`` always (overrides everything).
        * ``off`` → ``False`` always.
        * ``auto`` → Phase 1 = ``True``, Phase 2/3 = ``False`` UNLESS
          ``.streamlit_pref.json`` provides an override (key
          ``"save_replay"``: bool). The pref file is written by the UI
          sidebar (Task 18.1 ``SaveReplayPreference``) and is intentionally
          gitignored. Missing/invalid file → fall back to phase default.

    Args:
        value: One of :data:`VALID_SAVE_REPLAY`.
        phase: 1, 2, or 3.
        pref_file: Optional path to the preference JSON file (default
            ``.streamlit_pref.json`` in CWD).

    Returns:
        Resolved boolean.

    Raises:
        ValueError: If ``value`` not in :data:`VALID_SAVE_REPLAY` or phase
            invalid.
    """
    if value not in VALID_SAVE_REPLAY:
        raise ValueError(
            f"resolve_save_replay: value must be one of "
            f"{VALID_SAVE_REPLAY}; got {value!r}"
        )
    if phase not in VALID_PHASES:
        raise ValueError(
            f"resolve_save_replay: phase must be one of {VALID_PHASES}; "
            f"got {phase!r}"
        )

    if value == "on":
        return True
    if value == "off":
        return False

    # value == "auto"
    auto_default = phase == 1  # demo on by default; full off by default

    pref_path = Path(pref_file) if pref_file else Path(".streamlit_pref.json")
    if pref_path.exists():
        try:
            data = json.loads(pref_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "resolve_save_replay: failed to read %s (%s); "
                "falling back to phase default %s.",
                pref_path,
                exc,
                auto_default,
            )
            return auto_default
        ui_value = data.get("save_replay") if isinstance(data, dict) else None
        if isinstance(ui_value, bool):
            return ui_value
        logger.debug(
            "resolve_save_replay: %s did not contain a boolean "
            "'save_replay' key; using phase default %s.",
            pref_path,
            auto_default,
        )

    return auto_default


# ---------------------------------------------------------------------------
# CityFlow config builder
# ---------------------------------------------------------------------------


def build_cityflow_config_path(
    dataset: str,
    *,
    llmtscs_dir: str | os.PathLike[str],
    seed: int,
    interval: float = 1.0,
    rl_traffic_light: bool = True,
    output_dir: str | os.PathLike[str] | None = None,
) -> str:
    """Build a CityFlow ``config.json`` for ``dataset`` and return its path.

    The file is written to ``output_dir`` (default: a temp dir). Caller is
    responsible for deletion if needed; tests pass an explicit ``tmp_path``.

    The CityFlow config is the standard structure used by ``cityflow.Engine``::

        {
          "interval": 1.0,
          "seed": 42,
          "dir": "/path/to/LLMTSCS/data/Jinan/3_4/",
          "roadnetFile": "roadnet_3_4.json",
          "flowFile": "anon_3_4_jinan_real.json",
          "rlTrafficLight": true,
          "saveReplay": false,
          "roadnetLogFile": "/tmp/roadnet_log_<rand>.json",
          "replayLogFile": "/tmp/replay_log_<rand>.txt"
        }

    ``CityFlowEngine`` reads this file, and if ``save_replay=True`` is
    passed to its ``__init__``, it writes a replay-enabled COPY into
    ``results/replays/`` and uses that copy instead. So the file produced
    here always sets ``saveReplay=False`` — the engine flips it on demand.

    Args:
        dataset: Key in :data:`DATASET_FILES`.
        llmtscs_dir: Path to the cloned LLMTSCS repo (contains ``data/``).
        seed: CityFlow seed (passed to ``"seed"`` key).
        interval: Timestep interval in seconds. Default 1.0.
        rl_traffic_light: Enable RL-controlled signals. Default True (matches
            LLMLight paper convention).
        output_dir: Where to write the config JSON. Default: a temp dir.

    Returns:
        Absolute path to the generated config file.

    Raises:
        ValueError: ``dataset`` not in :data:`DATASET_FILES`, or ``seed``
            not int.
        FileNotFoundError: Resolved roadnet/flow file does not exist under
            ``llmtscs_dir/data/``.
    """
    if dataset not in DATASET_FILES:
        raise ValueError(
            f"build_cityflow_config_path: dataset {dataset!r} not in "
            f"{sorted(DATASET_FILES.keys())}"
        )
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(
            f"build_cityflow_config_path: seed must be int; "
            f"got {type(seed).__name__} ({seed!r})"
        )

    subdir, roadnet, flow = DATASET_FILES[dataset]
    llmtscs_path = Path(llmtscs_dir).resolve()
    data_dir = llmtscs_path / "data" / subdir

    roadnet_full = data_dir / roadnet
    flow_full = data_dir / flow
    if not roadnet_full.is_file():
        raise FileNotFoundError(
            "build_cityflow_config_path: roadnet file not found: "
            f"{roadnet_full}"
        )
    if not flow_full.is_file():
        raise FileNotFoundError(
            "build_cityflow_config_path: flow file not found: " f"{flow_full}"
        )

    # CityFlow's "dir" key MUST end with a separator so it concatenates
    # cleanly with roadnetFile / flowFile.
    cityflow_config: dict[str, Any] = {
        "interval": float(interval),
        "seed": int(seed),
        "dir": str(data_dir) + "/",
        "roadnetFile": roadnet,
        "flowFile": flow,
        "rlTrafficLight": bool(rl_traffic_light),
        "saveReplay": False,
        "roadnetLogFile": "_roadnet_log_unused.json",
        "replayLogFile": "_replay_log_unused.txt",
        "laneChange": False,
    }

    if output_dir is None:
        out_dir = Path(tempfile.mkdtemp(prefix="cityflow_cfg_"))
    else:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"cityflow_{dataset}_seed{int(seed)}.json"
    out_path.write_text(json.dumps(cityflow_config, indent=2), encoding="utf-8")
    return str(out_path.resolve())


# ---------------------------------------------------------------------------
# LLM prompt logging
# ---------------------------------------------------------------------------


def log_llm_prompt(
    *,
    dataset: str,
    method: str,
    run_id: int,
    timestep: int,
    intersection_id: str,
    prompt: str,
    raw_response: str,
    parsed_phase: str,
    base_dir: str | os.PathLike[str] = "results/logs/llm_prompts",
) -> str:
    """Append a single LLM prompt + raw response to the per-run log file.

    File path: ``{base_dir}/{dataset}_{method}_{run_id}_t{timestep}.txt``.

    The file is created on first call within a run; subsequent calls in
    the same run append (multiple intersections in the same timestep share
    one file).

    Format::

        ============================================================
        [<UTC ISO timestamp>] intersection=<id> parsed_phase=<phase>
        --- PROMPT ---
        <prompt body>
        --- RESPONSE ---
        <raw response body>
        ============================================================

    Args:
        dataset: ``jinan_1`` / ``hangzhou_1`` / ``newyork_1``.
        method: One of the ``ExperimentResult.method`` values.
        run_id: Run index.
        timestep: Sim timestep at which the prompt was issued.
        intersection_id: ID of the intersection this prompt is for.
        prompt: ``ObservationParser`` output.
        raw_response: Untouched LLM response text (before ``ResponseParser``).
        parsed_phase: Phase chosen by ``ResponseParser``.
        base_dir: Output directory (default
            ``results/logs/llm_prompts``).

    Returns:
        Absolute path of the log file written to.
    """
    out_dir = Path(base_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{dataset}_{method}_{int(run_id)}_t{int(timestep)}.txt"
    path = out_dir / fname

    timestamp = utc_iso_timestamp()
    border = "=" * 60
    block = (
        f"{border}\n"
        f"[{timestamp}] intersection={intersection_id} "
        f"parsed_phase={parsed_phase}\n"
        f"--- PROMPT ---\n{prompt}\n"
        f"--- RESPONSE ---\n{raw_response}\n"
        f"{border}\n"
    )
    with path.open("a", encoding="utf-8") as f:
        f.write(block)

    return str(path.resolve())


# ---------------------------------------------------------------------------
# ExperimentResult JSON output
# ---------------------------------------------------------------------------


def experiment_result_filename(result: ExperimentResult) -> str:
    """Compute the canonical filename for a ``ExperimentResult``.

    Format: ``{method}_{dataset}_phase{N}_run{M}.json``. Stable across
    machines and operating systems (only ASCII underscores + numbers).
    """
    phase_num = result.phase_label.removeprefix("Phase")
    return f"{result.method}_{result.dataset}_phase{phase_num}_run{result.run_id}.json"


def write_experiment_result(
    result: ExperimentResult,
    *,
    base_dir: str | os.PathLike[str] = "results/metrics",
) -> str:
    """Serialize an :class:`ExperimentResult` to JSON and return its path.

    The output dir is created if missing. JSON is written with
    ``indent=2`` for human readability + stable diff.

    Args:
        result: Validated :class:`ExperimentResult` instance.
        base_dir: Output directory (default ``results/metrics``).

    Returns:
        Absolute path of the written JSON file.
    """
    out_dir = Path(base_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = experiment_result_filename(result)
    path = out_dir / fname

    payload = asdict(result)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return str(path.resolve())


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------


def utc_iso_timestamp() -> str:
    """Return UTC timestamp in ISO 8601 (e.g. ``2025-04-15T13:42:00Z``)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def setup_basic_logging(level: int = logging.INFO) -> None:
    """Configure root logger if no handlers are attached yet.

    Idempotent — safe to call from every runner ``main()``. Uses a stable
    format that includes module name + level + message.
    """
    root = logging.getLogger()
    if root.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
    )
    root.addHandler(handler)
    root.setLevel(level)


__all__ = [
    "DATASET_FILES",
    "VALID_DATASETS",
    "VALID_PHASES",
    "VALID_MODES",
    "VALID_SAVE_REPLAY",
    "resolve_mode",
    "resolve_save_replay",
    "build_cityflow_config_path",
    "log_llm_prompt",
    "experiment_result_filename",
    "write_experiment_result",
    "utc_iso_timestamp",
    "setup_basic_logging",
]
