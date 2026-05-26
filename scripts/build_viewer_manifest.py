"""Build viewer/results/replays/manifest.json for the side-by-side viewer.

Scans `results/replays/` for matching pairs of `<base>.txt` (replay) +
`<base>_roadnet.json` (roadnet) and matches each pair with the corresponding
`results/metrics/<method>_<dataset>_phase<N>_run<N>.json` when present.

Run from project root:

    python scripts/build_viewer_manifest.py

The viewer's `main.js` fetches `../results/replays/manifest.json` relative to
`viewer/index.html`, so the file lands at `results/replays/manifest.json`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

REPLAY_DIR = Path("results/replays")
METRICS_DIR = Path("results/metrics")
SIM_CONFIG = Path("config/simulation.json")
OUTPUT = REPLAY_DIR / "manifest.json"


def _load_decision_cycle() -> int:
    """Read green + yellow + all_red from simulation config."""
    try:
        cfg = json.loads(SIM_CONFIG.read_text(encoding="utf-8"))
        return int(cfg["green_duration"]) + int(cfg["yellow_duration"]) + int(cfg["all_red_duration"])
    except (OSError, KeyError, json.JSONDecodeError):
        return 35  # safe default

# Base filename convention from src/cityflow_engine.py:
#   {dataset}_{method}_run{N}.txt
# where dataset = "hangzhou_1" / "jinan_1" / "newyork_1" and method may itself
# contain underscores (e.g. "gpt4o_codexhub", "advanced_maxpressure").
_DATASET_PREFIXES = ("hangzhou_", "jinan_", "newyork_")


def _split_base(base: str) -> tuple[str, str, int] | None:
    """Parse "<dataset>_<method>_run<N>" → (dataset, method, run_id)."""
    m = re.match(r"^(?P<dataset>[a-z]+_\d+)_(?P<method>.+)_run(?P<run>\d+)$", base)
    if not m:
        return None
    return m.group("dataset"), m.group("method"), int(m.group("run"))


def _find_metrics(method: str, dataset: str, run_id: int) -> Path | None:
    """Find metrics JSON. Try Phase2 first (current target), then Phase1/3."""
    for phase in (2, 1, 3):
        candidate = METRICS_DIR / f"{method}_{dataset}_phase{phase}_run{run_id}.json"
        if candidate.exists():
            return candidate
    return None


def main() -> int:
    if not REPLAY_DIR.exists():
        print(f"[error] {REPLAY_DIR} does not exist")
        return 1

    pairs = []
    for replay in sorted(REPLAY_DIR.glob("*.txt")):
        base = replay.stem
        if not any(base.startswith(p) for p in _DATASET_PREFIXES):
            continue
        roadnet = REPLAY_DIR / f"{base}_roadnet.json"
        if not roadnet.exists():
            print(f"[skip] {base}: no roadnet file")
            continue
        parsed = _split_base(base)
        if not parsed:
            print(f"[skip] {base}: cannot parse dataset/method/run")
            continue
        dataset, method, run_id = parsed
        metrics = _find_metrics(method, dataset, run_id)
        size_mb = replay.stat().st_size / (1024 * 1024)
        label = f"{method} · {dataset} · run{run_id} ({size_mb:.0f} MB)"
        pairs.append({
            "label": label,
            "dataset": dataset,
            "method": method,
            "run_id": run_id,
            "size_mb": round(size_mb, 1),
            # Paths are relative to viewer/index.html (one level up from viewer/).
            "roadnet": f"../{roadnet.as_posix()}",
            "replay": f"../{replay.as_posix()}",
            "metrics": f"../{metrics.as_posix()}" if metrics else None,
        })

    decision_cycle = _load_decision_cycle()
    manifest = {
        "generated_by": "scripts/build_viewer_manifest.py",
        "count": len(pairs),
        "decisionCycle": decision_cycle,
        "pairs": pairs,
    }
    OUTPUT.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[ok] wrote {OUTPUT} with {len(pairs)} pair(s)")
    for p in pairs:
        marker = "[m]" if p["metrics"] else "[ ]"
        print(f"  {marker} {p['label']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
