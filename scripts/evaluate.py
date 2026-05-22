"""Generate comparison reports from ExperimentResult JSON files."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts._runner_utils import VALID_DATASETS, VALID_PHASES  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="evaluate",
        description="Generate method comparison markdown from results/metrics JSON.",
    )
    parser.add_argument("--dataset", choices=VALID_DATASETS, required=True)
    parser.add_argument("--phase", type=int, choices=VALID_PHASES, required=True)
    parser.add_argument("--metrics-dir", default="results/metrics")
    parser.add_argument("--output", default=None)
    return parser.parse_args(argv)


def _load_results(metrics_dir: Path, dataset: str, phase: int) -> list[dict[str, Any]]:
    phase_label = f"Phase{phase}"
    rows: list[dict[str, Any]] = []
    for path in sorted(metrics_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("dataset") != dataset:
            continue
        if payload.get("phase_label") != phase_label:
            continue
        metrics = payload.get("metrics")
        if not isinstance(metrics, dict):
            continue
        rows.append(payload)
    return rows


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(statistics.mean(values)), float(statistics.stdev(values))


def _fmt(values: list[float]) -> str:
    mean, std = _mean_std(values)
    return f"{mean:.2f} +/- {std:.2f}"


def _backend_for(rows: list[dict[str, Any]]) -> str:
    for row in rows:
        usage = row.get("token_usage")
        if isinstance(usage, dict) and usage.get("backend"):
            return str(usage["backend"])
    return "-"


def generate_report(rows: list[dict[str, Any]], dataset: str, phase: int) -> str:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["method"]), []).append(row)

    lines = [
        f"# Comparison Report: {dataset} Phase {phase}",
        "",
        "| Method | ATT (s) | AQL (veh/lane) | AWT (s) | Runs | Backend |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for method in sorted(grouped):
        method_rows = grouped[method]
        att = [float(r["metrics"]["att"]) for r in method_rows]
        aql = [float(r["metrics"]["aql"]) for r in method_rows]
        awt = [float(r["metrics"]["awt"]) for r in method_rows]
        lines.append(
            f"| {method} | {_fmt(att)} | {_fmt(aql)} | {_fmt(awt)} | "
            f"{len(method_rows)} | {_backend_for(method_rows)} |"
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    metrics_dir = Path(args.metrics_dir)
    rows = _load_results(metrics_dir, args.dataset, args.phase)
    if not rows:
        print(
            f"No ExperimentResult JSON files found for dataset={args.dataset} "
            f"phase=Phase{args.phase} in {metrics_dir}",
            file=sys.stderr,
        )
        return 2

    report = generate_report(rows, args.dataset, args.phase)
    output = (
        Path(args.output)
        if args.output
        else metrics_dir / f"comparison_{args.dataset}_phase{args.phase}.md"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    print(report)
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
