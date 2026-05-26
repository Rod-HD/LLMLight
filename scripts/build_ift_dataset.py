"""Build IFT dataset from API replay cache JSONL files.

Collects (prompt, raw_response) pairs from all completed API calls in
results/api_cache/<dataset>/gpt4o_*/phase2/run_N.jsonl, filters by:

  - response must contain a valid <signal> tag (one of ETWT/NTST/ELWL/NLSL)
  - response must not be a fallback (fallback_used == False)
  - error_type must be None

Produces a single JSON file consumable by scripts/run_training.py:

    [
      {
        "prompt": "...",
        "response": "<signal>ETWT</signal>",
        "dataset": "hangzhou_1",
        "run_id": 0,
        "intersection_id": "intersection_1_1",
        "timestep": 0
      },
      ...
    ]

Also writes a sidecar `<output>.summary.json` with per-source counts and a
deterministic SHA-256 of the produced dataset for reproducibility.

Usage:
    python scripts/build_ift_dataset.py \\
        --cache-roots results/api_cache/hangzhou_1 results/api_cache/jinan_1 \\
        --output data/ift_dataset_v1.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("build_ift_dataset")

VALID_PHASES = {"ETWT", "NTST", "ELWL", "NLSL"}
_SIGNAL_RE = re.compile(r"<signal>\s*([A-Z]{4})\s*</signal>", re.IGNORECASE)


def _is_valid_response(raw: str) -> bool:
    if not isinstance(raw, str) or not raw:
        return False
    m = _SIGNAL_RE.search(raw)
    if m is None:
        return False
    return m.group(1).strip().upper() in VALID_PHASES


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("skip %s:%d JSON error: %s", path, line_no, exc)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--cache-roots",
        nargs="+",
        required=True,
        help="One or more directories like results/api_cache/<dataset>",
    )
    p.add_argument(
        "--output",
        required=True,
        help="Where to write the consolidated JSON dataset.",
    )
    p.add_argument(
        "--include-fallback",
        action="store_true",
        help="Include rows where fallback_used==True (default: skip).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args(argv)

    rows: list[dict] = []
    per_source: Counter[str] = Counter()
    per_dataset: Counter[str] = Counter()
    skipped: Counter[str] = Counter()

    for root_str in args.cache_roots:
        root = Path(root_str)
        if not root.is_dir():
            logger.warning("cache root missing: %s", root)
            continue
        for jsonl_path in sorted(root.rglob("*.jsonl")):
            logger.info("reading %s", jsonl_path)
            for rec in _iter_jsonl(jsonl_path):
                prompt = rec.get("prompt")
                raw = rec.get("raw_response")
                if not isinstance(prompt, str) or not isinstance(raw, str):
                    skipped["bad_types"] += 1
                    continue
                if rec.get("error_type"):
                    skipped["error_type"] += 1
                    continue
                if not args.include_fallback and rec.get("fallback_used"):
                    skipped["fallback"] += 1
                    continue
                if not _is_valid_response(raw):
                    skipped["invalid_signal"] += 1
                    continue
                rows.append(
                    {
                        "prompt": prompt,
                        "response": raw,
                        "dataset": rec.get("dataset", ""),
                        "run_id": rec.get("run_id", -1),
                        "intersection_id": rec.get("intersection_id", ""),
                        "timestep": rec.get("timestep", -1),
                    }
                )
                try:
                    rel = jsonl_path.resolve().relative_to(PROJECT_ROOT)
                except ValueError:
                    rel = jsonl_path
                per_source[str(rel)] += 1
                per_dataset[rec.get("dataset", "")] += 1

    if not rows:
        logger.error("no valid samples collected; abort.")
        return 2

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(rows, ensure_ascii=False, indent=2)
    output.write_text(payload, encoding="utf-8")
    sha = hashlib.sha256(payload.encode("utf-8")).hexdigest()

    summary = {
        "output": str(output),
        "sha256": sha,
        "total_samples": len(rows),
        "per_dataset": dict(per_dataset),
        "per_source": dict(per_source),
        "skipped": dict(skipped),
        "include_fallback": args.include_fallback,
    }
    summary_path = output.with_suffix(output.suffix + ".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    logger.info("wrote %d samples to %s", len(rows), output)
    logger.info("summary: %s", summary_path)
    logger.info("per dataset: %s", dict(per_dataset))
    logger.info("skipped: %s", dict(skipped))
    logger.info("sha256: %s", sha)
    return 0


if __name__ == "__main__":
    sys.exit(main())
