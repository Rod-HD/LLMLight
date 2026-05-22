"""Standalone trajectory collection wrapper."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts._runner_utils import (  # noqa: E402
    DATASET_FILES,
    VALID_DATASETS,
    VALID_MODES,
    VALID_PHASES,
    build_cityflow_config_path,
    resolve_mode,
    setup_basic_logging,
)
from src.cityflow_engine import CityFlowEngine  # noqa: E402
from src.observation_parser import ObservationParser  # noqa: E402
from src.phase_approval_gate import PhaseApprovalGate  # noqa: E402
from src.phase_index_mapper import PhaseIndexMapper  # noqa: E402
from src.response_parser import ResponseParser  # noqa: E402
from src.seed_manager import SeedManager  # noqa: E402
from src.training.multi_backend_api_client import MultiBackendAPIClient  # noqa: E402
from src.training.trajectory_collector import TrajectoryCollector  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="collect_trajectory",
        description="Collect teacher LLM trajectory pairs for IFT.",
    )
    parser.add_argument("--dataset", choices=VALID_DATASETS, required=True)
    parser.add_argument("--num-samples", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--phase", type=int, choices=VALID_PHASES, default=1)
    parser.add_argument("--mode", choices=VALID_MODES, default=None)
    parser.add_argument("--llmtscs-dir", default=None)
    parser.add_argument("--env-file", default=".env")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    setup_basic_logging()
    args = parse_args(argv)
    mode = resolve_mode(args.mode, args.phase)

    seed_manager = SeedManager()
    seed = seed_manager.seed_for_run(0)
    seed_manager.apply(seed)

    gate = PhaseApprovalGate()
    gate.validate_phase(args.phase, args.dataset, mode)
    gate.check_prerequisite(args.phase)

    llmtscs_dir = args.llmtscs_dir or os.environ.get("LLMTSCS_DIR")
    if not llmtscs_dir:
        print("LLMTSCS_DIR is required", file=sys.stderr)
        return 2

    cityflow_config = build_cityflow_config_path(
        args.dataset,
        llmtscs_dir=llmtscs_dir,
        seed=seed,
    )
    subdir, roadnet, _flow = DATASET_FILES[args.dataset]
    mapper = PhaseIndexMapper(str(Path(llmtscs_dir) / "data" / subdir / roadnet))
    intersections = mapper.all_intersections()
    if not intersections:
        print("No intersections found in roadnet", file=sys.stderr)
        return 3

    engine = CityFlowEngine(config_path=cityflow_config, seed=seed)
    client = MultiBackendAPIClient(env_path=args.env_file, mode=mode)
    collector = TrajectoryCollector(
        client,
        ObservationParser(),
        ResponseParser(),
        args.phase,
    )
    samples = collector.collect(
        engine,
        args.num_samples,
        intersection_id=intersections[0],
        phase_index_resolver=mapper.get_index,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            [{"prompt": p, "response": r} for p, r in samples],
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {len(samples)} trajectory samples to {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
