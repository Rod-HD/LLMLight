"""Run the IFT -> CGPR pair collection -> CGPR -> LoRA merge pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.collect_cgpr_pairs import JsonCritic  # noqa: E402
from scripts._runner_utils import VALID_DATASETS, VALID_MODES, VALID_PHASES  # noqa: E402
from src.phase_approval_gate import PhaseApprovalGate  # noqa: E402
from src.preflight_checker import PreflightChecker  # noqa: E402
from src.seed_manager import SeedManager  # noqa: E402
from src.training.cgpr_data_collector import CGPRDataCollector  # noqa: E402
from src.training.cgpr_trainer import CGPRTrainer  # noqa: E402
from src.training.ift_trainer import IFTTrainer, TrainingConfig  # noqa: E402
from src.training.lora_merger import LoRAMerger  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_training",
        description="Run LightGPT self-finetuning pipeline.",
    )
    parser.add_argument("--dataset", choices=VALID_DATASETS, required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--mode", choices=VALID_MODES, default="demo")
    parser.add_argument("--phase", type=int, choices=VALID_PHASES, default=1)
    parser.add_argument("--trajectory-file", required=True)
    parser.add_argument("--critic-file", required=True)
    parser.add_argument("--output-model", default="models/qwen2_finetuned/")
    parser.add_argument("--skip-preflight", action="store_true")
    return parser.parse_args(argv)


def _load_trajectory(path: Path) -> list[tuple[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("--trajectory-file must contain a JSON list")
    rows: list[tuple[str, str]] = []
    for item in payload:
        if isinstance(item, dict):
            prompt = item.get("prompt")
            response = item.get("response") or item.get("raw_response")
            if isinstance(prompt, str) and isinstance(response, str):
                rows.append((prompt, response))
        elif (
            isinstance(item, list)
            and len(item) == 2
            and isinstance(item[0], str)
            and isinstance(item[1], str)
        ):
            rows.append((item[0], item[1]))
    if not rows:
        raise ValueError("No valid (prompt, response) trajectory rows found")
    return rows


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    seed_manager = SeedManager()
    seed_manager.apply(seed_manager.seed_for_run(0))

    gate = PhaseApprovalGate()
    gate.validate_phase(args.phase, args.dataset, args.mode)
    gate.check_prerequisite(args.phase)

    if not args.skip_preflight:
        PreflightChecker().run_all(str(PROJECT_ROOT))

    config = TrainingConfig()
    trajectory = _load_trajectory(Path(args.trajectory_file))

    ift_checkpoint = IFTTrainer(config).train(trajectory, resume_from=args.resume)
    if not Path(ift_checkpoint).exists():
        raise FileNotFoundError(f"IFT checkpoint was not written: {ift_checkpoint}")

    prompts = [prompt for prompt, _response in trajectory]
    critic = JsonCritic(args.critic_file)
    pairs = CGPRDataCollector(ift_checkpoint, critic).collect(prompts)
    if not pairs:
        raise ValueError("CGPR pair collection produced no pairs")

    cgpr_adapter = CGPRTrainer(config).train(
        ift_checkpoint,
        [(p.prompt, p.positive_response, p.negative_response) for p in pairs],
    )
    if not Path(cgpr_adapter).exists():
        raise FileNotFoundError(f"CGPR adapter was not written: {cgpr_adapter}")

    output = LoRAMerger().merge(config.base_model, cgpr_adapter, args.output_model)
    print(f"Wrote merged model to {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
