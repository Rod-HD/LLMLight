"""Collect CGPR ranking pairs from prompts and an IFT checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.cgpr_data_collector import CGPRDataCollector  # noqa: E402


class JsonCritic:
    """Small file-backed critic used by the standalone collection script."""

    def __init__(self, path: str, default_phase: str = "ETWT") -> None:
        self.default_phase = default_phase
        self.mapping: dict[str, str] = {}
        p = Path(path)
        if p.is_file():
            payload = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                self.mapping = {str(k): str(v) for k, v in payload.items()}

    def predict_phase(self, prompt: str) -> str:
        return self.mapping.get(prompt, self.default_phase)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="collect_cgpr_pairs",
        description="Build CGPR pairs from prompts, IFT model, and critic outputs.",
    )
    parser.add_argument("--ift-checkpoint", required=True)
    parser.add_argument("--colight-critic", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--default-critic-phase", default="ETWT")
    return parser.parse_args(argv)


def _load_prompts(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        prompts: list[str] = []
        for item in payload:
            if isinstance(item, str):
                prompts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("prompt"), str):
                prompts.append(item["prompt"])
        return prompts
    raise ValueError("--prompts must be a JSON list of strings or objects")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    prompts = _load_prompts(Path(args.prompts))
    critic = JsonCritic(args.colight_critic, args.default_critic_phase)
    collector = CGPRDataCollector(args.ift_checkpoint, critic)
    pairs = collector.collect(prompts)
    out = [
        {
            "prompt": p.prompt,
            "positive_response": p.positive_response,
            "negative_response": p.negative_response,
        }
        for p in pairs
    ]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(out)} CGPR pairs to {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
