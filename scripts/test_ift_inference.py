"""Inference smoke test for the IFT-finetuned Qwen2-0.5B adapter.

Loads the base model + adapter, runs a handful of prompts from the IFT
dataset, and prints (prompt[:80], gold_response, model_output) so the user
can eyeball whether the model emits valid <signal>...</signal> tokens.

Usage:
    python scripts/test_ift_inference.py
    python scripts/test_ift_inference.py --adapter models/qwen2_finetuned_ift --n 10
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("test_ift_inference")

VALID_PHASES = {"ETWT", "NTST", "ELWL", "NLSL"}
_SIGNAL_RE = re.compile(r"<signal>\s*([A-Z]{4})\s*</signal>", re.IGNORECASE)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--adapter",
        default="models/qwen2_finetuned_ift/",
        help="Path to LoRA adapter directory.",
    )
    p.add_argument(
        "--base-model",
        default="Qwen/Qwen2-0.5B",
        help="HuggingFace base model id.",
    )
    p.add_argument(
        "--dataset",
        default="data/ift_dataset_v1.json",
        help="IFT dataset used to sample test prompts.",
    )
    p.add_argument("--n", type=int, default=10, help="Number of prompts to test.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
        help="Generation cap. <signal>XXXX</signal> is ~9 tokens.",
    )
    p.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=512,
        help="Head/tail prompt budget used during IFT training.",
    )
    return p.parse_args(argv)


def _encode_prompt_head_tail(tokenizer, prompt: str, max_tokens: int, device: str):
    import torch

    prompt_ids_full = list(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    if len(prompt_ids_full) <= max_tokens:
        return tokenizer(prompt, return_tensors="pt").to(device)

    head_keep = max(1, int(max_tokens * 0.3))
    tail_keep = max_tokens - head_keep
    prompt_ids = prompt_ids_full[:head_keep] + prompt_ids_full[-tail_keep:]
    return {
        "input_ids": torch.tensor([prompt_ids], dtype=torch.long, device=device),
        "attention_mask": torch.ones((1, len(prompt_ids)), dtype=torch.long, device=device),
    }


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args(argv)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("device: %s", device)

    logger.info("loading dataset: %s", args.dataset)
    rows = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    random.seed(args.seed)
    samples = random.sample(rows, min(args.n, len(rows)))

    logger.info("loading base model: %s", args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(args.base_model)
    base = base.to(device)

    logger.info("loading adapter: %s", args.adapter)
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()

    correct_format = 0
    correct_phase = 0
    per_phase: dict[str, dict[str, int]] = {p: {"total": 0, "match": 0} for p in VALID_PHASES}

    print("=" * 100)
    print(f"Testing {len(samples)} samples from {args.dataset}")
    print("=" * 100)

    for i, row in enumerate(samples, 1):
        prompt = row["prompt"]
        gold = row["response"]
        gold_phase_m = _SIGNAL_RE.search(gold)
        gold_phase = gold_phase_m.group(1).upper() if gold_phase_m else "?"

        inputs = _encode_prompt_head_tail(
            tokenizer, prompt, args.max_prompt_tokens, device
        )
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)

        pred_m = _SIGNAL_RE.search(response)
        pred_phase = pred_m.group(1).upper() if pred_m else "?"

        ok_format = pred_phase in VALID_PHASES
        ok_phase = ok_format and pred_phase == gold_phase
        correct_format += int(ok_format)
        correct_phase += int(ok_phase)
        if gold_phase in per_phase:
            per_phase[gold_phase]["total"] += 1
            if ok_phase:
                per_phase[gold_phase]["match"] += 1

        marker_fmt = "OK " if ok_format else "BAD"
        marker_match = "MATCH " if ok_phase else "DIFFER"
        print(f"\n[{i}/{len(samples)}] {marker_fmt} format / {marker_match}")
        print(f"  gold : {gold.strip()[:80]}  (phase={gold_phase})")
        print(f"  pred : {response.strip()[:80]}  (phase={pred_phase})")
        print(f"  ds   : {row.get('dataset', '?')}  ts={row.get('timestep', '?')}  int={row.get('intersection_id', '?')}")

    print("\n" + "=" * 100)
    print(f"Format valid : {correct_format}/{len(samples)}  ({100*correct_format/len(samples):.1f}%)")
    print(f"Phase match  : {correct_phase}/{len(samples)}  ({100*correct_phase/len(samples):.1f}%)")
    print("Per-gold-phase accuracy:")
    for ph in sorted(per_phase):
        s = per_phase[ph]
        if s["total"] > 0:
            print(f"  {ph}: {s['match']}/{s['total']}  ({100*s['match']/s['total']:.1f}%)")
        else:
            print(f"  {ph}: 0/0  (n/a)")
    print("=" * 100)
    return 0


if __name__ == "__main__":
    sys.exit(main())
