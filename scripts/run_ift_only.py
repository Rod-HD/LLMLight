"""Run *only* the IFT stage of LightGPT 0.5B fine-tuning.

This is the safe entry point for the "start training, allow pause, allow
resume" workflow. Unlike scripts/run_training.py which runs the full
IFT -> CGPR pair collection -> CGPR -> LoRA merge pipeline, this script
stops after IFT so the LoRA adapter checkpoints can be inspected before
moving on.

Saved artifacts (under --checkpoint-dir, default checkpoints/ift/):

  * checkpoint-<step>/        full HF Trainer state (model, optimizer,
                              scheduler, RNG, training_args.bin,
                              trainer_state.json). Created at every
                              epoch boundary (save_strategy="epoch")
                              plus the final step.
  * checkpoint-<step>/adapter_model.safetensors    LoRA weights only
  * <output-dir>/             final adapter saved by trainer.save_model()

Resume:
    python scripts/run_ift_only.py \\
        --dataset data/ift_dataset_v1.json \\
        --resume checkpoints/ift/checkpoint-XXXX

Run log (one per invocation):
    logs/training/ift_<timestamp>.log    full Python logging output
    logs/training/ift_<timestamp>.json   structured summary (config,
                                          dataset sha, sample count,
                                          start/end time, vram, exit_status)
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import logging
import os
import socket
import sys
import time
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.seed_manager import SeedManager  # noqa: E402
from src.training.ift_trainer import (  # noqa: E402
    IFTTrainer,
    OOMError,
    TrainingConfig,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset",
        required=True,
        help="JSON file produced by scripts/build_ift_dataset.py.",
    )
    p.add_argument(
        "--resume",
        default=None,
        help="Path to a checkpoint-XXXX directory to resume from. "
             "Omit to start a fresh run.",
    )
    p.add_argument(
        "--checkpoint-dir",
        default="checkpoints/ift/",
        help="Where Trainer writes per-epoch checkpoints. Re-use across "
             "runs so --resume can find them.",
    )
    p.add_argument(
        "--output-dir",
        default="models/qwen2_finetuned_ift/",
        help="Where the final LoRA adapter is saved after training.",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of IFT epochs (3..30).",
    )
    p.add_argument(
        "--lora-rank",
        type=int,
        default=8,
    )
    p.add_argument(
        "--lora-alpha",
        type=int,
        default=16,
    )
    p.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=128,
    )
    p.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
    )
    p.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
    )
    p.add_argument(
        "--no-train-on-inputs",
        action="store_true",
        help=(
            "Mask prompt tokens (-100) when computing loss. Default is "
            "paper-style train_on_inputs=True."
        ),
    )
    p.add_argument(
        "--save-total-limit",
        type=int,
        default=5,
        help="Number of checkpoint-XXXX folders to keep on disk.",
    )
    p.add_argument(
        "--logging-steps",
        type=int,
        default=10,
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    p.add_argument(
        "--log-dir",
        default="logs/training",
        help="Where to write the per-run .log + .json summary.",
    )
    return p.parse_args(argv)


def _load_dataset(path: Path) -> list[tuple[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path}: top-level JSON must be a list")
    rows: list[tuple[str, str]] = []
    for i, item in enumerate(payload):
        if isinstance(item, dict):
            prompt = item.get("prompt")
            response = item.get("response") or item.get("raw_response")
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            prompt, response = item
        else:
            raise ValueError(f"{path}[{i}]: unsupported row shape {type(item).__name__}")
        if not isinstance(prompt, str) or not isinstance(response, str):
            raise ValueError(f"{path}[{i}]: prompt/response must be str")
        rows.append((prompt, response))
    if not rows:
        raise ValueError(f"{path}: no rows")
    return rows


def _detect_resume_checkpoint(
    resume: str | None, checkpoint_dir: Path
) -> str | None:
    """If --resume not given, auto-detect latest checkpoint-XXXX/ dir.

    Returns None when nothing usable is found (caller will train from scratch).
    """
    if resume:
        return resume
    if not checkpoint_dir.is_dir():
        return None
    candidates = sorted(
        (p for p in checkpoint_dir.glob("checkpoint-*") if p.is_dir()),
        key=lambda p: int(p.name.split("-")[-1])
        if p.name.split("-")[-1].isdigit()
        else 0,
    )
    if not candidates:
        return None
    return str(candidates[-1])


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _env_snapshot() -> dict:
    info: dict = {
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "hostname": socket.gethostname(),
        "cwd": str(Path.cwd()),
        "env": {
            k: v for k, v in os.environ.items()
            if k.startswith(("CUDA_", "TORCH_", "HF_", "TRANSFORMERS_"))
        },
    }
    try:
        import torch  # type: ignore[import-not-found]
        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["cuda_device"] = torch.cuda.get_device_name(0)
            info["cuda_capability"] = torch.cuda.get_device_capability(0)
    except Exception as exc:  # noqa: BLE001
        info["torch_error"] = repr(exc)
    for mod in ("transformers", "peft", "datasets", "accelerate", "bitsandbytes"):
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "?")
        except Exception as exc:  # noqa: BLE001
            info[f"{mod}_error"] = repr(exc)
    return info


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"ift_{timestamp}.log"
    summary_file = log_dir / f"ift_{timestamp}.json"

    # ----- root logger -> stdout + file
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    root.handlers.clear()
    root.addHandler(sh)
    root.addHandler(fh)
    log = logging.getLogger("run_ift_only")

    log.info("=" * 72)
    log.info("LightGPT-0.5B IFT run %s", timestamp)
    log.info("log file: %s", log_file)
    log.info("=" * 72)

    # ----- seed for deterministic shuffling / init
    SeedManager().apply(args.seed)
    log.info("seed applied: %d", args.seed)

    # ----- env snapshot
    env = _env_snapshot()
    log.info("env: %s", json.dumps(env, indent=2, default=str))

    # ----- dataset
    dataset_path = Path(args.dataset)
    data_sha = _file_sha256(dataset_path)
    data = _load_dataset(dataset_path)
    log.info(
        "dataset: %s (%d samples, sha256=%s)",
        dataset_path,
        len(data),
        data_sha,
    )

    # ----- config
    config = TrainingConfig(
        ift_epochs=args.epochs,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_seq_length=args.max_seq_length,
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        save_total_limit=args.save_total_limit,
        logging_steps=args.logging_steps,
        train_on_inputs=not args.no_train_on_inputs,
    )
    log.info("training config: %s", json.dumps(asdict(config), indent=2))

    # ----- checkpoint resume detection
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    resume_path = _detect_resume_checkpoint(
        args.resume, Path(args.checkpoint_dir)
    )
    if resume_path:
        log.info("resuming from checkpoint: %s", resume_path)
    else:
        log.info("no checkpoint found -- training from scratch")

    # ----- summary scaffold (will be overwritten on exit)
    summary = {
        "timestamp": timestamp,
        "log_file": str(log_file),
        "dataset_path": str(dataset_path),
        "dataset_sha256": data_sha,
        "dataset_size": len(data),
        "resume_from": resume_path,
        "config": asdict(config),
        "env": env,
        "args": vars(args),
        "start_time": _dt.datetime.now().isoformat(),
        "end_time": None,
        "duration_seconds": None,
        "exit_status": "running",
        "final_checkpoint": None,
        "error": None,
    }
    summary_file.write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    start = time.time()
    try:
        final_output = IFTTrainer(config).train(data, resume_from=resume_path)
        summary["exit_status"] = "ok"
        summary["final_checkpoint"] = final_output
        log.info("IFT training complete. Final adapter at %s", final_output)
    except OOMError as exc:
        summary["exit_status"] = "oom"
        summary["error"] = repr(exc)
        log.error("OOM during training: %s", exc)
        return 3
    except KeyboardInterrupt:
        summary["exit_status"] = "interrupted"
        log.warning(
            "KeyboardInterrupt: trainer has saved up to the last completed "
            "epoch boundary in %s; rerun with --resume <checkpoint-XXXX>",
            args.checkpoint_dir,
        )
        return 130
    except Exception as exc:  # noqa: BLE001
        summary["exit_status"] = "error"
        summary["error"] = repr(exc)
        log.exception("training failed: %s", exc)
        return 1
    finally:
        end_time = _dt.datetime.now()
        summary["end_time"] = end_time.isoformat()
        summary["duration_seconds"] = round(time.time() - start, 2)
        summary_file.write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8"
        )
        log.info("summary written: %s", summary_file)

    return 0


if __name__ == "__main__":
    sys.exit(main())
