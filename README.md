# LLMLight Reproduction

Reproduction of **LLMLight: Large Language Models as Traffic Signal Control Agents** (Lai et al., KDD 2025 — arXiv 2312.16044). Implements the full pipeline from API-driven baseline collection through LightGPT-0.5B fine-tuning, side-by-side replay visualization, and CityFlow evaluation.

This is the CS106 course project at HCMUS.

---

## What is implemented

### Backbone & model
- **Inference backbone (baseline)**: GPT-5.5 (`cx/gpt-5.5`) accessed through the **CodexHub** API. Three additional backends are wired in for fallback / future use: OpenAI, Groq, and Puter.
- **Fine-tuning target**: `Qwen/Qwen2-0.5B` (matches the LightGPT-0.5B-Qwen2 variant published by the LLMLight authors on HuggingFace).
- **Adapter**: LoRA — `rank=8`, `alpha=16`, dropout `0.05`, target modules `q_proj` + `v_proj`. Saved as `models/qwen2_finetuned_ift/` (post-IFT) and `models/qwen2_finetuned/` (post LoRA-merge).

### Datasets
- **CityFlow scenarios**: Hangzhou (4×4 grid, `anon_4_4_hangzhou_real.json`) and Jinan (1×3 grid, `anon_3_4_jinan_real.json`) — the two real-world traffic networks from the paper.
- **IFT training data**: collected by calling GPT-5.5 on every intersection at every decision cycle for both maps × 2 runs. The replay cache lives in `results/api_cache/<dataset>/gpt4o_codexhub/phase2/run_*.jsonl`; the builder `scripts/build_ift_dataset.py` filters to 4725 valid `(prompt, response)` pairs and writes a versioned JSON dataset.
- **Replay traces**: `results/replays/*.txt` (CityFlow native format) + manifest for the viewer.

### Components present in `src/`
- `cityflow_engine.py` — CityFlow wrapper with replay-save patching, ASCII staging for Unicode paths, vehicle tracking.
- `observation_parser.py` — turns CityFlow state into the LLMLight 3-section prompt.
- `response_parser.py` — extracts `<signal>XXXX</signal>` with fallback to ETWT.
- `phase_index_mapper.py` — maps phase names to CityFlow phase indices per intersection.
- `metrics_evaluator.py` — computes ATT / AQL / AWT exactly as the paper defines them.
- `lightgpt_inference.py` — runs the fine-tuned model on traffic prompts.
- `training/multi_backend_api_client.py` — unified client over CodexHub / OpenAI / Groq / Puter with key rotation, parallel intersection calls, and SSE stream parsing for CodexHub.
- `training/api_replay_cache.py` — deterministic cache of all teacher API decisions for cost-free replay.
- `training/trajectory_collector.py` — collects `(prompt, raw_response)` pairs during a teacher run.
- `training/ift_trainer.py` — LoRA IFT on Qwen2-0.5B (paper-aligned settings, see below).
- `training/cgpr_data_collector.py` + `cgpr_trainer.py` — pair collection + pairwise margin loss for Critic-Guided Policy Refinement.
- `training/lora_merger.py` — merge LoRA adapter into base model, verify architecture preservation.
- `seed_manager.py` + `phase_approval_gate.py` + `preflight_checker.py` — reproducibility / safety scaffolding.

### Scripts
- `scripts/run_gpt4o.py` — teacher run on a CityFlow scenario via any LLM backend, writes replay + metrics + cache.
- `scripts/run_lightgpt.py` — student run with the fine-tuned LightGPT-0.5B.
- `scripts/build_ift_dataset.py` — gather IFT pairs from the API cache.
- `scripts/run_ift_only.py` — IFT-only entry point with full checkpoint/resume/logging.
- `scripts/run_training.py` — full pipeline (IFT → CGPR collect → CGPR → LoRA merge).
- `scripts/collect_cgpr_pairs.py` — CGPR pair-collection standalone.
- `scripts/build_viewer_manifest.py` + `scripts/serve_viewer.py` — viewer support.
- `scripts/test_ift_inference.py` — IFT adapter smoke test.

### Viewer (`viewer/`)
Browser-based side-by-side replay comparison built on **PixiJS v7**. Streams CityFlow replay files (~1 GB each for Jinan) without freezing the browser via a byte-offset index parser. Live AWT/AQL/Throughput tickers, per-intersection compare overlay, time-series charts. Launch with `python scripts/serve_viewer.py`.

### Tests
`tests/` covers metrics computation, response parsing, observation parser, API client, replay cache, IFT trainer (mocked), and CGPR training. Run with `pytest tests/`.

---

## Pipeline status (1 = done, 0 = pending)

| Step | Status | Notes |
|---|---|---|
| **0** — CityFlow build + WSL venv + Qwen2-0.5B cached | 1 | RTX 4060 Laptop GPU, torch 2.5.1+cu121 |
| **1** — Baseline runner (rule-based: MaxPressure, AdvancedMaxPressure) | 1 | Metrics in `results/metrics/` for Hangzhou + Jinan, runs 0-2 |
| **2** — LLM teacher runner (GPT-5.5 via CodexHub) | 1 | 2 runs × 2 maps. Cache replayable cost-free. |
| **3** — IFT dataset construction | 1 | 4725 valid samples, SHA-256-pinned (`data/ift_dataset_v1.json`) |
| **4** — LoRA IFT on Qwen2-0.5B | partial | First training run completed but tokenization bug discovered: prompt tokens (~3600) exceed `max_seq_length=512`, response truncated. Trainer config has now been aligned with the paper (see below); next training run will be re-executed. |
| **5** — Inference smoke test | 0 | Script ready, awaiting IFT rerun |
| **6** — CGPR pair collection (needs Advanced-CoLight critic) | 0 | Critic adapter implemented; Advanced-CoLight RL training not yet run |
| **7** — CGPR fine-tuning | 0 | Trainer implemented, blocked on (6) |
| **8** — LoRA merge | 0 | Merger implemented and unit-tested, blocked on (4)/(7) |
| **9** — CityFlow evaluation with LightGPT-0.5B | 0 | Runner ready, blocked on (8) |
| **10** — Side-by-side viewer | 1 | Streaming replay, metrics, compare overlay |

**Roughly 6 / 10 paper components delivered.** The remaining four depend on a clean IFT pass plus the optional Advanced-CoLight critic.

---

## Reproducing the paper — what matches and what differs

### Same as paper
- Base model family: Qwen2-0.5B (the smallest variant the LLMLight authors publish).
- LoRA configuration: `r=8`, `alpha=16`, dropout `0.05`, target `q_proj` + `v_proj`.
- Prompt structure: 3-section template (system + traffic observation + output-format example with `<signal>` tag).
- Loss: causal-LM cross-entropy with `train_on_inputs=True` (labels = input ids copy), identical to `LLMTSCS/finetune/run_imitation_finetune.py`.
- Effective batch size 128 via gradient accumulation.
- Learning rate `3e-4`, warmup `10`, `group_by_length=True`.
- Imitation Fine-Tuning → Policy Refinement Data Collection → CGPR → LoRA Merge pipeline order.
- Metrics: ATT (`get_average_travel_time`), AQL (mean of `get_lane_vehicle_count` across lanes × steps), AWT (cumulative low-speed time per vehicle, `speed < 0.1 m/s`).

### Different from paper
- **Teacher LLM**: paper uses GPT-3.5 / GPT-4 directly; this project uses **GPT-5.5 via CodexHub** (free CodexHub tier) so the teacher decisions are not identical to the paper's. Quality may shift; the framework treats teacher as a hyperparameter.
- **Datasets**: paper covers 9 datasets; this project uses 2 of them (Hangzhou 4×4 + Jinan 1×3) to fit within compute budget.
- **Quantization**: paper uses `load_in_8bit=True` for the 13B run; for the 0.5B model on RTX 4060 8GB this project uses fp32 base + fp16 mixed-precision (no bitsandbytes int8) — the 0.5B does not require quantization.
- **CGPR critic**: paper trains Advanced-CoLight (RL) as critic; this project ships the wrapper but the RL training has not been run yet. CGPR may be skipped initially and IFT-only adapter evaluated first.
- **API replay cache**: not in the paper. Added so re-running the simulation does not re-spend API quota.
- **Viewer**: not in the paper. Custom PixiJS comparison UI for thesis presentation.

### Hardware/environment differences
- Paper: server-class GPUs (assumed A100-class).
- This project: laptop RTX 4060 8GB → micro-batch 1 + grad-accum 128 to reach the paper's effective batch size 128.
- Project runs on Windows 11 with WSL2 Ubuntu (CityFlow C++ requires Linux).

---

## How to run

### Set up environment
```bash
# WSL Ubuntu, from project root
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# CityFlow must be built separately — see scripts/install_cityflow.sh
```

### Collect baseline (rule-based)
```bash
wsl bash -c "source venv/bin/activate && python scripts/run_baselines.py --dataset hangzhou --num-runs 3"
```

### Run GPT-5.5 teacher
```bash
# Configure CodexHub key in .env.codexhub
wsl bash -c "source venv/bin/activate && python scripts/run_gpt4o.py --dataset hangzhou --num-runs 2 --backend codexhub"
```

### Build IFT dataset
```bash
python scripts/build_ift_dataset.py \
    --cache-roots results/api_cache/hangzhou_1 results/api_cache/jinan_1 \
    --output data/ift_dataset_v1.json
```

### Run IFT
```bash
wsl bash -c "source venv/bin/activate && python scripts/run_ift_only.py --dataset data/ift_dataset_v1.json"
```
Resume after Ctrl+C is automatic — the script detects the latest `checkpoints/ift/checkpoint-*` and resumes optimizer/scheduler/RNG.

### Smoke-test the adapter
```bash
wsl bash -c "source venv/bin/activate && python scripts/test_ift_inference.py --n 20"
```

### LoRA merge
```bash
wsl bash -c "source venv/bin/activate && python -c \"
from src.training.lora_merger import LoRAMerger
LoRAMerger().merge('Qwen/Qwen2-0.5B', 'models/qwen2_finetuned_ift/', 'models/qwen2_finetuned/')
\""
```

### Evaluate on CityFlow
```bash
wsl bash -c "source venv/bin/activate && python scripts/run_lightgpt.py --dataset hangzhou --model-path models/qwen2_finetuned --num-runs 1"
```

### View replays side-by-side
```bash
python scripts/serve_viewer.py
# Open the printed URL
```

---

## Repository layout

```
.kiro/                # Kiro-style spec-driven development (requirements/design/tasks)
config/               # training.json, simulation.json
LLMTSCS/              # cloned reference repo (gitignored)
viewer/               # browser-based replay comparison UI
src/
  cityflow_engine.py
  metrics_evaluator.py
  observation_parser.py
  response_parser.py
  lightgpt_inference.py
  ...
  training/
    multi_backend_api_client.py
    api_replay_cache.py
    trajectory_collector.py
    ift_trainer.py
    cgpr_data_collector.py
    cgpr_trainer.py
    lora_merger.py
scripts/              # CLI entry points (training, evaluation, viewer)
tests/                # pytest suite
results/              # API cache, metrics, replays (gitignored — large)
data/                 # generated IFT datasets (gitignored — regenerate)
models/, checkpoints/ # trained weights (gitignored — large)
logs/                 # runtime logs (gitignored)
```

---

## References

- LLMLight paper: <https://arxiv.org/abs/2312.16044>
- Original code: <https://github.com/usail-hkust/LLMTSCS>
- Published LightGPT models: <https://huggingface.co/collections/usail-hkust/llmlight-lightgpt-673ac5a619cbbe309165b56d>
- CityFlow simulator: <https://github.com/cityflow-project/CityFlow>
