#!/bin/bash
# Install heavy Python dependencies into venv on D drive.
# Logs progress to /tmp/heavy_deps_install.log so we can monitor.
#
# Strategy:
#   1. torch (CUDA 12.1 wheels, ~2.5GB) — separate index URL
#   2. transformers stack (transformers, peft, accelerate, datasets)
#   3. bitsandbytes (4-bit quantization)
#   4. tensorflow-cpu==2.8.0 (LLMTSCS dep)
#   5. misc: wandb, sentencepiece, streamlit, altair

set -e  # exit on first failure
cd "/mnt/d/Duy/Docs/School/CS106 - Trí tuệ nhân tạo/Đồ án/LLMLight"

LOG=/tmp/heavy_deps_install.log
PIP="venv/bin/pip"
echo "=== Heavy deps install started: $(date) ===" > "$LOG"

run_step() {
    local label="$1"
    shift
    echo "" >> "$LOG"
    echo "=== STEP: $label ($(date)) ===" >> "$LOG"
    "$@" >> "$LOG" 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then
        echo "=== STEP DONE: $label ($(date)) ===" >> "$LOG"
    else
        echo "=== STEP FAILED: $label rc=$rc ($(date)) ===" >> "$LOG"
        exit $rc
    fi
}

# Step 1: torch CUDA
run_step "torch CUDA 12.1" \
    "$PIP" install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu121 \
        torch torchvision torchaudio

# Step 2: transformers stack (HF default index)
run_step "transformers stack" \
    "$PIP" install --no-cache-dir \
        "transformers==4.45.0" \
        "peft==0.7.1" \
        "accelerate==0.27.2" \
        "datasets==2.16.1" \
        sentencepiece \
        protobuf

# Step 3: bitsandbytes
run_step "bitsandbytes" \
    "$PIP" install --no-cache-dir \
        "bitsandbytes>=0.43.0,<0.45.0"

# Step 4: tensorflow-cpu (cũ — Python 3.10 OK)
run_step "tensorflow-cpu 2.8.0" \
    "$PIP" install --no-cache-dir \
        "tensorflow-cpu==2.8.0" \
        "protobuf<3.21"

# Step 5: misc utilities
run_step "wandb + UI" \
    "$PIP" install --no-cache-dir \
        wandb \
        streamlit \
        altair

echo "" >> "$LOG"
echo "=== ALL DONE: $(date) ===" >> "$LOG"
echo "" >> "$LOG"
echo "=== INSTALLED PACKAGES ===" >> "$LOG"
"$PIP" list >> "$LOG" 2>&1
