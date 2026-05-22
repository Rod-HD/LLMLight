#!/usr/bin/env bash
# =============================================================================
# LLMLight Reproduction — WSL2 environment setup script
# =============================================================================
# Cài đặt môi trường Python 3.10 + CityFlow + dependencies cho pipeline
# LLMLight reproduction trên WSL2 Ubuntu 24.04.
#
# Quy trình:
#   1. Resolve PROJECT_DIR (cho phép override qua biến môi trường)
#   2. Pre-flight check (nvidia-smi + Path Unicode + Disk space) BEFORE bất kỳ
#      cài đặt nào (Requirement 1.8, 11.5, 11.6)
#   3. Cài Python 3.10 (deadsnakes PPA) + build deps (cmake, g++, libboost-dev)
#      qua apt (Requirement 1.1, 1.2)
#   4. Tạo venv tại $PROJECT_DIR/venv (Requirement 1.1, 11.4)
#   5. Clone + build CityFlow từ source (Requirement 1.2, 1.3)
#   6. Cài Python packages với phiên bản pin chính xác (Requirement 1.4)
#   7. Xác nhận từng package import được (Requirement 1.4)
#
# vllm==0.6.2 được đánh dấu OPTIONAL — KHÔNG cài trong Phase 1 (Requirement 1.10).
# Cần dùng `--with-vllm` flag để cài thêm khi sử dụng run_open_LLM_with_vllm.py.
#
# Idempotent: chạy lại script nhiều lần là an toàn — apt skip package đã cài,
# pip skip package đã ở đúng version, CityFlow skip clone nếu thư mục tồn tại.
#
# Cách chạy (từ Windows host qua terminal):
#   wsl -d Ubuntu -e bash -c "bash '/mnt/d/.../LLMLight/scripts/setup_env.sh'"
#
# Override PROJECT_DIR (mặc định = parent của script directory):
#   PROJECT_DIR=/custom/path bash scripts/setup_env.sh
#
# _Requirements: 1.1, 1.2, 1.3, 1.4, 1.6, 1.7, 1.8, 1.10_
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
readonly PYTHON_VERSION="3.10"
readonly PYTHON_BIN="python${PYTHON_VERSION}"
readonly CITYFLOW_REPO="https://github.com/cityflow-project/CityFlow.git"
readonly CITYFLOW_DIRNAME="CityFlow"

# Lý do CUDA 12.1: tương thích với bitsandbytes>=0.43.0,<0.45.0 (Requirement 1.4).
readonly TORCH_INDEX_URL="https://download.pytorch.org/whl/cu121"

# Danh sách package + tên import (cho bước verify ở cuối).
# Format: "<pip_spec>|<import_name>"
readonly -a PIP_PACKAGES=(
    "tensorflow-cpu==2.8.0|tensorflow"
    "pandas==1.5.0|pandas"
    "numpy==1.26.2|numpy"
    "transformers==4.45.0|transformers"
    "peft==0.7.1|peft"
    "accelerate==0.27.2|accelerate"
    "bitsandbytes>=0.43.0,<0.45.0|bitsandbytes"
    "datasets==2.16.1|datasets"
    "wandb|wandb"
    "fire|fire"
    "requests|requests"
    "python-dotenv|dotenv"
    "hypothesis|hypothesis"
    "pytest|pytest"
)

# -----------------------------------------------------------------------------
# Logging helpers
# -----------------------------------------------------------------------------
log_info()  { printf '\033[1;34m[INFO]\033[0m  %s\n' "$*" >&2; }
log_warn()  { printf '\033[1;33m[WARN]\033[0m  %s\n' "$*" >&2; }
log_error() { printf '\033[1;31m[ERROR]\033[0m %s\n' "$*" >&2; }
log_step()  { printf '\n\033[1;36m===== %s =====\033[0m\n' "$*" >&2; }

# -----------------------------------------------------------------------------
# CLI parsing
# -----------------------------------------------------------------------------
WITH_VLLM=0
for arg in "$@"; do
    case "$arg" in
        --with-vllm)
            WITH_VLLM=1
            ;;
        --help|-h)
            sed -n '2,30p' "$0"
            exit 0
            ;;
        *)
            log_error "Unknown argument: $arg"
            log_error "Usage: $0 [--with-vllm]"
            exit 2
            ;;
    esac
done
readonly WITH_VLLM

# -----------------------------------------------------------------------------
# Resolve PROJECT_DIR
# -----------------------------------------------------------------------------
# Mặc định: parent của thư mục chứa script (script ở scripts/, project ở ..).
# Cho phép override qua biến môi trường PROJECT_DIR (ví dụ load từ .env).
SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
readonly SCRIPT_PATH
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"
readonly SCRIPT_DIR
DEFAULT_PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

PROJECT_DIR="${PROJECT_DIR:-$DEFAULT_PROJECT_DIR}"
readonly PROJECT_DIR

if [[ ! -d "$PROJECT_DIR" ]]; then
    log_error "PROJECT_DIR không tồn tại: $PROJECT_DIR"
    exit 1
fi

readonly VENV_DIR="$PROJECT_DIR/venv"
readonly CITYFLOW_DIR="$PROJECT_DIR/$CITYFLOW_DIRNAME"
readonly VENV_PYTHON="$VENV_DIR/bin/python"
readonly VENV_PIP="$VENV_DIR/bin/pip"

log_info "PROJECT_DIR = $PROJECT_DIR"
log_info "VENV_DIR    = $VENV_DIR"
log_info "WITH_VLLM   = $WITH_VLLM"

# -----------------------------------------------------------------------------
# Step 1: Pre-flight check (BEFORE bất cứ cài đặt nào)
# -----------------------------------------------------------------------------
# Gọi PreflightChecker.run_all() qua python (Requirement 1.8, 11.5-11.8).
# Yêu cầu: hệ thống đã có python3 sẵn (chuẩn trên Ubuntu 24.04).
# KHÔNG dùng venv Python ở đây vì venv chưa được tạo — preflight CHẠY TRƯỚC.
run_preflight() {
    log_step "Step 1/7: Pre-flight environment check"

    local system_python
    if command -v python3 >/dev/null 2>&1; then
        system_python="python3"
    elif command -v python >/dev/null 2>&1; then
        system_python="python"
    else
        log_error "Không tìm thấy python3 trên hệ thống — cần ít nhất Python 3 để chạy pre-flight."
        log_error "Cài bằng: sudo apt install -y python3"
        exit 1
    fi

    log_info "Sử dụng $system_python để chạy PreflightChecker"

    # Dùng PYTHONPATH=$PROJECT_DIR để import được src.preflight_checker
    # mà KHÔNG cần cài project.
    if ! PYTHONPATH="$PROJECT_DIR" "$system_python" -c "
from src.preflight_checker import PreflightChecker
PreflightChecker().run_all('$PROJECT_DIR')
"; then
        log_error "Pre-flight check FAILED — dừng cài đặt."
        log_error "Sửa các lỗi báo ở trên rồi chạy lại script."
        exit 1
    fi

    log_info "Pre-flight check passed."
}

# -----------------------------------------------------------------------------
# Step 2: APT dependencies (Python 3.10 + build tools)
# -----------------------------------------------------------------------------
install_apt_packages() {
    log_step "Step 2/7: APT packages (Python ${PYTHON_VERSION} + build deps)"

    # Yêu cầu sudo. Nếu user không có sudo, báo lỗi rõ ràng.
    if ! command -v sudo >/dev/null 2>&1; then
        log_error "Cần 'sudo' để cài apt packages trên WSL2 Ubuntu."
        exit 1
    fi

    log_info "Updating apt indexes..."
    sudo apt-get update -y

    # software-properties-common cung cấp add-apt-repository.
    log_info "Cài software-properties-common (cho add-apt-repository)..."
    sudo apt-get install -y software-properties-common

    # Thêm deadsnakes PPA nếu chưa có (Requirement 1.1).
    if ! grep -rq "^deb .*deadsnakes" /etc/apt/sources.list /etc/apt/sources.list.d/ 2>/dev/null; then
        log_info "Adding deadsnakes PPA cho Python ${PYTHON_VERSION}..."
        sudo add-apt-repository -y ppa:deadsnakes/ppa
        sudo apt-get update -y
    else
        log_info "deadsnakes PPA đã có sẵn — skip add."
    fi

    # Cài Python 3.10 + venv module + dev headers + build deps (Requirement 1.2).
    log_info "Cài ${PYTHON_BIN}, venv, dev headers, và build dependencies..."
    sudo apt-get install -y \
        "${PYTHON_BIN}" \
        "${PYTHON_BIN}-venv" \
        "${PYTHON_BIN}-dev" \
        "${PYTHON_BIN}-distutils" \
        build-essential \
        cmake \
        g++ \
        libboost-all-dev \
        git \
        curl \
        ca-certificates

    # Verify Python 3.10 thực sự khả dụng.
    if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
        log_error "${PYTHON_BIN} vẫn không có trên PATH sau khi cài apt."
        exit 1
    fi
    log_info "$(${PYTHON_BIN} --version) sẵn sàng."
}

# -----------------------------------------------------------------------------
# Step 3: Tạo virtual environment
# -----------------------------------------------------------------------------
create_venv() {
    log_step "Step 3/7: Tạo virtual environment tại $VENV_DIR"

    if [[ -x "$VENV_PYTHON" ]]; then
        local existing_version
        existing_version="$("$VENV_PYTHON" --version 2>&1 || true)"
        log_info "venv đã tồn tại ($existing_version) — skip tạo mới."
    else
        log_info "Tạo venv với ${PYTHON_BIN}..."
        if ! "${PYTHON_BIN}" -m venv "$VENV_DIR"; then
            log_error "Không tạo được venv tại $VENV_DIR."
            log_error "Kiểm tra quyền ghi (Requirement 1.7) và path tồn tại."
            exit 1
        fi
    fi

    # Upgrade pip / setuptools / wheel — cần thiết để cài bitsandbytes manylinux2014 wheels.
    log_info "Upgrade pip / setuptools / wheel trong venv..."
    "$VENV_PIP" install --upgrade pip setuptools wheel
}

# -----------------------------------------------------------------------------
# Step 4: Clone + build CityFlow từ source
# -----------------------------------------------------------------------------
build_cityflow() {
    log_step "Step 4/7: Clone + build CityFlow"

    if [[ -d "$CITYFLOW_DIR/.git" ]]; then
        log_info "CityFlow đã clone ($CITYFLOW_DIR) — skip clone."
    else
        log_info "Clone CityFlow từ $CITYFLOW_REPO..."
        git clone --depth 1 "$CITYFLOW_REPO" "$CITYFLOW_DIR"
    fi

    # Kiểm tra cityflow đã import được trong venv chưa — nếu rồi thì skip build.
    if "$VENV_PYTHON" -c "import cityflow" >/dev/null 2>&1; then
        log_info "cityflow đã import được trong venv — skip rebuild."
        return 0
    fi

    log_info "Build CityFlow trong venv (pip install ./CityFlow)..."
    # Dùng pip install . để build qua setup.py — pip sẽ gọi cmake + g++.
    if ! "$VENV_PIP" install --no-build-isolation "$CITYFLOW_DIR"; then
        log_error "Build CityFlow thất bại."
        log_error "Kiểm tra cmake, g++, libboost-all-dev đã cài (Requirement 1.6)."
        exit 1
    fi

    # Verify post-build (Requirement 1.3): cityflow.Engine khởi tạo được.
    log_info "Verify cityflow import sau build..."
    if ! "$VENV_PYTHON" -c "import cityflow; print('cityflow OK')"; then
        log_error "cityflow build xong nhưng KHÔNG import được."
        exit 1
    fi
}

# -----------------------------------------------------------------------------
# Step 5: Cài torch (CUDA) — tách riêng vì cần index URL đặc biệt
# -----------------------------------------------------------------------------
install_torch() {
    log_step "Step 5/7: Cài PyTorch (CUDA 12.1)"

    if "$VENV_PYTHON" -c "import torch; assert torch.cuda.is_available()" >/dev/null 2>&1; then
        local torch_ver
        torch_ver="$("$VENV_PYTHON" -c "import torch; print(torch.__version__)" 2>/dev/null || echo "?")"
        log_info "torch đã cài (${torch_ver}) với CUDA available — skip."
        return 0
    fi

    log_info "Cài torch + torchvision với CUDA 12.1 wheels..."
    "$VENV_PIP" install --index-url "$TORCH_INDEX_URL" torch torchvision

    log_info "Verify torch CUDA..."
    if ! "$VENV_PYTHON" -c "
import torch
assert torch.cuda.is_available(), 'torch.cuda.is_available() == False'
print(f'torch={torch.__version__} cuda={torch.version.cuda} device={torch.cuda.get_device_name(0)}')
"; then
        log_error "torch cài xong nhưng CUDA không khả dụng."
        log_error "Kiểm tra CUDA driver passthrough trên WSL2 (chạy nvidia-smi)."
        exit 1
    fi
}

# -----------------------------------------------------------------------------
# Step 6: Cài Python packages còn lại
# -----------------------------------------------------------------------------
install_python_packages() {
    log_step "Step 6/7: Cài Python packages (pinned versions)"

    # Tách list spec và list import-name song song.
    local -a pip_specs=()
    for entry in "${PIP_PACKAGES[@]}"; do
        pip_specs+=("${entry%%|*}")
    done

    log_info "Cài ${#pip_specs[@]} packages: ${pip_specs[*]}"
    # Cài tất cả trong một lệnh để pip resolve dependencies cùng lúc.
    "$VENV_PIP" install "${pip_specs[@]}"

    # vllm OPTIONAL (Requirement 1.10): chỉ cài khi --with-vllm.
    if [[ "$WITH_VLLM" -eq 1 ]]; then
        log_info "Cài vllm==0.6.2 (OPTIONAL — đã yêu cầu qua --with-vllm)..."
        "$VENV_PIP" install "vllm==0.6.2"
    else
        log_info "vllm KHÔNG được cài (Phase 1 mặc định bỏ qua, Requirement 1.10)."
        log_info "Truyền --with-vllm để cài thêm khi cần run_open_LLM_with_vllm.py."
    fi
}

# -----------------------------------------------------------------------------
# Step 7: Verify cài đặt — import từng package (Requirement 1.4)
# -----------------------------------------------------------------------------
verify_installation() {
    log_step "Step 7/7: Verify import từng package"

    # Verify cityflow + torch trước (đã verify trong step 4 và 5 nhưng kiểm tra lại
    # cùng nhau ở đây cho output đẹp).
    local -a verify_imports=(
        "cityflow"
        "torch"
    )
    for entry in "${PIP_PACKAGES[@]}"; do
        verify_imports+=("${entry##*|}")
    done

    if [[ "$WITH_VLLM" -eq 1 ]]; then
        verify_imports+=("vllm")
    fi

    local failed=0
    for import_name in "${verify_imports[@]}"; do
        if "$VENV_PYTHON" -c "import ${import_name}" >/dev/null 2>&1; then
            log_info "✓ import ${import_name}"
        else
            log_error "✗ import ${import_name} FAILED"
            failed=$((failed + 1))
        fi
    done

    if [[ "$failed" -gt 0 ]]; then
        log_error "$failed package(s) không import được — xem log ở trên."
        exit 1
    fi

    log_info "Tất cả packages import thành công."
}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
main() {
    log_info "===================================================================="
    log_info " LLMLight Reproduction — WSL2 setup"
    log_info "===================================================================="

    run_preflight
    install_apt_packages
    create_venv
    build_cityflow
    install_torch
    install_python_packages
    verify_installation

    log_info ""
    log_info "===================================================================="
    log_info " Setup hoàn tất."
    log_info "===================================================================="
    log_info "Activate venv:"
    log_info "  source $VENV_DIR/bin/activate"
    log_info ""
    log_info "Hoặc chạy trực tiếp:"
    log_info "  $VENV_PYTHON <script>"
    log_info "===================================================================="
}

main "$@"
