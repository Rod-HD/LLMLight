"""VRAM monitor utility (Task 6.1).

Helper module dùng chung cho các thao tác load model / inference / training
trên GPU để kiểm tra VRAM availability TRƯỚC khi load weights và tránh OOM
khi vượt budget 7.5GB của RTX 4060 8GB.

API design:

* :func:`get_vram_usage_mb`: VRAM hiện đang được PyTorch allocate (MB),
  trả về ``None`` nếu CUDA không khả dụng.
* :func:`get_vram_free_mb`: VRAM trống trên device (MB) theo
  ``torch.cuda.mem_get_info``, trả về ``None`` nếu CUDA không khả dụng.
* :func:`check_vram_available`: trả về ``True`` nếu free VRAM
  ``>= required_mb``, ``False`` nếu thiếu (kèm log lỗi định dạng
  ``[VRAM_ERROR] Required: {X}MB, Available: {Y}MB, Model: {name}``).

Behavior khi CUDA không khả dụng (không có GPU, torch chưa cài, hoặc
chạy trên Windows dev box):

* :func:`get_vram_usage_mb` và :func:`get_vram_free_mb` trả về ``None``
  để caller phía trên có thể skip check thay vì crash.
* :func:`check_vram_available` trả về ``False`` (fail-safe) và log
  cảnh báo — caller đang định load model lên GPU nhưng GPU không có,
  nên KHÔNG được phép tiếp tục. Caller có thể detect case này qua
  :func:`is_cuda_available` nếu cần phân biệt "không có CUDA" với
  "có CUDA nhưng thiếu VRAM".

Module này dùng ``torch.cuda.memory_allocated()`` cho usage và
``torch.cuda.mem_get_info()`` cho free/total. ``mem_get_info`` được
chọn (thay vì tính ``total - allocated``) vì nó phản ánh free VRAM
thực tế trên device kể cả khi process khác đang dùng GPU.

_Requirements_: 11.1, 11.2
"""

from __future__ import annotations

import logging
from typing import Union

logger = logging.getLogger(__name__)

# Conversion: PyTorch trả bytes; 1 MB = 1024 * 1024 bytes (binary MiB).
# Toàn bộ module dùng MiB để khớp với ``torch.cuda.memory_allocated``.
_BYTES_PER_MB: int = 1024 * 1024

# Type alias cho device parameter — chấp nhận cả int (device index) và
# str (e.g. ``"cuda:0"``).
DeviceLike = Union[int, str]

# Default VRAM budget cho RTX 4060 8GB (Requirement 11.1: < 7.5GB).
MAX_VRAM_MB: float = 7680.0


# =========================================================================
# torch import guard
# =========================================================================


def _import_torch():
    """Import torch lazily; trả về module hoặc ``None`` nếu chưa cài.

    Tách thành function để tests có thể mock dễ dàng và để module này
    import được trên Windows dev box (chưa có torch + CUDA).
    """
    try:
        import torch  # type: ignore[import-not-found]

        return torch
    except ImportError:
        logger.debug(
            "vram_monitor: torch không khả dụng; mọi query VRAM sẽ "
            "trả về None / False."
        )
        return None


def is_cuda_available() -> bool:
    """Trả về ``True`` nếu torch đã cài và CUDA detect được ít nhất 1 GPU."""
    torch = _import_torch()
    if torch is None:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception as exc:  # pragma: no cover — torch internal issue
        logger.warning(
            "vram_monitor: torch.cuda.is_available() raised %s; "
            "treating as no-CUDA.",
            exc,
        )
        return False


# =========================================================================
# queries
# =========================================================================


def get_vram_usage_mb(device: DeviceLike = 0) -> float | None:
    """Return VRAM hiện đang được PyTorch allocate trên ``device`` (MB).

    Args:
        device: Device index (int) hoặc string (``"cuda:0"``); mặc định 0.

    Returns:
        Float MB nếu CUDA khả dụng, ``None`` nếu không (torch chưa cài
        hoặc không có GPU).
    """
    if not is_cuda_available():
        return None
    torch = _import_torch()
    assert torch is not None  # is_cuda_available implies torch imported
    try:
        bytes_allocated = torch.cuda.memory_allocated(device)
    except Exception as exc:
        logger.warning(
            "vram_monitor: torch.cuda.memory_allocated(%r) raised %s; "
            "returning None.",
            device,
            exc,
        )
        return None
    return bytes_allocated / _BYTES_PER_MB


def get_vram_free_mb(device: DeviceLike = 0) -> float | None:
    """Return free VRAM trên ``device`` (MB) qua ``torch.cuda.mem_get_info``.

    ``mem_get_info`` phản ánh free VRAM thực trên device (kể cả khi
    process khác đang dùng GPU), thay vì chỉ ``total - allocated_by_us``.

    Args:
        device: Device index hoặc string; mặc định 0.

    Returns:
        Float MB free nếu CUDA khả dụng, ``None`` nếu không.
    """
    if not is_cuda_available():
        return None
    torch = _import_torch()
    assert torch is not None
    try:
        free_bytes, _total_bytes = torch.cuda.mem_get_info(device)
    except Exception as exc:
        logger.warning(
            "vram_monitor: torch.cuda.mem_get_info(%r) raised %s; "
            "returning None.",
            device,
            exc,
        )
        return None
    return free_bytes / _BYTES_PER_MB


# =========================================================================
# checks
# =========================================================================


def check_vram_available(
    required_mb: float,
    device: DeviceLike = 0,
    model_name: str | None = None,
) -> bool:
    """Verify free VRAM trên ``device`` đủ ``required_mb``.

    Args:
        required_mb: VRAM cần thiết để load model / chạy inference (MB).
            Phải >= 0.
        device: Device index hoặc string; mặc định 0.
        model_name: Tên model (chỉ dùng cho log message); mặc định
            ``"unknown"`` nếu ``None``.

    Returns:
        ``True`` nếu free VRAM ``>= required_mb``; ``False`` nếu thiếu
        HOẶC nếu CUDA không khả dụng (fail-safe — caller định load model
        lên GPU nhưng GPU không có).

    Log:
        Khi không đủ, log ở mức ERROR với format:
            ``[VRAM_ERROR] Required: {X}MB, Available: {Y}MB, Model: {name}``
        (X, Y làm tròn 2 chữ số thập phân).

    Raises:
        ValueError: Nếu ``required_mb < 0``.
    """
    if required_mb < 0:
        raise ValueError(f"required_mb must be >= 0, got {required_mb}")

    name = model_name if model_name is not None else "unknown"

    free_mb = get_vram_free_mb(device)
    if free_mb is None:
        # CUDA không khả dụng → log ERROR theo format spec với
        # Available: 0 (vì không có VRAM nào để dùng) và return False.
        logger.error(
            "[VRAM_ERROR] Required: %.2fMB, Available: 0.00MB, Model: %s "
            "(CUDA không khả dụng trên device %r)",
            required_mb,
            name,
            device,
        )
        return False

    if free_mb < required_mb:
        logger.error(
            "[VRAM_ERROR] Required: %.2fMB, Available: %.2fMB, Model: %s",
            required_mb,
            free_mb,
            name,
        )
        return False

    logger.info(
        "vram_monitor: VRAM OK on device %r (%.2fMB free, %.2fMB required, "
        "model: %s).",
        device,
        free_mb,
        required_mb,
        name,
    )
    return True
