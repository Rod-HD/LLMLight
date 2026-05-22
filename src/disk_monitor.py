"""Runtime Disk Space Monitor (Task 14.1).

Helper module dùng chung cho các script ghi file lớn (training, model
download, replay logging) để cảnh báo khi sắp hết dung lượng và yêu cầu
xác nhận user trước khi tiếp tục.

Khác với :class:`src.preflight_checker.PreflightChecker` (chạy ONE-SHOT
trước khi build CityFlow / cài torch), :class:`DiskMonitor` được runner
gọi NHIỀU LẦN trong runtime — trước mỗi thao tác I/O lớn:

* Trước khi tải HuggingFace model về ``models/hf_cache/``
* Trước khi ghi checkpoint mới sau mỗi epoch IFT/CGPR
* Trước khi mở CityFlow engine với ``save_replay=True``

API design:

* :meth:`get_free_gb` / :meth:`get_storage_used_gb`: stateless query
* :meth:`check_free_space` / :meth:`check_storage_quota`: log warning,
  trả về ``True`` nếu OK (under threshold), ``False`` nếu vượt
* :meth:`require_confirmation`: prompt ``y/N`` qua stdin (TTY) hoặc tự
  động fail an toàn nếu không phải TTY và caller không truyền ``auto_yes``
* :meth:`before_write`: high-level wrapper — kết hợp 3 method trên cho
  các runner script chỉ cần một call duy nhất

Module này ĐƯỢC PHÉP raise ``RuntimeError`` khi user từ chối xác nhận
(khác với :class:`PreflightChecker.check_disk_space` chỉ cảnh báo); hành
vi raise/no-raise được caller chọn qua flag ``required`` của
:meth:`before_write`.

Path Unicode pre-flight đã được chuyển vào :class:`PreflightChecker`
(Task 1.1) — module này KHÔNG xử lý Unicode path checks.

_Requirements_: 11.7, 11.8, 14.12
"""

from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path
from typing import Iterable

from .preflight_checker import PreflightChecker

logger = logging.getLogger(__name__)

# Convert factor giữa GB (decimal) và bytes (binary GiB như shutil dùng).
# Toàn bộ module dùng GiB để khớp với ``shutil.disk_usage`` và
# ``PreflightChecker``.
_BYTES_PER_GB: int = 1024 ** 3


class DiskMonitor:
    """Runtime disk-space monitor cho các thao tác ghi file lớn.

    Attributes:
        drive: Mount point cần kiểm tra free space (mặc định ``/mnt/d``).
        project_dir: Project root để tính tổng ``models/`` + ``checkpoints/``;
            nếu ``None``, dùng trực tiếp ``drive`` làm root.
        free_warn_gb: Ngưỡng cảnh báo free space (mặc định 10 GB).
        storage_warn_gb: Ngưỡng cảnh báo total weights + checkpoints (mặc
            định 250 GB).
        storage_subdirs: Tuple các thư mục con dưới ``project_dir`` được
            tính vào "tổng model + checkpoint". Mặc định ``("models",
            "checkpoints")`` — khớp với :class:`PreflightChecker`.
    """

    DEFAULT_FREE_WARN_GB: float = 10.0
    DEFAULT_STORAGE_WARN_GB: float = 250.0
    DEFAULT_STORAGE_SUBDIRS: tuple[str, ...] = ("models", "checkpoints")

    def __init__(
        self,
        drive: str = "/mnt/d",
        project_dir: str | None = None,
        free_warn_gb: float = DEFAULT_FREE_WARN_GB,
        storage_warn_gb: float = DEFAULT_STORAGE_WARN_GB,
        storage_subdirs: Iterable[str] | None = None,
    ) -> None:
        if free_warn_gb < 0:
            raise ValueError(
                f"free_warn_gb must be >= 0, got {free_warn_gb}"
            )
        if storage_warn_gb < 0:
            raise ValueError(
                f"storage_warn_gb must be >= 0, got {storage_warn_gb}"
            )

        self.drive: str = drive
        self.project_dir: str | None = project_dir
        self.free_warn_gb: float = float(free_warn_gb)
        self.storage_warn_gb: float = float(storage_warn_gb)
        self.storage_subdirs: tuple[str, ...] = (
            tuple(storage_subdirs)
            if storage_subdirs is not None
            else self.DEFAULT_STORAGE_SUBDIRS
        )

    # ----------------------------------------------------------- queries --

    def get_free_gb(self) -> float | None:
        """Return free GB on :attr:`drive`, or ``None`` if drive không tồn tại.

        Trả về ``None`` (thay vì raise) để caller phía trên có thể skip
        check trên platform không có ``/mnt/d`` (Windows host, CI...).
        """
        try:
            usage = shutil.disk_usage(self.drive)
        except (FileNotFoundError, NotADirectoryError) as exc:
            logger.warning(
                "DiskMonitor: cannot stat drive %r (%s). "
                "Free-space query returns None.",
                self.drive,
                exc,
            )
            return None
        except OSError as exc:
            logger.warning(
                "DiskMonitor: OS error while statting drive %r (%s). "
                "Free-space query returns None.",
                self.drive,
                exc,
            )
            return None
        return usage.free / _BYTES_PER_GB

    def get_storage_used_gb(self) -> float:
        """Tổng GB của ``models/`` + ``checkpoints/`` (đệ quy).

        Bỏ qua các thư mục con không tồn tại. Symlinks bị skip để tránh
        đếm trùng (giống :meth:`PreflightChecker._dir_size_bytes`).
        """
        root = Path(self.project_dir) if self.project_dir else Path(self.drive)
        total_bytes = 0
        for subdir in self.storage_subdirs:
            sub_path = root / subdir
            if not sub_path.exists():
                continue
            total_bytes += PreflightChecker._dir_size_bytes(sub_path)
        return total_bytes / _BYTES_PER_GB

    # ----------------------------------------------------------- checks --

    def check_free_space(self, required_gb: float | None = None) -> bool:
        """Cảnh báo nếu free space dưới ngưỡng hoặc không đủ ``required_gb``.

        Args:
            required_gb: Nếu truyền vào, cảnh báo thêm khi
                ``free < required_gb`` (kể cả khi free vẫn trên
                ``free_warn_gb``).

        Returns:
            ``True`` nếu free space OK (>= cả ``free_warn_gb`` và
            ``required_gb`` nếu có); ``False`` nếu vi phạm bất kỳ ngưỡng
            nào. Trả về ``True`` khi không xác định được free space (drive
            không tồn tại) — caller phải tự kiểm tra qua
            :meth:`get_free_gb` nếu muốn fail-safe.
        """
        free_gb = self.get_free_gb()
        if free_gb is None:
            # Không xác định được; coi như OK để không block runner trên
            # platform không có /mnt/d. Warning đã được log ở get_free_gb.
            return True

        ok = True

        if free_gb < self.free_warn_gb:
            logger.warning(
                "[DISK_WARN] Free: %.2fGB on %s, threshold: %.0fGB. "
                "Estimated need: ~10-20 GB for CityFlow build + Python deps + "
                "HF cache.",
                free_gb,
                self.drive,
                self.free_warn_gb,
            )
            ok = False

        if required_gb is not None and free_gb < required_gb:
            logger.warning(
                "[DISK_WARN] Free: %.2fGB on %s, Required: ~%.2fGB. "
                "Insufficient space for upcoming write.",
                free_gb,
                self.drive,
                required_gb,
            )
            ok = False

        if ok:
            logger.info(
                "DiskMonitor: free space OK on %s (%.2fGB free, threshold "
                "%.0fGB).",
                self.drive,
                free_gb,
                self.free_warn_gb,
            )
        return ok

    def check_storage_quota(self, estimated_additional_gb: float = 0.0) -> bool:
        """Cảnh báo nếu total models+checkpoints (+ additional) > ngưỡng.

        Args:
            estimated_additional_gb: Ước tính kích thước file sắp ghi
                (ví dụ model weights HF mới tải, checkpoint LoRA mới). Mặc
                định 0 (chỉ kiểm tra dung lượng đã có).

        Returns:
            ``True`` nếu total dưới ngưỡng; ``False`` nếu vượt.
        """
        if estimated_additional_gb < 0:
            raise ValueError(
                "estimated_additional_gb must be >= 0, got "
                f"{estimated_additional_gb}"
            )

        used_gb = self.get_storage_used_gb()
        projected_gb = used_gb + estimated_additional_gb

        root = (
            Path(self.project_dir).as_posix()
            if self.project_dir
            else self.drive
        )

        if projected_gb > self.storage_warn_gb:
            logger.warning(
                "[STORAGE_WARN] Total models+checkpoints under %s: "
                "%.2fGB (current) + %.2fGB (planned) = %.2fGB > %.0fGB "
                "threshold. Confirm before downloading more.",
                root,
                used_gb,
                estimated_additional_gb,
                projected_gb,
                self.storage_warn_gb,
            )
            return False

        logger.info(
            "DiskMonitor: storage quota OK under %s (%.2fGB used, %.2fGB "
            "after planned write, threshold %.0fGB).",
            root,
            used_gb,
            projected_gb,
            self.storage_warn_gb,
        )
        return True

    # -------------------------------------------------------- confirm --

    def require_confirmation(
        self,
        prompt: str,
        *,
        auto_yes: bool = False,
    ) -> bool:
        """Hiển thị prompt ``y/N`` qua stdin và trả về ``True`` nếu user
        gõ ``y``/``yes`` (case-insensitive).

        Args:
            prompt: Câu hỏi hiển thị cho user (sẽ được suffix
                ``" Continue? [y/N]: "``).
            auto_yes: Nếu ``True`` → trả về ``True`` ngay không hỏi
                (cho non-interactive runs / CI).

        Returns:
            ``True`` nếu user xác nhận tiếp tục, ``False`` nếu từ chối,
            EOF, hoặc stdin không phải TTY (fail-safe).
        """
        if auto_yes:
            logger.info(
                "DiskMonitor: auto_yes=True, skipping confirmation for: %s",
                prompt,
            )
            return True

        # Non-TTY (CI, redirected stdin) → fail-safe deny.
        if not (sys.stdin and sys.stdin.isatty()):
            logger.warning(
                "DiskMonitor: stdin is not a TTY; cannot prompt. "
                "Treating as decline. Pass auto_yes=True to override."
            )
            return False

        try:
            answer = input(f"{prompt} Continue? [y/N]: ")
        except (EOFError, KeyboardInterrupt):
            logger.warning(
                "DiskMonitor: input interrupted; treating as decline."
            )
            return False

        return answer.strip().lower() in {"y", "yes"}

    # ----------------------------------------------- high-level helpers --

    def before_write(
        self,
        operation_name: str,
        *,
        estimated_size_gb: float = 0.0,
        interactive: bool = True,
        auto_yes: bool = False,
        required: bool = False,
    ) -> bool:
        """High-level helper: kiểm tra free space + storage quota trước
        một thao tác ghi file lớn, prompt confirmation nếu vi phạm.

        Đây là entry point chính cho runner scripts (training, model
        download, replay save). Quy trình:

        1. Gọi :meth:`check_free_space` với ``required_gb=estimated_size_gb``
        2. Gọi :meth:`check_storage_quota` với ``estimated_additional_gb=
           estimated_size_gb``
        3. Nếu BẤT KỲ check nào fail VÀ ``interactive=True``: prompt user
           xác nhận; nếu user decline → return ``False`` (hoặc raise
           :class:`RuntimeError` nếu ``required=True``)

        Args:
            operation_name: Tên thao tác (dùng trong log + prompt). Ví dụ
                ``"download HuggingFace model"``, ``"save IFT checkpoint"``.
            estimated_size_gb: Ước tính kích thước file sẽ ghi (GB).
            interactive: Nếu ``True`` (default) và check fail, prompt user.
                Nếu ``False``, chỉ log warning và return ``False`` ngay.
            auto_yes: Tự động "yes" cho prompt (CI / batch jobs). Bỏ qua
                khi ``interactive=False``.
            required: Nếu ``True`` và operation bị decline (hoặc check
                fail trong non-interactive mode), raise :class:`RuntimeError`
                thay vì return ``False``. Dùng cho thao tác critical mà
                runner KHÔNG có cách fallback an toàn.

        Returns:
            ``True`` nếu operation được phép tiếp tục (mọi check pass HOẶC
            user xác nhận), ``False`` nếu user decline / non-interactive
            decline (chỉ khi ``required=False``).

        Raises:
            RuntimeError: Khi ``required=True`` và operation bị decline.
        """
        if estimated_size_gb < 0:
            raise ValueError(
                "estimated_size_gb must be >= 0, got "
                f"{estimated_size_gb}"
            )

        free_ok = self.check_free_space(
            required_gb=estimated_size_gb if estimated_size_gb > 0 else None
        )
        quota_ok = self.check_storage_quota(
            estimated_additional_gb=estimated_size_gb
        )

        if free_ok and quota_ok:
            logger.info(
                "DiskMonitor: pre-write checks OK for %r (estimated "
                "%.2fGB).",
                operation_name,
                estimated_size_gb,
            )
            return True

        # Có vi phạm — quyết định flow.
        prompt = (
            f"DiskMonitor: pre-write check FAILED for {operation_name!r} "
            f"(estimated {estimated_size_gb:.2f}GB). See warnings above."
        )

        if not interactive:
            logger.warning("%s Non-interactive mode → declining.", prompt)
            if required:
                raise RuntimeError(
                    f"{prompt} Required operation cannot proceed without "
                    "user confirmation."
                )
            return False

        approved = self.require_confirmation(prompt, auto_yes=auto_yes)
        if approved:
            logger.warning(
                "DiskMonitor: user approved %r despite warnings.",
                operation_name,
            )
            return True

        logger.warning(
            "DiskMonitor: user declined %r; aborting write.",
            operation_name,
        )
        if required:
            raise RuntimeError(
                f"User declined required operation {operation_name!r}; "
                "aborting."
            )
        return False
