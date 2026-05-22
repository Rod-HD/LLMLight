"""Environment Pre-Flight Checker (Component 11).

Chạy các kiểm tra môi trường BEFORE khi build CityFlow hoặc cài đặt
torch/bitsandbytes:

1. ``check_gpu_passthrough``: chạy ``nvidia-smi``, yêu cầu exit code = 0 và
   stdout chứa chuỗi nhận diện GPU (mặc định ``"RTX 4060"``); raise
   :class:`RuntimeError` chỉ rõ exit code + stdout/stderr nếu fail.
2. ``check_path_unicode``: ghi/đọc file ASCII tạm ``.preflight_test`` tại
   ``project_dir`` (đường dẫn chứa tiếng Việt + khoảng trắng) rồi xóa; raise
   :class:`OSError` (alias của ``IOError``) chỉ rõ đường dẫn + loại lỗi
   (``not_found`` / ``permission`` / ``encoding``) nếu fail.
3. ``check_disk_space``: cảnh báo (KHÔNG raise) nếu ổ D < 10GB trống hoặc tổng
   ``models/`` + ``checkpoints/`` > 250GB.

Module này được ``setup_env.sh`` gọi BEFORE cài torch/bitsandbytes và build
CityFlow; cũng được các runner scripts gọi như defensive check.

_Requirements_: 1.8, 11.5, 11.6, 11.7, 11.8
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


class PreflightChecker:
    """Kiểm tra môi trường trước khi build CityFlow / cài torch."""

    EXPECTED_GPU_SUBSTRING: str = "RTX 4060"
    DISK_WARN_GB: int = 10
    STORAGE_WARN_GB: int = 250
    PREFLIGHT_TEST_FILENAME: str = ".preflight_test"
    PREFLIGHT_TEST_CONTENT: str = "preflight ok"
    NVIDIA_SMI_TIMEOUT_SECONDS: int = 30
    # Các thư mục được tính vào "tổng model + checkpoint" cho check_disk_space.
    STORAGE_SUBDIRS: tuple[str, ...] = ("models", "checkpoints")

    # ------------------------------------------------------------------ GPU --

    def check_gpu_passthrough(self) -> None:
        """Verify CUDA passthrough qua ``nvidia-smi``.

        Yêu cầu:
          - exit code == 0
          - stdout chứa ``EXPECTED_GPU_SUBSTRING`` (mặc định ``"RTX 4060"``)

        Raises:
            RuntimeError: Nếu ``nvidia-smi`` không tồn tại, fail, hoặc stdout
                không chứa GPU substring kỳ vọng. Thông báo lỗi bao gồm
                exit code + stdout/stderr nhận được.
        """
        try:
            completed = subprocess.run(
                ["nvidia-smi"],
                capture_output=True,
                text=True,
                check=False,
                timeout=self.NVIDIA_SMI_TIMEOUT_SECONDS,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "GPU pre-flight FAILED: 'nvidia-smi' not found on PATH. "
                "On WSL2, ensure Windows NVIDIA driver is installed and "
                "CUDA passthrough is enabled. "
                f"Underlying error: {exc}"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                "GPU pre-flight FAILED: 'nvidia-smi' timed out after "
                f"{self.NVIDIA_SMI_TIMEOUT_SECONDS}s. "
                "GPU driver may be hung."
            ) from exc

        stdout = completed.stdout or ""
        stderr = completed.stderr or ""

        if completed.returncode != 0:
            raise RuntimeError(
                "GPU pre-flight FAILED: 'nvidia-smi' exited with code "
                f"{completed.returncode}.\n"
                f"stdout:\n{stdout}\n"
                f"stderr:\n{stderr}"
            )

        if self.EXPECTED_GPU_SUBSTRING not in stdout:
            raise RuntimeError(
                "GPU pre-flight FAILED: expected GPU substring "
                f"{self.EXPECTED_GPU_SUBSTRING!r} not found in nvidia-smi "
                f"stdout. Exit code = {completed.returncode}.\n"
                f"stdout:\n{stdout}\n"
                f"stderr:\n{stderr}"
            )

        logger.info(
            "GPU pre-flight OK: nvidia-smi exit=0, GPU substring %r found.",
            self.EXPECTED_GPU_SUBSTRING,
        )

    # ----------------------------------------------------------------- Path --

    def check_path_unicode(self, project_dir: str) -> None:
        """Verify rằng đường dẫn project (chứa tiếng Việt + khoảng trắng)
        có thể ghi/đọc file ASCII bình thường.

        Quy trình:
          1. Tạo file ASCII tạm ``.preflight_test`` tại ``project_dir``
          2. Ghi nội dung ASCII, đọc lại, so khớp byte-for-byte
          3. Xóa file sau khi xác nhận

        Args:
            project_dir: Đường dẫn tuyệt đối tới project root. Trên WSL2
                thường chứa ký tự Unicode tiếng Việt và khoảng trắng.

        Raises:
            OSError: Nếu ghi/đọc/xóa thất bại. ``OSError.args`` chứa
                ``error_kind`` (``"not_found"`` | ``"permission"``
                | ``"encoding"`` | ``"mismatch"`` | ``"unknown"``) và
                ``project_dir`` để dễ debug.
        """
        if project_dir is None:
            raise OSError(
                self._format_path_error(
                    "not_found",
                    project_dir="<None>",
                    detail="project_dir is None",
                )
            )

        project_path = Path(project_dir)

        if not project_path.exists():
            raise OSError(
                self._format_path_error(
                    "not_found",
                    project_dir=project_dir,
                    detail=f"Path does not exist: {project_path}",
                )
            )

        if not project_path.is_dir():
            raise OSError(
                self._format_path_error(
                    "not_found",
                    project_dir=project_dir,
                    detail=f"Not a directory: {project_path}",
                )
            )

        test_file = project_path / self.PREFLIGHT_TEST_FILENAME
        wrote = False
        try:
            try:
                test_file.write_text(
                    self.PREFLIGHT_TEST_CONTENT, encoding="ascii"
                )
                wrote = True
            except PermissionError as exc:
                raise OSError(
                    self._format_path_error(
                        "permission",
                        project_dir=project_dir,
                        detail=f"Cannot write {test_file}: {exc}",
                    )
                ) from exc
            except FileNotFoundError as exc:
                raise OSError(
                    self._format_path_error(
                        "not_found",
                        project_dir=project_dir,
                        detail=(
                            f"Cannot create {test_file} (parent missing or "
                            f"path invalid): {exc}"
                        ),
                    )
                ) from exc
            except UnicodeError as exc:
                raise OSError(
                    self._format_path_error(
                        "encoding",
                        project_dir=project_dir,
                        detail=(
                            "Filesystem rejected ASCII encoding when writing "
                            f"{test_file}: {exc}"
                        ),
                    )
                ) from exc
            except OSError as exc:
                raise OSError(
                    self._format_path_error(
                        "unknown",
                        project_dir=project_dir,
                        detail=f"Write failed for {test_file}: {exc}",
                    )
                ) from exc

            try:
                read_back = test_file.read_text(encoding="ascii")
            except UnicodeDecodeError as exc:
                raise OSError(
                    self._format_path_error(
                        "encoding",
                        project_dir=project_dir,
                        detail=(
                            f"Cannot read {test_file} as ASCII (encoding "
                            f"corruption): {exc}"
                        ),
                    )
                ) from exc
            except PermissionError as exc:
                raise OSError(
                    self._format_path_error(
                        "permission",
                        project_dir=project_dir,
                        detail=f"Cannot read {test_file}: {exc}",
                    )
                ) from exc
            except FileNotFoundError as exc:
                raise OSError(
                    self._format_path_error(
                        "not_found",
                        project_dir=project_dir,
                        detail=(
                            f"File vanished between write and read: "
                            f"{test_file}: {exc}"
                        ),
                    )
                ) from exc
            except OSError as exc:
                raise OSError(
                    self._format_path_error(
                        "unknown",
                        project_dir=project_dir,
                        detail=f"Read failed for {test_file}: {exc}",
                    )
                ) from exc

            if read_back != self.PREFLIGHT_TEST_CONTENT:
                raise OSError(
                    self._format_path_error(
                        "mismatch",
                        project_dir=project_dir,
                        detail=(
                            f"Round-trip mismatch at {test_file}: "
                            f"wrote {self.PREFLIGHT_TEST_CONTENT!r}, "
                            f"read back {read_back!r}"
                        ),
                    )
                )

            logger.info(
                "Path Unicode pre-flight OK at %s (wrote+read %s).",
                project_dir,
                self.PREFLIGHT_TEST_FILENAME,
            )

        finally:
            if wrote:
                try:
                    test_file.unlink()
                except OSError as exc:
                    # Cleanup failure is non-fatal but should be logged so users
                    # don't accumulate stray .preflight_test files.
                    logger.warning(
                        "Failed to delete pre-flight test file %s: %s",
                        test_file,
                        exc,
                    )

    @staticmethod
    def _format_path_error(
        error_kind: str, *, project_dir: str, detail: str
    ) -> str:
        """Format thông báo lỗi cho ``check_path_unicode``."""
        return (
            f"Path Unicode pre-flight FAILED [{error_kind}] at "
            f"project_dir={project_dir!r}: {detail}"
        )

    # ----------------------------------------------------------------- Disk --

    def check_disk_space(
        self,
        drive: str = "/mnt/d",
        project_dir: str | None = None,
    ) -> None:
        """Cảnh báo (qua logger) nếu disk free < 10GB hoặc tổng
        ``models/`` + ``checkpoints/`` > 250GB.

        KHÔNG raise — chỉ log warning. Caller có thể bật abort thủ công sau
        khi đọc warning.

        Args:
            drive: Mount point cần kiểm tra free space (mặc định ``/mnt/d``
                trên WSL2). Nếu drive không tồn tại trên hệ điều hành hiện
                tại, log warning và skip check.
            project_dir: Nếu truyền vào, ``models/`` và ``checkpoints/`` sẽ
                được tìm dưới đường dẫn này. Nếu ``None``, fallback về
                ``<drive>/models`` và ``<drive>/checkpoints``.
        """
        # ------- (a) Free space check ------------------------------------
        try:
            usage = shutil.disk_usage(drive)
        except (FileNotFoundError, NotADirectoryError) as exc:
            logger.warning(
                "Disk pre-flight: cannot stat drive %r (%s). "
                "Skipping free-space check.",
                drive,
                exc,
            )
            usage = None
        except OSError as exc:
            logger.warning(
                "Disk pre-flight: OS error while checking drive %r (%s). "
                "Skipping free-space check.",
                drive,
                exc,
            )
            usage = None

        if usage is not None:
            free_gb = usage.free / (1024 ** 3)
            if free_gb < self.DISK_WARN_GB:
                logger.warning(
                    "Disk pre-flight WARNING: drive %r has only %.2f GB free "
                    "(< %d GB threshold). Estimated need: ~10-20 GB for "
                    "CityFlow build + Python deps + HF cache.",
                    drive,
                    free_gb,
                    self.DISK_WARN_GB,
                )
            else:
                logger.info(
                    "Disk pre-flight OK: drive %r has %.2f GB free.",
                    drive,
                    free_gb,
                )

        # ------- (b) Model + checkpoint storage check --------------------
        storage_root = Path(project_dir) if project_dir else Path(drive)
        total_bytes = 0
        any_subdir_found = False
        for subdir in self.STORAGE_SUBDIRS:
            sub_path = storage_root / subdir
            if not sub_path.exists():
                continue
            any_subdir_found = True
            total_bytes += self._dir_size_bytes(sub_path)

        if any_subdir_found:
            total_gb = total_bytes / (1024 ** 3)
            if total_gb > self.STORAGE_WARN_GB:
                logger.warning(
                    "Disk pre-flight WARNING: total model + checkpoint "
                    "storage under %s is %.2f GB (> %d GB threshold). "
                    "Confirm before downloading more.",
                    storage_root,
                    total_gb,
                    self.STORAGE_WARN_GB,
                )
            else:
                logger.info(
                    "Disk pre-flight OK: total model + checkpoint storage "
                    "under %s is %.2f GB.",
                    storage_root,
                    total_gb,
                )

    @staticmethod
    def _dir_size_bytes(path: Path) -> int:
        """Tính tổng bytes của tất cả file dưới ``path`` (đệ quy)."""
        total = 0
        for dirpath, _dirnames, filenames in os.walk(path, followlinks=False):
            for filename in filenames:
                full = os.path.join(dirpath, filename)
                try:
                    # Symlinks đã bị skip qua followlinks=False ở mức directory;
                    # nhưng symlink ở cấp file vẫn cần xử lý — bỏ qua chúng.
                    if os.path.islink(full):
                        continue
                    total += os.path.getsize(full)
                except OSError:
                    # File có thể đã bị xóa giữa lúc walk và stat; bỏ qua.
                    continue
        return total

    # ------------------------------------------------------------- Run all --

    def run_all(self, project_dir: str) -> None:
        """Chạy 3 check theo thứ tự GPU → Path → Disk.

        Được gọi BEFORE cài torch/bitsandbytes và build CityFlow trong
        ``setup_env.sh``; cũng được runner scripts gọi như defensive check.

        Args:
            project_dir: Project root path (chứa tiếng Việt + khoảng trắng
                trên WSL2).

        Raises:
            RuntimeError: Nếu GPU check fail.
            OSError: Nếu Path Unicode check fail.

        Note:
            ``check_disk_space`` chỉ cảnh báo (không raise) — caller phải tự
            đọc log warning để quyết định abort.
        """
        self.check_gpu_passthrough()
        self.check_path_unicode(project_dir)
        self.check_disk_space(project_dir=project_dir)
