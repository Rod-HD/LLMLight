"""Unit tests for ``src.preflight_checker``.

Validates:
    - Requirement 1.8 (GPU passthrough check via nvidia-smi)
    - Requirement 11.5 (read/write at Unicode path)
    - Requirement 11.6 (pre-flight ASCII write/read at PROJECT_DIR)
    - Requirement 11.7 (storage warning at >250GB)
    - Requirement 11.8 (disk free warning at <10GB)

Tests use ``unittest.mock.patch`` to stub ``subprocess.run`` and
``shutil.disk_usage`` because real GPU / disk state are unavailable in CI.
The checks themselves are smoke checks (Testing Strategy section in
design.md), so unit tests focus on branch coverage of the logic.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import types
from pathlib import Path
from unittest import mock

import pytest

# Make ``src`` importable when running ``pytest`` from project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preflight_checker import PreflightChecker  # noqa: E402


# =========================================================================
# check_gpu_passthrough
# =========================================================================


def _make_completed(returncode: int, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(
        args=["nvidia-smi"], returncode=returncode, stdout=stdout, stderr=stderr
    )


class TestCheckGpuPassthrough:
    """Requirement 1.8: nvidia-smi exit=0 + stdout chứa 'RTX 4060'."""

    def test_passes_when_exit_zero_and_gpu_substring_present(self):
        checker = PreflightChecker()
        fake_stdout = "GPU 0: NVIDIA GeForce RTX 4060 Laptop GPU\n"
        with mock.patch(
            "src.preflight_checker.subprocess.run",
            return_value=_make_completed(0, stdout=fake_stdout),
        ):
            checker.check_gpu_passthrough()  # should not raise

    def test_raises_when_nonzero_exit_code(self):
        checker = PreflightChecker()
        with mock.patch(
            "src.preflight_checker.subprocess.run",
            return_value=_make_completed(
                9, stdout="", stderr="NVIDIA-SMI has failed"
            ),
        ):
            with pytest.raises(RuntimeError) as excinfo:
                checker.check_gpu_passthrough()
        msg = str(excinfo.value)
        assert "exited with code 9" in msg
        assert "NVIDIA-SMI has failed" in msg

    def test_raises_when_gpu_substring_missing(self):
        checker = PreflightChecker()
        # Exit 0 but stdout shows a different GPU.
        fake_stdout = "GPU 0: NVIDIA GeForce GTX 1080\n"
        with mock.patch(
            "src.preflight_checker.subprocess.run",
            return_value=_make_completed(0, stdout=fake_stdout),
        ):
            with pytest.raises(RuntimeError) as excinfo:
                checker.check_gpu_passthrough()
        msg = str(excinfo.value)
        assert "RTX 4060" in msg
        assert "GTX 1080" in msg  # echoes the actual stdout
        assert "Exit code = 0" in msg

    def test_raises_when_nvidia_smi_not_found(self):
        checker = PreflightChecker()
        with mock.patch(
            "src.preflight_checker.subprocess.run",
            side_effect=FileNotFoundError("No such file: nvidia-smi"),
        ):
            with pytest.raises(RuntimeError) as excinfo:
                checker.check_gpu_passthrough()
        assert "'nvidia-smi' not found" in str(excinfo.value)

    def test_raises_when_nvidia_smi_times_out(self):
        checker = PreflightChecker()
        with mock.patch(
            "src.preflight_checker.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=30),
        ):
            with pytest.raises(RuntimeError) as excinfo:
                checker.check_gpu_passthrough()
        assert "timed out" in str(excinfo.value)


# =========================================================================
# check_path_unicode
# =========================================================================


class TestCheckPathUnicode:
    """Requirement 11.5, 11.6: ASCII round-trip at Unicode + spaced path."""

    def test_passes_on_normal_directory(self, tmp_path: Path):
        checker = PreflightChecker()
        checker.check_path_unicode(str(tmp_path))
        # Cleanup must remove the test file.
        assert not (tmp_path / PreflightChecker.PREFLIGHT_TEST_FILENAME).exists()

    def test_passes_on_unicode_vietnamese_path_with_spaces(self, tmp_path: Path):
        # Real-world style path: "Trí tuệ nhân tạo/Đồ án/LLMLight".
        unicode_dir = tmp_path / "Trí tuệ nhân tạo" / "Đồ án" / "LLMLight"
        unicode_dir.mkdir(parents=True)
        checker = PreflightChecker()
        checker.check_path_unicode(str(unicode_dir))
        assert not (unicode_dir / PreflightChecker.PREFLIGHT_TEST_FILENAME).exists()

    def test_raises_not_found_when_path_missing(self, tmp_path: Path):
        checker = PreflightChecker()
        missing = tmp_path / "does_not_exist"
        with pytest.raises(OSError) as excinfo:
            checker.check_path_unicode(str(missing))
        msg = str(excinfo.value)
        assert "[not_found]" in msg
        assert str(missing) in msg

    def test_raises_not_found_when_path_is_file(self, tmp_path: Path):
        checker = PreflightChecker()
        regular_file = tmp_path / "not_a_dir.txt"
        regular_file.write_text("hello")
        with pytest.raises(OSError) as excinfo:
            checker.check_path_unicode(str(regular_file))
        assert "[not_found]" in str(excinfo.value)

    def test_raises_not_found_when_project_dir_is_none(self):
        checker = PreflightChecker()
        with pytest.raises(OSError) as excinfo:
            checker.check_path_unicode(None)  # type: ignore[arg-type]
        assert "[not_found]" in str(excinfo.value)
        assert "None" in str(excinfo.value)

    def test_raises_permission_when_write_denied(self, tmp_path: Path):
        checker = PreflightChecker()
        with mock.patch(
            "src.preflight_checker.Path.write_text",
            side_effect=PermissionError("denied"),
        ):
            with pytest.raises(OSError) as excinfo:
                checker.check_path_unicode(str(tmp_path))
        msg = str(excinfo.value)
        assert "[permission]" in msg
        assert str(tmp_path) in msg

    def test_raises_encoding_when_read_back_corrupted(self, tmp_path: Path):
        checker = PreflightChecker()
        # First write succeeds normally; force read_text to raise UnicodeDecodeError.
        with mock.patch(
            "src.preflight_checker.Path.read_text",
            side_effect=UnicodeDecodeError(
                "ascii", b"\xff", 0, 1, "ordinal not in range(128)"
            ),
        ):
            with pytest.raises(OSError) as excinfo:
                checker.check_path_unicode(str(tmp_path))
        assert "[encoding]" in str(excinfo.value)
        # Cleanup still happens (write succeeded, file should be removed).
        assert not (tmp_path / PreflightChecker.PREFLIGHT_TEST_FILENAME).exists()

    def test_raises_mismatch_when_read_returns_wrong_content(self, tmp_path: Path):
        checker = PreflightChecker()
        with mock.patch(
            "src.preflight_checker.Path.read_text",
            return_value="garbage",
        ):
            with pytest.raises(OSError) as excinfo:
                checker.check_path_unicode(str(tmp_path))
        msg = str(excinfo.value)
        assert "[mismatch]" in msg
        assert "garbage" in msg

    def test_cleanup_removes_test_file_on_success(self, tmp_path: Path):
        checker = PreflightChecker()
        checker.check_path_unicode(str(tmp_path))
        # Explicit assertion for the cleanup contract.
        leftover = tmp_path / PreflightChecker.PREFLIGHT_TEST_FILENAME
        assert not leftover.exists()


# =========================================================================
# check_disk_space
# =========================================================================


class TestCheckDiskSpace:
    """Requirement 11.7, 11.8: warnings for low free space + large storage."""

    def _usage(self, free_gb: float) -> os.terminal_size | object:
        # shutil.disk_usage returns a 3-tuple-like object with ``free``.
        return types.SimpleNamespace(
            total=int(500 * 1024 ** 3),
            used=int((500 - free_gb) * 1024 ** 3),
            free=int(free_gb * 1024 ** 3),
        )

    def test_warns_when_free_space_below_threshold(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ):
        checker = PreflightChecker()
        with mock.patch(
            "src.preflight_checker.shutil.disk_usage",
            return_value=self._usage(free_gb=5.0),
        ):
            with caplog.at_level(logging.WARNING, logger="src.preflight_checker"):
                checker.check_disk_space(drive="/mnt/d", project_dir=str(tmp_path))
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any("only 5.00 GB free" in r.getMessage() for r in warnings), (
            f"Expected free-space warning, got: {[r.getMessage() for r in warnings]}"
        )

    def test_no_warning_when_free_space_above_threshold(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ):
        checker = PreflightChecker()
        with mock.patch(
            "src.preflight_checker.shutil.disk_usage",
            return_value=self._usage(free_gb=100.0),
        ):
            with caplog.at_level(logging.WARNING, logger="src.preflight_checker"):
                checker.check_disk_space(drive="/mnt/d", project_dir=str(tmp_path))
        free_warnings = [
            r for r in caplog.records
            if r.levelno == logging.WARNING and "free" in r.getMessage()
        ]
        assert free_warnings == []

    def test_warns_when_total_storage_exceeds_threshold(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ):
        # Create a fake 'models/' directory; mock _dir_size_bytes so we don't
        # have to actually allocate 250GB on disk.
        (tmp_path / "models").mkdir()
        (tmp_path / "checkpoints").mkdir()
        checker = PreflightChecker()

        # 260 GB total split between the two subdirs (in bytes).
        big_total = int(260 * 1024 ** 3)
        with mock.patch(
            "src.preflight_checker.shutil.disk_usage",
            return_value=self._usage(free_gb=100.0),
        ), mock.patch.object(
            PreflightChecker,
            "_dir_size_bytes",
            side_effect=[big_total // 2, big_total // 2],
        ):
            with caplog.at_level(logging.WARNING, logger="src.preflight_checker"):
                checker.check_disk_space(drive="/mnt/d", project_dir=str(tmp_path))
        msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert any("model + checkpoint" in m and "260." in m for m in msgs), (
            f"Expected storage warning, got: {msgs}"
        )

    def test_skips_storage_check_when_no_subdirs_exist(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ):
        # tmp_path has no models/ or checkpoints/ — storage section silent.
        checker = PreflightChecker()
        with mock.patch(
            "src.preflight_checker.shutil.disk_usage",
            return_value=self._usage(free_gb=100.0),
        ):
            with caplog.at_level(logging.INFO, logger="src.preflight_checker"):
                checker.check_disk_space(drive="/mnt/d", project_dir=str(tmp_path))
        storage_msgs = [
            r.getMessage()
            for r in caplog.records
            if "model + checkpoint" in r.getMessage()
        ]
        assert storage_msgs == []

    def test_warns_when_drive_does_not_exist(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ):
        checker = PreflightChecker()
        # Real call against a path that cannot exist on Windows or Linux.
        with caplog.at_level(logging.WARNING, logger="src.preflight_checker"):
            checker.check_disk_space(
                drive="/__definitely_not_a_real_mount__", project_dir=str(tmp_path)
            )
        msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert any("cannot stat drive" in m for m in msgs), (
            f"Expected stat-drive warning, got: {msgs}"
        )

    def test_check_disk_space_does_not_raise(self, tmp_path: Path):
        """Disk check is warning-only by spec — should never raise."""
        checker = PreflightChecker()
        # Even with broken inputs, no raise.
        checker.check_disk_space(
            drive="/__definitely_not_a_real_mount__", project_dir=str(tmp_path)
        )

    def test_dir_size_bytes_computes_total(self, tmp_path: Path):
        (tmp_path / "a.bin").write_bytes(b"x" * 100)
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "b.bin").write_bytes(b"x" * 50)
        size = PreflightChecker._dir_size_bytes(tmp_path)
        assert size == 150


# =========================================================================
# run_all
# =========================================================================


class TestRunAll:
    """Verify run_all calls the three checks in the documented order."""

    def test_run_all_invokes_all_three_checks_in_order(self, tmp_path: Path):
        checker = PreflightChecker()
        call_order: list[str] = []

        def gpu_stub():
            call_order.append("gpu")

        def path_stub(project_dir):
            call_order.append("path")
            assert project_dir == str(tmp_path)

        def disk_stub(*, project_dir=None, **kwargs):
            call_order.append("disk")
            assert project_dir == str(tmp_path)

        with mock.patch.object(checker, "check_gpu_passthrough", side_effect=gpu_stub), \
             mock.patch.object(checker, "check_path_unicode", side_effect=path_stub), \
             mock.patch.object(checker, "check_disk_space", side_effect=disk_stub):
            checker.run_all(str(tmp_path))

        assert call_order == ["gpu", "path", "disk"]

    def test_run_all_propagates_gpu_failure(self, tmp_path: Path):
        checker = PreflightChecker()
        with mock.patch.object(
            checker,
            "check_gpu_passthrough",
            side_effect=RuntimeError("GPU bad"),
        ), mock.patch.object(checker, "check_path_unicode") as path_mock, \
             mock.patch.object(checker, "check_disk_space") as disk_mock:
            with pytest.raises(RuntimeError, match="GPU bad"):
                checker.run_all(str(tmp_path))
        # Subsequent checks must not run.
        path_mock.assert_not_called()
        disk_mock.assert_not_called()

    def test_run_all_propagates_path_failure(self, tmp_path: Path):
        checker = PreflightChecker()
        with mock.patch.object(checker, "check_gpu_passthrough"), \
             mock.patch.object(
                 checker,
                 "check_path_unicode",
                 side_effect=OSError("path bad"),
             ), \
             mock.patch.object(checker, "check_disk_space") as disk_mock:
            with pytest.raises(OSError, match="path bad"):
                checker.run_all(str(tmp_path))
        disk_mock.assert_not_called()
