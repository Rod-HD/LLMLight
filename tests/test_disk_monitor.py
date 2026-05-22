"""Unit tests for ``src.disk_monitor``.

Validates:
    - Requirement 11.7 (storage warning when total models+checkpoints > 250GB
      and confirmation flow)
    - Requirement 11.8 (free-space warning when drive < 10GB before writes)

Tests use ``unittest.mock.patch`` to stub ``shutil.disk_usage`` and
``sys.stdin.isatty`` so we don't depend on real disk state or interactive
TTY in CI. The disk monitor is a runtime helper (not a one-shot pre-flight
smoke check), so unit tests focus on:

* threshold branches (free space, storage quota)
* user confirmation flow (yes / no / EOF / non-TTY)
* the high-level ``before_write`` orchestration
"""

from __future__ import annotations

import io
import logging
import sys
import types
from pathlib import Path
from unittest import mock

import pytest

# Make ``src`` importable when running ``pytest`` from project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.disk_monitor import DiskMonitor  # noqa: E402
from src.preflight_checker import PreflightChecker  # noqa: E402


# =========================================================================
# helpers
# =========================================================================


def _usage(free_gb: float, total_gb: float = 500.0):
    """Build a fake ``shutil.disk_usage`` result with custom free GB."""
    return types.SimpleNamespace(
        total=int(total_gb * 1024 ** 3),
        used=int((total_gb - free_gb) * 1024 ** 3),
        free=int(free_gb * 1024 ** 3),
    )


# =========================================================================
# constructor validation
# =========================================================================


class TestInit:
    def test_default_thresholds_match_preflight_checker(self):
        """DiskMonitor defaults must match PreflightChecker constants
        (Requirements 11.7, 11.8)."""
        monitor = DiskMonitor()
        assert monitor.free_warn_gb == PreflightChecker.DISK_WARN_GB
        assert monitor.storage_warn_gb == PreflightChecker.STORAGE_WARN_GB
        assert set(monitor.storage_subdirs) == set(
            PreflightChecker.STORAGE_SUBDIRS
        )

    def test_rejects_negative_free_threshold(self):
        with pytest.raises(ValueError, match="free_warn_gb"):
            DiskMonitor(free_warn_gb=-1)

    def test_rejects_negative_storage_threshold(self):
        with pytest.raises(ValueError, match="storage_warn_gb"):
            DiskMonitor(storage_warn_gb=-1)

    def test_custom_subdirs_override_default(self):
        monitor = DiskMonitor(storage_subdirs=("models",))
        assert monitor.storage_subdirs == ("models",)


# =========================================================================
# get_free_gb
# =========================================================================


class TestGetFreeGb:
    def test_returns_free_gb_when_drive_exists(self):
        monitor = DiskMonitor(drive="/mnt/d")
        with mock.patch(
            "src.disk_monitor.shutil.disk_usage",
            return_value=_usage(free_gb=50.0),
        ):
            assert monitor.get_free_gb() == pytest.approx(50.0, rel=1e-6)

    def test_returns_none_when_drive_missing(self):
        monitor = DiskMonitor(drive="/__definitely_not_a_real_mount__")
        assert monitor.get_free_gb() is None

    def test_returns_none_on_os_error(self):
        monitor = DiskMonitor(drive="/mnt/d")
        with mock.patch(
            "src.disk_monitor.shutil.disk_usage",
            side_effect=OSError("permission denied"),
        ):
            assert monitor.get_free_gb() is None


# =========================================================================
# get_storage_used_gb
# =========================================================================


class TestGetStorageUsedGb:
    def test_returns_zero_when_no_subdirs_exist(self, tmp_path: Path):
        monitor = DiskMonitor(project_dir=str(tmp_path))
        assert monitor.get_storage_used_gb() == 0.0

    def test_sums_existing_subdirs(self, tmp_path: Path):
        (tmp_path / "models").mkdir()
        (tmp_path / "checkpoints").mkdir()
        (tmp_path / "models" / "a.bin").write_bytes(b"x" * 1024)
        (tmp_path / "checkpoints" / "b.bin").write_bytes(b"x" * 2048)
        monitor = DiskMonitor(project_dir=str(tmp_path))
        used_gb = monitor.get_storage_used_gb()
        # 3072 bytes total, expressed as GB (binary). Tiny number, so check
        # equality at byte granularity.
        assert used_gb == pytest.approx(3072 / (1024 ** 3), rel=1e-9)

    def test_skips_missing_subdir(self, tmp_path: Path):
        # Only models/ exists; checkpoints/ missing.
        (tmp_path / "models").mkdir()
        (tmp_path / "models" / "a.bin").write_bytes(b"x" * 100)
        monitor = DiskMonitor(project_dir=str(tmp_path))
        assert monitor.get_storage_used_gb() == pytest.approx(
            100 / (1024 ** 3), rel=1e-9
        )

    def test_uses_drive_when_project_dir_none(self, tmp_path: Path):
        # When project_dir is None, drive is used as the storage root.
        (tmp_path / "models").mkdir()
        (tmp_path / "models" / "a.bin").write_bytes(b"x" * 500)
        monitor = DiskMonitor(drive=str(tmp_path), project_dir=None)
        assert monitor.get_storage_used_gb() == pytest.approx(
            500 / (1024 ** 3), rel=1e-9
        )


# =========================================================================
# check_free_space
# =========================================================================


class TestCheckFreeSpace:
    def test_returns_true_and_logs_info_when_above_threshold(
        self, caplog: pytest.LogCaptureFixture
    ):
        monitor = DiskMonitor(drive="/mnt/d", free_warn_gb=10.0)
        with mock.patch(
            "src.disk_monitor.shutil.disk_usage",
            return_value=_usage(free_gb=100.0),
        ):
            with caplog.at_level(logging.INFO, logger="src.disk_monitor"):
                ok = monitor.check_free_space()
        assert ok is True
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert warnings == []

    def test_returns_false_and_warns_when_below_threshold(
        self, caplog: pytest.LogCaptureFixture
    ):
        monitor = DiskMonitor(drive="/mnt/d", free_warn_gb=10.0)
        with mock.patch(
            "src.disk_monitor.shutil.disk_usage",
            return_value=_usage(free_gb=5.0),
        ):
            with caplog.at_level(logging.WARNING, logger="src.disk_monitor"):
                ok = monitor.check_free_space()
        assert ok is False
        warns = [r.getMessage() for r in caplog.records
                 if r.levelno == logging.WARNING]
        assert any("[DISK_WARN]" in m and "5.00GB" in m for m in warns), (
            f"Expected DISK_WARN with 5.00GB, got: {warns}"
        )

    def test_returns_false_when_required_gb_exceeds_free(
        self, caplog: pytest.LogCaptureFixture
    ):
        # Free is above the global threshold but below the requested size.
        monitor = DiskMonitor(drive="/mnt/d", free_warn_gb=10.0)
        with mock.patch(
            "src.disk_monitor.shutil.disk_usage",
            return_value=_usage(free_gb=15.0),
        ):
            with caplog.at_level(logging.WARNING, logger="src.disk_monitor"):
                ok = monitor.check_free_space(required_gb=20.0)
        assert ok is False
        warns = [r.getMessage() for r in caplog.records
                 if r.levelno == logging.WARNING]
        assert any("Required: ~20.00GB" in m for m in warns), (
            f"Expected Required warning, got: {warns}"
        )

    def test_returns_true_when_free_unknown(
        self, caplog: pytest.LogCaptureFixture
    ):
        # Drive missing → get_free_gb returns None → check_free_space treats
        # as OK to avoid blocking on platforms without /mnt/d.
        monitor = DiskMonitor(drive="/__definitely_not_a_real_mount__")
        with caplog.at_level(logging.WARNING, logger="src.disk_monitor"):
            ok = monitor.check_free_space()
        assert ok is True


# =========================================================================
# check_storage_quota
# =========================================================================


class TestCheckStorageQuota:
    def test_returns_true_when_under_threshold(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ):
        monitor = DiskMonitor(
            project_dir=str(tmp_path), storage_warn_gb=250.0
        )
        with caplog.at_level(logging.INFO, logger="src.disk_monitor"):
            ok = monitor.check_storage_quota()
        assert ok is True
        warns = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert warns == []

    def test_returns_false_and_warns_when_storage_exceeds_threshold(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ):
        (tmp_path / "models").mkdir()
        (tmp_path / "checkpoints").mkdir()
        monitor = DiskMonitor(
            project_dir=str(tmp_path), storage_warn_gb=250.0
        )
        # Force used_gb computation to return 260 GB.
        big_total_bytes = int(260 * 1024 ** 3)
        with mock.patch.object(
            PreflightChecker,
            "_dir_size_bytes",
            side_effect=[big_total_bytes // 2, big_total_bytes // 2],
        ):
            with caplog.at_level(logging.WARNING, logger="src.disk_monitor"):
                ok = monitor.check_storage_quota()
        assert ok is False
        warns = [r.getMessage() for r in caplog.records
                 if r.levelno == logging.WARNING]
        assert any(
            "[STORAGE_WARN]" in m and "260." in m for m in warns
        ), f"Expected STORAGE_WARN at 260GB, got: {warns}"

    def test_includes_estimated_additional_in_projection(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ):
        # Used 240 GB, planning to add 20 GB → projected 260 GB exceeds 250.
        (tmp_path / "models").mkdir()
        monitor = DiskMonitor(
            project_dir=str(tmp_path), storage_warn_gb=250.0
        )
        with mock.patch.object(
            PreflightChecker,
            "_dir_size_bytes",
            return_value=int(240 * 1024 ** 3),
        ):
            with caplog.at_level(logging.WARNING, logger="src.disk_monitor"):
                ok = monitor.check_storage_quota(
                    estimated_additional_gb=20.0
                )
        assert ok is False
        warns = [r.getMessage() for r in caplog.records
                 if r.levelno == logging.WARNING]
        assert any("[STORAGE_WARN]" in m for m in warns)
        # Should mention both current and planned in the projection.
        assert any("240.00GB" in m and "20.00GB" in m for m in warns), (
            f"Expected projection breakdown, got: {warns}"
        )

    def test_returns_true_when_addition_keeps_under_threshold(
        self, tmp_path: Path
    ):
        (tmp_path / "models").mkdir()
        monitor = DiskMonitor(
            project_dir=str(tmp_path), storage_warn_gb=250.0
        )
        with mock.patch.object(
            PreflightChecker,
            "_dir_size_bytes",
            return_value=int(100 * 1024 ** 3),
        ):
            assert monitor.check_storage_quota(
                estimated_additional_gb=50.0
            ) is True

    def test_rejects_negative_estimated_additional(self, tmp_path: Path):
        monitor = DiskMonitor(project_dir=str(tmp_path))
        with pytest.raises(ValueError, match="estimated_additional_gb"):
            monitor.check_storage_quota(estimated_additional_gb=-1)


# =========================================================================
# require_confirmation
# =========================================================================


class TestRequireConfirmation:
    def test_auto_yes_returns_true_without_prompting(self):
        monitor = DiskMonitor()
        with mock.patch("builtins.input") as fake_input:
            assert (
                monitor.require_confirmation(
                    "test prompt", auto_yes=True
                )
                is True
            )
        fake_input.assert_not_called()

    def test_returns_false_when_stdin_not_tty(
        self, caplog: pytest.LogCaptureFixture
    ):
        monitor = DiskMonitor()
        # Replace stdin with a non-TTY stream (BytesIO/StringIO is non-TTY).
        fake_stdin = io.StringIO()
        with mock.patch.object(sys, "stdin", fake_stdin):
            with caplog.at_level(logging.WARNING, logger="src.disk_monitor"):
                ok = monitor.require_confirmation("test prompt")
        assert ok is False

    def test_returns_true_on_yes_input(self):
        monitor = DiskMonitor()
        fake_stdin = mock.MagicMock()
        fake_stdin.isatty.return_value = True
        with mock.patch.object(sys, "stdin", fake_stdin), mock.patch(
            "builtins.input", return_value="yes"
        ):
            assert monitor.require_confirmation("test") is True

    def test_returns_true_on_y_input_case_insensitive(self):
        monitor = DiskMonitor()
        fake_stdin = mock.MagicMock()
        fake_stdin.isatty.return_value = True
        with mock.patch.object(sys, "stdin", fake_stdin), mock.patch(
            "builtins.input", return_value="  Y  "
        ):
            assert monitor.require_confirmation("test") is True

    def test_returns_false_on_empty_input(self):
        monitor = DiskMonitor()
        fake_stdin = mock.MagicMock()
        fake_stdin.isatty.return_value = True
        with mock.patch.object(sys, "stdin", fake_stdin), mock.patch(
            "builtins.input", return_value=""
        ):
            assert monitor.require_confirmation("test") is False

    def test_returns_false_on_no_input(self):
        monitor = DiskMonitor()
        fake_stdin = mock.MagicMock()
        fake_stdin.isatty.return_value = True
        with mock.patch.object(sys, "stdin", fake_stdin), mock.patch(
            "builtins.input", return_value="n"
        ):
            assert monitor.require_confirmation("test") is False

    def test_returns_false_on_eof(self):
        monitor = DiskMonitor()
        fake_stdin = mock.MagicMock()
        fake_stdin.isatty.return_value = True
        with mock.patch.object(sys, "stdin", fake_stdin), mock.patch(
            "builtins.input", side_effect=EOFError
        ):
            assert monitor.require_confirmation("test") is False

    def test_returns_false_on_keyboard_interrupt(self):
        monitor = DiskMonitor()
        fake_stdin = mock.MagicMock()
        fake_stdin.isatty.return_value = True
        with mock.patch.object(sys, "stdin", fake_stdin), mock.patch(
            "builtins.input", side_effect=KeyboardInterrupt
        ):
            assert monitor.require_confirmation("test") is False


# =========================================================================
# before_write (high-level)
# =========================================================================


class TestBeforeWrite:
    def test_returns_true_when_all_checks_pass(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ):
        monitor = DiskMonitor(
            drive="/mnt/d",
            project_dir=str(tmp_path),
            free_warn_gb=10.0,
            storage_warn_gb=250.0,
        )
        with mock.patch(
            "src.disk_monitor.shutil.disk_usage",
            return_value=_usage(free_gb=100.0),
        ):
            with caplog.at_level(logging.INFO, logger="src.disk_monitor"):
                ok = monitor.before_write(
                    "test op", estimated_size_gb=1.0
                )
        assert ok is True

    def test_returns_false_in_non_interactive_when_check_fails(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ):
        monitor = DiskMonitor(
            drive="/mnt/d",
            project_dir=str(tmp_path),
            free_warn_gb=10.0,
        )
        with mock.patch(
            "src.disk_monitor.shutil.disk_usage",
            return_value=_usage(free_gb=2.0),
        ):
            with caplog.at_level(logging.WARNING, logger="src.disk_monitor"):
                ok = monitor.before_write(
                    "low-space op",
                    estimated_size_gb=1.0,
                    interactive=False,
                )
        assert ok is False

    def test_raises_runtime_error_when_required_and_non_interactive_fail(
        self, tmp_path: Path
    ):
        monitor = DiskMonitor(
            drive="/mnt/d",
            project_dir=str(tmp_path),
            free_warn_gb=10.0,
        )
        with mock.patch(
            "src.disk_monitor.shutil.disk_usage",
            return_value=_usage(free_gb=2.0),
        ):
            with pytest.raises(RuntimeError, match="cannot proceed"):
                monitor.before_write(
                    "critical op",
                    estimated_size_gb=1.0,
                    interactive=False,
                    required=True,
                )

    def test_returns_true_when_user_approves_after_warning(
        self, tmp_path: Path
    ):
        monitor = DiskMonitor(
            drive="/mnt/d",
            project_dir=str(tmp_path),
            free_warn_gb=10.0,
        )
        with mock.patch(
            "src.disk_monitor.shutil.disk_usage",
            return_value=_usage(free_gb=2.0),
        ), mock.patch.object(
            DiskMonitor, "require_confirmation", return_value=True
        ) as confirm:
            ok = monitor.before_write(
                "user-approves op",
                estimated_size_gb=1.0,
                interactive=True,
            )
        assert ok is True
        confirm.assert_called_once()

    def test_returns_false_when_user_declines(self, tmp_path: Path):
        monitor = DiskMonitor(
            drive="/mnt/d",
            project_dir=str(tmp_path),
            free_warn_gb=10.0,
        )
        with mock.patch(
            "src.disk_monitor.shutil.disk_usage",
            return_value=_usage(free_gb=2.0),
        ), mock.patch.object(
            DiskMonitor, "require_confirmation", return_value=False
        ):
            ok = monitor.before_write(
                "user-declines op",
                estimated_size_gb=1.0,
                interactive=True,
            )
        assert ok is False

    def test_raises_runtime_error_when_required_and_user_declines(
        self, tmp_path: Path
    ):
        monitor = DiskMonitor(
            drive="/mnt/d",
            project_dir=str(tmp_path),
            free_warn_gb=10.0,
        )
        with mock.patch(
            "src.disk_monitor.shutil.disk_usage",
            return_value=_usage(free_gb=2.0),
        ), mock.patch.object(
            DiskMonitor, "require_confirmation", return_value=False
        ):
            with pytest.raises(RuntimeError, match="declined"):
                monitor.before_write(
                    "required op",
                    estimated_size_gb=1.0,
                    interactive=True,
                    required=True,
                )

    def test_auto_yes_short_circuits_confirmation(self, tmp_path: Path):
        monitor = DiskMonitor(
            drive="/mnt/d",
            project_dir=str(tmp_path),
            free_warn_gb=10.0,
        )
        with mock.patch(
            "src.disk_monitor.shutil.disk_usage",
            return_value=_usage(free_gb=2.0),
        ):
            ok = monitor.before_write(
                "auto-yes op",
                estimated_size_gb=1.0,
                interactive=True,
                auto_yes=True,
            )
        assert ok is True

    def test_rejects_negative_estimated_size(self, tmp_path: Path):
        monitor = DiskMonitor(project_dir=str(tmp_path))
        with pytest.raises(ValueError, match="estimated_size_gb"):
            monitor.before_write("bad op", estimated_size_gb=-1.0)
