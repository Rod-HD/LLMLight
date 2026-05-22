"""Unit tests for ``src.vram_monitor``.

Validates:
    - Requirement 11.1 (VRAM usage stays below 7.5GB across train/inference)
    - Requirement 11.2 (raise informative error when VRAM exceeds limit)

Tests use ``unittest.mock`` to stub ``torch.cuda.*`` so we don't depend on
real CUDA hardware (Windows dev box does not have torch installed).

Strategy:

* The module imports torch lazily through ``_import_torch``. Tests patch
  this helper to inject a fake ``torch`` module with the cuda calls we
  care about (``is_available``, ``memory_allocated``, ``mem_get_info``).
* We document the intended behavior when CUDA is not available:
  ``get_vram_usage_mb`` and ``get_vram_free_mb`` return ``None``, and
  ``check_vram_available`` returns ``False`` (fail-safe — caller is
  about to load a GPU model but no GPU is available) and emits the
  ``[VRAM_ERROR]`` log line with ``Available: 0.00MB``.
"""

from __future__ import annotations

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

from src import vram_monitor  # noqa: E402
from src.vram_monitor import (  # noqa: E402
    check_vram_available,
    get_vram_free_mb,
    get_vram_usage_mb,
    is_cuda_available,
)


_BYTES_PER_MB = 1024 * 1024


# =========================================================================
# helpers
# =========================================================================


def _make_fake_torch(
    *,
    cuda_available: bool = True,
    allocated_bytes: int = 0,
    free_bytes: int = 8 * 1024 * _BYTES_PER_MB,
    total_bytes: int = 8 * 1024 * _BYTES_PER_MB,
    is_available_raises: BaseException | None = None,
    memory_allocated_raises: BaseException | None = None,
    mem_get_info_raises: BaseException | None = None,
) -> types.SimpleNamespace:
    """Build a fake ``torch`` module exposing the tiny API surface the
    monitor needs."""

    cuda_ns = types.SimpleNamespace()

    def _is_available():
        if is_available_raises is not None:
            raise is_available_raises
        return cuda_available

    def _memory_allocated(device=0):  # noqa: ARG001
        if memory_allocated_raises is not None:
            raise memory_allocated_raises
        return allocated_bytes

    def _mem_get_info(device=0):  # noqa: ARG001
        if mem_get_info_raises is not None:
            raise mem_get_info_raises
        return (free_bytes, total_bytes)

    cuda_ns.is_available = _is_available
    cuda_ns.memory_allocated = _memory_allocated
    cuda_ns.mem_get_info = _mem_get_info

    fake_torch = types.SimpleNamespace(cuda=cuda_ns)
    return fake_torch


def _patch_torch(fake_torch):
    """Patch ``vram_monitor._import_torch`` to return ``fake_torch``."""
    return mock.patch.object(
        vram_monitor, "_import_torch", return_value=fake_torch
    )


# =========================================================================
# is_cuda_available
# =========================================================================


class TestIsCudaAvailable:
    def test_returns_false_when_torch_not_installed(self):
        with _patch_torch(None):
            assert is_cuda_available() is False

    def test_returns_false_when_cuda_unavailable(self):
        fake = _make_fake_torch(cuda_available=False)
        with _patch_torch(fake):
            assert is_cuda_available() is False

    def test_returns_true_when_cuda_available(self):
        fake = _make_fake_torch(cuda_available=True)
        with _patch_torch(fake):
            assert is_cuda_available() is True

    def test_returns_false_on_internal_exception(
        self, caplog: pytest.LogCaptureFixture
    ):
        fake = _make_fake_torch(
            is_available_raises=RuntimeError("driver error")
        )
        with _patch_torch(fake), caplog.at_level(
            logging.WARNING, logger="src.vram_monitor"
        ):
            assert is_cuda_available() is False


# =========================================================================
# get_vram_usage_mb
# =========================================================================


class TestGetVramUsageMb:
    def test_returns_none_when_torch_not_installed(self):
        with _patch_torch(None):
            assert get_vram_usage_mb() is None

    def test_returns_none_when_cuda_unavailable(self):
        fake = _make_fake_torch(cuda_available=False)
        with _patch_torch(fake):
            assert get_vram_usage_mb() is None

    def test_returns_mb_when_cuda_available(self):
        # 2048 MB allocated.
        fake = _make_fake_torch(
            cuda_available=True,
            allocated_bytes=2048 * _BYTES_PER_MB,
        )
        with _patch_torch(fake):
            assert get_vram_usage_mb() == pytest.approx(2048.0)

    def test_returns_zero_when_nothing_allocated(self):
        fake = _make_fake_torch(
            cuda_available=True, allocated_bytes=0
        )
        with _patch_torch(fake):
            assert get_vram_usage_mb() == 0.0

    def test_passes_device_through(self):
        captured = {}

        def _memory_allocated(device=0):
            captured["device"] = device
            return 100 * _BYTES_PER_MB

        fake = _make_fake_torch(cuda_available=True)
        fake.cuda.memory_allocated = _memory_allocated
        with _patch_torch(fake):
            get_vram_usage_mb(device="cuda:1")
        assert captured["device"] == "cuda:1"

    def test_returns_none_on_internal_exception(
        self, caplog: pytest.LogCaptureFixture
    ):
        fake = _make_fake_torch(
            cuda_available=True,
            memory_allocated_raises=RuntimeError("driver error"),
        )
        with _patch_torch(fake), caplog.at_level(
            logging.WARNING, logger="src.vram_monitor"
        ):
            assert get_vram_usage_mb() is None


# =========================================================================
# get_vram_free_mb
# =========================================================================


class TestGetVramFreeMb:
    def test_returns_none_when_torch_not_installed(self):
        with _patch_torch(None):
            assert get_vram_free_mb() is None

    def test_returns_none_when_cuda_unavailable(self):
        fake = _make_fake_torch(cuda_available=False)
        with _patch_torch(fake):
            assert get_vram_free_mb() is None

    def test_returns_mb_when_cuda_available(self):
        # 4096 MB free.
        fake = _make_fake_torch(
            cuda_available=True,
            free_bytes=4096 * _BYTES_PER_MB,
            total_bytes=8192 * _BYTES_PER_MB,
        )
        with _patch_torch(fake):
            assert get_vram_free_mb() == pytest.approx(4096.0)

    def test_passes_device_through(self):
        captured = {}

        def _mem_get_info(device=0):
            captured["device"] = device
            return (1 * _BYTES_PER_MB, 8192 * _BYTES_PER_MB)

        fake = _make_fake_torch(cuda_available=True)
        fake.cuda.mem_get_info = _mem_get_info
        with _patch_torch(fake):
            get_vram_free_mb(device=2)
        assert captured["device"] == 2

    def test_returns_none_on_internal_exception(
        self, caplog: pytest.LogCaptureFixture
    ):
        fake = _make_fake_torch(
            cuda_available=True,
            mem_get_info_raises=RuntimeError("driver error"),
        )
        with _patch_torch(fake), caplog.at_level(
            logging.WARNING, logger="src.vram_monitor"
        ):
            assert get_vram_free_mb() is None


# =========================================================================
# check_vram_available
# =========================================================================


class TestCheckVramAvailable:
    def test_returns_true_when_free_exceeds_required(
        self, caplog: pytest.LogCaptureFixture
    ):
        fake = _make_fake_torch(
            cuda_available=True,
            free_bytes=4096 * _BYTES_PER_MB,
        )
        with _patch_torch(fake), caplog.at_level(
            logging.INFO, logger="src.vram_monitor"
        ):
            ok = check_vram_available(
                required_mb=2048.0, model_name="test-model"
            )
        assert ok is True
        errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert errors == []

    def test_returns_true_at_exact_boundary(self):
        # Edge case: free == required.
        fake = _make_fake_torch(
            cuda_available=True,
            free_bytes=2048 * _BYTES_PER_MB,
        )
        with _patch_torch(fake):
            assert (
                check_vram_available(
                    required_mb=2048.0, model_name="boundary"
                )
                is True
            )

    def test_returns_false_when_insufficient_and_logs_error(
        self, caplog: pytest.LogCaptureFixture
    ):
        fake = _make_fake_torch(
            cuda_available=True,
            free_bytes=1024 * _BYTES_PER_MB,
        )
        with _patch_torch(fake), caplog.at_level(
            logging.ERROR, logger="src.vram_monitor"
        ):
            ok = check_vram_available(
                required_mb=4000.0, model_name="big-model"
            )
        assert ok is False
        errors = [
            r.getMessage()
            for r in caplog.records
            if r.levelno == logging.ERROR
        ]
        # Spec format: [VRAM_ERROR] Required: {X}MB, Available: {Y}MB, Model: {name}
        assert any(
            "[VRAM_ERROR]" in m
            and "Required: 4000.00MB" in m
            and "Available: 1024.00MB" in m
            and "Model: big-model" in m
            for m in errors
        ), f"Expected formatted [VRAM_ERROR], got: {errors}"

    def test_log_includes_model_name_when_provided(
        self, caplog: pytest.LogCaptureFixture
    ):
        fake = _make_fake_torch(
            cuda_available=True, free_bytes=10 * _BYTES_PER_MB
        )
        with _patch_torch(fake), caplog.at_level(
            logging.ERROR, logger="src.vram_monitor"
        ):
            check_vram_available(
                required_mb=100.0,
                model_name="lightgpt/LightGPT-0.5B-Qwen2",
            )
        errors = [
            r.getMessage()
            for r in caplog.records
            if r.levelno == logging.ERROR
        ]
        assert any(
            "Model: lightgpt/LightGPT-0.5B-Qwen2" in m for m in errors
        ), f"Expected model name in log, got: {errors}"

    def test_log_uses_unknown_when_model_name_missing(
        self, caplog: pytest.LogCaptureFixture
    ):
        fake = _make_fake_torch(
            cuda_available=True, free_bytes=10 * _BYTES_PER_MB
        )
        with _patch_torch(fake), caplog.at_level(
            logging.ERROR, logger="src.vram_monitor"
        ):
            check_vram_available(required_mb=100.0)
        errors = [
            r.getMessage()
            for r in caplog.records
            if r.levelno == logging.ERROR
        ]
        assert any("Model: unknown" in m for m in errors), (
            f"Expected fallback Model: unknown, got: {errors}"
        )

    def test_returns_false_and_logs_error_when_cuda_unavailable(
        self, caplog: pytest.LogCaptureFixture
    ):
        # Documented behavior: caller is about to load GPU model but
        # CUDA missing → fail-safe deny.
        fake = _make_fake_torch(cuda_available=False)
        with _patch_torch(fake), caplog.at_level(
            logging.ERROR, logger="src.vram_monitor"
        ):
            ok = check_vram_available(
                required_mb=100.0, model_name="any-model"
            )
        assert ok is False
        errors = [
            r.getMessage()
            for r in caplog.records
            if r.levelno == logging.ERROR
        ]
        assert any(
            "[VRAM_ERROR]" in m
            and "Required: 100.00MB" in m
            and "Available: 0.00MB" in m
            and "Model: any-model" in m
            for m in errors
        ), f"Expected fail-safe [VRAM_ERROR], got: {errors}"

    def test_rejects_negative_required_mb(self):
        fake = _make_fake_torch(cuda_available=True)
        with _patch_torch(fake):
            with pytest.raises(ValueError, match="required_mb"):
                check_vram_available(
                    required_mb=-1.0, model_name="bad"
                )

    def test_zero_required_always_passes_when_cuda_available(self):
        # Edge case: required = 0 with any free VRAM (>= 0) → True.
        fake = _make_fake_torch(
            cuda_available=True, free_bytes=0
        )
        with _patch_torch(fake):
            assert (
                check_vram_available(required_mb=0.0, model_name="noop")
                is True
            )
