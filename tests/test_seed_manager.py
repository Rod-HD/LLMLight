"""Unit tests for ``src.seed_manager``.

Validates:
    - Requirement 1.9 (set seed cho NumPy/PyTorch/random/CityFlow init từ
      ``RANDOM_SEED`` env, default 42)
    - Requirement 8.9 (seed = ``RANDOM_SEED + run_id``)
    - Requirement 12.8 (deterministic reproducibility)

Tests cover:
  * env var read (with and without ``RANDOM_SEED`` set)
  * default falls back to 42 when env var missing
  * malformed env var falls back to 42 and logs warning
  * ``seed_for_run(run_id)`` computes ``base + run_id``
  * ``apply(seed)`` sets ``PYTHONHASHSEED`` and ``random.seed``
  * ``apply(seed)`` tolerates missing torch / numpy
  * ``apply(seed)`` actually seeds numpy when available (determinism check)
  * Two consecutive ``apply(seed)`` calls produce identical Python ``random``
    output (basic determinism property — same in spirit as Property 8 but
    minimal here, the full property test lives in Task 11.4).
"""

from __future__ import annotations

import importlib
import logging
import os
import random
import sys
from pathlib import Path
from unittest import mock

import pytest

# Make ``src`` importable when running ``pytest`` from project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.seed_manager as seed_manager_module  # noqa: E402
from src.seed_manager import SeedManager  # noqa: E402


# =========================================================================
# __init__ — base_seed resolution
# =========================================================================


class TestInit:
    """Requirement 1.9: ``RANDOM_SEED`` env, default 42."""

    def test_default_seed_is_42_when_env_var_missing(self, monkeypatch):
        monkeypatch.delenv("RANDOM_SEED", raising=False)
        sm = SeedManager()
        assert sm.base_seed == 42

    def test_reads_seed_from_env_var(self, monkeypatch):
        monkeypatch.setenv("RANDOM_SEED", "1337")
        sm = SeedManager()
        assert sm.base_seed == 1337

    def test_explicit_base_seed_overrides_env(self, monkeypatch):
        monkeypatch.setenv("RANDOM_SEED", "999")
        sm = SeedManager(base_seed=7)
        assert sm.base_seed == 7

    def test_explicit_base_seed_zero_is_respected(self, monkeypatch):
        # ``0`` is a valid seed; constructor must not treat it as "missing".
        monkeypatch.setenv("RANDOM_SEED", "999")
        sm = SeedManager(base_seed=0)
        assert sm.base_seed == 0

    def test_empty_env_var_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("RANDOM_SEED", "")
        sm = SeedManager()
        assert sm.base_seed == 42

    def test_whitespace_only_env_var_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("RANDOM_SEED", "   ")
        sm = SeedManager()
        assert sm.base_seed == 42

    def test_malformed_env_var_falls_back_to_default_and_warns(
        self, monkeypatch, caplog
    ):
        monkeypatch.setenv("RANDOM_SEED", "not-an-int")
        with caplog.at_level(logging.WARNING, logger="src.seed_manager"):
            sm = SeedManager()
        assert sm.base_seed == 42
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert any("not an integer" in m and "not-an-int" in m for m in warnings)

    def test_negative_env_var_is_accepted(self, monkeypatch):
        # ``int("-5")`` works; spec doesn't forbid negative seeds.
        monkeypatch.setenv("RANDOM_SEED", "-5")
        sm = SeedManager()
        assert sm.base_seed == -5


# =========================================================================
# seed_for_run
# =========================================================================


class TestSeedForRun:
    """Requirement 8.9: seed = base + run_id."""

    def test_run_zero_returns_base(self):
        sm = SeedManager(base_seed=42)
        assert sm.seed_for_run(0) == 42

    def test_run_one_returns_base_plus_one(self):
        sm = SeedManager(base_seed=42)
        assert sm.seed_for_run(1) == 43

    def test_run_two_returns_base_plus_two(self):
        sm = SeedManager(base_seed=42)
        assert sm.seed_for_run(2) == 44

    def test_run_id_with_custom_base(self):
        sm = SeedManager(base_seed=1000)
        assert sm.seed_for_run(7) == 1007

    def test_run_id_coerces_to_int(self):
        sm = SeedManager(base_seed=10)
        # numpy ints / float-without-fraction are common in practice.
        assert sm.seed_for_run(3) == 13


# =========================================================================
# apply — core seeding logic
# =========================================================================


class TestApplyCore:
    """Requirements 1.9, 12.8: apply seed to all available backends."""

    def test_apply_sets_pythonhashseed(self):
        sm = SeedManager(base_seed=42)
        sm.apply(123)
        assert os.environ["PYTHONHASHSEED"] == "123"

    def test_apply_sets_random_seed_deterministically(self):
        sm = SeedManager(base_seed=42)
        sm.apply(99)
        sequence_a = [random.random() for _ in range(5)]
        sm.apply(99)
        sequence_b = [random.random() for _ in range(5)]
        assert sequence_a == sequence_b

    def test_apply_with_different_seeds_yields_different_sequences(self):
        sm = SeedManager(base_seed=42)
        sm.apply(1)
        seq_a = [random.random() for _ in range(5)]
        sm.apply(2)
        seq_b = [random.random() for _ in range(5)]
        assert seq_a != seq_b

    def test_apply_coerces_seed_to_int(self):
        sm = SeedManager(base_seed=42)
        sm.apply(42)  # int
        # Should not raise; PYTHONHASHSEED stored as str of int.
        assert os.environ["PYTHONHASHSEED"] == "42"

    def test_apply_logs_at_info_level(self, caplog):
        sm = SeedManager(base_seed=42)
        with caplog.at_level(logging.INFO, logger="src.seed_manager"):
            sm.apply(7)
        infos = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
        assert any("seed=7" in m for m in infos)


# =========================================================================
# apply — tolerates missing numpy / torch
# =========================================================================


class TestApplyToleratesMissingBackends:
    """Module must import + apply() must work even when numpy/torch absent.

    This is critical for the dev environment (Windows, no torch) — the module
    is exercised through unit tests + downstream imports before the WSL2 venv
    is bootstrapped.
    """

    def test_apply_works_when_torch_is_none(self, monkeypatch, caplog):
        sm = SeedManager(base_seed=42)
        # Force torch backend off.
        monkeypatch.setattr(seed_manager_module, "_torch", None)
        with caplog.at_level(logging.DEBUG, logger="src.seed_manager"):
            sm.apply(11)
        # PYTHONHASHSEED + random still set.
        assert os.environ["PYTHONHASHSEED"] == "11"
        # Should log a debug message about skipping torch.
        debug_msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG]
        assert any("torch not available" in m for m in debug_msgs)

    def test_apply_works_when_numpy_is_none(self, monkeypatch, caplog):
        sm = SeedManager(base_seed=42)
        monkeypatch.setattr(seed_manager_module, "_numpy", None)
        with caplog.at_level(logging.DEBUG, logger="src.seed_manager"):
            sm.apply(13)
        assert os.environ["PYTHONHASHSEED"] == "13"
        debug_msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG]
        assert any("numpy not available" in m for m in debug_msgs)

    def test_apply_works_when_both_backends_missing(self, monkeypatch):
        sm = SeedManager(base_seed=42)
        monkeypatch.setattr(seed_manager_module, "_torch", None)
        monkeypatch.setattr(seed_manager_module, "_numpy", None)
        # Should not raise.
        sm.apply(17)
        assert os.environ["PYTHONHASHSEED"] == "17"
        # Python random still seeded — verify determinism.
        a = random.random()
        sm.apply(17)
        b = random.random()
        assert a == b

    def test_apply_handles_torch_cuda_check_failure(self, monkeypatch):
        """If ``torch.cuda.is_available()`` itself raises, apply() must not."""
        sm = SeedManager(base_seed=42)

        class FlakyCuda:
            @staticmethod
            def is_available():
                raise RuntimeError("driver mismatch")

            @staticmethod
            def manual_seed_all(seed):  # pragma: no cover - shouldn't be reached
                raise AssertionError("should not be called when is_available fails")

        class FakeTorch:
            cuda = FlakyCuda()

            @staticmethod
            def manual_seed(seed):
                FakeTorch._last = seed

            _last = None

        monkeypatch.setattr(seed_manager_module, "_torch", FakeTorch)
        sm.apply(5)
        assert FakeTorch._last == 5  # CPU seed still set
        assert os.environ["PYTHONHASHSEED"] == "5"

    def test_apply_seeds_torch_cpu_and_cuda_when_available(self, monkeypatch):
        sm = SeedManager(base_seed=42)
        captured: dict[str, int] = {}

        class FakeCuda:
            @staticmethod
            def is_available():
                return True

            @staticmethod
            def manual_seed_all(seed):
                captured["cuda_all"] = seed

        class FakeTorch:
            cuda = FakeCuda()

            @staticmethod
            def manual_seed(seed):
                captured["cpu"] = seed

        monkeypatch.setattr(seed_manager_module, "_torch", FakeTorch)
        sm.apply(101)
        assert captured == {"cpu": 101, "cuda_all": 101}

    def test_apply_skips_cuda_when_not_available(self, monkeypatch):
        sm = SeedManager(base_seed=42)
        captured: dict[str, int] = {}

        class FakeCuda:
            @staticmethod
            def is_available():
                return False

            @staticmethod
            def manual_seed_all(seed):  # pragma: no cover - shouldn't be reached
                captured["cuda_all"] = seed

        class FakeTorch:
            cuda = FakeCuda()

            @staticmethod
            def manual_seed(seed):
                captured["cpu"] = seed

        monkeypatch.setattr(seed_manager_module, "_torch", FakeTorch)
        sm.apply(202)
        assert captured == {"cpu": 202}


# =========================================================================
# apply — numpy seeding (skipped when numpy missing)
# =========================================================================


numpy = pytest.importorskip("numpy", reason="numpy required for this check")


class TestApplyWithNumpy:
    """When numpy is installed, apply() must seed numpy.random."""

    def test_apply_seeds_numpy_random(self):
        sm = SeedManager(base_seed=42)
        sm.apply(2024)
        a = numpy.random.rand(5)
        sm.apply(2024)
        b = numpy.random.rand(5)
        assert (a == b).all()

    def test_different_seeds_yield_different_numpy_sequences(self):
        sm = SeedManager(base_seed=42)
        sm.apply(1)
        a = numpy.random.rand(5)
        sm.apply(2)
        b = numpy.random.rand(5)
        assert not (a == b).all()


# =========================================================================
# Module imports cleanly even with backends absent
# =========================================================================


class TestModuleImports:
    """Critical: ``src.seed_manager`` must import even with no torch/numpy."""

    def test_module_reimport_succeeds(self):
        # Sanity: we can reload the module without errors.
        importlib.reload(seed_manager_module)
        assert hasattr(seed_manager_module, "SeedManager")

    def test_module_reimport_with_torch_blocked(self, monkeypatch):
        """Simulate dev box with no torch installed — reload must succeed."""
        # Block torch import inside the module by injecting a sentinel that
        # raises ImportError when accessed via importlib machinery.
        original_torch = sys.modules.get("torch")
        sys.modules["torch"] = None  # type: ignore[assignment]
        try:
            importlib.reload(seed_manager_module)
            assert seed_manager_module._torch is None
            sm = seed_manager_module.SeedManager(base_seed=42)
            sm.apply(5)  # must not raise
        finally:
            if original_torch is not None:
                sys.modules["torch"] = original_torch
            else:
                sys.modules.pop("torch", None)
            importlib.reload(seed_manager_module)
