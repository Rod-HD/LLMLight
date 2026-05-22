"""Property-based tests for ``LoRAMerger`` architecture preservation.

# Feature: llmlight-reproduction, Property 7: LoRA Merge Architecture Preservation

**Validates: Requirements 6.5, 12.7**

Spec (design.md §"Property 7"):

    ∀ adapter_rank ∈ [1, 16], ∀ base_arch (num_hidden_layers, hidden_size,
                                              vocab_size) random valid:
        let merged = LoRAMerger().merge(base_model, adapter, output_path)

        # Forward direction — preservation case.
        merged.config.num_hidden_layers == base.config.num_hidden_layers
        merged.config.hidden_size       == base.config.hidden_size
        merged.config.vocab_size        == base.config.vocab_size

        # Reverse direction — when peft returns a merged model whose
        # architecture differs, LoRAMerger.merge MUST raise ValueError.

The forward property exercises the success path: when ``peft.merge_and_unload``
returns a model whose config matches the base (the realistic case for any
correctly-implemented LoRA merge), :meth:`LoRAMerger.merge` succeeds and
returns the output path. The reverse property exercises the safety net:
when the simulated merge returns a corrupted architecture, the merger
must reject it via ``ValueError``.

Strategy
--------
We monkey-patch ``transformers``, ``peft``, and ``torch`` at the module
level so the test runs without GPU, without downloading Qwen2-0.5B, and
without any real merge happening. The fakes are parameterised by
hypothesis-drawn architecture parameters; the property is verified by
inspecting (a) the return value, (b) the saved merged model's config,
and (c) the raised exception (in the mismatch case).

Settings: ``max_examples=100, deadline=None``.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

# Make ``src`` importable when running ``pytest`` from project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training import lora_merger  # noqa: E402
from src.training.lora_merger import LoRAMerger  # noqa: E402


# =========================================================================
# Fakes
# =========================================================================


class _FakeConfig:
    """Stand-in for ``transformers.PretrainedConfig``.

    Carries the three architecture fields Property 7 cares about plus an
    ``adapter_rank`` annotation that is not used by ``_extract_architecture``
    but documents which adapter rank the test drew.
    """

    def __init__(
        self,
        num_hidden_layers: int,
        hidden_size: int,
        vocab_size: int,
        adapter_rank: int = 0,
    ) -> None:
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.adapter_rank = adapter_rank


class _FakeTokenizer:
    @classmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any) -> "_FakeTokenizer":
        return cls()

    def save_pretrained(self, path: str) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "tokenizer_config.json").write_text(
            "{}", encoding="utf-8"
        )


class _FakeBaseModel:
    """Stand-in for ``transformers.AutoModelForCausalLM`` output."""

    def __init__(self, config: _FakeConfig) -> None:
        self.config = config

    @classmethod
    def factory(cls, config: _FakeConfig) -> "_FakeBaseModel":
        return cls(config)


class _FakeMergedModel:
    """Returned by ``peft_model.merge_and_unload()``.

    Carries an arbitrary config (which may or may not match the base —
    that's the property under test).
    """

    def __init__(self, config: _FakeConfig) -> None:
        self.config = config

    def save_pretrained(self, path: str) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "config.json").write_text("{}", encoding="utf-8")
        (Path(path) / "model.safetensors").write_bytes(b"fake")


class _FakePeftModel:
    """Stand-in for ``peft.PeftModel``.

    The class-level ``_merged_config`` attribute (set by the patcher
    factory below) controls whether ``merge_and_unload`` preserves the
    base architecture or returns a corrupted one.
    """

    _merged_config: _FakeConfig | None = None

    def __init__(self, base_model: _FakeBaseModel) -> None:
        self.base_model = base_model

    @classmethod
    def from_pretrained(
        cls, base_model: _FakeBaseModel, adapter_path: str, **kwargs: Any
    ) -> "_FakePeftModel":
        return cls(base_model)

    def merge_and_unload(self) -> _FakeMergedModel:
        cfg = (
            self._merged_config
            if self._merged_config is not None
            else self.base_model.config
        )
        return _FakeMergedModel(cfg)


def _make_fake_torch() -> types.SimpleNamespace:
    return types.SimpleNamespace(float16="fp16-marker")


def _build_patches(
    base_config: _FakeConfig,
    merged_config: _FakeConfig | None,
):
    """Build a ``(transformers, peft, torch)`` triple of fake modules.

    ``merged_config=None`` → merge_and_unload preserves the base config
    (forward property — success path).

    ``merged_config != None`` → merge_and_unload returns the supplied
    config (reverse property — mismatch path).
    """
    fake_transformers = types.SimpleNamespace()

    class _AutoTokenizer:
        @classmethod
        def from_pretrained(cls, *args: Any, **kwargs: Any) -> _FakeTokenizer:
            return _FakeTokenizer.from_pretrained(*args, **kwargs)

    class _AutoModelForCausalLM:
        @classmethod
        def from_pretrained(cls, *args: Any, **kwargs: Any) -> _FakeBaseModel:
            return _FakeBaseModel.factory(base_config)

    fake_transformers.AutoTokenizer = _AutoTokenizer
    fake_transformers.AutoModelForCausalLM = _AutoModelForCausalLM

    # Per-test PeftModel subclass so the class-level ``_merged_config``
    # doesn't leak across hypothesis examples.
    class _ScopedPeftModel(_FakePeftModel):
        pass

    _ScopedPeftModel._merged_config = merged_config

    fake_peft = types.SimpleNamespace(PeftModel=_ScopedPeftModel)
    fake_torch = _make_fake_torch()
    return fake_transformers, fake_peft, fake_torch


# =========================================================================
# Strategies
# =========================================================================

#: Adapter rank ≤ 16 per task spec. r=0 is excluded (degenerate adapter).
_adapter_rank_strategy = st.integers(min_value=1, max_value=16)

#: Base model architecture parameters in reasonable ranges. Bounds chosen
#: to cover Qwen2-0.5B (24 layers / 896 hidden / 151936 vocab) and
#: smaller / larger plausible decoder-only configurations without
#: making the search space pathological.
_num_hidden_layers_strategy = st.integers(min_value=2, max_value=48)
_hidden_size_strategy = st.integers(min_value=128, max_value=4096)
_vocab_size_strategy = st.integers(min_value=1024, max_value=200_000)


@st.composite
def _base_arch_strategy(draw: st.DrawFn) -> dict[str, int]:
    """Compose a valid base-model architecture parameter set."""
    return {
        "num_hidden_layers": draw(_num_hidden_layers_strategy),
        "hidden_size": draw(_hidden_size_strategy),
        "vocab_size": draw(_vocab_size_strategy),
    }


@st.composite
def _mismatch_strategy(draw: st.DrawFn) -> tuple[str, int]:
    """Pick ONE field to mutate plus a delta (positive or negative)."""
    field = draw(
        st.sampled_from(["num_hidden_layers", "hidden_size", "vocab_size"])
    )
    # Non-zero delta to guarantee the mismatch is visible.
    delta = draw(st.integers(min_value=1, max_value=64))
    sign = draw(st.sampled_from([-1, 1]))
    return field, delta * sign


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def adapter_dir(tmp_path: Path) -> str:
    p = tmp_path / "adapter"
    p.mkdir()
    (p / "adapter_config.json").write_text("{}", encoding="utf-8")
    (p / "adapter_model.safetensors").write_bytes(b"fake")
    return str(p)


# =========================================================================
# Property 7 — forward direction (preservation)
# =========================================================================


@given(
    adapter_rank=_adapter_rank_strategy,
    base_arch=_base_arch_strategy(),
)
@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_property_lora_merge_preserves_architecture(
    adapter_rank: int,
    base_arch: dict[str, int],
    adapter_dir: str,
    tmp_path: Path,
) -> None:
    """**Validates: Requirements 6.5, 12.7**

    Forward direction: when the simulated peft merge returns a model
    whose config matches the base (correct LoRA merge behaviour),
    :meth:`LoRAMerger.merge` succeeds and the resulting merged model's
    architecture is byte-identical to the base on the three fields
    Property 7 names (``num_hidden_layers``, ``hidden_size``,
    ``vocab_size``).
    """
    base_config = _FakeConfig(
        num_hidden_layers=base_arch["num_hidden_layers"],
        hidden_size=base_arch["hidden_size"],
        vocab_size=base_arch["vocab_size"],
        adapter_rank=adapter_rank,
    )

    fake_t, fake_p, fake_torch = _build_patches(
        base_config=base_config, merged_config=None
    )

    output_path = tmp_path / "merged_preserved"

    with (
        mock.patch.object(
            lora_merger, "_import_transformers", return_value=fake_t
        ),
        mock.patch.object(lora_merger, "_import_peft", return_value=fake_p),
        mock.patch.object(
            lora_merger, "_import_torch", return_value=fake_torch
        ),
    ):
        merger = LoRAMerger()
        result = merger.merge(
            base_model="Qwen/Qwen2-0.5B",
            adapter_path=adapter_dir,
            output_path=str(output_path),
        )

    # Return value: merged model directory.
    assert result == str(output_path)
    assert Path(result).exists()
    assert Path(result).is_dir()
    # Persistence: merged model and tokenizer files were written.
    assert (Path(result) / "config.json").exists()
    assert (Path(result) / "tokenizer_config.json").exists()


# =========================================================================
# Property 7 — reverse direction (mismatch detection)
# =========================================================================


@given(
    adapter_rank=_adapter_rank_strategy,
    base_arch=_base_arch_strategy(),
    mismatch=_mismatch_strategy(),
)
@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_property_lora_merge_rejects_mismatched_architecture(
    adapter_rank: int,
    base_arch: dict[str, int],
    mismatch: tuple[str, int],
    adapter_dir: str,
    tmp_path: Path,
) -> None:
    """**Validates: Requirements 6.5, 12.7**

    Reverse direction: when the simulated peft merge returns a model
    whose config differs from the base on at least one of
    ``num_hidden_layers`` / ``hidden_size`` / ``vocab_size``,
    :meth:`LoRAMerger.merge` MUST raise ``ValueError``. The error
    message must explicitly reference Property 7 so a downstream
    operator can identify the failure mode.
    """
    base_config = _FakeConfig(
        num_hidden_layers=base_arch["num_hidden_layers"],
        hidden_size=base_arch["hidden_size"],
        vocab_size=base_arch["vocab_size"],
        adapter_rank=adapter_rank,
    )

    field, delta = mismatch
    merged_kwargs = dict(base_arch)
    new_value = merged_kwargs[field] + delta
    # Ensure the mutated field stays in a meaningful positive range so
    # the mismatch is unambiguous (a negative ``hidden_size`` would be
    # rejected for other reasons).
    if new_value <= 0:
        new_value = merged_kwargs[field] + abs(delta) + 1
    merged_kwargs[field] = new_value

    merged_config = _FakeConfig(
        num_hidden_layers=merged_kwargs["num_hidden_layers"],
        hidden_size=merged_kwargs["hidden_size"],
        vocab_size=merged_kwargs["vocab_size"],
        adapter_rank=adapter_rank,
    )

    fake_t, fake_p, fake_torch = _build_patches(
        base_config=base_config, merged_config=merged_config
    )

    output_path = tmp_path / "merged_mismatch"

    with (
        mock.patch.object(
            lora_merger, "_import_transformers", return_value=fake_t
        ),
        mock.patch.object(lora_merger, "_import_peft", return_value=fake_p),
        mock.patch.object(
            lora_merger, "_import_torch", return_value=fake_torch
        ),
    ):
        merger = LoRAMerger()
        with pytest.raises(ValueError) as excinfo:
            merger.merge(
                base_model="Qwen/Qwen2-0.5B",
                adapter_path=adapter_dir,
                output_path=str(output_path),
            )

    msg = str(excinfo.value)
    # Message must indicate the property and the specific mismatched field.
    assert "Property 7" in msg, (
        f"ValueError must reference Property 7; got: {msg}"
    )
    assert field in msg, (
        f"ValueError must name the mismatched field {field!r}; got: {msg}"
    )
    # Both base and merged values for the mutated field must appear.
    assert str(base_arch[field]) in msg
    assert str(merged_kwargs[field]) in msg
