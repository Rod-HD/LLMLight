"""Unit tests for ``src.training.lora_merger`` (Task 10.5).

Validates Requirement 6 AC 4, 5 (LoRA merge → merged model có cùng
architecture với base) và Property 7 (LoRA Merge Architecture
Preservation, Requirements 6.5/12.7).

Strategy:

* ``transformers`` (``AutoModelForCausalLM``, ``AutoTokenizer``) và
  ``peft`` (``PeftModel``) được monkey-patched ở module level để test
  chạy được trên CPU-only environment mà không cần download Qwen2-0.5B
  (~1GB) hay khởi tạo CUDA.
* ``torch`` cũng được patch (chỉ cần ``float16`` attribute).

Tests coverage:

1. :meth:`LoRAMerger.merge` raise ``ValueError`` khi ``base_model`` empty
   hoặc không phải str.
2. :meth:`LoRAMerger.merge` raise ``FileNotFoundError`` khi
   ``adapter_path`` không tồn tại.
3. :meth:`LoRAMerger.merge` raise ``ValueError`` khi ``output_path`` empty.
4. :meth:`LoRAMerger.merge` tạo ``output_path`` directory nếu chưa tồn tại.
5. :meth:`LoRAMerger.merge` gọi đúng :func:`transformers.AutoModelForCausalLM.from_pretrained`
   cho ``base_model``.
6. :meth:`LoRAMerger.merge` gọi đúng :func:`transformers.AutoTokenizer.from_pretrained`
   cho ``base_model``.
7. :meth:`LoRAMerger.merge` gọi :func:`peft.PeftModel.from_pretrained`
   với ``(base, adapter_path)``.
8. :meth:`LoRAMerger.merge` gọi ``merge_and_unload()`` trên peft model.
9. :meth:`LoRAMerger.merge` gọi ``save_pretrained(output_path)`` cho cả
   merged model lẫn tokenizer.
10. :meth:`LoRAMerger.merge` return ``output_path``.
11. Property 7: :meth:`LoRAMerger.merge` raise ``ValueError`` khi
    merged config có ``num_hidden_layers`` khác base.
12. Property 7: :meth:`LoRAMerger.merge` raise ``ValueError`` khi
    merged config có ``hidden_size`` khác base.
13. Property 7: :meth:`LoRAMerger.merge` raise ``ValueError`` khi
    merged config có ``vocab_size`` khác base.
14. :meth:`LoRAMerger.merge` raise ``ImportError`` khi transformers/peft/torch
    không có.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

# Make ``src`` importable when running pytest from project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training import lora_merger  # noqa: E402
from src.training.lora_merger import LoRAMerger  # noqa: E402


# =========================================================================
# Fakes
# =========================================================================


class _FakeConfig:
    """Stand-in for HuggingFace ``PretrainedConfig``.

    Holds ``num_hidden_layers``, ``hidden_size``, ``vocab_size``.
    """

    def __init__(
        self,
        num_hidden_layers: int = 24,
        hidden_size: int = 896,
        vocab_size: int = 151936,
    ):
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size


class _FakeTokenizer:
    """Stand-in for HuggingFace tokenizer."""

    instances: list["_FakeTokenizer"] = []

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.save_calls: list[str] = []
        _FakeTokenizer.instances.append(self)

    @classmethod
    def from_pretrained(cls, model_name: str, **kwargs):
        return cls(model_name)

    def save_pretrained(self, path: str):
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        self.save_calls.append(path)


class _FakeBaseModel:
    """Stand-in for ``transformers.AutoModelForCausalLM`` output."""

    instances: list["_FakeBaseModel"] = []

    def __init__(self, model_name: str, config: _FakeConfig | None = None):
        self.model_name = model_name
        self.config = config if config is not None else _FakeConfig()
        _FakeBaseModel.instances.append(self)

    @classmethod
    def from_pretrained(cls, model_name: str, **kwargs):
        return cls(model_name)

    def save_pretrained(self, path: str):
        # Base model itself shouldn't typically save (merged_model does);
        # but PeftModel.merge_and_unload returns the *underlying* base
        # whose .save_pretrained is the one called.
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "config.json").write_text("{}", encoding="utf-8")
        (Path(path) / "model.safetensors").write_bytes(b"fake")


class _FakeMergedModel:
    """Returned by ``peft_model.merge_and_unload()``.

    Carries the *same* config as the base model (architecture preserved).
    """

    instances: list["_FakeMergedModel"] = []

    def __init__(self, config: _FakeConfig):
        self.config = config
        self.save_calls: list[str] = []
        _FakeMergedModel.instances.append(self)

    def save_pretrained(self, path: str):
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "config.json").write_text("{}", encoding="utf-8")
        (Path(path) / "model.safetensors").write_bytes(b"fake")
        self.save_calls.append(path)


class _FakePeftModel:
    """Stand-in for ``peft.PeftModel``."""

    instances: list["_FakePeftModel"] = []

    def __init__(
        self,
        base_model: _FakeBaseModel,
        adapter_path: str,
        merged_config: _FakeConfig | None = None,
    ):
        self.base_model = base_model
        self.adapter_path = adapter_path
        # By default merge preserves architecture; tests can override
        # ``merged_config`` via the factory to simulate Property 7 violations.
        self._merged_config = (
            merged_config if merged_config is not None else base_model.config
        )
        self.merge_and_unload_calls = 0
        _FakePeftModel.instances.append(self)

    @classmethod
    def from_pretrained(cls, base_model, adapter_path: str, **kwargs):
        return cls(base_model, adapter_path)

    def merge_and_unload(self):
        self.merge_and_unload_calls += 1
        return _FakeMergedModel(self._merged_config)


def _make_fake_transformers():
    """Build a fake ``transformers`` module."""
    fake = types.SimpleNamespace()

    class _AutoTokenizer:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return _FakeTokenizer.from_pretrained(*args, **kwargs)

    class _AutoModelForCausalLM:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return _FakeBaseModel.from_pretrained(*args, **kwargs)

    fake.AutoTokenizer = _AutoTokenizer
    fake.AutoModelForCausalLM = _AutoModelForCausalLM
    return fake


def _make_fake_peft():
    """Build a fake ``peft`` module."""
    fake = types.SimpleNamespace()
    fake.PeftModel = _FakePeftModel
    return fake


def _make_fake_torch():
    return types.SimpleNamespace(float16="fp16-marker")


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture(autouse=True)
def reset_recorders():
    _FakeTokenizer.instances.clear()
    _FakeBaseModel.instances.clear()
    _FakePeftModel.instances.clear()
    _FakeMergedModel.instances.clear()
    yield


@pytest.fixture
def patch_heavy_imports():
    """Patch ``transformers``, ``peft``, ``torch`` for lora_merger."""
    fake_t = _make_fake_transformers()
    fake_p = _make_fake_peft()
    fake_torch = _make_fake_torch()
    with (
        mock.patch.object(
            lora_merger, "_import_transformers", return_value=fake_t
        ),
        mock.patch.object(lora_merger, "_import_peft", return_value=fake_p),
        mock.patch.object(
            lora_merger, "_import_torch", return_value=fake_torch
        ),
    ):
        yield


@pytest.fixture
def adapter_dir(tmp_path: Path) -> str:
    """Create a fake adapter checkpoint directory."""
    p = tmp_path / "adapter"
    p.mkdir()
    (p / "adapter_config.json").write_text("{}", encoding="utf-8")
    (p / "adapter_model.safetensors").write_bytes(b"fake")
    return str(p)


@pytest.fixture
def output_path(tmp_path: Path) -> str:
    return str(tmp_path / "qwen2_finetuned")


# =========================================================================
# Input validation
# =========================================================================


class TestInputValidation:
    """Validate ``base_model``, ``adapter_path``, ``output_path``."""

    @pytest.mark.parametrize("bad", ["", None, 123])
    def test_empty_base_model_rejected(
        self, patch_heavy_imports, adapter_dir, output_path, bad
    ):
        m = LoRAMerger()
        with pytest.raises(ValueError, match="base_model"):
            m.merge(
                base_model=bad,  # type: ignore[arg-type]
                adapter_path=adapter_dir,
                output_path=output_path,
            )

    def test_missing_adapter_path_raises_filenotfound(
        self, patch_heavy_imports, output_path, tmp_path
    ):
        m = LoRAMerger()
        bogus = str(tmp_path / "no_such_adapter")
        with pytest.raises(FileNotFoundError, match="adapter_path"):
            m.merge(
                base_model="Qwen/Qwen2-0.5B",
                adapter_path=bogus,
                output_path=output_path,
            )

    @pytest.mark.parametrize("bad", ["", None, 42])
    def test_invalid_adapter_path_type_rejected(
        self, patch_heavy_imports, output_path, bad
    ):
        m = LoRAMerger()
        with pytest.raises(ValueError, match="adapter_path"):
            m.merge(
                base_model="Qwen/Qwen2-0.5B",
                adapter_path=bad,  # type: ignore[arg-type]
                output_path=output_path,
            )

    @pytest.mark.parametrize("bad", ["", None, 0])
    def test_empty_output_path_rejected(
        self, patch_heavy_imports, adapter_dir, bad
    ):
        m = LoRAMerger()
        with pytest.raises(ValueError, match="output_path"):
            m.merge(
                base_model="Qwen/Qwen2-0.5B",
                adapter_path=adapter_dir,
                output_path=bad,  # type: ignore[arg-type]
            )

    def test_creates_output_path_if_missing(
        self, patch_heavy_imports, adapter_dir, tmp_path
    ):
        out = tmp_path / "deeply" / "nested" / "models" / "qwen2_finetuned"
        assert not out.exists()
        m = LoRAMerger()
        result = m.merge(
            base_model="Qwen/Qwen2-0.5B",
            adapter_path=adapter_dir,
            output_path=str(out),
        )
        assert Path(result).exists()
        assert Path(result).is_dir()


# =========================================================================
# Pipeline wiring
# =========================================================================


class TestMergePipelineWiring:
    """Verify merge() wires transformers + peft + save calls correctly."""

    def test_returns_output_path(
        self, patch_heavy_imports, adapter_dir, output_path
    ):
        m = LoRAMerger()
        result = m.merge(
            base_model="Qwen/Qwen2-0.5B",
            adapter_path=adapter_dir,
            output_path=output_path,
        )
        assert result == output_path
        assert Path(result).exists()
        assert Path(result).is_dir()

    def test_calls_auto_tokenizer_from_pretrained(
        self, patch_heavy_imports, adapter_dir, output_path
    ):
        m = LoRAMerger()
        m.merge(
            base_model="Qwen/Qwen2-0.5B",
            adapter_path=adapter_dir,
            output_path=output_path,
        )
        assert len(_FakeTokenizer.instances) == 1
        assert _FakeTokenizer.instances[0].model_name == "Qwen/Qwen2-0.5B"

    def test_calls_auto_model_from_pretrained(
        self, patch_heavy_imports, adapter_dir, output_path
    ):
        m = LoRAMerger()
        m.merge(
            base_model="Qwen/Qwen2-0.5B",
            adapter_path=adapter_dir,
            output_path=output_path,
        )
        assert len(_FakeBaseModel.instances) == 1
        assert _FakeBaseModel.instances[0].model_name == "Qwen/Qwen2-0.5B"

    def test_calls_peft_model_from_pretrained_with_base_and_adapter(
        self, patch_heavy_imports, adapter_dir, output_path
    ):
        m = LoRAMerger()
        m.merge(
            base_model="Qwen/Qwen2-0.5B",
            adapter_path=adapter_dir,
            output_path=output_path,
        )
        assert len(_FakePeftModel.instances) == 1
        peft = _FakePeftModel.instances[0]
        assert peft.base_model is _FakeBaseModel.instances[0]
        assert peft.adapter_path == adapter_dir

    def test_calls_merge_and_unload(
        self, patch_heavy_imports, adapter_dir, output_path
    ):
        m = LoRAMerger()
        m.merge(
            base_model="Qwen/Qwen2-0.5B",
            adapter_path=adapter_dir,
            output_path=output_path,
        )
        assert len(_FakePeftModel.instances) == 1
        assert _FakePeftModel.instances[0].merge_and_unload_calls == 1

    def test_saves_merged_model_to_output_path(
        self, patch_heavy_imports, adapter_dir, output_path
    ):
        m = LoRAMerger()
        m.merge(
            base_model="Qwen/Qwen2-0.5B",
            adapter_path=adapter_dir,
            output_path=output_path,
        )
        assert len(_FakeMergedModel.instances) == 1
        assert _FakeMergedModel.instances[0].save_calls == [output_path]
        # Sanity: actual files were written.
        assert (Path(output_path) / "config.json").exists()
        assert (Path(output_path) / "model.safetensors").exists()

    def test_saves_tokenizer_to_output_path(
        self, patch_heavy_imports, adapter_dir, output_path
    ):
        m = LoRAMerger()
        m.merge(
            base_model="Qwen/Qwen2-0.5B",
            adapter_path=adapter_dir,
            output_path=output_path,
        )
        assert len(_FakeTokenizer.instances) == 1
        assert _FakeTokenizer.instances[0].save_calls == [output_path]
        assert (Path(output_path) / "tokenizer_config.json").exists()

    def test_accepts_local_base_model_path(
        self, patch_heavy_imports, adapter_dir, output_path, tmp_path
    ):
        # base_model is allowed to be a local path (not just HF id).
        local_base = tmp_path / "local_base"
        local_base.mkdir()
        m = LoRAMerger()
        result = m.merge(
            base_model=str(local_base),
            adapter_path=adapter_dir,
            output_path=output_path,
        )
        assert result == output_path
        assert _FakeBaseModel.instances[0].model_name == str(local_base)


# =========================================================================
# Property 7 — Architecture preservation
# =========================================================================


class TestArchitecturePreservation:
    """Property 7 — merged model has same num_hidden_layers, hidden_size,
    vocab_size as base model.

    These tests bypass the standard fake (which preserves architecture)
    and inject a peft.PeftModel.from_pretrained that returns a model
    with mismatched config.
    """

    def _patched_with_mismatch(
        self, mismatch_field: str, base_value, merged_value
    ):
        """Build a custom peft module whose merge_and_unload returns
        a model with a DIFFERENT config than base on ``mismatch_field``.
        """
        fake_transformers = _make_fake_transformers()
        fake_torch = _make_fake_torch()

        # Construct a custom AutoModel that returns a base with a known
        # config, and a custom PeftModel.from_pretrained that returns a
        # peft instance whose merge_and_unload yields a config differing
        # only on ``mismatch_field``.
        base_config_kwargs = {
            "num_hidden_layers": 24,
            "hidden_size": 896,
            "vocab_size": 151936,
        }
        merged_config_kwargs = dict(base_config_kwargs)
        # Override base[mismatch_field] = base_value, merged = merged_value.
        base_config_kwargs[mismatch_field] = base_value
        merged_config_kwargs[mismatch_field] = merged_value

        base_cfg = _FakeConfig(**base_config_kwargs)
        merged_cfg = _FakeConfig(**merged_config_kwargs)

        class _MismatchAutoModelForCausalLM:
            @classmethod
            def from_pretrained(cls, model_name, **kwargs):
                return _FakeBaseModel(model_name, config=base_cfg)

        fake_transformers.AutoModelForCausalLM = _MismatchAutoModelForCausalLM

        class _MismatchPeftModel:
            @classmethod
            def from_pretrained(cls, base_model, adapter_path, **kwargs):
                return _FakePeftModel(
                    base_model, adapter_path, merged_config=merged_cfg
                )

        fake_peft = types.SimpleNamespace(PeftModel=_MismatchPeftModel)

        return fake_transformers, fake_peft, fake_torch

    @pytest.mark.parametrize(
        "field,base_v,merged_v",
        [
            ("num_hidden_layers", 24, 23),
            ("num_hidden_layers", 24, 25),
            ("hidden_size", 896, 1024),
            ("hidden_size", 896, 512),
            ("vocab_size", 151936, 32000),
            ("vocab_size", 151936, 200000),
        ],
    )
    def test_mismatched_field_raises_valueerror(
        self, adapter_dir, output_path, field, base_v, merged_v
    ):
        ft, fp, ftorch = self._patched_with_mismatch(field, base_v, merged_v)
        with (
            mock.patch.object(
                lora_merger, "_import_transformers", return_value=ft
            ),
            mock.patch.object(lora_merger, "_import_peft", return_value=fp),
            mock.patch.object(lora_merger, "_import_torch", return_value=ftorch),
        ):
            m = LoRAMerger()
            with pytest.raises(ValueError, match="Property 7"):
                m.merge(
                    base_model="Qwen/Qwen2-0.5B",
                    adapter_path=adapter_dir,
                    output_path=output_path,
                )

    def test_error_message_includes_field_name(
        self, adapter_dir, output_path
    ):
        ft, fp, ftorch = self._patched_with_mismatch(
            "hidden_size", 896, 1024
        )
        with (
            mock.patch.object(
                lora_merger, "_import_transformers", return_value=ft
            ),
            mock.patch.object(lora_merger, "_import_peft", return_value=fp),
            mock.patch.object(lora_merger, "_import_torch", return_value=ftorch),
        ):
            m = LoRAMerger()
            with pytest.raises(ValueError) as exc_info:
                m.merge(
                    base_model="Qwen/Qwen2-0.5B",
                    adapter_path=adapter_dir,
                    output_path=output_path,
                )
            msg = str(exc_info.value)
            assert "hidden_size" in msg
            assert "896" in msg
            assert "1024" in msg

    def test_matching_architecture_passes(
        self, patch_heavy_imports, adapter_dir, output_path
    ):
        # The default fake preserves architecture — should succeed.
        m = LoRAMerger()
        result = m.merge(
            base_model="Qwen/Qwen2-0.5B",
            adapter_path=adapter_dir,
            output_path=output_path,
        )
        assert result == output_path


# =========================================================================
# Import errors
# =========================================================================


class TestImportErrors:
    def test_missing_transformers_raises_importerror(
        self, adapter_dir, output_path
    ):
        with (
            mock.patch.object(
                lora_merger, "_import_transformers", return_value=None
            ),
            mock.patch.object(
                lora_merger, "_import_peft", return_value=_make_fake_peft()
            ),
            mock.patch.object(
                lora_merger, "_import_torch", return_value=_make_fake_torch()
            ),
        ):
            m = LoRAMerger()
            with pytest.raises(ImportError, match="transformers"):
                m.merge(
                    base_model="Qwen/Qwen2-0.5B",
                    adapter_path=adapter_dir,
                    output_path=output_path,
                )

    def test_missing_peft_raises_importerror(self, adapter_dir, output_path):
        with (
            mock.patch.object(
                lora_merger,
                "_import_transformers",
                return_value=_make_fake_transformers(),
            ),
            mock.patch.object(lora_merger, "_import_peft", return_value=None),
            mock.patch.object(
                lora_merger, "_import_torch", return_value=_make_fake_torch()
            ),
        ):
            m = LoRAMerger()
            with pytest.raises(ImportError, match="peft"):
                m.merge(
                    base_model="Qwen/Qwen2-0.5B",
                    adapter_path=adapter_dir,
                    output_path=output_path,
                )

    def test_missing_torch_raises_importerror(self, adapter_dir, output_path):
        with (
            mock.patch.object(
                lora_merger,
                "_import_transformers",
                return_value=_make_fake_transformers(),
            ),
            mock.patch.object(
                lora_merger, "_import_peft", return_value=_make_fake_peft()
            ),
            mock.patch.object(lora_merger, "_import_torch", return_value=None),
        ):
            m = LoRAMerger()
            with pytest.raises(ImportError, match="torch"):
                m.merge(
                    base_model="Qwen/Qwen2-0.5B",
                    adapter_path=adapter_dir,
                    output_path=output_path,
                )


# =========================================================================
# Architecture extractor helper
# =========================================================================


class TestExtractArchitecture:
    """Test the internal ``_extract_architecture`` helper."""

    def test_extracts_standard_qwen2_fields(self):
        cfg = _FakeConfig(num_hidden_layers=24, hidden_size=896, vocab_size=151936)
        arch = lora_merger._extract_architecture(cfg)
        assert arch == {
            "num_hidden_layers": 24,
            "hidden_size": 896,
            "vocab_size": 151936,
        }

    def test_falls_back_to_alternative_attr_names(self):
        # Some configs use ``num_layers`` instead of ``num_hidden_layers``.
        cfg = types.SimpleNamespace(
            num_layers=12,  # alt for num_hidden_layers
            n_embd=768,  # alt for hidden_size
            vocab_size=50257,
        )
        arch = lora_merger._extract_architecture(cfg)
        assert arch["num_hidden_layers"] == 12
        assert arch["hidden_size"] == 768
        assert arch["vocab_size"] == 50257

    def test_returns_none_if_no_attribute(self):
        cfg = types.SimpleNamespace()
        arch = lora_merger._extract_architecture(cfg)
        assert arch == {
            "num_hidden_layers": None,
            "hidden_size": None,
            "vocab_size": None,
        }


# =========================================================================
# Property-based test (Property 7)
# =========================================================================
# The property test for Property 7 lives in
# tests/test_property_lora_merge_architecture.py (Task 10.6) per the
# project convention of one PBT file per property. This file focuses on
# example-based unit tests; the parameterized matrix above already
# exercises the full architecture-preservation contract for fixed values.
