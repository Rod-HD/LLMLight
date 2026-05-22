"""Unit tests for ``src.training.ift_trainer`` (Task 10.2).

Validates Requirement 6 AC 1, 3, 6 — IFT Trainer cho Qwen2-0.5B với LoRA.

Strategy:

* ``transformers`` (``AutoModelForCausalLM``, ``AutoTokenizer``,
  ``Trainer``, ``TrainingArguments``, ``DataCollatorForLanguageModeling``),
  ``peft`` (``LoraConfig``, ``get_peft_model``), ``datasets`` (``Dataset``),
  và ``torch`` đều được monkey-patched ở module level để test chạy được
  trên CPU-only environment (Windows dev box, CI runners) mà không cần
  download Qwen2-0.5B (~1GB).
* ``vram_monitor.get_vram_usage_mb`` được patch để inject các kịch bản
  VRAM (None / dưới budget / vượt budget).

Tests coverage:

1. :class:`TrainingConfig` defaults match Requirement 6 AC 1, 3.
2. :class:`TrainingConfig` validates ``ift_epochs`` ∈ [3, 10].
3. :class:`TrainingConfig` validates ``lora_rank`` > 0,
   ``lora_alpha`` > 0, ``gradient_accumulation_steps`` >= 4.
4. :class:`TrainingConfig` validates ``lora_target_modules`` is non-empty
   tuple of str.
5. :class:`IFTTrainer.__init__` validates ``config`` type.
6. :meth:`IFTTrainer.train` empty data → ``ValueError``.
7. :meth:`IFTTrainer.train` calls ``Trainer.train`` với đúng arguments
   (epochs, batch_size, gradient_accumulation, learning_rate).
8. :meth:`IFTTrainer.train` builds :class:`peft.LoraConfig` với đúng
   rank/alpha/target_modules.
9. :meth:`IFTTrainer.train` returns ``output_dir`` (CGPRTrainer sẽ load).
10. :meth:`IFTTrainer.train` raises :class:`OOMError` khi VRAM vượt 7.5GB.
11. :meth:`IFTTrainer.train` resume support — passes ``resume_from_checkpoint``
    đến ``Trainer.train``.
12. :meth:`IFTTrainer.train` rejects ``resume_from`` không tồn tại.
13. :meth:`IFTTrainer.train` calls ``save_strategy="epoch"`` trong
    TrainingArguments (Requirement 6 AC 6).
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

from src import vram_monitor  # noqa: E402
from src.training import ift_trainer  # noqa: E402
from src.training.ift_trainer import (  # noqa: E402
    IFTTrainer,
    OOMError,
    TrainingConfig,
)


# =========================================================================
# Fakes
# =========================================================================


class _FakeTokenizer:
    """Stand-in for HuggingFace ``AutoTokenizer.from_pretrained`` output.

    ``__call__(text, ...)`` returns ``{"input_ids": [...]}`` where
    input_ids is a deterministic list-of-ints based on text length —
    enough for the trainer to build a Dataset.
    """

    eos_token = "<eos>"
    eos_token_id = 0
    pad_token = None

    @classmethod
    def from_pretrained(cls, model_name, **kwargs):
        return cls()

    def __call__(
        self,
        text: str,
        add_special_tokens: bool = True,
        truncation: bool = False,
        max_length: int | None = None,
        **kwargs,
    ):
        # Toy tokenization: 1 token per character (deterministic).
        ids = list(range(1, min(len(text), max_length or len(text)) + 1))
        return {"input_ids": ids}


class _RecordingTrainer:
    """Captures ``Trainer.__init__`` arguments and records ``train()`` calls.

    Test fixtures will read these to assert on the LoRA / training
    pipeline being constructed correctly.
    """

    instances: list["_RecordingTrainer"] = []

    def __init__(
        self,
        model=None,
        args=None,
        train_dataset=None,
        tokenizer=None,
        data_collator=None,
        **kwargs,
    ):
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.tokenizer = tokenizer
        self.data_collator = data_collator
        self.train_calls: list[dict[str, Any]] = []
        self.save_calls: list[str] = []
        _RecordingTrainer.instances.append(self)

    def train(self, resume_from_checkpoint=None, **kwargs):
        self.train_calls.append(
            {"resume_from_checkpoint": resume_from_checkpoint, **kwargs}
        )

    def save_model(self, output_dir: str):
        self.save_calls.append(output_dir)


class _RecordingTrainingArguments:
    """Captures all kwargs passed by IFTTrainer."""

    instances: list["_RecordingTrainingArguments"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        _RecordingTrainingArguments.instances.append(self)


class _FakeDataCollator:
    def __init__(self, tokenizer=None, mlm=False, **kwargs):
        self.tokenizer = tokenizer
        self.mlm = mlm


class _FakeBaseModel:
    def __init__(self, name="Qwen/Qwen2-0.5B"):
        self.name = name

    @classmethod
    def from_pretrained(cls, model_name, **kwargs):
        return cls(name=model_name)


class _FakePeftModel:
    """Returned by :func:`peft.get_peft_model`. Records the LoraConfig
    that was used."""

    def __init__(self, base_model, lora_config):
        self.base_model = base_model
        self.lora_config = lora_config


class _RecordingLoraConfig:
    """Captures kwargs passed to ``peft.LoraConfig``."""

    instances: list["_RecordingLoraConfig"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        _RecordingLoraConfig.instances.append(self)


class _FakeDatasets:
    """Stand-in for the ``datasets`` module."""

    class Dataset:
        @staticmethod
        def from_list(records):
            # Minimal shim that quacks like a Dataset for our assertions.
            return _FakeDataset(records)


class _FakeDataset:
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)


def _make_fake_transformers():
    """Build a fake ``transformers`` module with the API surface used."""
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
    fake.Trainer = _RecordingTrainer
    fake.TrainingArguments = _RecordingTrainingArguments
    fake.DataCollatorForLanguageModeling = _FakeDataCollator
    return fake


def _make_fake_peft():
    fake = types.SimpleNamespace()
    fake.LoraConfig = _RecordingLoraConfig

    def _get_peft_model(base_model, lora_config):
        return _FakePeftModel(base_model, lora_config)

    fake.get_peft_model = _get_peft_model
    return fake


def _make_fake_torch():
    return types.SimpleNamespace(float16="fp16-marker")


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture(autouse=True)
def reset_recorders():
    """Clear class-level recording lists between tests."""
    _RecordingTrainer.instances.clear()
    _RecordingTrainingArguments.instances.clear()
    _RecordingLoraConfig.instances.clear()
    yield


@pytest.fixture
def patch_heavy_imports():
    """Patch ``transformers``, ``peft``, ``datasets``, ``torch``."""
    fake_t = _make_fake_transformers()
    fake_p = _make_fake_peft()
    fake_d = _FakeDatasets()
    fake_torch = _make_fake_torch()
    with (
        mock.patch.object(
            ift_trainer, "_import_transformers", return_value=fake_t
        ),
        mock.patch.object(ift_trainer, "_import_peft", return_value=fake_p),
        mock.patch.object(
            ift_trainer, "_import_datasets", return_value=fake_d
        ),
        mock.patch.object(
            ift_trainer, "_import_torch", return_value=fake_torch
        ),
    ):
        yield


@pytest.fixture
def vram_ok():
    """Patch ``vram_monitor.get_vram_usage_mb`` to return None (CPU env).

    With ``None``, IFTTrainer skips the OOM check (production path on
    CPU-only test machines).
    """
    with mock.patch.object(
        vram_monitor, "get_vram_usage_mb", return_value=None
    ) as m:
        yield m


@pytest.fixture
def vram_under_budget():
    """Patch VRAM to return value comfortably under MAX_VRAM_MB."""
    with mock.patch.object(
        vram_monitor, "get_vram_usage_mb", return_value=4096.0
    ) as m:
        yield m


@pytest.fixture
def vram_over_budget():
    """Patch VRAM to return value above MAX_VRAM_MB → triggers OOMError."""
    with mock.patch.object(
        vram_monitor,
        "get_vram_usage_mb",
        return_value=IFTTrainer.MAX_VRAM_MB + 100.0,
    ) as m:
        yield m


@pytest.fixture
def runtime_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Provide writable checkpoint_dir and output_dir under tmp_path."""
    monkeypatch.chdir(tmp_path)
    ck = tmp_path / "ckpt_ift"
    out = tmp_path / "model_ift_out"
    return {"checkpoint_dir": str(ck), "output_dir": str(out)}


@pytest.fixture
def small_data() -> list[tuple[str, str]]:
    """Tiny trajectory dataset for unit tests."""
    return [
        ("Prompt #1: queue is high", "<signal>ETWT</signal>"),
        ("Prompt #2: north heavy", "<signal>NTST</signal>"),
        ("Prompt #3: empty", "<signal>ELWL</signal>"),
    ]


# =========================================================================
# TrainingConfig
# =========================================================================


class TestTrainingConfigDefaults:
    """Defaults match Requirement 6 AC 1, 3 and task spec."""

    def test_defaults_match_spec(self):
        c = TrainingConfig()
        assert c.base_model == "Qwen/Qwen2-0.5B"
        assert c.lora_rank == 8
        assert c.lora_alpha == 16
        assert c.lora_target_modules == ("q_proj", "v_proj")
        assert c.gradient_accumulation_steps >= 4
        assert c.batch_size == 1
        assert 3 <= c.ift_epochs <= 10
        assert c.checkpoint_dir
        assert c.output_dir
        assert c.max_seq_length >= 1
        assert c.learning_rate > 0


class TestTrainingConfigValidation:
    """``__post_init__`` raises on invalid values."""

    @pytest.mark.parametrize("bad_epochs", [0, 1, 2, 11, 100, -1])
    def test_invalid_epochs_rejected(self, bad_epochs):
        with pytest.raises(ValueError, match="ift_epochs"):
            TrainingConfig(ift_epochs=bad_epochs)

    @pytest.mark.parametrize("bad_type", ["5", 5.0, None])
    def test_non_int_epochs_rejected(self, bad_type):
        with pytest.raises(TypeError, match="ift_epochs"):
            TrainingConfig(ift_epochs=bad_type)

    @pytest.mark.parametrize("bad_rank", [0, -1, -8])
    def test_invalid_rank_rejected(self, bad_rank):
        with pytest.raises(ValueError, match="lora_rank"):
            TrainingConfig(lora_rank=bad_rank)

    @pytest.mark.parametrize("bad_alpha", [0, -1])
    def test_invalid_alpha_rejected(self, bad_alpha):
        with pytest.raises(ValueError, match="lora_alpha"):
            TrainingConfig(lora_alpha=bad_alpha)

    @pytest.mark.parametrize("bad_grad_acc", [0, 1, 2, 3])
    def test_grad_accumulation_below_4_rejected(self, bad_grad_acc):
        with pytest.raises(
            ValueError, match="gradient_accumulation_steps"
        ):
            TrainingConfig(gradient_accumulation_steps=bad_grad_acc)

    @pytest.mark.parametrize("bad_modules", [(), [], "q_proj"])
    def test_invalid_target_modules_rejected(self, bad_modules):
        with pytest.raises((TypeError, ValueError), match="target_modules"):
            TrainingConfig(lora_target_modules=bad_modules)

    @pytest.mark.parametrize("bad_batch", [0, -1])
    def test_invalid_batch_size_rejected(self, bad_batch):
        with pytest.raises(ValueError, match="batch_size"):
            TrainingConfig(batch_size=bad_batch)

    @pytest.mark.parametrize("bad_lr", [0, -1e-4])
    def test_invalid_learning_rate_rejected(self, bad_lr):
        with pytest.raises(ValueError, match="learning_rate"):
            TrainingConfig(learning_rate=bad_lr)

    def test_empty_checkpoint_dir_rejected(self):
        with pytest.raises(ValueError, match="checkpoint_dir"):
            TrainingConfig(checkpoint_dir="")

    def test_empty_output_dir_rejected(self):
        with pytest.raises(ValueError, match="output_dir"):
            TrainingConfig(output_dir="")

    def test_min_max_epochs_inclusive(self):
        # 3 and 10 must be accepted (boundary).
        TrainingConfig(ift_epochs=3)
        TrainingConfig(ift_epochs=10)


# =========================================================================
# IFTTrainer.__init__
# =========================================================================


class TestIFTTrainerInit:
    def test_accepts_valid_config(self):
        c = TrainingConfig()
        t = IFTTrainer(c)
        assert t.config is c

    def test_rejects_non_config_type(self):
        with pytest.raises(TypeError, match="config"):
            IFTTrainer({"epochs": 5})  # type: ignore[arg-type]

    def test_max_vram_mb_constant(self):
        assert IFTTrainer.MAX_VRAM_MB == 7680.0

    def test_min_max_epochs_constants(self):
        assert IFTTrainer.MIN_EPOCHS == 3
        assert IFTTrainer.MAX_EPOCHS == 10


# =========================================================================
# train(): input validation
# =========================================================================


class TestTrainInputValidation:
    def test_empty_data_raises_valueerror(
        self, patch_heavy_imports, vram_ok, runtime_dirs
    ):
        c = TrainingConfig(**runtime_dirs)
        t = IFTTrainer(c)
        with pytest.raises(ValueError, match="non-empty"):
            t.train([])

    def test_data_must_be_list(
        self, patch_heavy_imports, vram_ok, runtime_dirs
    ):
        c = TrainingConfig(**runtime_dirs)
        t = IFTTrainer(c)
        with pytest.raises(ValueError, match="list"):
            t.train("not a list")  # type: ignore[arg-type]

    def test_data_entries_must_be_str_str_tuples(
        self, patch_heavy_imports, vram_ok, runtime_dirs
    ):
        c = TrainingConfig(**runtime_dirs)
        t = IFTTrainer(c)
        bad_data = [("prompt only",)]  # Wrong arity
        with pytest.raises(ValueError, match=r"data\[0\]"):
            t.train(bad_data)  # type: ignore[arg-type]

        bad_data2 = [(1, "response")]  # Non-str prompt
        with pytest.raises(ValueError, match=r"data\[0\]"):
            t.train(bad_data2)  # type: ignore[arg-type]

    def test_resume_from_must_exist(
        self, patch_heavy_imports, vram_ok, runtime_dirs, small_data
    ):
        c = TrainingConfig(**runtime_dirs)
        t = IFTTrainer(c)
        with pytest.raises(FileNotFoundError, match="does not exist"):
            t.train(small_data, resume_from="/nonexistent/checkpoint-100")


# =========================================================================
# train(): Trainer / TrainingArguments / LoraConfig wiring
# =========================================================================


class TestTrainCallsTrainer:
    def test_trainer_train_called_once(
        self,
        patch_heavy_imports,
        vram_ok,
        runtime_dirs,
        small_data,
    ):
        c = TrainingConfig(**runtime_dirs)
        t = IFTTrainer(c)
        t.train(small_data)
        assert len(_RecordingTrainer.instances) == 1
        rec = _RecordingTrainer.instances[0]
        assert len(rec.train_calls) == 1
        # No resume → resume_from_checkpoint should be None.
        assert rec.train_calls[0]["resume_from_checkpoint"] is None

    def test_training_arguments_respect_config(
        self,
        patch_heavy_imports,
        vram_ok,
        runtime_dirs,
        small_data,
    ):
        c = TrainingConfig(
            ift_epochs=4,
            batch_size=1,
            gradient_accumulation_steps=8,
            learning_rate=1e-4,
            **runtime_dirs,
        )
        t = IFTTrainer(c)
        t.train(small_data)

        assert len(_RecordingTrainingArguments.instances) == 1
        ta = _RecordingTrainingArguments.instances[0]
        kw = ta.kwargs
        assert kw["num_train_epochs"] == 4
        assert kw["per_device_train_batch_size"] == 1
        assert kw["gradient_accumulation_steps"] == 8
        assert kw["learning_rate"] == 1e-4
        assert kw["save_strategy"] == "epoch"  # Requirement 6 AC 6
        assert kw["output_dir"] == runtime_dirs["checkpoint_dir"]

    def test_lora_config_uses_target_modules_q_v_proj(
        self,
        patch_heavy_imports,
        vram_ok,
        runtime_dirs,
        small_data,
    ):
        c = TrainingConfig(**runtime_dirs)
        t = IFTTrainer(c)
        t.train(small_data)

        assert len(_RecordingLoraConfig.instances) == 1
        kw = _RecordingLoraConfig.instances[0].kwargs
        assert kw["r"] == 8
        assert kw["lora_alpha"] == 16
        # target_modules is a list of strings inside LoraConfig.
        assert list(kw["target_modules"]) == ["q_proj", "v_proj"]
        assert kw["task_type"] == "CAUSAL_LM"

    def test_lora_config_honors_custom_rank(
        self,
        patch_heavy_imports,
        vram_ok,
        runtime_dirs,
        small_data,
    ):
        c = TrainingConfig(
            lora_rank=16,
            lora_alpha=32,
            lora_target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
            **runtime_dirs,
        )
        t = IFTTrainer(c)
        t.train(small_data)

        kw = _RecordingLoraConfig.instances[0].kwargs
        assert kw["r"] == 16
        assert kw["lora_alpha"] == 32
        assert list(kw["target_modules"]) == [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ]

    def test_returns_output_dir_path(
        self,
        patch_heavy_imports,
        vram_ok,
        runtime_dirs,
        small_data,
    ):
        c = TrainingConfig(**runtime_dirs)
        t = IFTTrainer(c)
        result = t.train(small_data)
        assert result == runtime_dirs["output_dir"]
        # And output_dir should now exist (created by IFTTrainer).
        assert Path(result).exists()

    def test_save_model_called_with_output_dir(
        self,
        patch_heavy_imports,
        vram_ok,
        runtime_dirs,
        small_data,
    ):
        c = TrainingConfig(**runtime_dirs)
        t = IFTTrainer(c)
        t.train(small_data)
        rec = _RecordingTrainer.instances[0]
        assert rec.save_calls == [runtime_dirs["output_dir"]]

    def test_dataset_built_from_data(
        self,
        patch_heavy_imports,
        vram_ok,
        runtime_dirs,
        small_data,
    ):
        c = TrainingConfig(**runtime_dirs)
        t = IFTTrainer(c)
        t.train(small_data)
        rec = _RecordingTrainer.instances[0]
        ds = rec.train_dataset
        assert ds is not None
        # Same number of records as input pairs.
        assert len(ds) == len(small_data)
        # Each record has the IFT-tuning columns.
        for rec_row in ds.records:
            assert set(rec_row.keys()) == {
                "input_ids",
                "attention_mask",
                "labels",
            }
            # Prompt portion must be masked with -100.
            assert -100 in rec_row["labels"]


# =========================================================================
# Resume support
# =========================================================================


class TestResumeFromCheckpoint:
    def test_resume_passes_path_to_trainer(
        self,
        patch_heavy_imports,
        vram_ok,
        runtime_dirs,
        small_data,
        tmp_path,
    ):
        # Create a fake existing checkpoint directory.
        ckpt = tmp_path / "checkpoint-100"
        ckpt.mkdir()

        c = TrainingConfig(**runtime_dirs)
        t = IFTTrainer(c)
        t.train(small_data, resume_from=str(ckpt))

        rec = _RecordingTrainer.instances[0]
        assert rec.train_calls[0]["resume_from_checkpoint"] == str(ckpt)


# =========================================================================
# OOM detection
# =========================================================================


class TestOOMDetection:
    def test_vram_over_budget_after_load_raises_oomerror(
        self,
        patch_heavy_imports,
        vram_over_budget,
        runtime_dirs,
        small_data,
    ):
        c = TrainingConfig(**runtime_dirs)
        t = IFTTrainer(c)
        with pytest.raises(OOMError, match="VRAM exceeded budget"):
            t.train(small_data)

    def test_vram_under_budget_passes(
        self,
        patch_heavy_imports,
        vram_under_budget,
        runtime_dirs,
        small_data,
    ):
        c = TrainingConfig(**runtime_dirs)
        t = IFTTrainer(c)
        # Should NOT raise — VRAM fully within budget.
        result = t.train(small_data)
        assert result == runtime_dirs["output_dir"]

    def test_oom_error_is_runtimeerror_subclass(self):
        assert issubclass(OOMError, RuntimeError)

    def test_cuda_oom_during_training_wraps_to_oomerror(
        self,
        patch_heavy_imports,
        vram_ok,
        runtime_dirs,
        small_data,
    ):
        """If ``Trainer.train`` raises a CUDA OOM, IFTTrainer wraps it
        into ``OOMError`` for caller convenience."""

        class _OomTrainer(_RecordingTrainer):
            def train(self, resume_from_checkpoint=None, **kwargs):
                # Mimic torch's CUDA OOM message.
                raise RuntimeError("CUDA out of memory. Tried to allocate ...")

        with mock.patch.object(
            ift_trainer,
            "_import_transformers",
            return_value=_make_fake_transformers_with_trainer(_OomTrainer),
        ):
            c = TrainingConfig(**runtime_dirs)
            t = IFTTrainer(c)
            with pytest.raises(OOMError, match="OOM"):
                t.train(small_data)


# Helper builder used in TestOOMDetection.test_cuda_oom_...
def _make_fake_transformers_with_trainer(trainer_cls):
    fake = _make_fake_transformers()
    fake.Trainer = trainer_cls
    return fake


# =========================================================================
# Constants exposed for callers (LoRAMerger, CGPRTrainer, run_training)
# =========================================================================


class TestExposedConstants:
    def test_lora_target_modules_default_is_q_v_proj(self):
        c = TrainingConfig()
        assert c.lora_target_modules == ("q_proj", "v_proj")

    def test_default_base_model_is_qwen2_0_5b(self):
        c = TrainingConfig()
        assert c.base_model == "Qwen/Qwen2-0.5B"

    def test_default_output_dir_is_qwen2_finetuned_ift(self):
        c = TrainingConfig()
        # Aligns with design: PRE-merge output (LoRAMerger reads from here)
        assert "qwen2_finetuned_ift" in c.output_dir
