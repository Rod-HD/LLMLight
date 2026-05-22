"""Unit tests for ``src.training.cgpr_trainer`` (Task 10.4).

Validates Requirement 6 AC 2, 6, 10 — CGPR Trainer pairwise margin
ranking loss training trên IFT checkpoint.

Strategy:

* ``transformers`` (``AutoModelForCausalLM``, ``AutoTokenizer``), ``peft``
  (``LoraConfig``, ``get_peft_model``), và ``torch`` đều được monkey-
  patched ở module level để test chạy được trên CPU-only environment
  mà không cần download Qwen2-0.5B (~1GB) hay khởi tạo CUDA.
* ``vram_monitor.get_vram_usage_mb`` được patch để inject các kịch bản
  VRAM (None / dưới budget / vượt budget).
* IFT checkpoint được tạo bằng tmp_path để verify pre-condition existence.

Tests coverage (matches task requirements):

1. :class:`TrainingConfig` validates ``cgpr_epochs`` ∈ [2, 5].
2. :class:`TrainingConfig` validates ``cgpr_learning_rate`` > 0.
3. :class:`TrainingConfig` validates ``cgpr_margin`` > 0.
4. :class:`CGPRTrainer.__init__` validates ``config`` type.
5. :meth:`CGPRTrainer.train` raises :class:`FileNotFoundError` khi
   ``ift_checkpoint`` không tồn tại trên disk (Requirement 6.10).
6. :meth:`CGPRTrainer.train` raises :class:`ValueError` khi
   ``ranking_pairs`` rỗng.
7. :meth:`CGPRTrainer.train` validates mỗi tuple có đúng 3 strings
   non-empty.
8. :meth:`CGPRTrainer.train` checkpoint được lưu mỗi epoch tại
   ``{cgpr_checkpoint_dir}/epoch_{N}/`` (Requirement 6 AC 6).
9. :meth:`CGPRTrainer.train` resume từ latest checkpoint nếu phát hiện.
10. :meth:`CGPRTrainer.train` return path tới folder hợp lệ
    (``cgpr_output_dir``).
11. :meth:`CGPRTrainer.train` raises :class:`OOMError` khi VRAM vượt
    7.5GB.
12. VRAM check is called sau model load và sau training.
13. :meth:`CGPRTrainer.train` LoRA config dùng đúng rank/alpha/target
    từ TrainingConfig.
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
from src.training import cgpr_trainer, ift_trainer  # noqa: E402
from src.training.cgpr_trainer import CGPRTrainer  # noqa: E402
from src.training.ift_trainer import (  # noqa: E402
    OOMError,
    TrainingConfig,
)


# =========================================================================
# Fakes
# =========================================================================


class _FakeTokenizer:
    """Stand-in for HuggingFace tokenizer."""

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
        ids = list(range(1, min(len(text), max_length or len(text)) + 1))
        return {"input_ids": ids}


class _FakeBaseModel:
    def __init__(self, name="ift_ckpt"):
        self.name = name

    @classmethod
    def from_pretrained(cls, model_name, **kwargs):
        return cls(name=model_name)


class _FakePeftModel:
    """Mock PEFT-wrapped model. Tracks save_pretrained / forward calls."""

    instances: list["_FakePeftModel"] = []

    def __init__(self, base_model, lora_config):
        self.base_model = base_model
        self.lora_config = lora_config
        self.save_calls: list[str] = []
        self.train_calls: int = 0
        self.eval_calls: int = 0
        self.forward_calls: int = 0
        self.adapter_loads: list[str] = []
        _FakePeftModel.instances.append(self)

    def train(self):
        self.train_calls += 1

    def eval(self):
        self.eval_calls += 1

    def parameters(self):
        # Return one fake "parameter" so the optimizer factory has something
        # to iterate over.
        return [_FakeParam()]

    def save_pretrained(self, path: str):
        Path(path).mkdir(parents=True, exist_ok=True)
        # Touch a marker file so checkpoint is detectable.
        (Path(path) / "adapter_model.bin").write_bytes(b"fake")
        self.save_calls.append(path)

    def load_adapter(self, path: str, adapter_name: str = "default"):
        self.adapter_loads.append(path)

    def __call__(self, input_ids=None, labels=None, **kwargs):
        self.forward_calls += 1
        # Return an object with a .logits tensor of correct shape.
        seq_len = input_ids.shape[1] if hasattr(input_ids, "shape") else 4
        vocab = 8
        # Use a fresh tensor each call to allow .backward()
        logits = _fake_torch().zeros((1, seq_len, vocab), requires_grad=True)
        return types.SimpleNamespace(logits=logits)


class _FakeParam:
    requires_grad = True


class _RecordingLoraConfig:
    instances: list["_RecordingLoraConfig"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        _RecordingLoraConfig.instances.append(self)


class _RecordingOptimizer:
    """Fake AdamW optimizer."""

    instances: list["_RecordingOptimizer"] = []

    def __init__(self, params, lr=1e-3, **kwargs):
        self.params = list(params)
        self.lr = lr
        self.step_calls = 0
        self.zero_grad_calls = 0
        _RecordingOptimizer.instances.append(self)

    def step(self):
        self.step_calls += 1

    def zero_grad(self):
        self.zero_grad_calls += 1


# ----- Fake torch with just enough surface for the trainer -----


_FAKE_TORCH_INSTANCE: Any = None


def _fake_torch() -> Any:
    """Return a singleton fake torch module (cached)."""
    global _FAKE_TORCH_INSTANCE
    if _FAKE_TORCH_INSTANCE is None:
        _FAKE_TORCH_INSTANCE = _build_fake_torch()
    return _FAKE_TORCH_INSTANCE


def _build_fake_torch() -> Any:
    fake = types.SimpleNamespace()
    fake.float16 = "fp16-marker"
    fake.long = "long-marker"

    class _FakeTensor:
        def __init__(self, data, shape=None, requires_grad=False):
            self._data = data
            self._shape = shape if shape is not None else (
                _shape_of(data)
            )
            self.requires_grad = requires_grad

        @property
        def shape(self):
            return self._shape

        def unsqueeze(self, dim):
            new_shape = list(self._shape)
            new_shape.insert(dim if dim >= 0 else len(new_shape) + dim + 1, 1)
            return _FakeTensor(self._data, tuple(new_shape))

        def squeeze(self, dim):
            new_shape = [s for i, s in enumerate(self._shape) if not (
                (i == dim or i - len(self._shape) == dim) and s == 1
            )]
            return _FakeTensor(self._data, tuple(new_shape))

        def contiguous(self):
            return self

        def clone(self):
            return _FakeTensor(self._data, self._shape, self.requires_grad)

        def __getitem__(self, key):
            # Slicing path used: logits[..., :-1, :], labels[..., 1:]
            new_shape = list(self._shape)
            if isinstance(key, tuple):
                # Trim last 2 dims by 1 for shift_logits
                # Ellipsis-aware
                if len(new_shape) >= 2:
                    if new_shape[-2] > 1:
                        new_shape[-2] -= 1
            return _FakeTensor(self._data, tuple(new_shape))

        def __setitem__(self, key, value):
            # No-op for safe_labels[shift_labels == -100] = 0
            pass

        def __ne__(self, other):
            return _FakeTensor(self._data, self._shape)

        def __eq__(self, other):
            return _FakeTensor(self._data, self._shape)

        def float(self):
            return self

        def gather(self, dim, index):
            new_shape = list(index._shape)
            return _FakeTensor(self._data, tuple(new_shape))

        def sum(self):
            return _FakeScalar(0.5)

        def __mul__(self, other):
            return self

        __rmul__ = __mul__

        def detach(self):
            return self

        def item(self):
            return 0.5

        def backward(self):
            pass

    class _FakeScalar:
        """A scalar-like object supporting arithmetic + .backward()."""

        def __init__(self, value: float):
            self._value = float(value)
            self.requires_grad = True

        def __sub__(self, other):
            return _FakeScalar(self._value - _scalar(other))

        def __rsub__(self, other):
            return _FakeScalar(_scalar(other) - self._value)

        def __add__(self, other):
            return _FakeScalar(self._value + _scalar(other))

        def __radd__(self, other):
            return _FakeScalar(_scalar(other) + self._value)

        def __truediv__(self, other):
            return _FakeScalar(self._value / _scalar(other))

        def __rtruediv__(self, other):
            return _FakeScalar(_scalar(other) / self._value)

        def __mul__(self, other):
            return _FakeScalar(self._value * _scalar(other))

        __rmul__ = __mul__

        def __float__(self):
            return float(self._value)

        def detach(self):
            return self

        def item(self):
            return self._value

        def backward(self):
            pass

    def _scalar(x):
        if isinstance(x, _FakeScalar):
            return x._value
        if isinstance(x, (int, float)):
            return float(x)
        return 0.0

    def _shape_of(data):
        if hasattr(data, "__len__") and not isinstance(data, str):
            return (len(data),)
        return (1,)

    def tensor(data, dtype=None):
        return _FakeTensor(data)

    def clamp(x, min=0.0):
        if isinstance(x, _FakeScalar):
            return _FakeScalar(max(min, x._value))
        return x

    def zeros(shape, requires_grad=False):
        return _FakeTensor([0] * (shape[-1] if shape else 1), shape, requires_grad)

    fake.tensor = tensor
    fake.clamp = clamp
    fake.zeros = zeros
    fake._FakeTensor = _FakeTensor
    fake._FakeScalar = _FakeScalar

    # nn.functional.log_softmax
    nn = types.SimpleNamespace()
    functional = types.SimpleNamespace()
    functional.log_softmax = lambda x, dim: x
    nn.functional = functional
    fake.nn = nn

    # optim.AdamW
    optim = types.SimpleNamespace()
    optim.AdamW = _RecordingOptimizer
    fake.optim = optim

    return fake


def _make_fake_transformers():
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
    fake = types.SimpleNamespace()
    fake.LoraConfig = _RecordingLoraConfig

    def _get_peft_model(base_model, lora_config):
        return _FakePeftModel(base_model, lora_config)

    fake.get_peft_model = _get_peft_model
    return fake


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture(autouse=True)
def reset_recorders():
    _FakePeftModel.instances.clear()
    _RecordingLoraConfig.instances.clear()
    _RecordingOptimizer.instances.clear()
    global _FAKE_TORCH_INSTANCE
    _FAKE_TORCH_INSTANCE = None
    yield


@pytest.fixture
def patch_heavy_imports():
    """Patch ``transformers``, ``peft``, and ``torch`` for cgpr_trainer."""
    fake_t = _make_fake_transformers()
    fake_p = _make_fake_peft()
    fake_torch = _fake_torch()
    with (
        mock.patch.object(
            cgpr_trainer, "_import_transformers", return_value=fake_t
        ),
        mock.patch.object(cgpr_trainer, "_import_peft", return_value=fake_p),
        mock.patch.object(
            cgpr_trainer, "_import_torch", return_value=fake_torch
        ),
        # Also patch ift_trainer module-level loaders since cgpr_trainer
        # imports them via re-export.
        mock.patch.object(
            ift_trainer, "_import_transformers", return_value=fake_t
        ),
        mock.patch.object(ift_trainer, "_import_peft", return_value=fake_p),
        mock.patch.object(
            ift_trainer, "_import_torch", return_value=fake_torch
        ),
    ):
        yield


@pytest.fixture
def vram_ok():
    """VRAM monitor returns None (CPU env) so OOM check is skipped."""
    with mock.patch.object(
        vram_monitor, "get_vram_usage_mb", return_value=None
    ) as m:
        yield m


@pytest.fixture
def vram_under_budget():
    with mock.patch.object(
        vram_monitor, "get_vram_usage_mb", return_value=4096.0
    ) as m:
        yield m


@pytest.fixture
def vram_over_budget():
    with mock.patch.object(
        vram_monitor,
        "get_vram_usage_mb",
        return_value=CGPRTrainer.MAX_VRAM_MB + 100.0,
    ) as m:
        yield m


@pytest.fixture
def ift_ckpt(tmp_path: Path) -> str:
    """Create a fake IFT checkpoint directory."""
    p = tmp_path / "ift_ckpt"
    p.mkdir()
    (p / "adapter_config.json").write_text("{}", encoding="utf-8")
    return str(p)


@pytest.fixture
def runtime_dirs(tmp_path: Path):
    """Provide writable cgpr_checkpoint_dir / cgpr_output_dir."""
    return {
        "cgpr_checkpoint_dir": str(tmp_path / "ckpt_cgpr"),
        "cgpr_output_dir": str(tmp_path / "model_cgpr_out"),
        "checkpoint_dir": str(tmp_path / "ckpt_ift"),
        "output_dir": str(tmp_path / "model_ift_out"),
    }


@pytest.fixture
def small_pairs() -> list[tuple[str, str, str]]:
    return [
        (
            "Prompt #1: queue is high",
            "<signal>NTST</signal>",
            "<signal>ETWT</signal>",
        ),
        (
            "Prompt #2: north heavy",
            "<signal>ELWL</signal>",
            "<signal>NLSL</signal>",
        ),
    ]


# =========================================================================
# TrainingConfig CGPR validation
# =========================================================================


class TestTrainingConfigCGPRDefaults:
    def test_cgpr_defaults_match_spec(self):
        c = TrainingConfig()
        assert 2 <= c.cgpr_epochs <= 5
        assert c.cgpr_learning_rate > 0
        assert c.cgpr_margin > 0
        assert c.cgpr_checkpoint_dir
        assert c.cgpr_output_dir


class TestTrainingConfigCGPRValidation:
    @pytest.mark.parametrize("bad", [0, 1, 6, 100, -1])
    def test_invalid_cgpr_epochs_rejected(self, bad):
        with pytest.raises(ValueError, match="cgpr_epochs"):
            TrainingConfig(cgpr_epochs=bad)

    @pytest.mark.parametrize("bad", ["3", 3.0, None])
    def test_non_int_cgpr_epochs_rejected(self, bad):
        with pytest.raises(TypeError, match="cgpr_epochs"):
            TrainingConfig(cgpr_epochs=bad)

    def test_min_max_cgpr_epochs_inclusive(self):
        # 2 and 5 must both be accepted (boundary).
        TrainingConfig(cgpr_epochs=2)
        TrainingConfig(cgpr_epochs=5)

    @pytest.mark.parametrize("bad", [0, -1e-5, -1])
    def test_invalid_cgpr_learning_rate_rejected(self, bad):
        with pytest.raises(ValueError, match="cgpr_learning_rate"):
            TrainingConfig(cgpr_learning_rate=bad)

    @pytest.mark.parametrize("bad", [0, -0.1, -1.0])
    def test_invalid_cgpr_margin_rejected(self, bad):
        with pytest.raises(ValueError, match="cgpr_margin"):
            TrainingConfig(cgpr_margin=bad)

    def test_empty_cgpr_checkpoint_dir_rejected(self):
        with pytest.raises(ValueError, match="cgpr_checkpoint_dir"):
            TrainingConfig(cgpr_checkpoint_dir="")

    def test_empty_cgpr_output_dir_rejected(self):
        with pytest.raises(ValueError, match="cgpr_output_dir"):
            TrainingConfig(cgpr_output_dir="")


# =========================================================================
# CGPRTrainer.__init__
# =========================================================================


class TestCGPRTrainerInit:
    def test_accepts_valid_config(self):
        c = TrainingConfig()
        t = CGPRTrainer(c)
        assert t.config is c

    def test_rejects_non_config_type(self):
        with pytest.raises(TypeError, match="config"):
            CGPRTrainer({"epochs": 3})  # type: ignore[arg-type]

    def test_max_vram_mb_constant(self):
        assert CGPRTrainer.MAX_VRAM_MB == 7680.0

    def test_min_max_epochs_constants(self):
        assert CGPRTrainer.MIN_EPOCHS == 2
        assert CGPRTrainer.MAX_EPOCHS == 5


# =========================================================================
# train(): input validation
# =========================================================================


class TestTrainInputValidation:
    def test_missing_ift_checkpoint_raises(
        self, patch_heavy_imports, vram_ok, runtime_dirs, small_pairs, tmp_path
    ):
        c = TrainingConfig(**runtime_dirs)
        t = CGPRTrainer(c)
        bogus = str(tmp_path / "no_such_ckpt")
        with pytest.raises(FileNotFoundError, match="ift_checkpoint"):
            t.train(ift_checkpoint=bogus, ranking_pairs=small_pairs)

    @pytest.mark.parametrize("bad", ["", None, 123])
    def test_invalid_ift_checkpoint_type_rejected(
        self, patch_heavy_imports, vram_ok, runtime_dirs, small_pairs, bad
    ):
        c = TrainingConfig(**runtime_dirs)
        t = CGPRTrainer(c)
        with pytest.raises(ValueError, match="ift_checkpoint"):
            t.train(ift_checkpoint=bad, ranking_pairs=small_pairs)  # type: ignore[arg-type]

    def test_empty_ranking_pairs_raises(
        self, patch_heavy_imports, vram_ok, runtime_dirs, ift_ckpt
    ):
        c = TrainingConfig(**runtime_dirs)
        t = CGPRTrainer(c)
        with pytest.raises(ValueError, match="non-empty"):
            t.train(ift_checkpoint=ift_ckpt, ranking_pairs=[])

    def test_ranking_pairs_must_be_list(
        self, patch_heavy_imports, vram_ok, runtime_dirs, ift_ckpt
    ):
        c = TrainingConfig(**runtime_dirs)
        t = CGPRTrainer(c)
        with pytest.raises(ValueError, match="must be list"):
            t.train(
                ift_checkpoint=ift_ckpt,
                ranking_pairs="not a list",  # type: ignore[arg-type]
            )

    def test_each_tuple_must_have_3_elements(
        self, patch_heavy_imports, vram_ok, runtime_dirs, ift_ckpt
    ):
        c = TrainingConfig(**runtime_dirs)
        t = CGPRTrainer(c)
        # 2-tuple (missing negative)
        bad = [("prompt", "positive")]
        with pytest.raises(ValueError, match=r"ranking_pairs\[0\]"):
            t.train(
                ift_checkpoint=ift_ckpt, ranking_pairs=bad  # type: ignore[arg-type]
            )

    def test_each_tuple_field_must_be_str(
        self, patch_heavy_imports, vram_ok, runtime_dirs, ift_ckpt
    ):
        c = TrainingConfig(**runtime_dirs)
        t = CGPRTrainer(c)
        bad = [("prompt", 42, "neg")]
        with pytest.raises(ValueError, match=r"ranking_pairs\[0\]\[1\]"):
            t.train(
                ift_checkpoint=ift_ckpt, ranking_pairs=bad  # type: ignore[arg-type]
            )

    def test_each_tuple_field_must_be_non_empty(
        self, patch_heavy_imports, vram_ok, runtime_dirs, ift_ckpt
    ):
        c = TrainingConfig(**runtime_dirs)
        t = CGPRTrainer(c)
        bad = [("prompt", "", "neg")]
        with pytest.raises(ValueError, match=r"ranking_pairs\[0\]\[1\]"):
            t.train(ift_checkpoint=ift_ckpt, ranking_pairs=bad)

    def test_explicit_resume_must_exist(
        self, patch_heavy_imports, vram_ok, runtime_dirs, ift_ckpt, small_pairs
    ):
        c = TrainingConfig(**runtime_dirs)
        t = CGPRTrainer(c)
        with pytest.raises(FileNotFoundError, match="resume_from"):
            t.train(
                ift_checkpoint=ift_ckpt,
                ranking_pairs=small_pairs,
                resume_from="/nonexistent/epoch_1",
            )


# =========================================================================
# train(): pipeline wiring (LoRA + checkpoints + return path)
# =========================================================================


class TestTrainPipelineWiring:
    def test_returns_cgpr_output_dir(
        self, patch_heavy_imports, vram_ok, runtime_dirs, ift_ckpt, small_pairs
    ):
        c = TrainingConfig(**runtime_dirs)
        t = CGPRTrainer(c)
        result = t.train(ift_checkpoint=ift_ckpt, ranking_pairs=small_pairs)
        assert result == runtime_dirs["cgpr_output_dir"]
        assert Path(result).exists()
        assert Path(result).is_dir()

    def test_lora_config_uses_q_v_proj_defaults(
        self, patch_heavy_imports, vram_ok, runtime_dirs, ift_ckpt, small_pairs
    ):
        c = TrainingConfig(**runtime_dirs)
        t = CGPRTrainer(c)
        t.train(ift_checkpoint=ift_ckpt, ranking_pairs=small_pairs)

        assert len(_RecordingLoraConfig.instances) == 1
        kw = _RecordingLoraConfig.instances[0].kwargs
        assert kw["r"] == 8
        assert kw["lora_alpha"] == 16
        assert list(kw["target_modules"]) == ["q_proj", "v_proj"]
        assert kw["task_type"] == "CAUSAL_LM"

    def test_lora_config_honors_custom_rank(
        self, patch_heavy_imports, vram_ok, runtime_dirs, ift_ckpt, small_pairs
    ):
        c = TrainingConfig(
            lora_rank=16,
            lora_alpha=32,
            lora_target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
            **runtime_dirs,
        )
        t = CGPRTrainer(c)
        t.train(ift_checkpoint=ift_ckpt, ranking_pairs=small_pairs)
        kw = _RecordingLoraConfig.instances[0].kwargs
        assert kw["r"] == 16
        assert kw["lora_alpha"] == 32
        assert list(kw["target_modules"]) == [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ]

    def test_optimizer_built_with_cgpr_learning_rate(
        self, patch_heavy_imports, vram_ok, runtime_dirs, ift_ckpt, small_pairs
    ):
        c = TrainingConfig(cgpr_learning_rate=2.5e-5, **runtime_dirs)
        t = CGPRTrainer(c)
        t.train(ift_checkpoint=ift_ckpt, ranking_pairs=small_pairs)
        assert len(_RecordingOptimizer.instances) == 1
        opt = _RecordingOptimizer.instances[0]
        assert opt.lr == 2.5e-5


# =========================================================================
# Checkpoint per epoch (Requirement 6 AC 6)
# =========================================================================


class TestCheckpointPerEpoch:
    def test_checkpoint_per_epoch_saved(
        self, patch_heavy_imports, vram_ok, runtime_dirs, ift_ckpt, small_pairs
    ):
        c = TrainingConfig(cgpr_epochs=3, **runtime_dirs)
        t = CGPRTrainer(c)
        t.train(ift_checkpoint=ift_ckpt, ranking_pairs=small_pairs)

        ckpt_dir = Path(runtime_dirs["cgpr_checkpoint_dir"])
        # epoch_1, epoch_2, epoch_3 should exist (or at least the final
        # ones up to save_total_limit).
        epoch_dirs = sorted(
            [p.name for p in ckpt_dir.iterdir() if p.is_dir()]
        )
        # save_total_limit defaults to 2 → only epoch_2 and epoch_3 kept.
        assert "epoch_3" in epoch_dirs

    def test_checkpoint_directory_named_epoch_N(
        self, patch_heavy_imports, vram_ok, runtime_dirs, ift_ckpt, small_pairs
    ):
        c = TrainingConfig(cgpr_epochs=2, save_total_limit=5, **runtime_dirs)
        t = CGPRTrainer(c)
        t.train(ift_checkpoint=ift_ckpt, ranking_pairs=small_pairs)

        ckpt_dir = Path(runtime_dirs["cgpr_checkpoint_dir"])
        names = sorted([p.name for p in ckpt_dir.iterdir() if p.is_dir()])
        assert names == ["epoch_1", "epoch_2"]

    def test_save_pretrained_called_per_epoch(
        self, patch_heavy_imports, vram_ok, runtime_dirs, ift_ckpt, small_pairs
    ):
        c = TrainingConfig(cgpr_epochs=2, save_total_limit=5, **runtime_dirs)
        t = CGPRTrainer(c)
        t.train(ift_checkpoint=ift_ckpt, ranking_pairs=small_pairs)
        # Final adapter save + 2 epoch saves = 3 calls.
        assert len(_FakePeftModel.instances) == 1
        m = _FakePeftModel.instances[0]
        # 2 epochs + 1 final save
        assert len(m.save_calls) == 3


# =========================================================================
# Resume from latest checkpoint
# =========================================================================


class TestResume:
    def test_resume_explicit_path(
        self,
        patch_heavy_imports,
        vram_ok,
        runtime_dirs,
        ift_ckpt,
        small_pairs,
        tmp_path,
    ):
        # Create a fake epoch_1 checkpoint outside cgpr_checkpoint_dir.
        resume_dir = tmp_path / "resume_dir" / "epoch_1"
        resume_dir.mkdir(parents=True)
        (resume_dir / "adapter_model.bin").write_bytes(b"fake")

        c = TrainingConfig(cgpr_epochs=3, save_total_limit=5, **runtime_dirs)
        t = CGPRTrainer(c)
        t.train(
            ift_checkpoint=ift_ckpt,
            ranking_pairs=small_pairs,
            resume_from=str(resume_dir),
        )

        # Should call load_adapter and resume from epoch index 1, so only
        # epoch_2 and epoch_3 are saved (epoch_1 already exists).
        m = _FakePeftModel.instances[0]
        assert str(resume_dir) in m.adapter_loads
        names = sorted(
            [p.name for p in Path(runtime_dirs["cgpr_checkpoint_dir"]).iterdir()
             if p.is_dir()]
        )
        # Only epoch_2 and epoch_3 should have been saved by training.
        assert "epoch_2" in names
        assert "epoch_3" in names

    def test_auto_detect_latest_checkpoint(
        self,
        patch_heavy_imports,
        vram_ok,
        runtime_dirs,
        ift_ckpt,
        small_pairs,
    ):
        # Pre-create epoch_1 in cgpr_checkpoint_dir.
        ckpt_dir = Path(runtime_dirs["cgpr_checkpoint_dir"])
        existing = ckpt_dir / "epoch_1"
        existing.mkdir(parents=True)
        (existing / "adapter_model.bin").write_bytes(b"fake")

        c = TrainingConfig(cgpr_epochs=3, save_total_limit=5, **runtime_dirs)
        t = CGPRTrainer(c)
        t.train(ift_checkpoint=ift_ckpt, ranking_pairs=small_pairs)

        m = _FakePeftModel.instances[0]
        # Should have auto-detected the existing epoch_1 and called load_adapter.
        assert any("epoch_1" in p for p in m.adapter_loads)


# =========================================================================
# OOM detection
# =========================================================================


class TestOOMDetection:
    def test_vram_over_budget_after_load_raises_oomerror(
        self,
        patch_heavy_imports,
        vram_over_budget,
        runtime_dirs,
        ift_ckpt,
        small_pairs,
    ):
        c = TrainingConfig(**runtime_dirs)
        t = CGPRTrainer(c)
        with pytest.raises(OOMError, match="VRAM exceeded budget"):
            t.train(ift_checkpoint=ift_ckpt, ranking_pairs=small_pairs)

    def test_vram_under_budget_passes(
        self,
        patch_heavy_imports,
        vram_under_budget,
        runtime_dirs,
        ift_ckpt,
        small_pairs,
    ):
        c = TrainingConfig(**runtime_dirs)
        t = CGPRTrainer(c)
        result = t.train(ift_checkpoint=ift_ckpt, ranking_pairs=small_pairs)
        assert result == runtime_dirs["cgpr_output_dir"]

    def test_oom_error_is_runtimeerror_subclass(self):
        assert issubclass(OOMError, RuntimeError)


# =========================================================================
# VRAM check is called
# =========================================================================


class TestVRAMCheckIsCalled:
    def test_vram_check_called_after_load_and_after_training(
        self,
        patch_heavy_imports,
        runtime_dirs,
        ift_ckpt,
        small_pairs,
    ):
        c = TrainingConfig(**runtime_dirs)
        t = CGPRTrainer(c)

        with mock.patch.object(
            vram_monitor, "get_vram_usage_mb", return_value=4096.0
        ) as m:
            t.train(ift_checkpoint=ift_ckpt, ranking_pairs=small_pairs)

        # Should be called at least twice: after model load + after training.
        assert m.call_count >= 2
