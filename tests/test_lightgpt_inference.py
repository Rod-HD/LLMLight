"""Unit tests for ``src.lightgpt_inference`` (Task 6.2).

Validates Requirement 5 (LightGPT inference, two variants) và Requirement
11 (VRAM budget).

Strategy:

* ``transformers`` / ``torch`` được mock ở module level để test chạy
  được trên CPU-only environment (Windows dev box, CI runners).
* ``vram_monitor.check_vram_available`` được patch để inject các kịch bản
  VRAM (đủ / không đủ).
* ``transformers.AutoModelForCausalLM.from_pretrained`` được patch để
  tránh download thật từ HuggingFace.

Tests coverage:

1. ``__init__`` accepts cả 2 variants với args hợp lệ.
2. ``__init__`` reject variant không hợp lệ.
3. ``lightgpt_mine`` raise ``FileNotFoundError`` khi
   ``models/qwen2_finetuned/`` không tồn tại; KHÔNG fallback sang HF.
4. ``lightgpt_hf`` thử HF priority chain theo đúng thứ tự (mock download).
5. ``_get_quantization_config`` trả về ``BitsAndBytesConfig`` với đúng
   field NF4 + double-quant + float16 compute dtype.
6. VRAM check raise ``RuntimeError`` khi không đủ VRAM.
7. ``get_vram_usage_mb`` returns float (kể cả khi CUDA không khả dụng).
8. ``generate`` returns generated text bằng mock tokenizer/model.
9. ``generate`` raises khi chưa load_model.
"""

from __future__ import annotations

import logging
import sys
import types
from pathlib import Path
from unittest import mock

import pytest

# Make ``src`` importable when running pytest from project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import lightgpt_inference, vram_monitor  # noqa: E402
from src.lightgpt_inference import (  # noqa: E402
    LightGPTInference,
    VALID_VARIANTS,
)


# =========================================================================
# Fakes
# =========================================================================


def _make_fake_torch():
    """Build a fake ``torch`` module with the API surface used by the engine."""
    cuda_ns = types.SimpleNamespace(
        is_available=lambda: True,
        empty_cache=lambda: None,
    )
    fake_torch = types.SimpleNamespace(
        float16="fp16-marker",  # any sentinel; we just check it is forwarded
        cuda=cuda_ns,
        no_grad=lambda: _NoGradContext(),
    )
    return fake_torch


class _NoGradContext:
    def __enter__(self):
        return None

    def __exit__(self, *_):
        return False


class _FakeBitsAndBytesConfig:
    """Stand-in for ``transformers.BitsAndBytesConfig`` that records args."""

    def __init__(
        self,
        *,
        load_in_4bit: bool,
        bnb_4bit_quant_type: str,
        bnb_4bit_compute_dtype,
        bnb_4bit_use_double_quant: bool,
    ) -> None:
        self.load_in_4bit = load_in_4bit
        self.bnb_4bit_quant_type = bnb_4bit_quant_type
        self.bnb_4bit_compute_dtype = bnb_4bit_compute_dtype
        self.bnb_4bit_use_double_quant = bnb_4bit_use_double_quant


class _FakeTokenizer:
    eos_token_id = 0

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def __call__(self, prompt, return_tensors="pt"):
        # Return dict-like with input_ids tensor stub
        return {"input_ids": _FakeTensor(shape=(1, 4), data=[[1, 2, 3, 4]])}

    def decode(self, ids, skip_special_tokens=True):
        if hasattr(ids, "data"):
            return f"<signal>ETWT</signal> tokens={list(ids.data)}"
        return "<signal>ETWT</signal>"


class _FakeTensor:
    def __init__(self, shape, data):
        self.shape = shape
        self.data = data

    def to(self, device):
        return self

    def __getitem__(self, idx):
        # support output_ids[0] (int) and output_ids[0][input_len:] (slice)
        if isinstance(idx, slice):
            sliced = self.data[idx]
            return _FakeTensor(shape=(len(sliced),), data=sliced)
        # int index — pick row
        row = self.data[idx]
        if isinstance(row, list):
            return _FakeTensor(shape=(len(row),), data=row)
        # scalar
        return _FakeTensor(shape=(), data=[row])


class _FakeModel:
    def __init__(self, name: str = "fake-model", param_bytes: int = 100 * 1024 * 1024):
        self.name = name
        self._param_bytes = param_bytes

    @classmethod
    def from_pretrained(cls, model_name, *args, **kwargs):
        return cls(name=model_name)

    def parameters(self):
        # one fake parameter with desired size
        yield _FakeParam(self._param_bytes)

    def buffers(self):
        return iter([])

    def generate(self, **kwargs):
        # Return a 2D tensor-like with shape (1, input_len + 4)
        return _FakeTensor(shape=(1, 8), data=[[1, 2, 3, 4, 5, 6, 7, 8]])


class _FakeParam:
    def __init__(self, total_bytes):
        self._bytes = total_bytes

    def element_size(self):
        return 1

    def nelement(self):
        return self._bytes


def _make_fake_transformers(model_factory=None, tokenizer_cls=None):
    """Build fake transformers module exposing the 3 symbols used."""
    fake = types.SimpleNamespace()
    fake.BitsAndBytesConfig = _FakeBitsAndBytesConfig

    class _AutoTokenizer:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            tokenizer = (tokenizer_cls or _FakeTokenizer)()
            return tokenizer

    class _AutoModel:
        @classmethod
        def from_pretrained(cls, model_name, *args, **kwargs):
            if model_factory is not None:
                return model_factory(model_name, *args, **kwargs)
            return _FakeModel(name=model_name)

    fake.AutoTokenizer = _AutoTokenizer
    fake.AutoModelForCausalLM = _AutoModel
    return fake


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def cache_dir(tmp_path: Path) -> str:
    """Provide a writable cache_dir under pytest's tmp_path."""
    return str(tmp_path / "hf_cache")


@pytest.fixture
def vram_ok():
    """Patch ``check_vram_available`` to always return True."""
    with mock.patch.object(
        vram_monitor, "check_vram_available", return_value=True
    ) as m:
        yield m


@pytest.fixture
def vram_fail():
    """Patch ``check_vram_available`` to always return False."""
    with mock.patch.object(
        vram_monitor, "check_vram_available", return_value=False
    ) as m:
        yield m


@pytest.fixture
def patch_torch():
    """Patch ``_import_torch`` to return a fake torch module."""
    fake = _make_fake_torch()
    with mock.patch.object(
        lightgpt_inference, "_import_torch", return_value=fake
    ):
        yield fake


@pytest.fixture
def patch_transformers():
    """Patch ``_import_transformers`` to return a fake module."""
    fake = _make_fake_transformers()
    with mock.patch.object(
        lightgpt_inference, "_import_transformers", return_value=fake
    ):
        yield fake


@pytest.fixture
def self_finetuned_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create a stub ``models/qwen2_finetuned/`` and chdir there.

    Returns the directory (already exists)."""
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "models" / "qwen2_finetuned"
    target.mkdir(parents=True)
    # Pretend a model is saved here.
    (target / "config.json").write_text("{}", encoding="utf-8")
    return target


@pytest.fixture
def no_self_finetuned(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Chdir to a tmp_path WITHOUT models/qwen2_finetuned/."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


# =========================================================================
# __init__ tests
# =========================================================================


class TestInit:
    def test_lightgpt_hf_with_valid_args(self, cache_dir, vram_ok):
        engine = LightGPTInference(
            variant="lightgpt_hf",
            cache_dir=cache_dir,
            hf_token="test-hf-token",
            device="cuda:0",
        )
        assert engine.variant == "lightgpt_hf"
        assert engine.cache_dir == cache_dir
        assert engine.hf_token == "test-hf-token"
        assert engine.device == "cuda:0"
        # No model loaded yet
        assert engine.model is None
        assert engine.tokenizer is None

    def test_lightgpt_mine_with_valid_args(
        self, cache_dir, vram_ok, self_finetuned_dir
    ):
        engine = LightGPTInference(
            variant="lightgpt_mine",
            cache_dir=cache_dir,
            hf_token="test-hf-token",  # Will be cleared (only used for hf variant)
            device="cuda:0",
        )
        assert engine.variant == "lightgpt_mine"
        # hf_token must NOT be retained for lightgpt_mine variant
        assert engine.hf_token is None
        # AC 5: variant lightgpt_mine KHÔNG dùng HF_TOKEN
        assert engine.model is None

    def test_invalid_variant_raises_valueerror(self, cache_dir, vram_ok):
        with pytest.raises(ValueError, match="variant"):
            LightGPTInference(
                variant="lightgpt_invalid",  # type: ignore[arg-type]
                cache_dir=cache_dir,
                hf_token=None,
            )

    def test_empty_cache_dir_raises_valueerror(self, vram_ok):
        with pytest.raises(ValueError, match="cache_dir"):
            LightGPTInference(
                variant="lightgpt_hf",
                cache_dir="",
                hf_token=None,
            )

    def test_invalid_hf_token_type_raises_valueerror(self, cache_dir, vram_ok):
        with pytest.raises(ValueError, match="hf_token"):
            LightGPTInference(
                variant="lightgpt_hf",
                cache_dir=cache_dir,
                hf_token=123,  # type: ignore[arg-type]
            )

    def test_empty_device_raises_valueerror(self, cache_dir, vram_ok):
        with pytest.raises(ValueError, match="device"):
            LightGPTInference(
                variant="lightgpt_hf",
                cache_dir=cache_dir,
                hf_token=None,
                device="",
            )

    def test_valid_variants_constant(self):
        assert VALID_VARIANTS == frozenset({"lightgpt_hf", "lightgpt_mine"})


# =========================================================================
# lightgpt_mine: FileNotFoundError if models/qwen2_finetuned/ missing
# =========================================================================


class TestLightgptMineMissingModel:
    def test_raises_filenotfounderror_with_clear_message(
        self, cache_dir, vram_ok, no_self_finetuned
    ):
        with pytest.raises(FileNotFoundError) as exc_info:
            LightGPTInference(
                variant="lightgpt_mine",
                cache_dir=cache_dir,
                hf_token=None,
            )
        msg = str(exc_info.value)
        assert "qwen2_finetuned" in msg
        assert "scripts/run_training.py" in msg

    def test_does_not_fallback_to_hf(
        self, cache_dir, vram_ok, no_self_finetuned, patch_transformers
    ):
        """Even with HF mocks installed, lightgpt_mine must NOT silently
        load from HuggingFace."""
        with pytest.raises(FileNotFoundError):
            LightGPTInference(
                variant="lightgpt_mine",
                cache_dir=cache_dir,
                hf_token=None,
            )

    def test_lightgpt_mine_load_model_raises_when_dir_disappears(
        self, cache_dir, vram_ok, self_finetuned_dir, patch_torch, patch_transformers, monkeypatch
    ):
        """Setup with dir, then remove dir before load_model() — must raise
        FileNotFoundError, not silently fall back."""
        engine = LightGPTInference(
            variant="lightgpt_mine",
            cache_dir=cache_dir,
            hf_token=None,
        )
        # Remove the directory to simulate it being deleted.
        import shutil

        shutil.rmtree(self_finetuned_dir)
        with pytest.raises(FileNotFoundError):
            engine.load_model()


# =========================================================================
# lightgpt_hf: HF_MODEL_PRIORITY ordering
# =========================================================================


class TestHFPriorityChain:
    def test_priority_chain_constant(self):
        assert LightGPTInference.HF_MODEL_PRIORITY == (
            "lightgpt/LightGPT-8B-Llama3",
            "lightgpt/LightGPT-7B-Qwen2",
            "lightgpt/LightGPT-7B-Llama2",
            "lightgpt/LightGPT-0.5B-Qwen2",
        )

    def test_excludes_13b_model(self):
        assert (
            "lightgpt/LightGPT-13B-Llama2"
            not in LightGPTInference.HF_MODEL_PRIORITY
        )

    def test_picks_first_priority_when_it_loads(
        self, cache_dir, vram_ok, patch_torch
    ):
        """First model in HF_MODEL_PRIORITY must be tried first; if
        loaded successfully, no fallback is attempted."""
        attempted: list[str] = []

        def model_factory(name, *args, **kwargs):
            attempted.append(name)
            # Small model fits VRAM
            return _FakeModel(name=name, param_bytes=100 * 1024 * 1024)

        fake_transformers = _make_fake_transformers(
            model_factory=model_factory
        )
        with mock.patch.object(
            lightgpt_inference,
            "_import_transformers",
            return_value=fake_transformers,
        ):
            engine = LightGPTInference(
                variant="lightgpt_hf",
                cache_dir=cache_dir,
                hf_token="test-hf-token",
            )
            engine.load_model()

        assert attempted == ["lightgpt/LightGPT-8B-Llama3"]
        assert engine.loaded_model_name == "lightgpt/LightGPT-8B-Llama3"

    def test_falls_back_to_next_when_first_fails(
        self, cache_dir, vram_ok, patch_torch
    ):
        """If first variant raises, must try next in priority."""
        attempted: list[str] = []

        def model_factory(name, *args, **kwargs):
            attempted.append(name)
            if name == "lightgpt/LightGPT-8B-Llama3":
                raise RuntimeError("download failed")
            return _FakeModel(name=name, param_bytes=100 * 1024 * 1024)

        fake_transformers = _make_fake_transformers(
            model_factory=model_factory
        )
        with mock.patch.object(
            lightgpt_inference,
            "_import_transformers",
            return_value=fake_transformers,
        ):
            engine = LightGPTInference(
                variant="lightgpt_hf",
                cache_dir=cache_dir,
                hf_token="test-hf-token",
            )
            engine.load_model()

        # Must try first then second
        assert attempted == [
            "lightgpt/LightGPT-8B-Llama3",
            "lightgpt/LightGPT-7B-Qwen2",
        ]
        assert engine.loaded_model_name == "lightgpt/LightGPT-7B-Qwen2"

    def test_falls_back_when_first_exceeds_vram(
        self, cache_dir, vram_ok, patch_torch
    ):
        """If first variant load OK but VRAM > MAX_VRAM_MB, fall back."""
        attempted: list[str] = []
        big_bytes = int((LightGPTInference.MAX_VRAM_MB + 1000) * 1024 * 1024)

        def model_factory(name, *args, **kwargs):
            attempted.append(name)
            if name == "lightgpt/LightGPT-8B-Llama3":
                return _FakeModel(name=name, param_bytes=big_bytes)
            return _FakeModel(name=name, param_bytes=100 * 1024 * 1024)

        fake_transformers = _make_fake_transformers(
            model_factory=model_factory
        )
        with mock.patch.object(
            lightgpt_inference,
            "_import_transformers",
            return_value=fake_transformers,
        ):
            engine = LightGPTInference(
                variant="lightgpt_hf",
                cache_dir=cache_dir,
                hf_token="test-hf-token",
            )
            engine.load_model()

        assert attempted[0] == "lightgpt/LightGPT-8B-Llama3"
        assert attempted[1] == "lightgpt/LightGPT-7B-Qwen2"
        assert engine.loaded_model_name == "lightgpt/LightGPT-7B-Qwen2"

    def test_raises_when_all_variants_fail(self, cache_dir, vram_ok, patch_torch):
        def model_factory(name, *args, **kwargs):
            raise RuntimeError(f"download failed for {name}")

        fake_transformers = _make_fake_transformers(
            model_factory=model_factory
        )
        with mock.patch.object(
            lightgpt_inference,
            "_import_transformers",
            return_value=fake_transformers,
        ):
            engine = LightGPTInference(
                variant="lightgpt_hf",
                cache_dir=cache_dir,
                hf_token="test-hf-token",
            )
            with pytest.raises(RuntimeError, match="failed to load any variant"):
                engine.load_model()


# =========================================================================
# _get_quantization_config
# =========================================================================


class TestQuantizationConfig:
    def test_returns_nf4_4bit_double_quant_fp16(
        self, cache_dir, vram_ok, patch_torch, patch_transformers
    ):
        engine = LightGPTInference(
            variant="lightgpt_hf",
            cache_dir=cache_dir,
            hf_token=None,
        )
        config = engine._get_quantization_config()

        assert config.load_in_4bit is True
        assert config.bnb_4bit_quant_type == "nf4"
        # Compute dtype must be torch.float16 (we forwarded sentinel "fp16-marker")
        assert config.bnb_4bit_compute_dtype == "fp16-marker"
        assert config.bnb_4bit_use_double_quant is True

    def test_raises_importerror_when_transformers_missing(
        self, cache_dir, vram_ok
    ):
        with mock.patch.object(
            lightgpt_inference, "_import_transformers", return_value=None
        ), mock.patch.object(
            lightgpt_inference, "_import_torch", return_value=None
        ):
            engine = LightGPTInference(
                variant="lightgpt_hf",
                cache_dir=cache_dir,
                hf_token=None,
            )
            with pytest.raises(ImportError):
                engine._get_quantization_config()


# =========================================================================
# VRAM check
# =========================================================================


class TestVRAMCheck:
    def test_init_raises_when_vram_insufficient(self, cache_dir, vram_fail):
        with pytest.raises(RuntimeError, match="VRAM insufficient"):
            LightGPTInference(
                variant="lightgpt_hf",
                cache_dir=cache_dir,
                hf_token=None,
            )

    def test_init_passes_min_vram_to_check(self, cache_dir, vram_ok):
        LightGPTInference(
            variant="lightgpt_hf",
            cache_dir=cache_dir,
            hf_token=None,
            device="cuda:0",
        )
        # Should have been called with required_mb >= MIN_VRAM_REQUIRED_MB
        vram_ok.assert_called_once()
        kwargs = vram_ok.call_args.kwargs
        assert kwargs["required_mb"] >= LightGPTInference.MIN_VRAM_REQUIRED_MB
        assert kwargs["device"] == "cuda:0"
        assert "lightgpt_hf" in kwargs["model_name"]

    def test_get_vram_usage_mb_returns_float_when_cuda_unavailable(
        self, cache_dir, vram_ok
    ):
        """Even on CPU-only environment, must return a float (0.0)."""
        engine = LightGPTInference(
            variant="lightgpt_hf",
            cache_dir=cache_dir,
            hf_token=None,
        )
        with mock.patch.object(
            vram_monitor, "get_vram_usage_mb", return_value=None
        ):
            value = engine.get_vram_usage_mb()
        assert isinstance(value, float)
        assert value == 0.0

    def test_get_vram_usage_mb_returns_float_when_cuda_available(
        self, cache_dir, vram_ok
    ):
        engine = LightGPTInference(
            variant="lightgpt_hf",
            cache_dir=cache_dir,
            hf_token=None,
        )
        with mock.patch.object(
            vram_monitor, "get_vram_usage_mb", return_value=1234.5
        ):
            value = engine.get_vram_usage_mb()
        assert isinstance(value, float)
        assert value == pytest.approx(1234.5)


# =========================================================================
# Generation
# =========================================================================


class TestGenerate:
    def test_generate_returns_text_with_mock_model(
        self, cache_dir, vram_ok, patch_torch, patch_transformers
    ):
        engine = LightGPTInference(
            variant="lightgpt_hf",
            cache_dir=cache_dir,
            hf_token=None,
        )
        engine.load_model()
        out = engine.generate("Test prompt asking about phase")
        assert isinstance(out, str)
        # Mock tokenizer.decode returns "<signal>ETWT</signal> ..."
        assert "<signal>" in out
        assert "ETWT" in out

    def test_generate_raises_when_not_loaded(
        self, cache_dir, vram_ok
    ):
        engine = LightGPTInference(
            variant="lightgpt_hf",
            cache_dir=cache_dir,
            hf_token=None,
        )
        with pytest.raises(RuntimeError, match="load_model"):
            engine.generate("hello")

    def test_generate_rejects_non_string(
        self, cache_dir, vram_ok, patch_torch, patch_transformers
    ):
        engine = LightGPTInference(
            variant="lightgpt_hf",
            cache_dir=cache_dir,
            hf_token=None,
        )
        engine.load_model()
        with pytest.raises(ValueError, match="prompt"):
            engine.generate(123)  # type: ignore[arg-type]

    def test_generate_rejects_empty_string(
        self, cache_dir, vram_ok, patch_torch, patch_transformers
    ):
        engine = LightGPTInference(
            variant="lightgpt_hf",
            cache_dir=cache_dir,
            hf_token=None,
        )
        engine.load_model()
        with pytest.raises(ValueError, match="non-empty"):
            engine.generate("")


# =========================================================================
# Constants
# =========================================================================


class TestConstants:
    def test_max_vram_budget_is_7_5gb(self):
        assert LightGPTInference.MAX_VRAM_MB == 7680.0

    def test_max_new_tokens_is_256(self):
        assert LightGPTInference.MAX_NEW_TOKENS == 256

    def test_timeout_is_10s(self):
        assert LightGPTInference.TIMEOUT_SECONDS == 10.0

    def test_self_finetuned_path_is_qwen2_finetuned(self):
        assert (
            LightGPTInference.SELF_FINETUNED_PATH == "models/qwen2_finetuned/"
        )

