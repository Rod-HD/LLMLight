"""LightGPT local inference engine với NF4 4-bit quantization (Task 6.2).

Component 5 trong design.md. Hỗ trợ HAI biến thể chạy song song:

* ``lightgpt_hf``  — load model đã merged từ HuggingFace theo
  ``HF_MODEL_PRIORITY`` fallback chain. Tất cả model trong chain MIT-licensed,
  không gated; ``HF_TOKEN`` chỉ dùng để giảm rate-limit khi tải xuống.
* ``lightgpt_mine`` — load model self-finetuned từ
  ``models/qwen2_finetuned/`` (output của Training Pipeline IFT + CGPR +
  LoRA Merge). **TUYỆT ĐỐI KHÔNG fallback sang model HF** nếu thư mục
  không tồn tại — phải raise ``FileNotFoundError`` để tránh nhầm lẫn
  kết quả ``lightgpt_hf`` với ``lightgpt_mine`` (làm mất ý nghĩa khoa học
  của so sánh "verify pipeline fine-tune").

Cả hai biến thể dùng chung infrastructure NF4 4-bit quantization
(:meth:`LightGPTInference._get_quantization_config`) và shared
generation pipeline (max 256 tokens, timeout 10s/intersection/timestep).

Module chỉ import ``transformers`` / ``torch`` lazily trong
:func:`_import_transformers` / :func:`_import_torch` để có thể được test
trên môi trường không có CUDA / chưa cài torch (Windows dev box).

_Requirements_: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 5.10,
5.11, 5.12, 11.1, 11.2
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any, Literal

from src import vram_monitor

logger = logging.getLogger(__name__)


# =========================================================================
# Constants
# =========================================================================

#: Variant literal type — duy nhất hai giá trị hợp lệ (Requirement 5 AC 9).
Variant = Literal["lightgpt_hf", "lightgpt_mine"]

#: Variant set hợp lệ — runtime check.
VALID_VARIANTS: frozenset[str] = frozenset({"lightgpt_hf", "lightgpt_mine"})


# =========================================================================
# Lazy imports — keep module importable on Windows dev box
# =========================================================================


def _import_torch():
    """Import torch lazily; trả về module hoặc ``None`` nếu chưa cài."""
    try:
        import torch  # type: ignore[import-not-found]

        return torch
    except ImportError:
        logger.debug(
            "lightgpt_inference: torch không khả dụng; mọi load/generate "
            "sẽ raise ImportError tại runtime."
        )
        return None


def _import_transformers():
    """Import transformers lazily; trả về module hoặc ``None`` nếu chưa cài."""
    try:
        import transformers  # type: ignore[import-not-found]

        return transformers
    except ImportError:
        logger.debug(
            "lightgpt_inference: transformers không khả dụng; mọi "
            "load/generate sẽ raise ImportError tại runtime."
        )
        return None


# =========================================================================
# LightGPTInference
# =========================================================================


class LightGPTInference:
    """Inference wrapper cho LightGPT 4-bit quantized với 2 biến thể.

    Một instance bind vào MỘT variant duy nhất tại ``__init__``. Pipeline
    runner sẽ tạo TWO instances (lightgpt_hf + lightgpt_mine) ở Phase 2/3
    và hiển thị thành HAI dòng riêng biệt trong bảng comparison
    (Requirement 5 AC 10).

    Lifecycle:

    1. ``__init__(variant, cache_dir, hf_token, device)`` — validate args
       + VRAM check + (cho ``lightgpt_mine``) verify thư mục tồn tại.
       KHÔNG load model ở đây để runner có thể detect lỗi sớm.
    2. ``load_model()`` — tải model với NF4 quantization.
       ``lightgpt_hf`` thử từng entry trong :attr:`HF_MODEL_PRIORITY`;
       ``lightgpt_mine`` load trực tiếp từ :attr:`SELF_FINETUNED_PATH`.
    3. ``generate(prompt)`` — sinh response (≤ :attr:`MAX_NEW_TOKENS`,
       timeout :attr:`TIMEOUT_SECONDS`).
    4. ``get_vram_usage_mb()`` — query VRAM hiện đang được model chiếm.
    """

    # -------------------------------------------------------------- Const --

    #: Fallback chain cho variant ``lightgpt_hf`` (ưu tiên cao → thấp).
    #: Tất cả là model đã merged trên HuggingFace, MIT-licensed, không gated.
    #: LOẠI TRỪ ``lightgpt/LightGPT-13B-Llama2`` (vượt 8GB VRAM kể cả với 4-bit).
    HF_MODEL_PRIORITY: tuple[str, ...] = (
        "lightgpt/LightGPT-8B-Llama3",      # primary
        "lightgpt/LightGPT-7B-Qwen2",       # fallback 1
        "lightgpt/LightGPT-7B-Llama2",      # fallback 2
        "lightgpt/LightGPT-0.5B-Qwen2",     # emergency fallback
    )

    #: Đường dẫn cố định cho variant ``lightgpt_mine`` (output của
    #: Training Pipeline; relative đến project root).
    SELF_FINETUNED_PATH: str = "models/qwen2_finetuned/"

    #: VRAM budget cho RTX 4060 8GB (Requirement 11.1: < 7.5GB).
    MAX_VRAM_MB: float = 7680.0

    #: Tokens tối đa generate cho mỗi prompt (Requirement 5 AC 3).
    MAX_NEW_TOKENS: int = 256

    #: Timeout per generate call, second (Requirement 5 AC 3).
    TIMEOUT_SECONDS: float = 10.0

    #: Estimate VRAM tối thiểu cần để load model nhỏ nhất ở 4-bit
    #: (~600MB cho weights + headroom ~200MB cho activations + KV cache
    #: short prompts). Dùng cho ``check_vram_available`` trước khi load
    #: bất kỳ variant nào (Requirement 5 AC 6).
    MIN_VRAM_REQUIRED_MB: float = 800.0

    # -------------------------------------------------------------- Init --

    def __init__(
        self,
        variant: Variant,
        cache_dir: str,
        hf_token: str | None,
        device: str = "cuda:0",
    ) -> None:
        """Khởi tạo inference engine.

        Args:
            variant: ``"lightgpt_hf"`` (load từ HF) hoặc ``"lightgpt_mine"``
                (load self-finetuned từ :attr:`SELF_FINETUNED_PATH`).
            cache_dir: Thư mục cache HuggingFace (``HF_HOME`` từ ``.env``).
                Phải là string không rỗng. Được tạo nếu chưa tồn tại.
            hf_token: HuggingFace token để giảm rate-limit khi tải xuống
                (chỉ dùng cho ``lightgpt_hf``). ``None`` được phép.
            device: CUDA device, mặc định ``"cuda:0"``.

        Raises:
            ValueError: ``variant`` không hợp lệ, ``cache_dir`` rỗng, hoặc
                ``hf_token`` không phải str/None.
            FileNotFoundError: ``variant == "lightgpt_mine"`` và
                :attr:`SELF_FINETUNED_PATH` không tồn tại. KHÔNG fallback
                sang HF (Requirement 5 AC 11).
            RuntimeError: VRAM không đủ để load variant nhỏ nhất ở
                4-bit (Requirement 5 AC 6, 11.2).
        """
        # ----- variant
        if variant not in VALID_VARIANTS:
            raise ValueError(
                f"LightGPTInference.variant must be one of "
                f"{sorted(VALID_VARIANTS)}; got {variant!r}"
            )
        self.variant: Variant = variant

        # ----- cache_dir
        if not isinstance(cache_dir, str) or not cache_dir:
            raise ValueError(
                "LightGPTInference.cache_dir must be a non-empty str; "
                f"got {cache_dir!r}"
            )
        self.cache_dir: str = cache_dir
        # Tạo thư mục cache nếu chưa có (idempotent).
        try:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ValueError(
                f"LightGPTInference: cannot create cache_dir={cache_dir!r}: "
                f"{exc}"
            ) from exc

        # ----- hf_token (chỉ áp dụng cho lightgpt_hf; với lightgpt_mine
        # ta vẫn chấp nhận giá trị nhưng KHÔNG dùng để load).
        if hf_token is not None and not isinstance(hf_token, str):
            raise ValueError(
                "LightGPTInference.hf_token must be str or None; "
                f"got {type(hf_token).__name__}"
            )
        self.hf_token: str | None = hf_token if variant == "lightgpt_hf" else None

        # ----- device
        if not isinstance(device, str) or not device:
            raise ValueError(
                "LightGPTInference.device must be a non-empty str "
                f"(e.g. 'cuda:0'); got {device!r}"
            )
        self.device: str = device

        # ----- variant == lightgpt_mine: verify path tồn tại NGAY
        # (Requirement 5 AC 11). KHÔNG fallback sang HF.
        if variant == "lightgpt_mine":
            self_finetuned = Path(self.SELF_FINETUNED_PATH)
            if not self_finetuned.exists() or not self_finetuned.is_dir():
                raise FileNotFoundError(
                    "Model qwen2_finetuned chưa được train; chạy "
                    "`scripts/run_training.py` trước"
                )

        # ----- VRAM check (Requirement 5 AC 6, Requirement 11.1).
        # Dùng tên model phụ thuộc variant cho log message rõ nghĩa.
        check_name = (
            f"lightgpt_mine ({self.SELF_FINETUNED_PATH})"
            if variant == "lightgpt_mine"
            else f"lightgpt_hf ({self.HF_MODEL_PRIORITY[0]})"
        )
        if not vram_monitor.check_vram_available(
            required_mb=self.MIN_VRAM_REQUIRED_MB,
            device=device,
            model_name=check_name,
        ):
            raise RuntimeError(
                "LightGPTInference: VRAM insufficient for any variant at "
                f"4-bit quantization (required >= {self.MIN_VRAM_REQUIRED_MB:.0f}"
                f"MB on device={device!r}). See [VRAM_ERROR] log above."
            )

        # ----- runtime state, populated by load_model()
        self.model: Any | None = None
        self.tokenizer: Any | None = None
        #: Tên/đường dẫn model đã load thành công (set bởi ``load_model``).
        self.loaded_model_name: str | None = None

    # ------------------------------------------------------------- Config --

    def _get_quantization_config(self) -> Any:
        """Build :class:`transformers.BitsAndBytesConfig` cho NF4 4-bit.

        NF4 (4-bit NormalFloat) + double-quantization là cấu hình tối ưu
        cho memory budget 7.5GB của RTX 4060 (Requirement 5 AC 2). Compute
        dtype ``float16`` (không bfloat16) vì RTX 4060 hỗ trợ FP16 native
        và FP16 có range animation tốt hơn cho text generation ngắn.

        Returns:
            ``BitsAndBytesConfig`` instance.

        Raises:
            ImportError: nếu ``transformers`` / ``torch`` không khả dụng.
        """
        transformers = _import_transformers()
        torch = _import_torch()
        if transformers is None or torch is None:
            raise ImportError(
                "LightGPTInference: transformers and torch must be installed "
                "to build BitsAndBytesConfig. Install via "
                "`venv/bin/pip install transformers torch bitsandbytes`."
            )
        return transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    # --------------------------------------------------------------- Load --

    def load_model(self) -> None:
        """Tải model với NF4 4-bit quantization.

        * ``variant == "lightgpt_hf"``: thử từng entry trong
          :attr:`HF_MODEL_PRIORITY`. Variant đầu tiên load thành công và
          fit < :attr:`MAX_VRAM_MB` được chọn.
        * ``variant == "lightgpt_mine"``: load trực tiếp từ
          :attr:`SELF_FINETUNED_PATH`.

        Sau khi load thành công, set ``self.model``, ``self.tokenizer``, và
        ``self.loaded_model_name``.

        Raises:
            ImportError: ``transformers`` / ``torch`` chưa cài.
            RuntimeError: ``lightgpt_hf`` không load được variant nào trong
                priority chain (Requirement 5 AC 6).
            FileNotFoundError: ``lightgpt_mine`` mà
                :attr:`SELF_FINETUNED_PATH` biến mất giữa ``__init__`` và
                ``load_model``.
        """
        transformers = _import_transformers()
        torch = _import_torch()
        if transformers is None or torch is None:
            raise ImportError(
                "LightGPTInference.load_model: transformers and torch must "
                "be installed."
            )

        quant_config = self._get_quantization_config()

        if self.variant == "lightgpt_mine":
            self._load_self_finetuned(transformers, quant_config)
        else:
            self._load_from_hf_priority(transformers, quant_config)

    def _load_self_finetuned(self, transformers, quant_config) -> None:
        """Load model từ :attr:`SELF_FINETUNED_PATH` (variant lightgpt_mine).

        KHÔNG fallback sang HF nếu fail (Requirement 5 AC 11).
        """
        path = Path(self.SELF_FINETUNED_PATH)
        if not path.exists() or not path.is_dir():
            raise FileNotFoundError(
                "Model qwen2_finetuned chưa được train; chạy "
                "`scripts/run_training.py` trước"
            )
        path_str = str(path)
        logger.info(
            "LightGPTInference: loading self-finetuned model from %s",
            path_str,
        )
        try:
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                path_str, cache_dir=self.cache_dir
            )
            model = transformers.AutoModelForCausalLM.from_pretrained(
                path_str,
                quantization_config=quant_config,
                device_map=self.device,
                cache_dir=self.cache_dir,
            )
        except Exception as exc:
            raise RuntimeError(
                f"LightGPTInference: failed to load self-finetuned model "
                f"from {path_str!r}: {exc}"
            ) from exc

        self._verify_vram_after_load(path_str)
        self.tokenizer = tokenizer
        self.model = model
        self.loaded_model_name = path_str
        logger.info(
            "LightGPTInference: loaded self-finetuned model %s "
            "(VRAM=%.2fMB)",
            path_str,
            self.get_vram_usage_mb() or -1.0,
        )

    def _load_from_hf_priority(self, transformers, quant_config) -> None:
        """Thử từng model trong :attr:`HF_MODEL_PRIORITY`.

        Variant đầu tiên load thành công VÀ fit ``<= MAX_VRAM_MB`` được
        chọn. Nếu không variant nào fit → raise ``RuntimeError``
        (Requirement 5 AC 6).
        """
        last_error: Exception | None = None
        for model_name in self.HF_MODEL_PRIORITY:
            logger.info(
                "LightGPTInference: trying HF model %s (variant=lightgpt_hf)",
                model_name,
            )
            try:
                tokenizer = transformers.AutoTokenizer.from_pretrained(
                    model_name,
                    cache_dir=self.cache_dir,
                    token=self.hf_token,
                )
                model = transformers.AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=quant_config,
                    device_map=self.device,
                    cache_dir=self.cache_dir,
                    token=self.hf_token,
                )
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "LightGPTInference: failed to load %s: %s; "
                    "trying next variant",
                    model_name,
                    exc,
                )
                continue

            # Verify model fit VRAM budget.
            usage_mb = self.get_vram_usage_mb_for_module(model)
            if usage_mb is not None and usage_mb > self.MAX_VRAM_MB:
                logger.warning(
                    "LightGPTInference: %s exceeds VRAM budget "
                    "(%.2fMB > %.2fMB); freeing and trying next variant",
                    model_name,
                    usage_mb,
                    self.MAX_VRAM_MB,
                )
                # Best-effort free
                self._free_module(model)
                last_error = RuntimeError(
                    f"{model_name} exceeds {self.MAX_VRAM_MB:.0f}MB "
                    f"VRAM budget at 4-bit (used {usage_mb:.2f}MB)"
                )
                continue

            # Success
            self.tokenizer = tokenizer
            self.model = model
            self.loaded_model_name = model_name
            logger.info(
                "LightGPTInference: loaded %s (VRAM=%.2fMB)",
                model_name,
                usage_mb if usage_mb is not None else -1.0,
            )
            return

        # Hết priority chain mà chưa load được.
        raise RuntimeError(
            "LightGPTInference: failed to load any variant from "
            f"HF_MODEL_PRIORITY={list(self.HF_MODEL_PRIORITY)}; "
            f"last_error={last_error!r}"
        )

    def _verify_vram_after_load(self, model_name: str) -> None:
        """Sau load, kiểm tra ``self.model`` không vượt budget.

        Raises:
            RuntimeError: nếu vượt budget.
        """
        # ``self.model`` chưa được gán; param truyền vào là tên display.
        # Caller phải gọi sau khi tự đo (cho ``lightgpt_mine``).
        usage_mb = self.get_vram_usage_mb()
        if usage_mb is not None and usage_mb > self.MAX_VRAM_MB:
            raise RuntimeError(
                f"LightGPTInference: {model_name} exceeds {self.MAX_VRAM_MB:.0f}MB "
                f"VRAM budget at 4-bit (used {usage_mb:.2f}MB)"
            )

    def _free_module(self, module: Any) -> None:
        """Best-effort free a torch module + cuda cache."""
        torch = _import_torch()
        try:
            del module
        except Exception:  # pragma: no cover
            pass
        try:
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # pragma: no cover
            pass

    # ----------------------------------------------------------- Generate --

    def generate(self, prompt: str) -> str:
        """Generate response từ prompt với timeout 10s.

        Args:
            prompt: Output của ``ObservationParser.parse``.

        Returns:
            Response text (đã decode), tối đa :attr:`MAX_NEW_TOKENS` tokens.
            Nếu timeout → trả về chuỗi rỗng (caller dùng default phase).

        Raises:
            RuntimeError: ``load_model`` chưa được gọi.
            ValueError: ``prompt`` không phải str hoặc rỗng.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError(
                "LightGPTInference.generate: call load_model() first."
            )
        if not isinstance(prompt, str):
            raise ValueError(
                "LightGPTInference.generate: prompt must be str; "
                f"got {type(prompt).__name__}"
            )
        if not prompt:
            raise ValueError(
                "LightGPTInference.generate: prompt must be non-empty."
            )

        torch = _import_torch()
        assert torch is not None  # load_model would have failed otherwise

        result_holder: dict[str, str] = {}
        error_holder: dict[str, BaseException] = {}

        def _run() -> None:
            try:
                inputs = self.tokenizer(prompt, return_tensors="pt")
                # Move tensors to same device as model.
                try:
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                except Exception:
                    # If device move fails (e.g. mocked tensors), fall back
                    # to using inputs as-is.
                    pass
                with torch.no_grad():
                    output_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=self.MAX_NEW_TOKENS,
                        do_sample=False,
                        temperature=0.0,
                        pad_token_id=getattr(
                            self.tokenizer,
                            "eos_token_id",
                            None,
                        ),
                    )
                # Strip prompt tokens from output (causal LM convention).
                input_len = inputs["input_ids"].shape[-1] if "input_ids" in inputs else 0
                generated_ids = output_ids[0][input_len:]
                text = self.tokenizer.decode(
                    generated_ids, skip_special_tokens=True
                )
                result_holder["text"] = text
            except BaseException as exc:  # noqa: BLE001
                error_holder["err"] = exc

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        thread.join(timeout=self.TIMEOUT_SECONDS)
        if thread.is_alive():
            logger.warning(
                "LightGPTInference.generate: timeout after %.1fs; "
                "returning empty string (caller should use default phase)",
                self.TIMEOUT_SECONDS,
            )
            return ""
        if "err" in error_holder:
            err = error_holder["err"]
            logger.warning(
                "LightGPTInference.generate: %s: %s; returning empty string",
                type(err).__name__,
                err,
            )
            return ""
        return result_holder.get("text", "")

    # --------------------------------------------------------------- VRAM --

    def get_vram_usage_mb(self) -> float:
        """Return VRAM hiện đang được PyTorch allocate (MB).

        Trả về ``0.0`` nếu CUDA không khả dụng (thay vì ``None`` để giữ
        signature ``-> float`` đơn giản cho caller).
        """
        usage = vram_monitor.get_vram_usage_mb(self.device)
        return float(usage) if usage is not None else 0.0

    @staticmethod
    def get_vram_usage_mb_for_module(module: Any) -> float | None:
        """Đo VRAM của một module cụ thể (sum parameter + buffer bytes).

        Dùng cho HF priority loop khi cần biết module mới load chiếm bao
        nhiêu VRAM (vs query toàn process via ``vram_monitor``). Trả về
        ``None`` nếu module không có ``.parameters()`` hoặc torch không có.
        """
        torch = _import_torch()
        if torch is None or module is None:
            return None
        try:
            total_bytes = 0
            for p in module.parameters():
                # quantized 4-bit weights expose `.element_size()` * `.nelement()`
                try:
                    total_bytes += p.element_size() * p.nelement()
                except Exception:
                    continue
            for b in module.buffers():
                try:
                    total_bytes += b.element_size() * b.nelement()
                except Exception:
                    continue
            return total_bytes / (1024 * 1024)
        except Exception as exc:
            logger.debug(
                "LightGPTInference.get_vram_usage_mb_for_module: %s", exc
            )
            return None
