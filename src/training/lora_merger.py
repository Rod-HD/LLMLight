"""LoRA Merger (Component 7 — Task 10.5).

Merge LoRA adapter weights vào base model Qwen2-0.5B để xuất ra một model
hoàn chỉnh dùng được cho inference (4-bit quantized) qua method
``lightgpt_mine`` trong :mod:`src.lightgpt_inference`.

Pipeline:

1. Validate inputs:

   * ``base_model``: non-empty str (HuggingFace ID hoặc local path).
   * ``adapter_path``: phải tồn tại trên disk (đầu ra của
     :class:`~src.training.cgpr_trainer.CGPRTrainer.train` hoặc
     :class:`~src.training.ift_trainer.IFTTrainer.train`).
   * ``output_path``: parent directory phải tồn tại hoặc tạo được.

2. Load base model + tokenizer qua
   :func:`transformers.AutoModelForCausalLM.from_pretrained` và
   :func:`transformers.AutoTokenizer.from_pretrained`.

3. Lưu lại ``num_hidden_layers`` / ``hidden_size`` / ``vocab_size`` của
   base config gốc để verify Property 7 sau merge.

4. Wrap base model với LoRA adapter qua
   :func:`peft.PeftModel.from_pretrained(base_model, adapter_path)`.

5. Gọi ``peft_model.merge_and_unload()`` để fold các LoRA delta weights
   ``A @ B`` vào các linear layers tương ứng của base model. Output là
   một :class:`transformers.PreTrainedModel` thuần (không còn LoRA wrap).

6. **Property 7 — LoRA Merge Architecture Preservation** (Requirements
   6.5, 12.7): assert merged model có cùng ``num_hidden_layers``,
   ``hidden_size``, ``vocab_size`` với base config gốc; raise
   :class:`ValueError` nếu khác biệt (chỉ rõ field bị thay đổi).

7. ``merged_model.save_pretrained(output_path)`` + tokenizer.

8. Return ``output_path`` để caller (script ``run_training.py``) chạy
   ``LightGPTInference(variant="lightgpt_mine", ...)`` đọc từ đường dẫn
   này.

Module import ``transformers`` / ``peft`` / ``torch`` lazily theo cùng
pattern với :mod:`src.training.ift_trainer` / :mod:`src.training.cgpr_trainer`,
cho phép unit tests chạy trên CPU-only env mà không cần download
Qwen2-0.5B (~1GB).

_Requirements_: 6.4, 6.5, 12.7
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Final

logger = logging.getLogger(__name__)

__all__ = [
    "LoRAMerger",
]


# =========================================================================
# Lazy imports — keep module importable on CPU-only env
# =========================================================================


def _import_transformers():
    """Import ``transformers`` lazily; ``None`` nếu chưa cài."""
    try:
        import transformers  # type: ignore[import-not-found]

        return transformers
    except ImportError:
        logger.debug(
            "lora_merger: transformers không khả dụng; merge() sẽ raise "
            "ImportError tại runtime."
        )
        return None


def _import_peft():
    """Import ``peft`` lazily; ``None`` nếu chưa cài."""
    try:
        import peft  # type: ignore[import-not-found]

        return peft
    except ImportError:
        logger.debug("lora_merger: peft không khả dụng.")
        return None


def _import_torch():
    """Import ``torch`` lazily; ``None`` nếu chưa cài."""
    try:
        import torch  # type: ignore[import-not-found]

        return torch
    except ImportError:
        logger.debug("lora_merger: torch không khả dụng.")
        return None


# =========================================================================
# Helpers
# =========================================================================


# Tên các field architecture trong HuggingFace ``PretrainedConfig``.
# Một số config (Llama, Qwen2) dùng ``num_hidden_layers``; một số config
# cũ hơn (GPT2) dùng ``num_layers`` / ``n_layer``. Ta thử lần lượt.
_NUM_LAYERS_ATTRS: Final[tuple[str, ...]] = (
    "num_hidden_layers",
    "num_layers",
    "n_layer",
)
_HIDDEN_SIZE_ATTRS: Final[tuple[str, ...]] = (
    "hidden_size",
    "n_embd",
    "d_model",
)
_VOCAB_SIZE_ATTRS: Final[tuple[str, ...]] = (
    "vocab_size",
)


def _get_first_attr(config: Any, names: tuple[str, ...]) -> Any:
    """Trả về giá trị attribute đầu tiên có trên ``config`` từ ``names``.

    Trả về ``None`` nếu không tìm thấy attribute nào.
    """
    for name in names:
        if hasattr(config, name):
            value = getattr(config, name)
            if value is not None:
                return value
    return None


def _extract_architecture(config: Any) -> dict[str, Any]:
    """Extract ``num_hidden_layers`` / ``hidden_size`` / ``vocab_size``
    từ một :class:`transformers.PretrainedConfig`.

    Returns:
        Dict với keys ``num_hidden_layers``, ``hidden_size``,
        ``vocab_size``. Giá trị có thể là ``None`` nếu config không có
        field tương ứng (rất hiếm với decoder-only LLM).
    """
    return {
        "num_hidden_layers": _get_first_attr(config, _NUM_LAYERS_ATTRS),
        "hidden_size": _get_first_attr(config, _HIDDEN_SIZE_ATTRS),
        "vocab_size": _get_first_attr(config, _VOCAB_SIZE_ATTRS),
    }


# =========================================================================
# LoRAMerger
# =========================================================================


class LoRAMerger:
    """Merge LoRA adapter vào base model và xuất merged model.

    Lifecycle (single :meth:`merge` call):

    1. Validate inputs (base_model non-empty str, adapter exists,
       output_path parent writable).
    2. Load base model + tokenizer.
    3. Snapshot base config architecture (``num_hidden_layers``,
       ``hidden_size``, ``vocab_size``).
    4. Wrap với :class:`peft.PeftModel` từ adapter.
    5. ``merge_and_unload()`` để fold weights.
    6. Verify Property 7: merged config khớp base config snapshot.
    7. Save merged model + tokenizer vào ``output_path``.
    8. Return ``output_path``.

    Module-level ``transformers`` / ``peft`` / ``torch`` được import
    lazily; trong unit tests chúng được monkey-patched.
    """

    def __init__(self) -> None:
        """Khởi tạo merger (no state). Mọi tham số được truyền vào
        :meth:`merge`."""

    # ------------------------------------------------------------ Public --

    def merge(
        self,
        base_model: str,
        adapter_path: str,
        output_path: str,
    ) -> str:
        """Merge LoRA adapter weights vào base model và lưu kết quả.

        Args:
            base_model: HuggingFace model ID (e.g. ``"Qwen/Qwen2-0.5B"``)
                hoặc local path tới base model. Phải là non-empty string.
            adapter_path: Đường dẫn tới LoRA adapter directory đã được
                lưu bởi :meth:`peft.PeftModel.save_pretrained` (hoặc
                :meth:`transformers.Trainer.save_model`). Phải tồn tại
                trên disk.
            output_path: Đường dẫn tới directory đích để lưu merged
                model (e.g. ``"models/qwen2_finetuned/"``). Tạo mới nếu
                chưa tồn tại.

        Returns:
            ``output_path`` (cùng giá trị với input). Caller có thể
            truyền thẳng vào :class:`LightGPTInference(variant="lightgpt_mine")`.

        Raises:
            ValueError: ``base_model`` empty hoặc không phải str;
                ``output_path`` empty hoặc không phải str; merged
                architecture không khớp base (Property 7 vi phạm).
            FileNotFoundError: ``adapter_path`` không tồn tại trên disk.
            ImportError: ``transformers`` / ``peft`` / ``torch`` chưa cài.
        """
        # ----- Validate base_model
        if not isinstance(base_model, str) or not base_model:
            raise ValueError(
                "LoRAMerger.merge: base_model must be non-empty str "
                f"(HuggingFace ID hoặc local path); got {type(base_model).__name__} "
                f"({base_model!r})"
            )

        # ----- Validate adapter_path
        if not isinstance(adapter_path, str) or not adapter_path:
            raise ValueError(
                "LoRAMerger.merge: adapter_path must be non-empty str; "
                f"got {type(adapter_path).__name__} ({adapter_path!r})"
            )
        if not os.path.exists(adapter_path):
            raise FileNotFoundError(
                f"LoRAMerger.merge: adapter_path does not exist: "
                f"{adapter_path!r}. Đảm bảo IFTTrainer (Task 10.2) hoặc "
                "CGPRTrainer (Task 10.4) đã hoàn tất và lưu adapter "
                "thành công trước khi gọi merge."
            )

        # ----- Validate output_path
        if not isinstance(output_path, str) or not output_path:
            raise ValueError(
                "LoRAMerger.merge: output_path must be non-empty str; "
                f"got {type(output_path).__name__} ({output_path!r})"
            )

        # Tạo parent dir nếu chưa tồn tại (raise OSError nếu không
        # tạo được do permission/path lỗi).
        Path(output_path).mkdir(parents=True, exist_ok=True)

        # ----- Lazy imports
        transformers = _import_transformers()
        peft = _import_peft()
        torch = _import_torch()
        if transformers is None or peft is None or torch is None:
            raise ImportError(
                "LoRAMerger.merge: requires transformers, peft, and torch. "
                "Install via venv/bin/pip install transformers peft torch."
            )

        logger.info(
            "LoRAMerger.merge: starting (base_model=%s, adapter_path=%s, "
            "output_path=%s)",
            base_model,
            adapter_path,
            output_path,
        )

        # ----- Load base model + tokenizer
        tokenizer = transformers.AutoTokenizer.from_pretrained(base_model)
        base = transformers.AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=getattr(torch, "float16", None),
        )

        # ----- Snapshot base architecture (BEFORE wrap/merge)
        base_arch = _extract_architecture(base.config)
        logger.info(
            "LoRAMerger: base architecture: num_hidden_layers=%s, "
            "hidden_size=%s, vocab_size=%s",
            base_arch["num_hidden_layers"],
            base_arch["hidden_size"],
            base_arch["vocab_size"],
        )

        # ----- Wrap with PEFT adapter
        peft_model = peft.PeftModel.from_pretrained(base, adapter_path)
        logger.info(
            "LoRAMerger: loaded PEFT adapter from %s", adapter_path
        )

        # ----- Merge LoRA delta weights into base linear layers
        merged_model = peft_model.merge_and_unload()
        logger.info("LoRAMerger: merged LoRA delta weights into base model")

        # ----- Property 7: verify architecture preservation
        merged_arch = _extract_architecture(merged_model.config)
        self._verify_architecture_preserved(base_arch, merged_arch)

        # ----- Save merged model + tokenizer
        merged_model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        logger.info(
            "LoRAMerger.merge: saved merged model + tokenizer to %s",
            output_path,
        )

        return output_path

    # ------------------------------------------------------------ Helpers --

    @staticmethod
    def _verify_architecture_preserved(
        base_arch: dict[str, Any],
        merged_arch: dict[str, Any],
    ) -> None:
        """Property 7 — assert merged config matches base config trên
        ``num_hidden_layers``, ``hidden_size``, ``vocab_size``.

        Raises:
            ValueError: Nếu bất kỳ field nào khác giữa base và merged.
                Message chứa tất cả field bị thay đổi và giá trị before/after.
        """
        diffs: list[str] = []
        for field in ("num_hidden_layers", "hidden_size", "vocab_size"):
            base_v = base_arch.get(field)
            merged_v = merged_arch.get(field)
            if base_v != merged_v:
                diffs.append(
                    f"{field}: base={base_v!r} merged={merged_v!r}"
                )

        if diffs:
            raise ValueError(
                "LoRAMerger.merge: merged model architecture KHÔNG khớp "
                "base model (vi phạm Property 7 — LoRA Merge Architecture "
                "Preservation, Requirements 6.5/12.7). Differences: "
                + "; ".join(diffs)
            )

        logger.info(
            "LoRAMerger: architecture preserved (Property 7 OK) — "
            "num_hidden_layers=%s, hidden_size=%s, vocab_size=%s",
            base_arch["num_hidden_layers"],
            base_arch["hidden_size"],
            base_arch["vocab_size"],
        )
