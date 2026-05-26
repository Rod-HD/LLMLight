"""IFT Trainer (Component 7 — Task 10.2).

Instruction Fine-Tuning Qwen2-0.5B với LoRA trên trajectory data thu thập
được bởi :class:`~src.training.trajectory_collector.TrajectoryCollector`.

Pipeline:

1. Load base model (``Qwen/Qwen2-0.5B``) và tokenizer.
2. Bọc model với LoRA adapter (rank=8, alpha=16, target_modules=
   ``("q_proj", "v_proj")``) qua :func:`peft.get_peft_model`.
3. Tokenize ``(prompt, response)`` pairs thành ``input_ids`` và
   ``labels`` (mask prompt tokens với ``-100`` để loss chỉ tính trên
   response).
4. Dùng :class:`transformers.Trainer` với ``save_strategy="epoch"`` để
   checkpoint mỗi epoch boundary (Requirement 6 AC 6 — resume support).
5. Monitor VRAM sau model load + mỗi epoch boundary; raise
   :class:`OOMError` nếu vượt 7.5GB (Requirement 6 AC 3, 11.1).
6. Trả về đường dẫn final checkpoint (``checkpoint-<step>`` cuối cùng
   trong ``checkpoint_dir``) — :class:`CGPRDataCollector` (Task 10.3) và
   :class:`CGPRTrainer` (Task 10.4) sẽ load từ đường dẫn này.

LoRA config được fix-vào-design để tuân thủ Requirement 6 AC 1, 3:
``rank=8`` + ``alpha=16`` + ``gradient_accumulation_steps>=4`` + ``batch_size=1``
giữ VRAM dưới 7.5GB trên RTX 4060 8GB. Caller có thể override qua
:class:`TrainingConfig` nhưng giá trị mặc định khớp paper LLMLight và
spec của task.

Module import ``transformers`` / ``peft`` / ``torch`` / ``datasets``
lazily để có thể test trên CPU-only environment / mock heavy ops mà
không cần download Qwen2-0.5B (≈1GB).

_Requirements_: 6.1, 6.3, 6.6
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from src import vram_monitor

logger = logging.getLogger(__name__)

__all__ = [
    "OOMError",
    "TrainingConfig",
    "IFTTrainer",
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
            "ift_trainer: transformers không khả dụng; train() sẽ raise "
            "ImportError tại runtime."
        )
        return None


def _import_peft():
    """Import ``peft`` lazily; ``None`` nếu chưa cài."""
    try:
        import peft  # type: ignore[import-not-found]

        return peft
    except ImportError:
        logger.debug("ift_trainer: peft không khả dụng.")
        return None


def _import_torch():
    """Import ``torch`` lazily; ``None`` nếu chưa cài."""
    try:
        import torch  # type: ignore[import-not-found]

        return torch
    except ImportError:
        logger.debug("ift_trainer: torch không khả dụng.")
        return None


def _import_datasets():
    """Import ``datasets`` lazily; ``None`` nếu chưa cài."""
    try:
        import datasets  # type: ignore[import-not-found]

        return datasets
    except ImportError:
        logger.debug("ift_trainer: datasets không khả dụng.")
        return None


# =========================================================================
# Errors
# =========================================================================


class OOMError(RuntimeError):
    """Raised khi VRAM vượt :attr:`IFTTrainer.MAX_VRAM_MB` (7.5GB) trong
    quá trình huấn luyện hoặc sau khi load base model.

    Subclass của :class:`RuntimeError` để runner có thể catch ở mức rộng
    nếu cần fallback (ví dụ giảm ``gradient_accumulation_steps`` hoặc
    chuyển sang batch_size nhỏ hơn).
    """


# =========================================================================
# TrainingConfig
# =========================================================================


@dataclass(frozen=True)
class TrainingConfig:
    """Cấu hình IFT training (Component 7 trong design.md).

    Mặc định khớp với paper LLMLight và Requirement 6 AC 1, 3:
    LoRA rank=8/alpha=16, target_modules ``q_proj``/``v_proj`` (Qwen2
    attention projections), gradient_accumulation_steps>=4, batch_size=1
    để giữ VRAM dưới 7.5GB trên RTX 4060 8GB.

    Validation tại ``__post_init__``:

    * ``ift_epochs`` ∈ [3, 10] (AC 1).
    * ``lora_rank`` > 0.
    * ``lora_alpha`` > 0.
    * ``gradient_accumulation_steps`` >= 4 (AC 3).
    * ``batch_size`` >= 1.
    * ``lora_target_modules`` non-empty tuple of str.
    * ``max_seq_length`` >= 1.
    * ``learning_rate`` > 0.
    * ``checkpoint_dir`` / ``output_dir`` non-empty str.
    """

    #: Base model HuggingFace ID. Paper dùng Qwen2-0.5B cho LightGPT 0.5B
    #: variant.
    base_model: str = "Qwen/Qwen2-0.5B"

    #: LoRA rank — paper LLMLight dùng 8 (Requirement 6 AC 1).
    lora_rank: int = 8

    #: LoRA alpha — paper dùng 16 (typically ``2 * rank``).
    lora_alpha: int = 16

    #: Modules để áp LoRA. Cho Qwen2 attention, ``q_proj`` + ``v_proj``
    #: là minimal config tiết kiệm VRAM (alternative: ``q/k/v/o_proj``
    #: cho coverage cao hơn nhưng VRAM cũng cao hơn). Spec task chỉ định
    #: ``("q_proj", "v_proj")``.
    lora_target_modules: tuple[str, ...] = ("q_proj", "v_proj")

    #: LoRA dropout — light regularization (paper không nêu cụ thể;
    #: 0.05 là default phổ biến trong PEFT examples).
    lora_dropout: float = 0.05

    #: Gradient accumulation — paper effective batch size is 128.
    gradient_accumulation_steps: int = 128

    #: Per-device batch size. Với 8GB VRAM và Qwen2-0.5B, batch_size=1
    #: là an toàn (effective batch size = 1 * gradient_accumulation = 128).
    batch_size: int = 1

    #: AdamW learning rate from the LLMLight imitation fine-tuning script.
    learning_rate: float = 3e-4

    #: Max sequence length from the LLMLight imitation fine-tuning script.
    max_seq_length: int = 2048

    #: Số epoch IFT. Requirement 6 AC 1: tối thiểu 3, tối đa 30.
    ift_epochs: int = 30

    #: If True, labels copy input_ids exactly, matching LLMLight paper code.
    #: If False, prompt tokens are masked with -100 for response-only loss.
    train_on_inputs: bool = True

    #: Thư mục lưu checkpoint mỗi epoch (Requirement 6 AC 6).
    #: Trainer sẽ tạo subdirectories ``checkpoint-<step>`` ở đây.
    checkpoint_dir: str = "checkpoints/ift/"

    #: Thư mục lưu adapter cuối cùng (PRE LoRA-merge). LoRAMerger
    #: (Task 10.5) sẽ đọc từ đường dẫn này để merge vào base model
    #: và xuất ra ``models/qwen2_finetuned/``.
    output_dir: str = "models/qwen2_finetuned_ift/"

    #: Logging cadence (steps).
    logging_steps: int = 10

    #: Số checkpoint giữ lại trên disk (giảm I/O và disk usage).
    save_total_limit: int = 2

    # ------------------------------------------------------------ CGPR --
    # CGPR (Critic-Guided Policy Refinement) fields — Task 10.4.
    # Ranking loss training trên ``CGPR_pairs`` đã thu thập bởi Task 10.3.
    # Pipeline: load IFT checkpoint làm starting point, apply pairwise
    # margin loss ``max(0, margin - log P(pos) + log P(neg))``, save adapter
    # cuối cùng vào ``cgpr_output_dir`` để LoRAMerger (Task 10.5) đọc.

    #: Số epoch CGPR. Requirement 6 AC 2: tối thiểu 2, tối đa 5.
    cgpr_epochs: int = 3

    #: Learning rate cho CGPR (nhỏ hơn IFT vì là refinement step;
    #: design.md training.json schema mặc định 1e-5).
    cgpr_learning_rate: float = 1e-5

    #: Margin cho pairwise ranking loss
    #: ``max(0, margin - log P(pos) + log P(neg))``. Design.md training.json
    #: schema mặc định 0.5.
    cgpr_margin: float = 0.5

    #: Thư mục lưu checkpoint CGPR mỗi epoch boundary (Requirement 6 AC 6).
    #: Mỗi epoch tạo sub-directory ``epoch_{N}/`` ở đây.
    cgpr_checkpoint_dir: str = "checkpoints/cgpr/"

    #: Thư mục lưu adapter CGPR cuối cùng (PRE LoRA-merge). LoRAMerger
    #: (Task 10.5) sẽ đọc từ đường dẫn này để merge vào base model.
    cgpr_output_dir: str = "models/qwen2_finetuned_cgpr/"

    def __post_init__(self) -> None:
        # Validate epochs (Requirement 6 AC 1: 3-10).
        if not isinstance(self.ift_epochs, int) or isinstance(
            self.ift_epochs, bool
        ):
            raise TypeError(
                f"TrainingConfig.ift_epochs must be int; "
                f"got {type(self.ift_epochs).__name__}"
            )
        if not (
            IFTTrainer.MIN_EPOCHS
            <= self.ift_epochs
            <= IFTTrainer.MAX_EPOCHS
        ):
            raise ValueError(
                f"TrainingConfig.ift_epochs must be in "
                f"[{IFTTrainer.MIN_EPOCHS}, {IFTTrainer.MAX_EPOCHS}]; "
                f"got {self.ift_epochs}"
            )

        # Validate LoRA rank/alpha.
        if not isinstance(self.lora_rank, int) or isinstance(
            self.lora_rank, bool
        ):
            raise TypeError(
                f"TrainingConfig.lora_rank must be int; "
                f"got {type(self.lora_rank).__name__}"
            )
        if self.lora_rank <= 0:
            raise ValueError(
                f"TrainingConfig.lora_rank must be > 0; got {self.lora_rank}"
            )
        if not isinstance(self.lora_alpha, int) or isinstance(
            self.lora_alpha, bool
        ):
            raise TypeError(
                f"TrainingConfig.lora_alpha must be int; "
                f"got {type(self.lora_alpha).__name__}"
            )
        if self.lora_alpha <= 0:
            raise ValueError(
                f"TrainingConfig.lora_alpha must be > 0; got {self.lora_alpha}"
            )

        # Validate LoRA target_modules.
        if not isinstance(self.lora_target_modules, tuple):
            raise TypeError(
                "TrainingConfig.lora_target_modules must be tuple; "
                f"got {type(self.lora_target_modules).__name__}"
            )
        if not self.lora_target_modules:
            raise ValueError(
                "TrainingConfig.lora_target_modules must be non-empty"
            )
        for m in self.lora_target_modules:
            if not isinstance(m, str) or not m:
                raise ValueError(
                    "TrainingConfig.lora_target_modules entries must be "
                    f"non-empty str; got {m!r}"
                )

        # Validate gradient_accumulation_steps (Requirement 6 AC 3: >= 4).
        if (
            not isinstance(self.gradient_accumulation_steps, int)
            or isinstance(self.gradient_accumulation_steps, bool)
        ):
            raise TypeError(
                "TrainingConfig.gradient_accumulation_steps must be int; "
                f"got {type(self.gradient_accumulation_steps).__name__}"
            )
        if self.gradient_accumulation_steps < 4:
            raise ValueError(
                "TrainingConfig.gradient_accumulation_steps must be >= 4 "
                f"(Requirement 6 AC 3); got {self.gradient_accumulation_steps}"
            )

        # Validate batch_size.
        if not isinstance(self.batch_size, int) or isinstance(
            self.batch_size, bool
        ):
            raise TypeError(
                f"TrainingConfig.batch_size must be int; "
                f"got {type(self.batch_size).__name__}"
            )
        if self.batch_size < 1:
            raise ValueError(
                f"TrainingConfig.batch_size must be >= 1; got {self.batch_size}"
            )

        # Validate max_seq_length.
        if not isinstance(self.max_seq_length, int) or isinstance(
            self.max_seq_length, bool
        ):
            raise TypeError(
                "TrainingConfig.max_seq_length must be int; "
                f"got {type(self.max_seq_length).__name__}"
            )
        if self.max_seq_length < 1:
            raise ValueError(
                "TrainingConfig.max_seq_length must be >= 1; "
                f"got {self.max_seq_length}"
            )

        # Validate learning_rate.
        if not isinstance(self.learning_rate, (int, float)) or isinstance(
            self.learning_rate, bool
        ):
            raise TypeError(
                "TrainingConfig.learning_rate must be float; "
                f"got {type(self.learning_rate).__name__}"
            )
        if self.learning_rate <= 0:
            raise ValueError(
                "TrainingConfig.learning_rate must be > 0; "
                f"got {self.learning_rate}"
            )

        # Validate train_on_inputs.
        if not isinstance(self.train_on_inputs, bool):
            raise TypeError(
                "TrainingConfig.train_on_inputs must be bool; "
                f"got {type(self.train_on_inputs).__name__}"
            )

        # Validate paths.
        for fname in (
            "base_model",
            "checkpoint_dir",
            "output_dir",
            "cgpr_checkpoint_dir",
            "cgpr_output_dir",
        ):
            v = getattr(self, fname)
            if not isinstance(v, str) or not v:
                raise ValueError(
                    f"TrainingConfig.{fname} must be non-empty str; "
                    f"got {v!r}"
                )

        # Validate lora_dropout.
        if not isinstance(self.lora_dropout, (int, float)) or isinstance(
            self.lora_dropout, bool
        ):
            raise TypeError(
                "TrainingConfig.lora_dropout must be float; "
                f"got {type(self.lora_dropout).__name__}"
            )
        if not (0.0 <= float(self.lora_dropout) < 1.0):
            raise ValueError(
                "TrainingConfig.lora_dropout must be in [0, 1); "
                f"got {self.lora_dropout}"
            )

        # ----- CGPR-specific validation -----
        # Requirement 6 AC 2: cgpr_epochs ∈ [2, 5].
        if not isinstance(self.cgpr_epochs, int) or isinstance(
            self.cgpr_epochs, bool
        ):
            raise TypeError(
                "TrainingConfig.cgpr_epochs must be int; "
                f"got {type(self.cgpr_epochs).__name__}"
            )
        if not (2 <= self.cgpr_epochs <= 5):
            raise ValueError(
                "TrainingConfig.cgpr_epochs must be in [2, 5] "
                f"(Requirement 6 AC 2); got {self.cgpr_epochs}"
            )

        # cgpr_learning_rate > 0.
        if not isinstance(
            self.cgpr_learning_rate, (int, float)
        ) or isinstance(self.cgpr_learning_rate, bool):
            raise TypeError(
                "TrainingConfig.cgpr_learning_rate must be float; "
                f"got {type(self.cgpr_learning_rate).__name__}"
            )
        if self.cgpr_learning_rate <= 0:
            raise ValueError(
                "TrainingConfig.cgpr_learning_rate must be > 0; "
                f"got {self.cgpr_learning_rate}"
            )

        # cgpr_margin > 0.
        if not isinstance(self.cgpr_margin, (int, float)) or isinstance(
            self.cgpr_margin, bool
        ):
            raise TypeError(
                "TrainingConfig.cgpr_margin must be float; "
                f"got {type(self.cgpr_margin).__name__}"
            )
        if self.cgpr_margin <= 0:
            raise ValueError(
                "TrainingConfig.cgpr_margin must be > 0; "
                f"got {self.cgpr_margin}"
            )



# =========================================================================
# IFTTrainer
# =========================================================================


class IFTTrainer:
    """LoRA Instruction Fine-Tuning của Qwen2-0.5B trên trajectory data.

    Lifecycle (single ``train`` call):

    1. Validate inputs (empty data → ``ValueError``; resume path → check tồn tại).
    2. Load base model + tokenizer (with optional 4-bit quantization for
       memory; ``transformers`` chọn quantization tự động qua
       ``BitsAndBytesConfig`` nếu được cài).
    3. Verify VRAM sau load < :attr:`MAX_VRAM_MB` (raise ``OOMError``).
    4. Wrap model với LoRA adapter (:func:`peft.get_peft_model`).
    5. Build :class:`~datasets.Dataset` từ ``(prompt, response)`` pairs;
       tokenize với ``max_seq_length`` truncation; mask prompt tokens với
       ``-100`` trong ``labels``.
    6. Build :class:`transformers.TrainingArguments` với:

       * ``num_train_epochs = config.ift_epochs``
       * ``save_strategy = "epoch"`` (Requirement 6 AC 6 — checkpoint
         mỗi epoch boundary).
       * ``per_device_train_batch_size = config.batch_size``
       * ``gradient_accumulation_steps = config.gradient_accumulation_steps``
       * ``learning_rate = config.learning_rate``
       * ``output_dir = config.checkpoint_dir``

    7. ``Trainer.train(resume_from_checkpoint=resume_from)``.
    8. Verify VRAM sau training < :attr:`MAX_VRAM_MB`.
    9. ``Trainer.save_model(config.output_dir)`` để lưu adapter cuối cùng.
    10. Return đường dẫn checkpoint cuối — ``output_dir`` (CGPRDataCollector
        và CGPRTrainer sẽ đọc từ đây).

    Module-level ``transformers`` / ``peft`` / ``datasets`` / ``torch`` được
    import lazily; trong unit tests các symbol đó được monkey-patched để
    tránh download Qwen2-0.5B (~1GB) và không cần GPU.
    """

    #: VRAM budget — Requirement 6 AC 3 / Requirement 11.1.
    MAX_VRAM_MB: Final[float] = 7680.0

    #: Min epochs IFT (Requirement 6 AC 1).
    MIN_EPOCHS: Final[int] = 3

    #: Max epochs IFT (Requirement 6 AC 1).
    MAX_EPOCHS: Final[int] = 30

    #: Default device.
    _DEVICE: Final[str] = "cuda:0"

    #: Hằng số ``-100`` để mask token trong loss của ``CrossEntropyLoss``
    #: theo convention của HuggingFace.
    _LABEL_IGNORE_INDEX: Final[int] = -100

    def __init__(self, config: TrainingConfig) -> None:
        """Khởi tạo trainer với config đã validate.

        Args:
            config: :class:`TrainingConfig`. Sẽ được validate ở
                ``__post_init__`` (đã chạy do ``frozen=True``).

        Raises:
            TypeError: nếu ``config`` không phải :class:`TrainingConfig`.
        """
        if not isinstance(config, TrainingConfig):
            raise TypeError(
                "IFTTrainer.config must be TrainingConfig; "
                f"got {type(config).__name__}"
            )
        self.config: TrainingConfig = config

    # ------------------------------------------------------------ Public --

    def train(
        self,
        data: list[tuple[str, str]],
        resume_from: str | None = None,
    ) -> str:
        """Run LoRA IFT trên trajectory data.

        Args:
            data: List ``(prompt, response)`` pairs (output của
                :class:`TrajectoryCollector.collect`).
            resume_from: Đường dẫn checkpoint để resume (đã được
                ``Trainer`` lưu ở lần chạy trước). Nếu ``None``, train
                from scratch.

        Returns:
            Đường dẫn final checkpoint (``output_dir`` của config).
            CGPRDataCollector và CGPRTrainer sẽ load từ đường dẫn này.

        Raises:
            ValueError: ``data`` rỗng hoặc entries không phải
                ``(str, str)``.
            FileNotFoundError: ``resume_from`` được cung cấp nhưng
                không tồn tại trên disk.
            ImportError: ``transformers`` / ``peft`` / ``datasets`` /
                ``torch`` chưa cài.
            OOMError: VRAM vượt :attr:`MAX_VRAM_MB` sau model load
                hoặc sau training.
        """
        # ----- Validate data
        if not isinstance(data, list):
            raise ValueError(
                "IFTTrainer.train: data must be list of (prompt, response) "
                f"tuples; got {type(data).__name__}"
            )
        if not data:
            raise ValueError(
                "IFTTrainer.train: data must be non-empty; got 0 samples. "
                "Run TrajectoryCollector.collect first to gather >= "
                "min_valid_samples per phase."
            )
        for i, item in enumerate(data):
            if (
                not isinstance(item, tuple)
                or len(item) != 2
                or not isinstance(item[0], str)
                or not isinstance(item[1], str)
            ):
                raise ValueError(
                    f"IFTTrainer.train: data[{i}] must be (str, str) tuple; "
                    f"got {item!r}"
                )

        # ----- Validate resume_from
        if resume_from is not None:
            if not isinstance(resume_from, str) or not resume_from:
                raise ValueError(
                    "IFTTrainer.train: resume_from must be non-empty str "
                    f"or None; got {resume_from!r}"
                )
            if not Path(resume_from).exists():
                raise FileNotFoundError(
                    f"IFTTrainer.train: resume_from checkpoint does not "
                    f"exist: {resume_from!r}"
                )

        # ----- Lazy imports
        transformers = _import_transformers()
        peft = _import_peft()
        datasets_mod = _import_datasets()
        torch = _import_torch()
        if (
            transformers is None
            or peft is None
            or datasets_mod is None
            or torch is None
        ):
            raise ImportError(
                "IFTTrainer.train: requires transformers, peft, datasets, "
                "and torch. Install via venv/bin/pip install transformers "
                "peft datasets torch."
            )

        logger.info(
            "IFTTrainer.train: starting (samples=%d, epochs=%d, "
            "lora_rank=%d, lora_alpha=%d, target_modules=%s, "
            "gradient_accumulation=%d, batch_size=%d, resume=%s)",
            len(data),
            self.config.ift_epochs,
            self.config.lora_rank,
            self.config.lora_alpha,
            self.config.lora_target_modules,
            self.config.gradient_accumulation_steps,
            self.config.batch_size,
            resume_from,
        )

        # ----- Tokenizer + base model
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.config.base_model
        )
        if getattr(tokenizer, "pad_token", None) is None:
            # Qwen2 không có pad_token mặc định; dùng eos_token để pad.
            tokenizer.pad_token = getattr(tokenizer, "eos_token", None)

        # Load base model in fp32 — TrainingArguments(fp16=True) handles the
        # mixed-precision autocast wrapping at training time. Loading in fp16
        # here breaks LoRA training because PEFT wraps trainable adapters as
        # fp16 parameters which the GradScaler then refuses to unscale
        # ("Attempting to unscale FP16 gradients" — torch/amp/grad_scaler.py).
        base_model = transformers.AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
        )

        # ----- VRAM check sau model load (Requirement 6 AC 3)
        self._check_vram(stage="after_model_load")

        # ----- LoRA wrap
        lora_config = peft.LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            target_modules=list(self.config.lora_target_modules),
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = peft.get_peft_model(base_model, lora_config)
        logger.info(
            "IFTTrainer: LoRA wrapped (rank=%d, alpha=%d, target=%s)",
            self.config.lora_rank,
            self.config.lora_alpha,
            self.config.lora_target_modules,
        )

        # ----- Tokenize dataset
        dataset = self._build_dataset(data, tokenizer, datasets_mod)

        # ----- TrainingArguments
        training_args = transformers.TrainingArguments(
            output_dir=self.config.checkpoint_dir,
            num_train_epochs=self.config.ift_epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_steps=10,
            group_by_length=True,
            save_strategy="epoch",
            save_total_limit=self.config.save_total_limit,
            logging_steps=self.config.logging_steps,
            report_to="none",
            remove_unused_columns=False,
            fp16=True,
        )

        # ----- DataCollator (keeps explicit labels and pads dynamically)
        data_collator = transformers.DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            padding=True,
            return_tensors="pt",
        )

        # ----- Trainer
        trainer = transformers.Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

        # ----- Train (with optional resume)
        try:
            if resume_from is not None:
                logger.info(
                    "IFTTrainer: resuming from checkpoint %s", resume_from
                )
                trainer.train(resume_from_checkpoint=resume_from)
            else:
                trainer.train()
        except Exception as exc:
            # Try to detect OOM patterns to give a clearer error message.
            msg = str(exc).lower()
            if "out of memory" in msg or "oom" in msg:
                raise OOMError(
                    f"IFTTrainer.train: CUDA OOM during training: {exc}"
                ) from exc
            raise

        # ----- VRAM check sau training (Requirement 6 AC 3)
        self._check_vram(stage="after_training")

        # ----- Save final adapter
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        trainer.save_model(self.config.output_dir)
        logger.info(
            "IFTTrainer.train: saved final adapter to %s",
            self.config.output_dir,
        )

        return self.config.output_dir

    # ------------------------------------------------------------ Helpers --

    def _check_vram(self, *, stage: str) -> None:
        """Verify VRAM usage < :attr:`MAX_VRAM_MB`; raise ``OOMError`` else.

        ``stage`` đi vào log message để debug dễ hơn (ví dụ
        ``"after_model_load"`` vs ``"after_training"``).

        Skip check khi CUDA không khả dụng — :func:`vram_monitor.get_vram_usage_mb`
        trả về ``None``, tránh fail trên CPU-only test envs.
        """
        usage = vram_monitor.get_vram_usage_mb(self._DEVICE)
        if usage is None:
            logger.debug(
                "IFTTrainer._check_vram[%s]: CUDA unavailable; skipping",
                stage,
            )
            return
        if usage > self.MAX_VRAM_MB:
            raise OOMError(
                f"IFTTrainer[{stage}]: VRAM exceeded budget "
                f"({usage:.2f}MB > {self.MAX_VRAM_MB:.2f}MB). Try lowering "
                "lora_rank, increasing gradient_accumulation_steps, or "
                "reducing max_seq_length."
            )
        logger.info(
            "IFTTrainer._check_vram[%s]: VRAM OK (%.2fMB / %.2fMB budget)",
            stage,
            usage,
            self.MAX_VRAM_MB,
        )

    def _build_dataset(
        self,
        data: list[tuple[str, str]],
        tokenizer: Any,
        datasets_mod: Any,
    ) -> Any:
        """Build :class:`datasets.Dataset` từ ``(prompt, response)`` pairs.

        Mỗi sample được tokenize thành chuỗi liên tục
        ``prompt + response + eos`` với ``input_ids`` và ``labels``.
        Mặc định ``labels`` copy toàn bộ ``input_ids`` theo LLMLight paper
        (``train_on_inputs=True``). Có thể bật legacy response-only loss bằng
        ``train_on_inputs=False``.

        Args:
            data: List of ``(prompt, response)``.
            tokenizer: HuggingFace tokenizer.
            datasets_mod: ``datasets`` module (lazy imported).

        Returns:
            ``datasets.Dataset`` với cột ``input_ids``, ``attention_mask``,
            ``labels``.
        """
        max_len = self.config.max_seq_length
        ignore = self._LABEL_IGNORE_INDEX
        eos_id = getattr(tokenizer, "eos_token_id", None)
        train_on_inputs = self.config.train_on_inputs

        records: list[dict[str, list[int]]] = []
        for prompt, response in data:
            response_ids = list(
                tokenizer(response, add_special_tokens=False)["input_ids"]
            )
            if eos_id is not None:
                response_ids.append(eos_id)
            response_ids = response_ids[:max_len]

            prompt_budget = max(0, max_len - len(response_ids))
            prompt_ids_full = list(
                tokenizer(prompt, add_special_tokens=False)["input_ids"]
            )
            if len(prompt_ids_full) <= prompt_budget:
                prompt_ids = prompt_ids_full
            elif prompt_budget <= 0:
                prompt_ids = []
            else:
                # Project prompts are longer than the paper prompts. Keep
                # both the beginning and the tail so observation context,
                # current phase, output instructions, and the final response
                # all survive inside cutoff_len.
                head_keep = max(1, int(prompt_budget * 0.3))
                tail_keep = prompt_budget - head_keep
                prompt_ids = (
                    prompt_ids_full[:head_keep]
                    + prompt_ids_full[-tail_keep:]
                    if tail_keep > 0
                    else prompt_ids_full[:head_keep]
                )

            input_ids = (prompt_ids + response_ids)[:max_len]
            attention_mask = [1] * len(input_ids)

            if train_on_inputs:
                labels = list(input_ids)
            else:
                n_prompt = min(len(prompt_ids), len(input_ids))
                labels = [ignore] * n_prompt + list(input_ids[n_prompt:])

            records.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }
            )

        # ``Dataset.from_list`` accepts ragged rows; the HuggingFace collator
        # pads dynamically at batch time.
        return datasets_mod.Dataset.from_list(records)
