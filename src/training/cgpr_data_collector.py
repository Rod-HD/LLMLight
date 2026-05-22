"""CGPR Data Collector (Component 7 — Task 10.3).

Step 2 trong workflow CGPR theo README LLMTSCS / Component 7 trong design:

    1. IFT đã chạy xong và checkpoint đã lưu (Task 10.2).
    2. **(THIS MODULE)** Cho cả model_after_IFT lẫn Advanced-CoLight critic
       chạy trên cùng tập prompt. Với mỗi prompt nơi 2 output KHÁC nhau,
       ghép thành bộ ba (prompt, positive=critic_output,
       negative=ift_model_output).
    3. CGPR Trainer (Task 10.4) tiêu thụ output này làm input ranking pairs.

Hành vi key:

* :meth:`CGPRDataCollector.__init__` xác minh ``ift_model_path`` tồn tại
  trên disk (file hoặc directory). Raise :class:`FileNotFoundError` nếu
  không — runner KHÔNG được cho phép chạy CGPR Pair Collection trước khi
  IFT checkpoint đã được lưu hoàn tất (Requirement 6.10).

* :meth:`CGPRDataCollector.collect` skip cặp khi:

  * Critic raise exception → log warning, không fail toàn bộ collection.
  * Critic trả về phase không thuộc ``VALID_PHASES`` → log warning, skip.
  * Critic phase trùng với phase parsed từ IFT response → skip vì không
    có signal phân biệt cho ranking loss.

* Output:

  * ``positive_response`` luôn được normalize về dạng
    ``<signal>{critic_phase}</signal>`` để CGPR Trainer có format nhất
    quán làm "preferred answer".
  * ``negative_response`` là raw text từ IFT model — KHÔNG strip /
    reformat. CGPR Trainer cần text gốc để tạo ranking signal đúng.

* Empty ``prompts`` → return ``[]`` (KHÔNG raise, vì chỉ là edge case
  hợp lệ — runner có thể quyết định handle hoặc bỏ qua).

Default IFT model loader (qua :func:`_default_ift_model_factory`) lazily
import ``transformers`` để module có thể được test trên CPU-only env mà
không cần download Qwen2-0.5B (~1GB). Tests inject một
``ift_model_adapter_factory`` trả về mock adapter trực tiếp, bypass
heavy load.

_Requirements_: 6.2, 6.9
"""

from __future__ import annotations

import logging
import os
from typing import Callable, Final, NamedTuple, Protocol, runtime_checkable

from src.response_parser import ResponseParser

logger = logging.getLogger(__name__)

__all__ = [
    "CriticAdapter",
    "IFTModelAdapter",
    "CGPRPair",
    "CGPRDataCollector",
]


# =========================================================================
# Protocols (structural typing — mock adapters are runtime-compatible)
# =========================================================================


@runtime_checkable
class CriticAdapter(Protocol):
    """Protocol cho Advanced-CoLight critic adapter.

    Trong production (Task 13.4 main runner), một wrapper mỏng adapt
    LLMTSCS Advanced-CoLight Q-network → :meth:`predict_phase`. Adapter
    nhận prompt text (không phải state vector) và trả về 1 trong
    ``{"ETWT", "NTST", "ELWL", "NLSL"}`` dựa trên state ẩn trong prompt.

    Implementer được phép raise bất kỳ exception nào — collector sẽ
    catch và skip prompt đó (không fail toàn bộ collection).
    """

    def predict_phase(self, prompt: str) -> str:  # pragma: no cover - protocol
        ...


@runtime_checkable
class IFTModelAdapter(Protocol):
    """Protocol cho IFT-finetuned model adapter.

    Trong production, default factory load Qwen2-0.5B + LoRA adapter từ
    ``ift_model_path`` qua ``transformers``. Tests inject mock adapter
    bypass heavy load.

    :meth:`generate` trả về raw response text (CHƯA parse) để collector
    capture nguyên trạng làm ``negative_response`` cho ranking pair.
    """

    def generate(self, prompt: str) -> str:  # pragma: no cover - protocol
        ...


# =========================================================================
# Data type
# =========================================================================


class CGPRPair(NamedTuple):
    """Bộ ba ranking pair cho CGPR Trainer.

    Attributes:
        prompt: Prompt text input cho cả critic và IFT model.
        positive_response: Output preferred (từ critic), đã được
            normalize về dạng ``<signal>{phase}</signal>`` để CGPR
            Trainer có format nhất quán.
        negative_response: Output dispreferred (raw text từ IFT model).
            KHÔNG được strip / reformat — giữ nguyên text gốc.
    """

    prompt: str
    positive_response: str
    negative_response: str


# =========================================================================
# Default IFT model factory (lazy)
# =========================================================================


def _default_ift_model_factory(ift_model_path: str) -> IFTModelAdapter:
    """Default factory load IFT model qua ``transformers``.

    Lazy import để module có thể được test trên CPU-only env.

    Args:
        ift_model_path: Đường dẫn tới IFT checkpoint (file hoặc directory)
            do :class:`~src.training.ift_trainer.IFTTrainer.train` xuất ra
            (mặc định ``models/qwen2_finetuned_ift/``).

    Returns:
        Adapter implements :class:`IFTModelAdapter` protocol.

    Raises:
        ImportError: Nếu ``transformers`` chưa cài.
    """
    try:
        from transformers import (  # type: ignore[import-not-found]
            AutoModelForCausalLM,
            AutoTokenizer,
        )
    except ImportError as exc:
        raise ImportError(
            "_default_ift_model_factory: requires transformers. Install via "
            "venv/bin/pip install transformers, or pass a custom "
            "ift_model_adapter_factory to CGPRDataCollector."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(ift_model_path)
    model = AutoModelForCausalLM.from_pretrained(ift_model_path)

    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = getattr(tokenizer, "eos_token", None)

    class _HFModelAdapter:
        """Adapter wrap HuggingFace model + tokenizer thành IFTModelAdapter."""

        def __init__(self, _model, _tokenizer):
            self._model = _model
            self._tokenizer = _tokenizer

        def generate(self, prompt: str) -> str:
            inputs = self._tokenizer(prompt, return_tensors="pt")
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=CGPRDataCollector.MAX_NEW_TOKENS,
                do_sample=False,
            )
            # Strip prompt portion from generated ids.
            input_len = inputs["input_ids"].shape[1]
            new_tokens = output_ids[0][input_len:]
            return self._tokenizer.decode(new_tokens, skip_special_tokens=True)

    return _HFModelAdapter(model, tokenizer)


# =========================================================================
# CGPRDataCollector
# =========================================================================


class CGPRDataCollector:
    """Collect CGPR ranking pairs by comparing IFT model vs Advanced-CoLight critic.

    Lifecycle:

    1. ``__init__`` validate ``ift_model_path`` tồn tại trên disk;
       lazy-load IFT model qua factory (default uses ``transformers``,
       tests inject mock).
    2. ``collect(prompts)`` chạy cả critic + IFT model trên mỗi prompt,
       skip cặp trùng phase hoặc invalid critic output, ghép divergent
       cặp thành :class:`CGPRPair`.
    3. Output được CGPR Trainer (Task 10.4) tiêu thụ.

    Class attribute:
        VALID_PHASES: Tập 4 pha hợp lệ ``{ETWT, NTST, ELWL, NLSL}`` —
            thống nhất với :class:`ResponseParser.VALID_PHASES`.

        MAX_NEW_TOKENS: Cap số token IFT model generate per prompt.
            Cùng giá trị với ``LightGPTInference.MAX_NEW_TOKENS``.

        _LOG_EVERY: Tần suất log progress (mỗi N prompts xử lý).
    """

    VALID_PHASES: Final[frozenset[str]] = ResponseParser.VALID_PHASES
    MAX_NEW_TOKENS: Final[int] = 256
    _LOG_EVERY: Final[int] = 50

    def __init__(
        self,
        ift_model_path: str,
        colight_critic: CriticAdapter,
        *,
        ift_model_adapter_factory: Callable[[str], IFTModelAdapter] | None = None,
    ) -> None:
        """Khởi tạo collector và load IFT model.

        Args:
            ift_model_path: Đường dẫn tới IFT checkpoint (file hoặc
                directory) do :class:`IFTTrainer.train` xuất ra. Phải
                tồn tại trên disk; raise :class:`FileNotFoundError`
                nếu không (Requirement 6.10 — CGPR phụ thuộc IFT
                checkpoint, không được chạy trước khi IFT lưu xong).
            colight_critic: Critic adapter implements
                :class:`CriticAdapter` protocol. Trong production là
                wrapper Advanced-CoLight Q-network từ Task 11.3; trong
                tests là mock object.
            ift_model_adapter_factory: Optional callable
                ``(path: str) -> IFTModelAdapter``. Default
                :func:`_default_ift_model_factory` load qua
                ``transformers``. Inject custom factory cho unit tests
                để bypass heavy load.

        Raises:
            FileNotFoundError: Nếu ``ift_model_path`` không tồn tại.
            ValueError: Nếu ``ift_model_path`` không phải non-empty str.
            TypeError: Nếu ``colight_critic`` không expose
                ``predict_phase`` callable.
        """
        if not isinstance(ift_model_path, str) or not ift_model_path:
            raise ValueError(
                "CGPRDataCollector: ift_model_path must be non-empty str; "
                f"got {type(ift_model_path).__name__} ({ift_model_path!r})"
            )

        if not os.path.exists(ift_model_path):
            raise FileNotFoundError(
                f"CGPRDataCollector: ift_model_path does not exist: "
                f"{ift_model_path!r}. Run IFTTrainer.train (Task 10.2) "
                "first and verify the checkpoint was saved successfully "
                "(Requirement 6.10 — CGPR phụ thuộc IFT checkpoint)."
            )

        if not callable(getattr(colight_critic, "predict_phase", None)):
            raise TypeError(
                "CGPRDataCollector: colight_critic must implement "
                "predict_phase(prompt: str) -> str; got "
                f"{type(colight_critic).__name__} without callable "
                "predict_phase attribute."
            )

        self._ift_model_path: str = ift_model_path
        self._critic: CriticAdapter = colight_critic

        factory = (
            ift_model_adapter_factory
            if ift_model_adapter_factory is not None
            else _default_ift_model_factory
        )
        self._ift_model: IFTModelAdapter = factory(ift_model_path)

        if not callable(getattr(self._ift_model, "generate", None)):
            raise TypeError(
                "CGPRDataCollector: ift_model_adapter_factory returned an "
                "object without callable generate(prompt: str) -> str; got "
                f"{type(self._ift_model).__name__}."
            )

        logger.info(
            "CGPRDataCollector: initialized with ift_model_path=%r, "
            "critic=%s, ift_model=%s",
            ift_model_path,
            type(colight_critic).__name__,
            type(self._ift_model).__name__,
        )

    # ---------------------------------------------------------- Properties --

    @property
    def ift_model_path(self) -> str:
        """Đường dẫn tới IFT checkpoint đang được sử dụng."""
        return self._ift_model_path

    # ------------------------------------------------------------- collect --

    def collect(self, prompts: list[str]) -> list[CGPRPair]:
        """Collect divergent ranking pairs từ list prompts.

        Algorithm (mỗi prompt):

        1. Validate prompt là str (skip + warning nếu không).
        2. Gọi ``critic.predict_phase(prompt)``:

           * Raise → log warning, skip prompt.
           * Trả về phase ∉ ``VALID_PHASES`` → log warning, skip.

        3. Gọi ``ift_model.generate(prompt)`` để lấy raw IFT response.
        4. Parse phase từ raw response qua :meth:`ResponseParser.parse`
           (fallback ETWT khi invalid — đây là behavior mong muốn vì
           "IFT model không có signal hợp lệ" cũng là tín hiệu để CGPR
           học cách phân biệt).
        5. Nếu ``critic_phase == ift_phase`` → skip (no learning signal
           cho ranking loss).
        6. Else → emit :class:`CGPRPair`.

        Args:
            prompts: List prompt text. Empty list → return ``[]``.

        Returns:
            ``list[CGPRPair]`` chỉ chứa các cặp divergent. Có thể nhỏ
            hơn ``len(prompts)`` (thực tế nhỏ hơn nhiều vì IFT model
            đã học khá tốt sau giai đoạn 1).

        Raises:
            ValueError: Nếu ``prompts`` không phải list.
        """
        if not isinstance(prompts, list):
            raise ValueError(
                "CGPRDataCollector.collect: prompts must be list[str]; "
                f"got {type(prompts).__name__}"
            )

        if not prompts:
            logger.info("CGPRDataCollector.collect: empty prompts list, returning []")
            return []

        # Lazy resp_parser instance — used only for phase extraction from
        # IFT raw response. ResponseParser is stateless apart from its
        # logger; cheap to instantiate once per collect() call.
        resp_parser = ResponseParser()

        pairs: list[CGPRPair] = []
        skipped_critic_error = 0
        skipped_invalid_phase = 0
        skipped_same_phase = 0
        skipped_invalid_prompt = 0

        for idx, prompt in enumerate(prompts):
            if not isinstance(prompt, str):
                logger.warning(
                    "CGPRDataCollector.collect: prompts[%d] is not str "
                    "(got %s); skipping",
                    idx,
                    type(prompt).__name__,
                )
                skipped_invalid_prompt += 1
                continue

            # 1. Critic phase
            try:
                critic_phase_raw = self._critic.predict_phase(prompt)
            except Exception as exc:  # noqa: BLE001 - catch all by design
                logger.warning(
                    "CGPRDataCollector.collect: critic.predict_phase "
                    "raised %s on prompts[%d]: %s; skipping",
                    type(exc).__name__,
                    idx,
                    exc,
                )
                skipped_critic_error += 1
                continue

            if not isinstance(critic_phase_raw, str):
                logger.warning(
                    "CGPRDataCollector.collect: critic returned non-str "
                    "phase (got %s) on prompts[%d]; skipping",
                    type(critic_phase_raw).__name__,
                    idx,
                )
                skipped_invalid_phase += 1
                continue

            critic_phase = critic_phase_raw.strip().upper()
            if critic_phase not in self.VALID_PHASES:
                logger.warning(
                    "CGPRDataCollector.collect: critic returned invalid "
                    "phase %r (∉ VALID_PHASES) on prompts[%d]; skipping",
                    critic_phase_raw,
                    idx,
                )
                skipped_invalid_phase += 1
                continue

            # 2. IFT response
            try:
                ift_response = self._ift_model.generate(prompt)
            except Exception as exc:  # noqa: BLE001 - defensive
                logger.warning(
                    "CGPRDataCollector.collect: ift_model.generate raised "
                    "%s on prompts[%d]: %s; skipping",
                    type(exc).__name__,
                    idx,
                    exc,
                )
                continue

            if not isinstance(ift_response, str):
                logger.warning(
                    "CGPRDataCollector.collect: ift_model.generate returned "
                    "non-str (got %s) on prompts[%d]; skipping",
                    type(ift_response).__name__,
                    idx,
                )
                continue

            # 3. Parse IFT phase. ResponseParser falls back to ETWT on
            #    invalid output — that behavior is desired here:
            #    "IFT model can't produce valid signal" still gives us a
            #    learning signal (when critic returns a different phase).
            ift_phase = resp_parser.parse(ift_response)

            # 4. Skip same-phase (no ranking signal).
            if critic_phase == ift_phase:
                skipped_same_phase += 1
                continue

            # 5. Emit divergent pair.
            positive_response = f"<signal>{critic_phase}</signal>"
            pairs.append(
                CGPRPair(
                    prompt=prompt,
                    positive_response=positive_response,
                    negative_response=ift_response,
                )
            )

            if (idx + 1) % self._LOG_EVERY == 0:
                logger.info(
                    "CGPRDataCollector.collect: processed %d/%d prompts, "
                    "%d divergent pairs collected",
                    idx + 1,
                    len(prompts),
                    len(pairs),
                )

        logger.info(
            "CGPRDataCollector.collect: completed. prompts=%d, pairs=%d, "
            "skipped_critic_error=%d, skipped_invalid_phase=%d, "
            "skipped_same_phase=%d, skipped_invalid_prompt=%d",
            len(prompts),
            len(pairs),
            skipped_critic_error,
            skipped_invalid_phase,
            skipped_same_phase,
            skipped_invalid_prompt,
        )
        return pairs
