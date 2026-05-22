"""Unit tests for ``src.training.cgpr_data_collector`` (Task 10.3).

Validates Requirement 6 AC 2, 9, 10 — CGPR Data Collector ghép divergent
pairs ``(prompt, critic_output, ift_model_output)`` cho CGPR Trainer.

Strategy:

* Mock :class:`CriticAdapter` và :class:`IFTModelAdapter` bằng các class
  đơn giản expose ``predict_phase(prompt)`` / ``generate(prompt)`` —
  không cần load Qwen2-0.5B (~1GB) hay download HuggingFace model.
* :class:`CGPRDataCollector.__init__` xác minh đường dẫn IFT checkpoint
  tồn tại trên disk; tests dùng ``tmp_path`` của pytest để tạo path
  hợp lệ và đường dẫn không tồn tại để verify error.
* Inject custom ``ift_model_adapter_factory`` để bypass default
  ``transformers`` loader — tránh side effect download.

Tests coverage:

1. ``__init__`` raise :class:`FileNotFoundError` khi ``ift_model_path``
   không tồn tại (Requirement 6.10).
2. ``__init__`` reject empty / non-str ``ift_model_path``.
3. ``__init__`` reject critic không có ``predict_phase`` callable.
4. ``__init__`` accepts valid path + mock factory; load IFT model qua
   factory injected.
5. ``__init__`` reject factory trả về object không có ``generate``.
6. ``collect([])`` returns ``[]`` — không gọi critic / IFT model.
7. ``collect`` reject ``prompts`` không phải list.
8. ``collect`` chỉ trả về pairs khi critic và IFT khác phase
   (Requirement 6.9).
9. ``collect`` skip pair khi critic phase trùng IFT phase (cùng
   normalize qua ResponseParser fallback).
10. ``positive_response`` chứa ``<signal>{critic_phase}</signal>``;
    ``negative_response`` là raw IFT output (không strip / reformat).
11. ``collect`` skip prompt khi ``critic.predict_phase`` raise.
12. ``collect`` skip prompt khi critic trả về phase ∉ VALID_PHASES.
13. ``collect`` skip prompt khi critic trả về non-str.
14. ``collect`` skip prompts[idx] không phải str.
15. ``collect`` 100 prompts với 30 divergent → trả về đúng 30 pairs.
16. ``collect`` IFT raw response invalid (no <signal>) → ResponseParser
    fallback ETWT; nếu critic_phase != ETWT → vẫn emit pair với raw
    response làm negative.
17. Critic output được normalize (lowercase / whitespace) trước so sánh.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Callable

import pytest

# Make ``src`` importable when running pytest from project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.cgpr_data_collector import (  # noqa: E402
    CGPRDataCollector,
    CGPRPair,
    CriticAdapter,
    IFTModelAdapter,
)


# =========================================================================
# Mock adapters
# =========================================================================


class _StubCritic:
    """Configurable critic stub.

    Args:
        responses: List of items where each is either:
            * str → returned by predict_phase
            * Exception subclass / instance → raised by predict_phase
            * Anything else (e.g. None, int) → returned as-is to test
              non-str handling.
        cycle_default: If ``True`` and prompts > responses, repeat the
            last entry; if ``False``, raise ``IndexError`` (default).
    """

    def __init__(self, responses: list, cycle_default: bool = False):
        self._responses = list(responses)
        self._idx = 0
        self.calls: list[str] = []
        self._cycle_default = cycle_default

    def predict_phase(self, prompt: str) -> str:
        self.calls.append(prompt)
        if self._idx >= len(self._responses):
            if self._cycle_default and self._responses:
                value = self._responses[-1]
            else:
                raise IndexError(
                    f"_StubCritic: no more responses configured "
                    f"(idx={self._idx})"
                )
        else:
            value = self._responses[self._idx]
            self._idx += 1

        if isinstance(value, BaseException):
            raise value
        return value


class _StubIFTModel:
    """Configurable IFT model stub.

    Args:
        responses: Per-prompt raw responses (str). Repeats last entry
            if prompts > responses.
    """

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self._idx = 0
        self.calls: list[str] = []

    def generate(self, prompt: str) -> str:
        self.calls.append(prompt)
        if self._idx >= len(self._responses):
            value = self._responses[-1] if self._responses else ""
        else:
            value = self._responses[self._idx]
            self._idx += 1
        if isinstance(value, BaseException):
            raise value
        return value  # type: ignore[return-value]


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def ift_path(tmp_path: Path) -> str:
    """Create a valid IFT checkpoint directory under tmp_path."""
    p = tmp_path / "ift_ckpt"
    p.mkdir()
    # Create a marker file so directory looks like a real checkpoint.
    (p / "adapter_config.json").write_text("{}", encoding="utf-8")
    return str(p)


@pytest.fixture
def make_factory() -> Callable[[_StubIFTModel], Callable[[str], IFTModelAdapter]]:
    """Helper: turn a stub IFT model into a factory accepted by collector."""

    def _factory(stub: _StubIFTModel) -> Callable[[str], IFTModelAdapter]:
        def _build(path: str) -> IFTModelAdapter:
            return stub  # type: ignore[return-value]

        return _build

    return _factory


# =========================================================================
# Tests: __init__
# =========================================================================


class TestInit:
    """``__init__`` validation and dependency wiring."""

    def test_raises_on_missing_path(self, tmp_path: Path):
        bogus = str(tmp_path / "does_not_exist")
        critic = _StubCritic([])
        with pytest.raises(FileNotFoundError, match="does not exist"):
            CGPRDataCollector(
                ift_model_path=bogus,
                colight_critic=critic,
                ift_model_adapter_factory=lambda p: _StubIFTModel([]),
            )

    @pytest.mark.parametrize("bad_path", ["", None, 123, [], object()])
    def test_rejects_invalid_path_type(self, bad_path):
        critic = _StubCritic([])
        with pytest.raises(ValueError, match="non-empty str"):
            CGPRDataCollector(
                ift_model_path=bad_path,  # type: ignore[arg-type]
                colight_critic=critic,
                ift_model_adapter_factory=lambda p: _StubIFTModel([]),
            )

    def test_rejects_critic_without_predict_phase(self, ift_path: str):
        class _Bad:
            pass

        with pytest.raises(TypeError, match="predict_phase"):
            CGPRDataCollector(
                ift_model_path=ift_path,
                colight_critic=_Bad(),  # type: ignore[arg-type]
                ift_model_adapter_factory=lambda p: _StubIFTModel([]),
            )

    def test_rejects_critic_with_non_callable_predict_phase(
        self, ift_path: str
    ):
        class _Bad:
            predict_phase = "not callable"

        with pytest.raises(TypeError, match="predict_phase"):
            CGPRDataCollector(
                ift_model_path=ift_path,
                colight_critic=_Bad(),  # type: ignore[arg-type]
                ift_model_adapter_factory=lambda p: _StubIFTModel([]),
            )

    def test_rejects_factory_returning_no_generate(self, ift_path: str):
        class _BadAdapter:
            pass

        critic = _StubCritic([])
        with pytest.raises(TypeError, match="generate"):
            CGPRDataCollector(
                ift_model_path=ift_path,
                colight_critic=critic,
                ift_model_adapter_factory=lambda p: _BadAdapter(),  # type: ignore[return-value]
            )

    def test_accepts_valid_inputs_and_calls_factory(
        self, ift_path: str, make_factory
    ):
        ift_stub = _StubIFTModel([])
        captured: list[str] = []

        def factory(p: str) -> IFTModelAdapter:
            captured.append(p)
            return ift_stub  # type: ignore[return-value]

        collector = CGPRDataCollector(
            ift_model_path=ift_path,
            colight_critic=_StubCritic([]),
            ift_model_adapter_factory=factory,
        )
        assert collector.ift_model_path == ift_path
        assert captured == [ift_path]

    def test_accepts_file_path_not_just_directory(
        self, tmp_path: Path
    ):
        """Path can be a file (some checkpoint formats) or directory."""
        f = tmp_path / "model.bin"
        f.write_bytes(b"fake")
        critic = _StubCritic([])
        # Should not raise.
        CGPRDataCollector(
            ift_model_path=str(f),
            colight_critic=critic,
            ift_model_adapter_factory=lambda p: _StubIFTModel([]),
        )


# =========================================================================
# Tests: collect — input validation
# =========================================================================


class TestCollectInputValidation:
    """``collect`` input validation."""

    def test_empty_prompts_returns_empty_list(self, ift_path: str):
        critic = _StubCritic([])
        ift = _StubIFTModel([])
        collector = CGPRDataCollector(
            ift_model_path=ift_path,
            colight_critic=critic,
            ift_model_adapter_factory=lambda p: ift,
        )
        result = collector.collect([])
        assert result == []
        # Critical: must NOT call critic / IFT on empty input.
        assert critic.calls == []
        assert ift.calls == []

    @pytest.mark.parametrize(
        "bad",
        [None, "not a list", 42, {"a": 1}, ("tuple",)],
    )
    def test_rejects_non_list_prompts(self, ift_path: str, bad):
        critic = _StubCritic([])
        ift = _StubIFTModel([])
        collector = CGPRDataCollector(
            ift_model_path=ift_path,
            colight_critic=critic,
            ift_model_adapter_factory=lambda p: ift,
        )
        with pytest.raises(ValueError, match="prompts must be list"):
            collector.collect(bad)  # type: ignore[arg-type]

    def test_skips_non_str_prompt_entries(
        self, ift_path: str, caplog: pytest.LogCaptureFixture
    ):
        """prompts[idx] không phải str → skip + warning."""
        critic = _StubCritic(["NTST"])  # only used for the str entry
        ift = _StubIFTModel(["<signal>ETWT</signal>"])
        collector = CGPRDataCollector(
            ift_model_path=ift_path,
            colight_critic=critic,
            ift_model_adapter_factory=lambda p: ift,
        )
        with caplog.at_level(logging.WARNING):
            result = collector.collect(["valid prompt", 42, None])  # type: ignore[list-item]
        # Only the str prompt produces a pair (critic NTST vs IFT ETWT
        # → divergent).
        assert len(result) == 1
        assert result[0].prompt == "valid prompt"
        # Verify warning fired for the non-str entries.
        assert any("not str" in rec.message for rec in caplog.records)


# =========================================================================
# Tests: collect — divergence logic (Requirement 6.9)
# =========================================================================


class TestCollectDivergence:
    """Pair collection emits ONLY divergent (critic_phase != ift_phase)."""

    def test_divergent_pair_emitted(self, ift_path: str):
        critic = _StubCritic(["NTST"])
        ift = _StubIFTModel(["<signal>ETWT</signal>"])
        collector = CGPRDataCollector(
            ift_model_path=ift_path,
            colight_critic=critic,
            ift_model_adapter_factory=lambda p: ift,
        )
        result = collector.collect(["traffic state A"])
        assert len(result) == 1
        pair = result[0]
        assert isinstance(pair, CGPRPair)
        assert pair.prompt == "traffic state A"
        assert pair.positive_response == "<signal>NTST</signal>"
        assert pair.negative_response == "<signal>ETWT</signal>"

    def test_same_phase_skipped(self, ift_path: str):
        critic = _StubCritic(["ETWT"])
        ift = _StubIFTModel(["<signal>ETWT</signal>"])
        collector = CGPRDataCollector(
            ift_model_path=ift_path,
            colight_critic=critic,
            ift_model_adapter_factory=lambda p: ift,
        )
        result = collector.collect(["traffic state A"])
        # No divergence → no pair.
        assert result == []
        # Both critic and IFT model called exactly once.
        assert len(critic.calls) == 1
        assert len(ift.calls) == 1

    def test_positive_response_format_is_signal_tagged(self, ift_path: str):
        critic = _StubCritic(["ELWL"])
        ift = _StubIFTModel(["raw IFT model output without signal tag"])
        collector = CGPRDataCollector(
            ift_model_path=ift_path,
            colight_critic=critic,
            ift_model_adapter_factory=lambda p: ift,
        )
        result = collector.collect(["prompt"])
        # IFT raw has no signal → ResponseParser falls back to ETWT.
        # critic=ELWL != ift_phase=ETWT → pair emitted.
        assert len(result) == 1
        assert result[0].positive_response == "<signal>ELWL</signal>"
        # negative_response is RAW (no signal tag, not normalized).
        assert (
            result[0].negative_response
            == "raw IFT model output without signal tag"
        )

    def test_negative_response_preserves_raw_text(self, ift_path: str):
        """negative MUST be raw (no strip / reformat)."""
        raw = "  <signal>NTST</signal>\n  Some explanation here.  "
        critic = _StubCritic(["ETWT"])
        ift = _StubIFTModel([raw])
        collector = CGPRDataCollector(
            ift_model_path=ift_path,
            colight_critic=critic,
            ift_model_adapter_factory=lambda p: ift,
        )
        result = collector.collect(["prompt"])
        assert len(result) == 1
        # Raw preservation: byte-for-byte equality.
        assert result[0].negative_response == raw

    def test_critic_phase_normalized_lowercase_and_whitespace(
        self, ift_path: str
    ):
        """Critic returning ``"  ntst\\n"`` should be treated as NTST."""
        critic = _StubCritic(["  ntst\n"])
        ift = _StubIFTModel(["<signal>ETWT</signal>"])
        collector = CGPRDataCollector(
            ift_model_path=ift_path,
            colight_critic=critic,
            ift_model_adapter_factory=lambda p: ift,
        )
        result = collector.collect(["prompt"])
        assert len(result) == 1
        assert result[0].positive_response == "<signal>NTST</signal>"

    @pytest.mark.parametrize(
        "critic_phase,ift_phase",
        [
            ("ETWT", "NTST"),
            ("NTST", "ELWL"),
            ("ELWL", "NLSL"),
            ("NLSL", "ETWT"),
        ],
    )
    def test_all_phase_combinations_divergent(
        self, ift_path: str, critic_phase: str, ift_phase: str
    ):
        critic = _StubCritic([critic_phase])
        ift = _StubIFTModel([f"<signal>{ift_phase}</signal>"])
        collector = CGPRDataCollector(
            ift_model_path=ift_path,
            colight_critic=critic,
            ift_model_adapter_factory=lambda p: ift,
        )
        result = collector.collect(["prompt"])
        assert len(result) == 1
        assert result[0].positive_response == f"<signal>{critic_phase}</signal>"
        assert result[0].negative_response == f"<signal>{ift_phase}</signal>"


# =========================================================================
# Tests: collect — error handling (Requirement 6.9 robustness)
# =========================================================================


class TestCollectErrorHandling:
    """Critic / IFT failures are isolated to the offending prompt."""

    def test_critic_exception_skips_prompt(
        self, ift_path: str, caplog: pytest.LogCaptureFixture
    ):
        critic = _StubCritic(
            [RuntimeError("critic boom"), "NTST"], cycle_default=False
        )
        ift = _StubIFTModel(
            ["<signal>ETWT</signal>", "<signal>ETWT</signal>"]
        )
        collector = CGPRDataCollector(
            ift_model_path=ift_path,
            colight_critic=critic,
            ift_model_adapter_factory=lambda p: ift,
        )
        with caplog.at_level(logging.WARNING):
            result = collector.collect(["bad prompt", "good prompt"])
        # First prompt skipped (critic raised); second produces pair.
        assert len(result) == 1
        assert result[0].prompt == "good prompt"
        # Warning logged for the failure.
        assert any("predict_phase raised" in rec.message for rec in caplog.records)

    def test_critic_invalid_phase_skipped(
        self, ift_path: str, caplog: pytest.LogCaptureFixture
    ):
        critic = _StubCritic(["BOGUS_PHASE", "NTST"])
        ift = _StubIFTModel(
            ["<signal>ETWT</signal>", "<signal>ETWT</signal>"]
        )
        collector = CGPRDataCollector(
            ift_model_path=ift_path,
            colight_critic=critic,
            ift_model_adapter_factory=lambda p: ift,
        )
        with caplog.at_level(logging.WARNING):
            result = collector.collect(["p1", "p2"])
        assert len(result) == 1
        assert result[0].prompt == "p2"
        assert any("invalid phase" in rec.message for rec in caplog.records)

    def test_critic_returns_non_str_skipped(
        self, ift_path: str, caplog: pytest.LogCaptureFixture
    ):
        critic = _StubCritic([42, "NTST"])
        ift = _StubIFTModel(
            ["<signal>ETWT</signal>", "<signal>ETWT</signal>"]
        )
        collector = CGPRDataCollector(
            ift_model_path=ift_path,
            colight_critic=critic,
            ift_model_adapter_factory=lambda p: ift,
        )
        with caplog.at_level(logging.WARNING):
            result = collector.collect(["p1", "p2"])
        assert len(result) == 1
        assert result[0].prompt == "p2"
        assert any("non-str" in rec.message for rec in caplog.records)

    def test_ift_exception_skips_prompt(
        self, ift_path: str, caplog: pytest.LogCaptureFixture
    ):
        critic = _StubCritic(["NTST", "NTST"])
        ift = _StubIFTModel(
            [RuntimeError("ift boom"), "<signal>ETWT</signal>"]
        )
        collector = CGPRDataCollector(
            ift_model_path=ift_path,
            colight_critic=critic,
            ift_model_adapter_factory=lambda p: ift,
        )
        with caplog.at_level(logging.WARNING):
            result = collector.collect(["p1", "p2"])
        assert len(result) == 1
        assert result[0].prompt == "p2"
        assert any("ift_model.generate raised" in rec.message for rec in caplog.records)


# =========================================================================
# Tests: scale — 100 prompts, 30 divergent
# =========================================================================


class TestCollectScale:
    """Realistic-sized collection scenario."""

    def test_100_prompts_30_divergent(self, ift_path: str):
        # Construct: first 30 prompts have critic=NTST vs IFT=ETWT
        # (divergent), remaining 70 have critic=ETWT vs IFT=ETWT (same).
        critic_responses = ["NTST"] * 30 + ["ETWT"] * 70
        ift_responses = ["<signal>ETWT</signal>"] * 100
        critic = _StubCritic(critic_responses)
        ift = _StubIFTModel(ift_responses)
        collector = CGPRDataCollector(
            ift_model_path=ift_path,
            colight_critic=critic,
            ift_model_adapter_factory=lambda p: ift,
        )
        prompts = [f"prompt_{i}" for i in range(100)]
        result = collector.collect(prompts)

        assert len(result) == 30
        # Every result pair has critic NTST vs IFT ETWT.
        for pair in result:
            assert pair.positive_response == "<signal>NTST</signal>"
            assert pair.negative_response == "<signal>ETWT</signal>"
        # Both adapters called exactly 100 times (every prompt processed).
        assert len(critic.calls) == 100
        # IFT called only when critic returned valid phase (all 100 here).
        assert len(ift.calls) == 100

    def test_mixed_skips_and_pairs(self, ift_path: str):
        """Mix: errors, same-phase, divergent — verify counts."""
        critic_responses = [
            "NTST",                       # 0: divergent vs ETWT
            RuntimeError("err"),          # 1: skipped (critic error)
            "ETWT",                       # 2: same-phase, skipped
            "INVALID",                    # 3: invalid phase, skipped
            "ELWL",                       # 4: divergent vs ETWT
        ]
        ift_responses = [
            "<signal>ETWT</signal>",      # 0
            "<signal>ETWT</signal>",      # 1 (won't be reached)
            "<signal>ETWT</signal>",      # 2
            "<signal>ETWT</signal>",      # 3 (won't be reached)
            "<signal>ETWT</signal>",      # 4
        ]
        critic = _StubCritic(critic_responses)
        ift = _StubIFTModel(ift_responses)
        collector = CGPRDataCollector(
            ift_model_path=ift_path,
            colight_critic=critic,
            ift_model_adapter_factory=lambda p: ift,
        )
        prompts = [f"p{i}" for i in range(5)]
        result = collector.collect(prompts)

        assert len(result) == 2
        assert result[0].prompt == "p0"
        assert result[0].positive_response == "<signal>NTST</signal>"
        assert result[1].prompt == "p4"
        assert result[1].positive_response == "<signal>ELWL</signal>"


# =========================================================================
# Tests: CGPRPair semantics
# =========================================================================


class TestCGPRPair:
    """``CGPRPair`` is a NamedTuple with the expected fields."""

    def test_named_tuple_fields(self):
        pair = CGPRPair(
            prompt="p",
            positive_response="<signal>ETWT</signal>",
            negative_response="raw",
        )
        assert pair.prompt == "p"
        assert pair.positive_response == "<signal>ETWT</signal>"
        assert pair.negative_response == "raw"
        # NamedTuple → tuple-iterable, immutable.
        assert tuple(pair) == ("p", "<signal>ETWT</signal>", "raw")

    def test_pair_is_immutable(self):
        pair = CGPRPair(prompt="p", positive_response="x", negative_response="y")
        with pytest.raises(AttributeError):
            pair.prompt = "new"  # type: ignore[misc]
