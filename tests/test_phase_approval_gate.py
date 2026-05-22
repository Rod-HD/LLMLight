"""Unit tests for ``src.phase_approval_gate``.

Validates:
    - Requirement 13.1 (CLI flag --phase 1|2|3 with default 1; phase enum)
    - Requirement 13.2 (Phase 2 prerequisite: warn + confirm)
    - Requirement 13.3 (Phase 3 prerequisite: hard block when both Phase 2
      files missing)
    - Requirement 13.4 (Phase 3 manual approval: must type "yes")
    - Requirement 13.5 (Phase 3 dataset = newyork_1 only; Phase 1/2 reject
      newyork_1)
    - Requirement 13.6 (phase_label helper for ExperimentResult)
    - Requirement 13.7 (mode/phase mismatch warning)

Tests use ``input_fn`` constructor injection rather than ``monkeypatch`` of
``builtins.input``, but include a ``monkeypatch`` test as well to satisfy
the spec's request.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

# Make ``src`` importable when running ``pytest`` from project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.phase_approval_gate import PhaseApprovalGate  # noqa: E402


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def empty_results_dir(tmp_path: Path) -> Path:
    """Empty results/metrics dir (no comparison_*.md files)."""
    d = tmp_path / "metrics"
    d.mkdir()
    return d


@pytest.fixture
def phase1_done_dir(tmp_path: Path) -> Path:
    """results/metrics with both Phase 1 comparison files present."""
    d = tmp_path / "metrics"
    d.mkdir()
    (d / "comparison_jinan_1_phase1.md").write_text("phase1 jinan", encoding="utf-8")
    (d / "comparison_hangzhou_1_phase1.md").write_text(
        "phase1 hangzhou", encoding="utf-8"
    )
    return d


@pytest.fixture
def phase2_done_dir(tmp_path: Path) -> Path:
    """results/metrics with both Phase 2 comparison files present."""
    d = tmp_path / "metrics"
    d.mkdir()
    (d / "comparison_jinan_1_phase2.md").write_text("phase2 jinan", encoding="utf-8")
    (d / "comparison_hangzhou_1_phase2.md").write_text(
        "phase2 hangzhou", encoding="utf-8"
    )
    return d


@pytest.fixture(autouse=True)
def _no_auto_approve(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure ``LLMLIGHT_AUTO_APPROVE`` is not leaking from the test env."""
    monkeypatch.delenv("LLMLIGHT_AUTO_APPROVE", raising=False)


def _make_gate(
    results_dir: Path | None = None,
    *,
    answers: list[str] | None = None,
) -> tuple[PhaseApprovalGate, list[str]]:
    """Helper: build a gate with a scripted ``input_fn`` returning ``answers``
    in order. Returns ``(gate, prompts_seen)``.
    """
    answers = list(answers or [])
    prompts_seen: list[str] = []

    def fake_input(prompt: str) -> str:
        prompts_seen.append(prompt)
        if not answers:
            raise AssertionError(
                f"input_fn called more times than scripted answers; "
                f"prompt={prompt!r}"
            )
        return answers.pop(0)

    gate = PhaseApprovalGate(
        results_dir=results_dir if results_dir is not None else "results/metrics",
        input_fn=fake_input,
    )
    return gate, prompts_seen


# =========================================================================
# validate_phase — invalid phase number
# =========================================================================


@pytest.mark.parametrize("bad_phase", [0, 4, -1, 99])
def test_validate_phase_rejects_invalid_phase_number(bad_phase: int) -> None:
    """Validates: Requirement 13.1"""
    gate, _ = _make_gate()
    with pytest.raises(ValueError, match="Invalid phase"):
        gate.validate_phase(bad_phase, "jinan_1", "demo")


def test_validate_phase_rejects_invalid_mode() -> None:
    gate, _ = _make_gate()
    with pytest.raises(ValueError, match="Invalid mode"):
        gate.validate_phase(1, "jinan_1", "turbo")


# =========================================================================
# validate_phase — dataset/phase mismatch
# =========================================================================


def test_validate_phase_phase1_rejects_newyork() -> None:
    """Validates: Requirement 13.5"""
    gate, _ = _make_gate()
    with pytest.raises(ValueError, match="newyork_1"):
        gate.validate_phase(1, "newyork_1", "demo")


def test_validate_phase_phase2_rejects_newyork() -> None:
    """Validates: Requirement 13.5"""
    gate, _ = _make_gate()
    with pytest.raises(ValueError, match="newyork_1"):
        gate.validate_phase(2, "newyork_1", "full")


def test_validate_phase_phase3_rejects_jinan() -> None:
    """Validates: Requirement 13.5"""
    gate, _ = _make_gate()
    with pytest.raises(ValueError, match="jinan_1"):
        gate.validate_phase(3, "jinan_1", "full")


def test_validate_phase_phase3_rejects_hangzhou() -> None:
    """Validates: Requirement 13.5"""
    gate, _ = _make_gate()
    with pytest.raises(ValueError, match="hangzhou_1"):
        gate.validate_phase(3, "hangzhou_1", "full")


@pytest.mark.parametrize(
    "phase, dataset, mode",
    [
        (1, "jinan_1", "demo"),
        (1, "hangzhou_1", "demo"),
        (2, "jinan_1", "full"),
        (2, "hangzhou_1", "full"),
        (3, "newyork_1", "full"),
    ],
)
def test_validate_phase_accepts_valid_combinations(
    phase: int, dataset: str, mode: str
) -> None:
    gate, _ = _make_gate()
    # Should not raise.
    gate.validate_phase(phase, dataset, mode)


# =========================================================================
# validate_phase — mode warnings (Requirement 13.7)
# =========================================================================


def test_validate_phase_warns_on_phase1_full(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Validates: Requirement 13.7"""
    gate, _ = _make_gate()
    with caplog.at_level(logging.WARNING, logger="src.phase_approval_gate"):
        gate.validate_phase(1, "jinan_1", "full")
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("Mode override" in r.message for r in warnings), (
        f"Expected mode-override warning, got records: {[r.message for r in warnings]}"
    )


def test_validate_phase_warns_on_phase2_demo(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Validates: Requirement 13.7"""
    gate, _ = _make_gate()
    with caplog.at_level(logging.WARNING, logger="src.phase_approval_gate"):
        gate.validate_phase(2, "jinan_1", "demo")
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("Mode override" in r.message for r in warnings)


def test_validate_phase_warns_on_phase3_demo(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Validates: Requirement 13.7"""
    gate, _ = _make_gate()
    with caplog.at_level(logging.WARNING, logger="src.phase_approval_gate"):
        gate.validate_phase(3, "newyork_1", "demo")
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("Mode override" in r.message for r in warnings)


def test_validate_phase_no_warning_when_mode_matches_default(
    caplog: pytest.LogCaptureFixture,
) -> None:
    gate, _ = _make_gate()
    with caplog.at_level(logging.WARNING, logger="src.phase_approval_gate"):
        gate.validate_phase(1, "jinan_1", "demo")
        gate.validate_phase(2, "jinan_1", "full")
        gate.validate_phase(3, "newyork_1", "full")
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert not any("Mode override" in r.message for r in warnings), (
        f"Did not expect Mode override warning, got: {[r.message for r in warnings]}"
    )


# =========================================================================
# check_prerequisite
# =========================================================================


def test_check_prerequisite_phase1_is_noop(empty_results_dir: Path) -> None:
    """Validates: Requirement 13.2 (Phase 1 has no prerequisite)"""
    gate, _ = _make_gate(empty_results_dir)
    # Should not raise even though dir is empty.
    gate.check_prerequisite(1)


def test_check_prerequisite_phase2_passes_when_files_present(
    phase1_done_dir: Path,
) -> None:
    """Validates: Requirement 13.2"""
    gate, prompts = _make_gate(phase1_done_dir)
    gate.check_prerequisite(2)
    assert prompts == [], "Should not prompt user when Phase 1 files present."


def test_check_prerequisite_phase2_warns_and_confirms(
    empty_results_dir: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Validates: Requirement 13.2 (warn + require user confirmation)"""
    gate, prompts = _make_gate(empty_results_dir, answers=["yes"])
    with caplog.at_level(logging.WARNING, logger="src.phase_approval_gate"):
        gate.check_prerequisite(2)
    assert len(prompts) == 1, "Should have prompted user once."
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("Phase 1 chưa hoàn tất" in r.message for r in warnings)


def test_check_prerequisite_phase2_aborts_when_user_says_no(
    empty_results_dir: Path,
) -> None:
    """Validates: Requirement 13.2 (no → abort)"""
    gate, _ = _make_gate(empty_results_dir, answers=["no"])
    with pytest.raises(RuntimeError, match="Phase 2 bị huỷ"):
        gate.check_prerequisite(2)


def test_check_prerequisite_phase2_partial_files_still_warns(
    tmp_path: Path,
) -> None:
    """Validates: Requirement 13.2"""
    d = tmp_path / "metrics"
    d.mkdir()
    (d / "comparison_jinan_1_phase1.md").write_text("only jinan", encoding="utf-8")
    # Hangzhou Phase 1 file missing.

    gate, prompts = _make_gate(d, answers=["yes"])
    gate.check_prerequisite(2)
    assert len(prompts) == 1, "Should still prompt when one file is missing."


def test_check_prerequisite_phase3_hard_blocks_when_both_files_missing(
    empty_results_dir: Path,
) -> None:
    """Validates: Requirement 13.3 (HARD BLOCK with RuntimeError)"""
    gate, _ = _make_gate(empty_results_dir)
    with pytest.raises(RuntimeError, match="Phase 3 yêu cầu Phase 2"):
        gate.check_prerequisite(3)


def test_check_prerequisite_phase3_passes_when_both_files_present(
    phase2_done_dir: Path,
) -> None:
    """Validates: Requirement 13.3"""
    gate, _ = _make_gate(phase2_done_dir)
    # Should not raise.
    gate.check_prerequisite(3)


def test_check_prerequisite_phase3_partial_warns_but_does_not_block(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Validates: Requirement 13.3 (hard block ONLY when both missing)"""
    d = tmp_path / "metrics"
    d.mkdir()
    (d / "comparison_jinan_1_phase2.md").write_text("phase2 jinan", encoding="utf-8")
    # Hangzhou Phase 2 missing.

    gate, _ = _make_gate(d)
    with caplog.at_level(logging.WARNING, logger="src.phase_approval_gate"):
        # Should NOT raise.
        gate.check_prerequisite(3)
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any(
        "Phase 3 prerequisite cảnh báo" in r.message for r in warnings
    ), f"Expected partial-prerequisite warning, got: {[r.message for r in warnings]}"


def test_check_prerequisite_invalid_phase_raises() -> None:
    gate, _ = _make_gate()
    with pytest.raises(ValueError, match="Invalid phase"):
        gate.check_prerequisite(99)


# =========================================================================
# request_manual_approval
# =========================================================================


@pytest.mark.parametrize("phase", [1, 2])
def test_request_manual_approval_phases_1_and_2_always_true(phase: int) -> None:
    """Validates: Requirement 13.4 (only Phase 3 needs approval)"""
    gate, prompts = _make_gate()
    assert gate.request_manual_approval(phase) is True
    assert prompts == [], "Phase 1/2 should not prompt user."


def test_request_manual_approval_phase3_yes_returns_true(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Validates: Requirement 13.4"""
    gate, prompts = _make_gate(answers=["yes"])
    result = gate.request_manual_approval(
        phase=3, estimated_cost_usd=12.5, estimated_time_hours=4.0
    )
    assert result is True
    assert len(prompts) == 1
    captured = capsys.readouterr()
    # Cost / time hints should be displayed in the printed warning.
    assert "$12.50" in captured.out
    assert "4.00 giờ" in captured.out


@pytest.mark.parametrize(
    "answer", ["YES", "  yes  ", "yes\n", "Yes", "yEs"]
)
def test_request_manual_approval_phase3_yes_case_insensitive_and_stripped(
    answer: str,
) -> None:
    """Validates: Requirement 13.4 (case-insensitive, strip whitespace)"""
    gate, _ = _make_gate(answers=[answer])
    assert gate.request_manual_approval(phase=3) is True


@pytest.mark.parametrize(
    "answer", ["no", "n", "", " ", "yess", "y", "ok", "Y", "yep"]
)
def test_request_manual_approval_phase3_anything_else_returns_false(
    answer: str,
) -> None:
    """Validates: Requirement 13.4 (anything other than 'yes' aborts)"""
    gate, _ = _make_gate(answers=[answer])
    assert gate.request_manual_approval(phase=3) is False


def test_request_manual_approval_phase3_via_monkeypatch_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validates: Requirement 13.4 — also support monkeypatch on input()."""
    monkeypatch.setattr("builtins.input", lambda prompt="": "yes")
    gate = PhaseApprovalGate(results_dir="results/metrics")
    assert gate.request_manual_approval(phase=3) is True


def test_request_manual_approval_phase3_monkeypatch_input_no(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("builtins.input", lambda prompt="": "")
    gate = PhaseApprovalGate(results_dir="results/metrics")
    assert gate.request_manual_approval(phase=3) is False


def test_request_manual_approval_phase3_eof_returns_false(
    capsys: pytest.CaptureFixture[str],
) -> None:
    def raises_eof(prompt: str) -> str:
        raise EOFError

    gate = PhaseApprovalGate(results_dir="results/metrics", input_fn=raises_eof)
    assert gate.request_manual_approval(phase=3) is False


def test_request_manual_approval_invalid_phase_raises() -> None:
    gate, _ = _make_gate()
    with pytest.raises(ValueError, match="Invalid phase"):
        gate.request_manual_approval(phase=4)


def test_auto_approve_env_var_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``LLMLIGHT_AUTO_APPROVE=yes`` should auto-confirm without prompting."""
    monkeypatch.setenv("LLMLIGHT_AUTO_APPROVE", "yes")

    def must_not_be_called(prompt: str) -> str:
        raise AssertionError("input_fn should not be called when auto-approve is set")

    gate = PhaseApprovalGate(
        results_dir="results/metrics", input_fn=must_not_be_called
    )
    assert gate.request_manual_approval(phase=3) is True


def test_auto_approve_env_var_case_insensitive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLMLIGHT_AUTO_APPROVE", "  YES  ")
    gate = PhaseApprovalGate(
        results_dir="results/metrics", input_fn=lambda p: "no"
    )
    assert gate.request_manual_approval(phase=3) is True


def test_auto_approve_env_var_other_values_do_not_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLMLIGHT_AUTO_APPROVE", "maybe")
    gate, _ = _make_gate(answers=["no"])
    # Reset env via fixture is re-applied, but here we want it set; rebuild gate
    # locally so we use the monkeypatched env.
    gate = PhaseApprovalGate(
        results_dir="results/metrics", input_fn=lambda p: "no"
    )
    assert gate.request_manual_approval(phase=3) is False


# =========================================================================
# phase_label
# =========================================================================


@pytest.mark.parametrize(
    "phase, expected",
    [(1, "Phase1"), (2, "Phase2"), (3, "Phase3")],
)
def test_phase_label_correctness(phase: int, expected: str) -> None:
    """Validates: Requirement 13.6"""
    gate, _ = _make_gate()
    assert gate.phase_label(phase) == expected


@pytest.mark.parametrize("bad_phase", [0, 4, -1])
def test_phase_label_rejects_invalid(bad_phase: int) -> None:
    gate, _ = _make_gate()
    with pytest.raises(ValueError, match="Invalid phase"):
        gate.phase_label(bad_phase)
