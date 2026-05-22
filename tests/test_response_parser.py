"""Unit tests for ``src.response_parser``.

Validates:
    - Requirement 4.1: trích xuất tag ``<signal>`` đầu tiên.
    - Requirement 4.2: case-insensitive comparison, return uppercase.
    - Requirement 4.3: fallback ``ETWT`` + warning với tối đa 500 ký tự đầu
      khi không có tag hoặc giá trị không hợp lệ.
    - Requirement 4.4: strip whitespace (space/tab/newline/CR) đầu/cuối nội
      dung tag.
    - Requirement 4.5: cảnh báo riêng khi phát hiện nhiều tag, dùng tag đầu.
    - Requirement 12.1, 12.4: parser không bao giờ raise; output luôn thuộc
      ``VALID_PHASES``.

Property-based tests (Property 1, Property 4) sống ở subtask 4.2 và 4.3
trong các file riêng — file này CHỈ chứa unit tests điểm/edge case.
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

from src.response_parser import ResponseParser  # noqa: E402


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def parser() -> ResponseParser:
    """Default parser instance using the module-level logger."""
    return ResponseParser()


# =========================================================================
# Class constants — sanity
# =========================================================================


class TestClassConstants:
    """Spec-mandated constants must be exact."""

    def test_valid_phases_exact_set(self):
        assert ResponseParser.VALID_PHASES == frozenset(
            {"ETWT", "NTST", "ELWL", "NLSL"}
        )

    def test_default_phase_is_etwt(self):
        assert ResponseParser.DEFAULT_PHASE == "ETWT"

    def test_max_log_length_is_500(self):
        assert ResponseParser.MAX_LOG_LENGTH == 500


# =========================================================================
# Happy path — valid tag in canonical form
# =========================================================================


class TestValidTag:
    """Requirement 4.1, 4.2: extract first tag, return uppercase phase."""

    @pytest.mark.parametrize("phase", ["ETWT", "NTST", "ELWL", "NLSL"])
    def test_canonical_tag_returns_phase(self, parser, phase):
        assert parser.parse(f"<signal>{phase}</signal>") == phase

    def test_tag_inside_longer_response(self, parser):
        response = (
            "Looking at the queue lengths, the best decision is to "
            "<signal>NTST</signal> for the next 30s."
        )
        assert parser.parse(response) == "NTST"

    def test_tag_with_surrounding_text(self, parser):
        response = "Decision: <signal>ELWL</signal>. Reason: blah."
        assert parser.parse(response) == "ELWL"

    def test_tag_with_text_before_only(self, parser):
        assert parser.parse("Final answer <signal>NLSL</signal>") == "NLSL"

    def test_tag_with_text_after_only(self, parser):
        assert parser.parse("<signal>ETWT</signal> selected.") == "ETWT"


# =========================================================================
# Case-insensitive comparison (Requirement 4.2)
# =========================================================================


class TestCaseInsensitive:
    """AC 4.2: case-insensitive comparison, return uppercase."""

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("etwt", "ETWT"),
            ("Etwt", "ETWT"),
            ("eTwT", "ETWT"),
            ("ntst", "NTST"),
            ("NtSt", "NTST"),
            ("elwl", "ELWL"),
            ("nlsl", "NLSL"),
            ("ETWT", "ETWT"),  # already uppercase
        ],
    )
    def test_lowercase_and_mixed_case_phases(self, parser, raw, expected):
        assert parser.parse(f"<signal>{raw}</signal>") == expected

    def test_lowercase_signal_tag_name_works(self, parser):
        # Tag name itself can be uppercase too because of re.IGNORECASE.
        assert parser.parse("<SIGNAL>etwt</SIGNAL>") == "ETWT"
        assert parser.parse("<Signal>NTST</Signal>") == "NTST"


# =========================================================================
# Whitespace handling (Requirement 4.4)
# =========================================================================


class TestWhitespaceWithinTag:
    """AC 4.4: strip space/tab/newline/CR at start AND end of tag content."""

    def test_leading_spaces(self, parser):
        assert parser.parse("<signal>   ETWT</signal>") == "ETWT"

    def test_trailing_spaces(self, parser):
        assert parser.parse("<signal>ETWT   </signal>") == "ETWT"

    def test_leading_and_trailing_spaces(self, parser):
        assert parser.parse("<signal>   NTST   </signal>") == "NTST"

    def test_tabs(self, parser):
        assert parser.parse("<signal>\tELWL\t</signal>") == "ELWL"

    def test_newlines(self, parser):
        assert parser.parse("<signal>\nNLSL\n</signal>") == "NLSL"

    def test_carriage_returns(self, parser):
        assert parser.parse("<signal>\rETWT\r</signal>") == "ETWT"

    def test_mixed_whitespace_combinations(self, parser):
        assert parser.parse("<signal>\n\t  NTST \r\n</signal>") == "NTST"

    def test_whitespace_with_lowercase_phase(self, parser):
        assert parser.parse("<signal>  etwt  </signal>") == "ETWT"


# =========================================================================
# Multiple tags (Requirement 4.5)
# =========================================================================


class TestMultipleTags:
    """AC 4.5: use first tag, log warning about multiple tags."""

    def test_first_tag_wins(self, parser):
        response = "<signal>NTST</signal> wait, actually <signal>ELWL</signal>"
        assert parser.parse(response) == "NTST"

    def test_three_tags_returns_first(self, parser):
        response = (
            "<signal>ETWT</signal> or <signal>NTST</signal> "
            "or <signal>ELWL</signal>?"
        )
        assert parser.parse(response) == "ETWT"

    def test_multiple_tags_logs_warning(self, parser, caplog):
        response = "<signal>NTST</signal> and <signal>ELWL</signal>"
        with caplog.at_level(logging.WARNING, logger="src.response_parser"):
            parser.parse(response)
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert any("Multiple <signal> tags" in m for m in warnings)
        # Count should be 2.
        assert any("count=2" in m for m in warnings)

    def test_multiple_tags_count_in_warning(self, parser, caplog):
        response = (
            "<signal>NTST</signal> <signal>ELWL</signal> <signal>NLSL</signal>"
        )
        with caplog.at_level(logging.WARNING, logger="src.response_parser"):
            parser.parse(response)
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert any("count=3" in m for m in warnings)

    def test_multiple_tags_first_invalid_still_falls_back(self, parser, caplog):
        """First tag wins even if invalid → fallback to ETWT.

        AC 4.5 says use FIRST tag. If first tag is garbage and the second is
        valid, parser still uses first → fallback. This is the documented
        behavior; downstream callers should ensure prompts request a single
        tag.
        """
        response = "<signal>BOGUS</signal> <signal>NTST</signal>"
        with caplog.at_level(logging.WARNING, logger="src.response_parser"):
            assert parser.parse(response) == "ETWT"
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        # Two warnings expected: multi-tag + fallback for BOGUS.
        assert any("Multiple <signal> tags" in m for m in warnings)
        assert any("BOGUS" in m for m in warnings)


# =========================================================================
# No tag → fallback (Requirement 4.3)
# =========================================================================


class TestNoTagFallback:
    """AC 4.3: no <signal> tag → ETWT + warning."""

    def test_empty_string_returns_default(self, parser):
        assert parser.parse("") == "ETWT"

    def test_garbage_text_returns_default(self, parser):
        assert parser.parse("garbage response with no tag") == "ETWT"

    def test_other_xml_tags_return_default(self, parser):
        assert parser.parse("<phase>NTST</phase>") == "ETWT"

    def test_only_opening_tag_returns_default(self, parser):
        assert parser.parse("<signal>NTST") == "ETWT"

    def test_only_closing_tag_returns_default(self, parser):
        assert parser.parse("NTST</signal>") == "ETWT"

    def test_no_tag_logs_warning_with_response_prefix(self, parser, caplog):
        with caplog.at_level(logging.WARNING, logger="src.response_parser"):
            parser.parse("garbage response")
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert any("no <signal> tag found" in m for m in warnings)
        assert any("garbage response" in m for m in warnings)

    def test_unicode_response_handled(self, parser):
        """Non-ASCII content without a tag → fallback, no crash."""
        assert parser.parse("Trả lời bằng tiếng Việt 🚦") == "ETWT"


# =========================================================================
# Invalid phase value inside tag → fallback
# =========================================================================


class TestInvalidPhaseValue:
    """AC 4.3: tag value not in VALID_PHASES → ETWT + warning."""

    @pytest.mark.parametrize(
        "raw",
        ["FOO", "RED", "GREEN", "etwtx", "et w t", "12345", "", "  "],
    )
    def test_invalid_value_falls_back(self, parser, raw):
        assert parser.parse(f"<signal>{raw}</signal>") == "ETWT"

    def test_invalid_value_logs_warning_with_value(self, parser, caplog):
        with caplog.at_level(logging.WARNING, logger="src.response_parser"):
            parser.parse("<signal>BOGUS</signal>")
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert any("BOGUS" in m for m in warnings)
        assert any("not in VALID_PHASES" in m for m in warnings)


# =========================================================================
# Long response — only first 500 chars logged (Requirement 4.3)
# =========================================================================


class TestResponseTruncationInLogs:
    """AC 4.3: log includes at most 500 first chars of raw response."""

    def test_response_under_500_chars_logged_in_full(self, parser, caplog):
        response = "x" * 100 + " no tag here"  # well under 500
        with caplog.at_level(logging.WARNING, logger="src.response_parser"):
            parser.parse(response)
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert any("no tag here" in m for m in warnings)

    def test_response_over_500_chars_logs_only_first_500(self, parser, caplog):
        # Build a 2000-char response where the truncation boundary is clear.
        head = "A" * 500
        tail = "B" * 1500  # sentinel: should NOT appear in any log message
        response = head + tail
        with caplog.at_level(logging.WARNING, logger="src.response_parser"):
            parser.parse(response)
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        # The truncated prefix (head) should be present.
        assert any("A" * 50 in m for m in warnings), warnings
        # The tail should never appear in any warning record.
        for w in warnings:
            assert "B" * 50 not in w, "Tail past 500-char boundary leaked into log"

    def test_truncation_boundary_exact_500(self, parser, caplog):
        """At the boundary, exactly 500 chars are kept."""
        response = "C" * 600  # 600 'C', no tag
        with caplog.at_level(logging.WARNING, logger="src.response_parser"):
            parser.parse(response)
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        # 500 'C' must be present...
        assert any("C" * 500 in m for m in warnings)
        # ...but 501 'C' must not (truncation exact).
        for w in warnings:
            assert "C" * 501 not in w


# =========================================================================
# Output domain invariant (Requirement 12.1) — minimal sanity
# =========================================================================


class TestOutputDomainInvariant:
    """AC 12.1: parser output must always lie in VALID_PHASES.

    Property test 4.2 covers this exhaustively with hypothesis; here we add
    a handful of pathological inputs as smoke tests.
    """

    @pytest.mark.parametrize(
        "response",
        [
            "",
            None,  # non-string still falls back, no raise
            "<signal></signal>",  # empty content
            "<signal>   </signal>",  # whitespace-only content
            "<signal>\n\n\n</signal>",
            "no tag",
            "<signal>FOO</signal>",
            "<SIGNAL>etwt</SIGNAL>",
            123,  # non-string scalar
            ["<signal>NTST</signal>"],  # list, not str
        ],
    )
    def test_always_returns_valid_phase(self, parser, response):
        out = parser.parse(response)
        assert out in ResponseParser.VALID_PHASES

    def test_non_string_input_does_not_raise(self, parser):
        # Defensive: covers cases where API client returns None on timeout.
        assert parser.parse(None) == "ETWT"


# =========================================================================
# Custom logger injection
# =========================================================================


class TestCustomLogger:
    """Constructor accepts an optional logger (per task spec)."""

    def test_custom_logger_receives_warnings(self, caplog):
        custom = logging.getLogger("custom.response.parser.test")
        parser = ResponseParser(logger_=custom)
        with caplog.at_level(logging.WARNING, logger="custom.response.parser.test"):
            assert parser.parse("garbage") == "ETWT"
        warnings = [
            r.getMessage()
            for r in caplog.records
            if r.levelno == logging.WARNING
            and r.name == "custom.response.parser.test"
        ]
        assert len(warnings) >= 1

    def test_default_logger_name(self, parser):
        # The internal logger should be the module-level one when not injected.
        assert parser._logger.name == "src.response_parser"


# =========================================================================
# Tag with attributes (defensive — regex allows them)
# =========================================================================


class TestTagWithAttributes:
    """Regex allows ``<signal foo="bar">VALUE</signal>`` for robustness."""

    def test_tag_with_attribute_extracts_value(self, parser):
        assert parser.parse('<signal kind="x">NTST</signal>') == "NTST"
