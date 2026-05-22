"""Property-based test cho Response Parser Output Validity.

**Property 1: Response Parser Output Validity**

**Validates: Requirements 4.2, 4.3, 12.1**

``ResponseParser.parse(response)`` PHẢI:
  * Không bao giờ raise (Requirement 12.1).
  * Luôn trả về một giá trị thuộc ``VALID_PHASES = {ETWT, NTST, ELWL, NLSL}``
    (Requirement 4.2 — case-insensitive comparison + uppercase output;
    Requirement 4.3 — fallback ``ETWT`` khi parse thất bại).

Strategy: hợp nhiều generator để bao phủ:
  * ``st.text()`` — random text (chuỗi rỗng, garbage, không có tag, unicode).
  * ``<signal>{random_content}</signal>`` — tag với nội dung ngẫu nhiên (đa
    số sẽ KHÔNG hợp lệ → kích hoạt fallback path).
  * ``<signal>{biased_phase}</signal>`` — tag với giá trị lấy từ tập trộn
    valid phases (mọi case) + invalid sentinels (rỗng, whitespace, gần
    đúng) để đảm bảo cả happy path và đường dẫn chuẩn hóa được khám phá.

Kích thước input lên đến 10000 ký tự (theo task spec).

Tối thiểu 100 iterations qua ``@settings(max_examples=100)``; ``deadline=None``
để bypass mọi giới hạn thời gian khi máy CI chậm.
"""

from __future__ import annotations

import sys
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

# Make ``src`` importable when running ``pytest`` from project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.response_parser import ResponseParser  # noqa: E402


# =========================================================================
# Constants
# =========================================================================

VALID_PHASES = frozenset({"ETWT", "NTST", "ELWL", "NLSL"})

# Mixed-case + invalid sentinels for the biased phase strategy. Includes:
#   * Canonical uppercase phases (happy path).
#   * Lowercase/mixed-case (case-insensitive comparison path).
#   * Empty / whitespace-only (fallback path).
#   * Near-miss tokens like "ETW", "ETWTX", "RED" (fallback path).
#   * Numeric-looking tokens (fallback path).
_MIXED_PHASE_TOKENS: tuple[str, ...] = (
    # Valid, all cases.
    "ETWT", "NTST", "ELWL", "NLSL",
    "etwt", "ntst", "elwl", "nlsl",
    "Etwt", "nTsT", "ElWl", "NlSl",
    # Invalid sentinels.
    "",
    "   ",
    "\t\n",
    "ETW",
    "ETWTX",
    "ET WT",
    "RED",
    "GREEN",
    "FOO",
    "1234",
    "phase1",
    # Unicode that looks like a phase but isn't ASCII.
    "ETWТ",  # Cyrillic Т, NOT ASCII T.
)


# =========================================================================
# Strategies
# =========================================================================


def _wrap_signal(content: str) -> str:
    """Helper: wrap arbitrary content inside a ``<signal>`` tag."""
    return f"<signal>{content}</signal>"


# Pure random text — covers empty string, garbage, unicode, no-tag responses.
# ``max_size=10000`` per the task spec for extreme-length coverage; using
# default text alphabet which already exercises unicode (incl. surrogates
# filtered out by hypothesis).
_pure_text = st.text(min_size=0, max_size=10000)

# Random content wrapped in a <signal> tag. Most outputs will be invalid →
# exercises the fallback + warning path.
_random_tag_content = st.builds(_wrap_signal, st.text(max_size=20))

# Phase tag with biased content (valid mixed-case + invalid sentinels).
_biased_phase_tag = st.builds(_wrap_signal, st.sampled_from(_MIXED_PHASE_TOKENS))

# Whitespace-padded biased phase: leading/trailing spaces/tabs/newlines
# around a sampled token. Exercises the AC 4.4 strip path.
_ws_chars = st.text(alphabet=" \t\n\r", min_size=0, max_size=8)
_padded_biased_tag = st.builds(
    lambda lead, tok, trail: _wrap_signal(f"{lead}{tok}{trail}"),
    _ws_chars,
    st.sampled_from(_MIXED_PHASE_TOKENS),
    _ws_chars,
)

# Multi-tag noise: two random tags concatenated with random text in between.
# AC 4.5 says first tag wins; we just need to verify the OUTPUT remains in
# VALID_PHASES regardless of what the second tag contains.
_multi_tag = st.builds(
    lambda a, mid, b: f"{_wrap_signal(a)}{mid}{_wrap_signal(b)}",
    st.text(max_size=20),
    st.text(max_size=20),
    st.text(max_size=20),
)

# Final union — every generated example is a string but spans multiple
# code paths inside ``ResponseParser.parse``.
_response_strategy = st.one_of(
    _pure_text,
    _random_tag_content,
    _biased_phase_tag,
    _padded_biased_tag,
    _multi_tag,
)


# =========================================================================
# Property test
# =========================================================================


@given(response=_response_strategy)
@settings(max_examples=100, deadline=None)
def test_parse_output_always_in_valid_phases(response: str) -> None:
    """Parser output for ANY string is always a member of ``VALID_PHASES``.

    Validates:
        Requirements 4.2, 4.3, 12.1.

    Bao phủ:
        * Chuỗi rỗng → fallback ETWT.
        * Garbage không có tag → fallback ETWT.
        * Tag với giá trị không hợp lệ → fallback ETWT.
        * Tag với phase hợp lệ (mọi case) → uppercase valid phase.
        * Tag có whitespace bao quanh phase → strip & uppercase.
        * Nhiều tag → tag đầu tiên thắng (output vẫn thuộc VALID_PHASES).
        * Unicode arbitrary → fallback an toàn (KHÔNG raise).
    """
    parser = ResponseParser()
    result = parser.parse(response)
    assert result in VALID_PHASES, (
        f"parse() returned {result!r} which is not in VALID_PHASES "
        f"for response={response!r}"
    )
