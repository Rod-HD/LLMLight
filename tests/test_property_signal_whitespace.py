"""Property-based tests for ``ResponseParser`` whitespace robustness.

**Property 4: Whitespace-Robust Signal Extraction**

**Validates: Requirements 4.1, 4.4, 12.4**

Spec (design.md §"Property 4"):

    ∀ phase ∈ VALID_PHASES, ∀ ws_lead, ws_trail ∈ {' ', '\\t', '\\n', '\\r'}*:
        let response = f"<signal>{ws_lead}{phase}{ws_trail}</signal>"
        ResponseParser.parse(response) == phase

Tests in this module:

* ``test_property_whitespace_robust_signal_extraction`` — the core named
  property from the design document. Generates ``(phase, ws_lead, ws_trail)``
  triples from the exact strategies mandated by the task description and
  asserts byte-equality with ``phase``.
* ``test_property_whitespace_robust_case_insensitive`` — extension required
  by the task description: combines arbitrary whitespace padding with
  case-insensitive phase variants (e.g. ``etwt``, ``EtWt``) and asserts the
  parser still returns the canonical uppercase phase. This complements
  Property 4 by guaranteeing AC 4.2 (case-insensitive comparison) holds
  jointly with AC 4.4 (whitespace robustness).

Both tests run with ``@settings(max_examples=100, deadline=None)`` per the
spec's "Tối thiểu 100 iterations" requirement; ``deadline=None`` avoids
flaky failures on slow CI (the parser itself runs in microseconds).
"""

from __future__ import annotations

import sys
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

# Make ``src`` importable when running ``pytest`` from project root or from
# inside the ``tests`` directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.response_parser import ResponseParser  # noqa: E402


# =========================================================================
# Strategies (exactly as specified in the task description)
# =========================================================================

# Canonical, uppercase phases per the LLMLight paper.
VALID_PHASES: tuple[str, ...] = ("ETWT", "NTST", "ELWL", "NLSL")

# Whitespace alphabet specified by AC 4.4: space, tab, newline, carriage return.
WHITESPACE_CHARS: tuple[str, ...] = (" ", "\t", "\n", "\r")

# Phase generator — uniform sample over the four valid phases.
phase_strategy = st.sampled_from(VALID_PHASES)

# Whitespace generator — variable-length string built from the allowed
# whitespace alphabet. ``min_size=0`` includes the empty string (no padding)
# as a valid input. ``max_size=20`` keeps generated responses small enough
# to shrink quickly while still exercising mixed/long whitespace cases.
whitespace_strategy = st.lists(
    st.sampled_from(WHITESPACE_CHARS),
    min_size=0,
    max_size=20,
).map("".join)


# =========================================================================
# Property 4 — core
# =========================================================================


@given(
    phase=phase_strategy,
    ws_lead=whitespace_strategy,
    ws_trail=whitespace_strategy,
)
@settings(max_examples=100, deadline=None)
def test_property_whitespace_robust_signal_extraction(
    phase: str, ws_lead: str, ws_trail: str
) -> None:
    """**Validates: Requirements 4.1, 4.4, 12.4**

    For every canonical phase and every combination of leading/trailing
    whitespace drawn from ``{' ', '\\t', '\\n', '\\r'}*``, the parser must
    extract the phase verbatim.

    This is the property exactly as specified in design.md §Property 4.
    """
    parser = ResponseParser()
    response = f"<signal>{ws_lead}{phase}{ws_trail}</signal>"

    result = parser.parse(response)

    assert result == phase, (
        f"Whitespace-robust extraction failed: "
        f"phase={phase!r}, ws_lead={ws_lead!r}, ws_trail={ws_trail!r}, "
        f"got={result!r}"
    )
    # Belt-and-suspenders: the output must lie in the valid set (AC 12.1).
    assert result in ResponseParser.VALID_PHASES


# =========================================================================
# Property 4 — extension: case-insensitive phase combined with whitespace
# =========================================================================


def _case_variants(phase: str) -> list[str]:
    """All useful case variants of a phase token.

    We intentionally enumerate a small representative set rather than
    Cartesian-product over every character (16 variants per phase) so the
    Hypothesis search space stays tractable while still covering the cases
    most likely to break a buggy implementation.
    """
    return [
        phase,           # canonical uppercase
        phase.lower(),   # all lowercase
        phase.title(),   # Title-Case (e.g. "Etwt")
        phase.swapcase(),  # inverted case
        # Mixed: alternate upper/lower per character.
        "".join(
            ch.lower() if i % 2 == 0 else ch.upper()
            for i, ch in enumerate(phase)
        ),
    ]


# Build a flat strategy over (canonical_phase, raw_variant) pairs so each
# example carries both the variant being parsed and the expected canonical
# answer. ``st.sampled_from`` over a precomputed list keeps the strategy
# deterministic and fast to shrink.
_PHASE_VARIANT_PAIRS: list[tuple[str, str]] = [
    (phase, variant)
    for phase in VALID_PHASES
    for variant in _case_variants(phase)
]
phase_variant_strategy = st.sampled_from(_PHASE_VARIANT_PAIRS)


@given(
    phase_variant=phase_variant_strategy,
    ws_lead=whitespace_strategy,
    ws_trail=whitespace_strategy,
)
@settings(max_examples=100, deadline=None)
def test_property_whitespace_robust_case_insensitive(
    phase_variant: tuple[str, str], ws_lead: str, ws_trail: str
) -> None:
    """**Validates: Requirements 4.1, 4.2, 4.4, 12.4**

    Extension required by the task description. For every canonical phase,
    every case variant of that phase, and every combination of
    leading/trailing whitespace, the parser must return the canonical
    uppercase phase.

    Joint coverage of:
      * AC 4.2 — case-insensitive comparison, return uppercase.
      * AC 4.4 — whitespace robustness inside the tag.
    """
    canonical, variant = phase_variant
    parser = ResponseParser()
    response = f"<signal>{ws_lead}{variant}{ws_trail}</signal>"

    result = parser.parse(response)

    assert result == canonical, (
        f"Case-insensitive whitespace extraction failed: "
        f"variant={variant!r}, canonical={canonical!r}, "
        f"ws_lead={ws_lead!r}, ws_trail={ws_trail!r}, got={result!r}"
    )
    assert result in ResponseParser.VALID_PHASES
