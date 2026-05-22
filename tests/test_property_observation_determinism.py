"""Property-based test: Observation Parser Determinism (Property 2).

**Validates: Requirements 3.2, 12.2**

Universal property under test (from ``design.md``):

```
∀ state ∈ ValidState, ∀ i, j ∈ [1..N], N ≥ 10:
  parse(state)_i == parse(state)_j   (byte-for-byte equality)
```

In words: for any valid CityFlow state dict, calling
:meth:`ObservationParser.parse` ten or more times in succession must
return strings that are identical down to the byte. This is the
foundation of caching, hashing, and reproducibility for the prompt
pipeline — any non-determinism (e.g. iterating over a dict in insertion
order, embedding `id()` of objects, leaking timestamps) would break it.

The test uses :func:`hypothesis.given` to generate valid states from a
deliberately constrained strategy:

- ``lane_vehicle_count``: ``dict[str, int]`` with 0-20 entries; keys are
  ASCII alphanumeric (mirrors CityFlow lane id format
  ``road_<dir>_<idx>``); values are non-negative ints up to 10 000.
- ``current_phase``: one of the four phases ``ObservationParser`` accepts.
- ``current_phase_time``: 0-3600 seconds (Full mode upper bound).

Per the task spec we run at least 100 hypothesis examples and at least 10
parse calls per example, comparing both the ``str`` form and its raw
``utf-8`` bytes to catch any encoding-level non-determinism.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

# Make ``src`` importable when running ``pytest`` from project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.observation_parser import ObservationParser  # noqa: E402

# =========================================================================
# Constants
# =========================================================================

#: Number of consecutive ``parse`` calls per generated state. Property 2
#: requires N ≥ 10; we use exactly 10 to keep example runtime predictable.
PARSE_CALLS_PER_STATE: int = 10

#: Phase set accepted by the parser (mirrors ``ObservationParser.VALID_PHASES``
#: but as a tuple so we can pass it to ``sampled_from``).
VALID_PHASES: tuple[str, ...] = ("ETWT", "NTST", "ELWL", "NLSL")


# =========================================================================
# Strategies
# =========================================================================

# Lane id alphabet: ASCII alphanumeric only. Hypothesis' ``text`` strategy
# with this whitelist matches the lane-id format CityFlow emits
# (``road_E_in_0``, ``road_W_out_1``, ...). Stripping unicode keeps the
# property focused on parser internals rather than encoding edge cases —
# those are covered by ResponseParser property tests (Task 4.2).
_LANE_ID_ALPHABET: str = (
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    "_"
)

_lane_id_strategy = st.text(
    alphabet=_LANE_ID_ALPHABET,
    min_size=1,
    max_size=20,
)

_lane_count_value_strategy = st.integers(min_value=0, max_value=10_000)

_lane_vehicle_count_strategy = st.dictionaries(
    keys=_lane_id_strategy,
    values=_lane_count_value_strategy,
    min_size=0,
    max_size=20,
)

_current_phase_strategy = st.sampled_from(VALID_PHASES)

_current_phase_time_strategy = st.integers(min_value=0, max_value=3600)


@st.composite
def valid_state_dicts(draw: st.DrawFn) -> dict:
    """Compose a valid state dict that ``ObservationParser`` will accept."""
    return {
        "lane_vehicle_count": draw(_lane_vehicle_count_strategy),
        "current_phase": draw(_current_phase_strategy),
        "current_phase_time": draw(_current_phase_time_strategy),
    }


# =========================================================================
# Property test
# =========================================================================


@pytest.fixture(scope="module")
def parser() -> ObservationParser:
    """One parser per module — ``parse`` is pure, no per-test state."""
    return ObservationParser()


@settings(
    max_examples=100,
    deadline=None,
    # ``parser`` fixture is module-scoped and immutable; the function-scoped
    # fixture warning would be a false positive here.
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(state=valid_state_dicts())
def test_parse_is_deterministic_byte_for_byte(
    parser: ObservationParser, state: dict
) -> None:
    """Property 2 — Observation Parser Determinism.

    For any valid state dict, ``PARSE_CALLS_PER_STATE`` (≥10) consecutive
    calls to :meth:`ObservationParser.parse` produce identical ``str``
    results AND identical ``utf-8``-encoded byte sequences.

    Validates: Requirements 3.2, 12.2
    """
    first_str = parser.parse(state)
    first_bytes = first_str.encode("utf-8")

    for call_index in range(1, PARSE_CALLS_PER_STATE):
        nth_str = parser.parse(state)
        nth_bytes = nth_str.encode("utf-8")

        assert nth_str == first_str, (
            f"Non-deterministic str output at call #{call_index + 1}/"
            f"{PARSE_CALLS_PER_STATE}.\n"
            f"State: {state!r}\n"
            f"First call: {first_str!r}\n"
            f"This call: {nth_str!r}"
        )
        assert nth_bytes == first_bytes, (
            f"Non-deterministic utf-8 bytes at call #{call_index + 1}/"
            f"{PARSE_CALLS_PER_STATE}.\n"
            f"State: {state!r}\n"
            f"First bytes: {first_bytes!r}\n"
            f"This bytes:  {nth_bytes!r}"
        )
