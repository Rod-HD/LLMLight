"""Unit tests for :mod:`src.observation_parser`.

Validates the behavioural contract of :class:`ObservationParser`:

- Valid state produces a prompt with the 3 required section headers
  (Requirement 3 AC 3).
- Missing field raises :class:`ValueError` whose message names the field
  (Requirement 3 AC 5).
- Invalid phase raises :class:`ValueError` whose message names the value
  (Requirement 3 AC 6).
- Negative queue length raises :class:`ValueError` (Requirement 3 AC 4).
- Two parse calls on the same state produce identical bytes
  (Requirement 3 AC 2 — sanity check; full property in Task 3.2).

Property-based tests for determinism (Task 3.2) and prompt structure
(Task 3.3) live in separate files.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.observation_parser import ObservationParser  # noqa: E402
from src.sim_config import IntersectionState  # noqa: E402


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def parser() -> ObservationParser:
    return ObservationParser()


@pytest.fixture
def valid_state_dict() -> dict:
    return {
        "lane_vehicle_count": {
            "road_E_in_0": 3,
            "road_W_in_0": 0,
            "road_N_in_0": 7,
            "road_S_in_0": 12,
        },
        "current_phase": "ETWT",
        "current_phase_time": 15,
    }


# =========================================================================
# Happy-path: 3 sections present (Requirement 3 AC 3)
# =========================================================================


def test_parse_returns_three_section_headers(
    parser: ObservationParser, valid_state_dict: dict
) -> None:
    prompt = parser.parse(valid_state_dict)

    assert "## Observation" in prompt
    assert "## Instruction" in prompt
    assert "## Output Format" in prompt


def test_parse_observation_section_lists_lane_counts(
    parser: ObservationParser, valid_state_dict: dict
) -> None:
    prompt = parser.parse(valid_state_dict)

    # Each lane must appear with its queue count.
    for lane_id, count in valid_state_dict["lane_vehicle_count"].items():
        assert f"- {lane_id}: {count}" in prompt


def test_parse_observation_section_lists_phase_and_phase_time(
    parser: ObservationParser, valid_state_dict: dict
) -> None:
    prompt = parser.parse(valid_state_dict)

    assert "Current phase: ETWT" in prompt
    assert "Current phase time: 15 seconds" in prompt


def test_parse_output_spec_mentions_signal_tag(
    parser: ObservationParser, valid_state_dict: dict
) -> None:
    prompt = parser.parse(valid_state_dict)

    assert "<signal>" in prompt
    assert "</signal>" in prompt


def test_parse_section_order_observation_instruction_output(
    parser: ObservationParser, valid_state_dict: dict
) -> None:
    prompt = parser.parse(valid_state_dict)

    idx_obs = prompt.index("## Observation")
    idx_ins = prompt.index("## Instruction")
    idx_out = prompt.index("## Output Format")

    assert idx_obs < idx_ins < idx_out


# =========================================================================
# Determinism sanity check (Requirement 3 AC 2)
# =========================================================================


def test_parse_two_calls_same_state_byte_identical(
    parser: ObservationParser, valid_state_dict: dict
) -> None:
    a = parser.parse(valid_state_dict)
    b = parser.parse(valid_state_dict)

    assert a == b
    assert a.encode("utf-8") == b.encode("utf-8")


def test_parse_dict_insertion_order_does_not_change_output(
    parser: ObservationParser,
) -> None:
    """Lane order in the output must be alphabetical regardless of the
    dict insertion order in the input — needed for byte-for-byte
    determinism (Property 2)."""
    state_a = {
        "lane_vehicle_count": {"a_lane": 1, "b_lane": 2, "c_lane": 3},
        "current_phase": "NTST",
        "current_phase_time": 5,
    }
    state_b = {
        "lane_vehicle_count": {"c_lane": 3, "a_lane": 1, "b_lane": 2},
        "current_phase": "NTST",
        "current_phase_time": 5,
    }

    assert parser.parse(state_a) == parser.parse(state_b)


# =========================================================================
# IntersectionState input (dual-format support)
# =========================================================================


def test_parse_accepts_intersection_state_dataclass(
    parser: ObservationParser,
) -> None:
    state = IntersectionState(
        intersection_id="intersection_1_1",
        lane_vehicle_count={"road_E_in_0": 2, "road_N_in_0": 4},
        current_phase="ELWL",
        current_phase_time=8,
    )

    prompt = parser.parse(state)

    assert "## Observation" in prompt
    assert "Current phase: ELWL" in prompt
    assert "Current phase time: 8 seconds" in prompt
    assert "- road_E_in_0: 2" in prompt
    assert "- road_N_in_0: 4" in prompt


def test_parse_dict_and_intersection_state_produce_same_output(
    parser: ObservationParser,
) -> None:
    lane_counts = {"a": 1, "b": 0, "c": 9}
    state_obj = IntersectionState(
        intersection_id="intersection_1_1",
        lane_vehicle_count=lane_counts,
        current_phase="ETWT",
        current_phase_time=12,
    )
    state_dict = {
        "lane_vehicle_count": lane_counts,
        "current_phase": "ETWT",
        "current_phase_time": 12,
    }

    assert parser.parse(state_obj) == parser.parse(state_dict)


# =========================================================================
# Validation: missing fields (Requirement 3 AC 5)
# =========================================================================


@pytest.mark.parametrize(
    "missing_field",
    ["lane_vehicle_count", "current_phase", "current_phase_time"],
)
def test_parse_missing_field_raises_value_error_naming_field(
    parser: ObservationParser,
    valid_state_dict: dict,
    missing_field: str,
) -> None:
    bad_state = dict(valid_state_dict)
    bad_state.pop(missing_field)

    with pytest.raises(ValueError) as exc_info:
        parser.parse(bad_state)

    assert missing_field in str(exc_info.value)


def test_parse_non_dict_state_raises_value_error(
    parser: ObservationParser,
) -> None:
    with pytest.raises(ValueError):
        parser.parse("not a dict")  # type: ignore[arg-type]


# =========================================================================
# Validation: invalid phase (Requirement 3 AC 6)
# =========================================================================


def test_parse_invalid_phase_raises_value_error_naming_value(
    parser: ObservationParser,
    valid_state_dict: dict,
) -> None:
    bad_state = dict(valid_state_dict)
    bad_state["current_phase"] = "BOGUS"

    with pytest.raises(ValueError) as exc_info:
        parser.parse(bad_state)

    msg = str(exc_info.value)
    assert "current_phase" in msg
    assert "BOGUS" in msg


def test_parse_phase_lowercase_is_rejected(
    parser: ObservationParser,
    valid_state_dict: dict,
) -> None:
    """ObservationParser does not normalize case (case-insensitive matching
    is the responsibility of ResponseParser, not the input contract here)."""
    bad_state = dict(valid_state_dict)
    bad_state["current_phase"] = "etwt"

    with pytest.raises(ValueError):
        parser.parse(bad_state)


# =========================================================================
# Validation: queue length contract (Requirement 3 AC 4)
# =========================================================================


def test_parse_negative_queue_length_raises_value_error(
    parser: ObservationParser,
    valid_state_dict: dict,
) -> None:
    bad_state = dict(valid_state_dict)
    bad_state["lane_vehicle_count"] = {"road_E_in_0": -1}

    with pytest.raises(ValueError) as exc_info:
        parser.parse(bad_state)

    msg = str(exc_info.value)
    assert "lane_vehicle_count" in msg
    assert "road_E_in_0" in msg


def test_parse_non_int_queue_length_raises_value_error(
    parser: ObservationParser,
    valid_state_dict: dict,
) -> None:
    bad_state = dict(valid_state_dict)
    bad_state["lane_vehicle_count"] = {"road_E_in_0": 2.5}

    with pytest.raises(ValueError):
        parser.parse(bad_state)


def test_parse_bool_queue_length_raises_value_error(
    parser: ObservationParser,
    valid_state_dict: dict,
) -> None:
    """``bool`` is a subclass of ``int`` in Python; the parser must reject
    it explicitly so ``True``/``False`` don't slip through as 1/0."""
    bad_state = dict(valid_state_dict)
    bad_state["lane_vehicle_count"] = {"road_E_in_0": True}

    with pytest.raises(ValueError):
        parser.parse(bad_state)


def test_parse_no_upper_bound_on_queue_length(
    parser: ObservationParser,
    valid_state_dict: dict,
) -> None:
    """Requirement 3 AC 4: NO fixed upper bound on queue length."""
    bad_state = dict(valid_state_dict)
    bad_state["lane_vehicle_count"] = {"road_E_in_0": 10_000_000}

    prompt = parser.parse(bad_state)

    assert "- road_E_in_0: 10000000" in prompt


def test_parse_zero_queue_length_is_valid(
    parser: ObservationParser,
    valid_state_dict: dict,
) -> None:
    bad_state = dict(valid_state_dict)
    bad_state["lane_vehicle_count"] = {"road_E_in_0": 0}

    prompt = parser.parse(bad_state)

    assert "- road_E_in_0: 0" in prompt


# =========================================================================
# Validation: current_phase_time contract
# =========================================================================


def test_parse_negative_phase_time_raises_value_error(
    parser: ObservationParser,
    valid_state_dict: dict,
) -> None:
    bad_state = dict(valid_state_dict)
    bad_state["current_phase_time"] = -3

    with pytest.raises(ValueError) as exc_info:
        parser.parse(bad_state)

    assert "current_phase_time" in str(exc_info.value)


def test_parse_non_int_phase_time_raises_value_error(
    parser: ObservationParser,
    valid_state_dict: dict,
) -> None:
    bad_state = dict(valid_state_dict)
    bad_state["current_phase_time"] = 12.7

    with pytest.raises(ValueError) as exc_info:
        parser.parse(bad_state)

    assert "current_phase_time" in str(exc_info.value)
