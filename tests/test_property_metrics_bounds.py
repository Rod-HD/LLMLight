"""Property-based tests for ``MetricsEvaluator`` bounds (Property 6).

# Feature: llmlight-reproduction, Property 6: Metrics Bounds

**Validates: Requirements 9.4, 12.6**

Spec (design.md §"Property 6 — Metrics Bounds"):

    ∀ travel_times, queues, wait_times generated for total_timesteps T ∈ {250, 3600}:
        compute_att(travel_times) ∈ [0, T]                        (when input non-empty)
        compute_awt(wait_times)   ∈ [0, T]
        compute_aql(queues)       ∈ [0, max_lane_capacity_observed]

In words: for any well-formed simulation data targeting either Demo
mode (``total_timesteps = 250``) or Full mode (``total_timesteps =
3600``), the three metrics must respect their per-mode bounds. The
bound for AQL is **per-lane** (the maximum queue observed on any
single lane at any single timestep), NOT the total vehicle count —
this is the central correctness invariant for AQL.

Strategy
--------
Generators are constrained to the input space the evaluator legally
accepts:

* ``vehicle_travel_times``: ``list[float]`` with values in
  ``[0, T]``. We deliberately bound the upper end at ``T`` because
  any value ``> T`` would itself be a corrupt input (the evaluator
  raises on it — see ``MetricsEvaluator.compute_att`` "exceeds
  total_timesteps"). The empty list is excluded for ``compute_att``
  because it has a documented fallback to ``T`` (Req 9 AC 5) which
  trivially satisfies the bound but isn't the property we're testing.
* ``lane_queues_per_step``: variable-length list of dicts mapping
  ASCII alphanumeric lane ids to non-negative integer queue counts.
* ``vehicle_wait_times``: same shape as travel times.
* ``total_timesteps``: sampled from ``{250, 3600}`` (the only values
  ``MetricsEvaluator.__init__`` accepts).

Settings: ``max_examples=100, deadline=None``.
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

from src.metrics_evaluator import MetricsEvaluator  # noqa: E402

# =========================================================================
# Strategies
# =========================================================================

#: Two valid total_timesteps values the evaluator accepts.
TOTAL_TIMESTEPS_VALUES: tuple[int, ...] = (250, 3600)

#: Lane id alphabet — ASCII alphanumeric + underscore (CityFlow lane id
#: convention, e.g. ``road_E_in_0``). Keeps the focus on bounds rather
#: than encoding edge cases.
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

# Per-lane queue count: bounded at 1000 to keep generators reasonable
# while still exercising the upper end of plausible CityFlow lane queues.
_queue_count_strategy = st.integers(min_value=0, max_value=1_000)


def _travel_times_strategy(total_timesteps: int) -> st.SearchStrategy[list[float]]:
    """Travel times in ``[0, total_timesteps]``. ``min_size=1`` because
    the empty-list case has a documented fallback (Req 9 AC 5) and isn't
    the bound property we're testing.
    """
    return st.lists(
        st.floats(
            min_value=0.0,
            max_value=float(total_timesteps),
            allow_nan=False,
            allow_infinity=False,
        ),
        min_size=1,
        max_size=200,
    )


def _wait_times_strategy(total_timesteps: int) -> st.SearchStrategy[list[float]]:
    """Wait times in ``[0, total_timesteps]``. ``min_size=0`` allowed
    (empty → 0.0, still in bound)."""
    return st.lists(
        st.floats(
            min_value=0.0,
            max_value=float(total_timesteps),
            allow_nan=False,
            allow_infinity=False,
        ),
        min_size=0,
        max_size=200,
    )


def _lane_queues_strategy() -> st.SearchStrategy[list[dict[str, int]]]:
    """List of per-timestep lane → queue dicts.

    Outer length up to 60 timesteps × inner up to 12 lanes is enough to
    cover Jinan 1 (12 intersections × ~12 lanes) without blowing up
    Hypothesis runtime.
    """
    return st.lists(
        st.dictionaries(
            keys=_lane_id_strategy,
            values=_queue_count_strategy,
            min_size=0,
            max_size=12,
        ),
        min_size=0,
        max_size=60,
    )


# =========================================================================
# Property 6.A — ATT bounds
# =========================================================================


@given(
    total_timesteps=st.sampled_from(TOTAL_TIMESTEPS_VALUES),
    data=st.data(),
)
@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_property_att_bounded_by_total_timesteps(
    total_timesteps: int, data: st.DataObject
) -> None:
    """**Validates: Requirements 9.4, 12.6**

    For any non-empty travel time list whose values lie in
    ``[0, total_timesteps]``, ``compute_att`` returns a value in
    ``[0, total_timesteps]``.
    """
    travel_times = data.draw(_travel_times_strategy(total_timesteps))
    evaluator = MetricsEvaluator(total_timesteps=total_timesteps)

    att = evaluator.compute_att(travel_times)

    assert 0.0 <= att <= float(total_timesteps), (
        f"ATT bound violated: att={att}, total_timesteps={total_timesteps}, "
        f"travel_times[:5]={travel_times[:5]!r} (n={len(travel_times)})"
    )


# =========================================================================
# Property 6.B — AWT bounds
# =========================================================================


@given(
    total_timesteps=st.sampled_from(TOTAL_TIMESTEPS_VALUES),
    data=st.data(),
)
@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_property_awt_bounded_by_total_timesteps(
    total_timesteps: int, data: st.DataObject
) -> None:
    """**Validates: Requirements 9.4, 12.6**

    For any wait time list (including empty) whose values lie in
    ``[0, total_timesteps]``, ``compute_awt`` returns a value in
    ``[0, total_timesteps]``.
    """
    wait_times = data.draw(_wait_times_strategy(total_timesteps))
    evaluator = MetricsEvaluator(total_timesteps=total_timesteps)

    awt = evaluator.compute_awt(wait_times)

    assert 0.0 <= awt <= float(total_timesteps), (
        f"AWT bound violated: awt={awt}, total_timesteps={total_timesteps}, "
        f"wait_times[:5]={wait_times[:5]!r} (n={len(wait_times)})"
    )


# =========================================================================
# Property 6.C — AQL bounds (per-lane, NOT total vehicles)
# =========================================================================


@given(
    total_timesteps=st.sampled_from(TOTAL_TIMESTEPS_VALUES),
    lane_queues_per_step=_lane_queues_strategy(),
)
@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_property_aql_bounded_by_max_lane_capacity_observed(
    total_timesteps: int, lane_queues_per_step: list[dict[str, int]]
) -> None:
    """**Validates: Requirements 9.4, 12.6**

    AQL is the per-lane mean queue length across all (lane, timestep)
    cells. Therefore ``0 ≤ AQL ≤ max_lane_capacity_observed``, where
    ``max_lane_capacity_observed`` is the maximum queue count appearing
    in any single (lane, timestep) entry — NOT the total number of
    vehicles in the network.
    """
    evaluator = MetricsEvaluator(total_timesteps=total_timesteps)

    # Compute the actual upper bound from the input data, mirroring the
    # invariant in ``MetricsEvaluator.compute_aql``.
    observed_max = 0
    for step in lane_queues_per_step:
        for queue in step.values():
            if queue > observed_max:
                observed_max = queue

    aql = evaluator.compute_aql(lane_queues_per_step)

    assert aql >= 0.0, f"AQL must be non-negative; got {aql}"
    # Allow tiny float epsilon from rounding to 2 decimals (e.g. mean
    # 5.005 rounds up to 5.01 while the underlying max is 5).
    assert aql <= float(observed_max) + 1e-9, (
        f"AQL bound violated: aql={aql}, max_lane_capacity_observed="
        f"{observed_max}, num_steps={len(lane_queues_per_step)}"
    )


# =========================================================================
# Property 6.D — combined evaluate() respects all bounds simultaneously
# =========================================================================


@given(
    total_timesteps=st.sampled_from(TOTAL_TIMESTEPS_VALUES),
    data=st.data(),
)
@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_property_metrics_bounds_combined(
    total_timesteps: int, data: st.DataObject
) -> None:
    """**Validates: Requirements 9.4, 12.6**

    When all three data sources are provided simultaneously,
    ``evaluate()`` returns a ``MetricsResult`` whose three fields each
    respect their respective bounds. This is the universal form of
    Property 6 stated in design.md.
    """
    travel_times = data.draw(_travel_times_strategy(total_timesteps))
    wait_times = data.draw(_wait_times_strategy(total_timesteps))
    lane_queues_per_step = data.draw(_lane_queues_strategy())

    observed_max = 0
    for step in lane_queues_per_step:
        for queue in step.values():
            if queue > observed_max:
                observed_max = queue

    evaluator = MetricsEvaluator(total_timesteps=total_timesteps)
    result = evaluator.evaluate(
        vehicle_travel_times=travel_times,
        lane_queues_per_step=lane_queues_per_step,
        vehicle_wait_times=wait_times,
    )

    assert 0.0 <= result.att <= float(total_timesteps), (
        f"ATT bound violated in evaluate(): att={result.att}, "
        f"total_timesteps={total_timesteps}"
    )
    assert 0.0 <= result.awt <= float(total_timesteps), (
        f"AWT bound violated in evaluate(): awt={result.awt}, "
        f"total_timesteps={total_timesteps}"
    )
    assert result.aql >= 0.0, (
        f"AQL must be non-negative; got {result.aql}"
    )
    assert result.aql <= float(observed_max) + 1e-9, (
        f"AQL bound violated in evaluate(): aql={result.aql}, "
        f"max_lane_capacity_observed={observed_max}"
    )
