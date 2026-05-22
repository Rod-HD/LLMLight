"""Unit tests for ``src.training.trajectory_collector`` (Task 10.1).

Validates Requirement 6 AC 7-8 — Trajectory Collector for LLM teacher
data with phase-aware thresholds (50 / 200 valid samples).

Strategy:

* Engine, ObservationParser, and MultiBackendAPIClient are mocked (no
  CityFlow build, no real HTTP calls).
* ResponseParser is REAL — validates that the regex-based ``<signal>``
  detection in :func:`TrajectoryCollector._is_response_valid` agrees
  with the production parser.
* Phase-index resolver is injected as a lambda — production uses
  ``PhaseIndexMapper.get_index`` but Task 1.5 already covers that.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

# Make ``src`` importable when running ``pytest`` from project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.observation_parser import ObservationParser  # noqa: E402
from src.response_parser import ResponseParser  # noqa: E402
from src.training.multi_backend_api_client import (  # noqa: E402
    APIBackend,
    APIResponse,
    RequestLimitExceeded,
)
from src.training.trajectory_collector import (  # noqa: E402
    InsufficientDataError,
    TrajectoryCollector,
)


# =========================================================================
# Helpers
# =========================================================================


def _api_response(content: str) -> APIResponse:
    """Build an APIResponse with the given content (from a fake backend)."""
    return APIResponse(
        content=content,
        input_tokens=10,
        output_tokens=5,
        backend=APIBackend.OPENAI,
    )


def _signal(phase: str) -> str:
    """Wrap a phase name in a ``<signal>`` tag."""
    return f"<signal>{phase}</signal>"


class _FakeEngine:
    """Minimal stand-in for ``CityFlowEngine`` used in collect() loops.

    * ``get_lane_vehicle_count()`` returns the same dict every call (fine
      for testing collector control flow — ObservationParser is fully
      deterministic over identical state).
    * ``set_phase(intersection_id, phase_index)`` records the call but
      does not touch any real simulation state.
    * ``next_step()`` is a no-op counter — collector falls back to it
      when ObservationParser rejects state or set_phase fails.
    """

    def __init__(
        self,
        lane_counts: dict[str, int] | None = None,
    ) -> None:
        self.lane_counts = (
            dict(lane_counts) if lane_counts else {"lane_a": 1, "lane_b": 0}
        )
        self.set_phase_calls: list[tuple[str, int]] = []
        self.next_step_calls: int = 0

    def get_lane_vehicle_count(self) -> dict[str, int]:
        return dict(self.lane_counts)

    def set_phase(self, intersection_id: str, phase_index: int) -> None:
        self.set_phase_calls.append((intersection_id, phase_index))

    def next_step(self) -> None:
        self.next_step_calls += 1


class _ScriptedAPIClient:
    """Mimics MultiBackendAPIClient.chat_completion with a scripted queue.

    Items in ``responses`` may be:

    * a ``str``  → wrapped into ``APIResponse(content=...)``.
    * an ``APIResponse`` → returned as-is.
    * an ``Exception`` instance → raised on that call.

    Tracks ``call_count`` and ``last_prompt`` for assertions.
    """

    def __init__(self, responses: list[Any]) -> None:
        self._responses = list(responses)
        self.call_count = 0
        self.last_prompt: str | None = None

    def chat_completion(self, prompt: str) -> APIResponse:
        self.call_count += 1
        self.last_prompt = prompt
        if not self._responses:
            # Default: fall back to a valid signal so loops don't stall
            # if the test forgets to enqueue more responses.
            return _api_response(_signal("ETWT"))
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, str):
            return _api_response(item)
        if isinstance(item, APIResponse):
            return item
        raise TypeError(
            f"_ScriptedAPIClient: unsupported response type "
            f"{type(item).__name__}"
        )


# =========================================================================
# Constructor — phase validation
# =========================================================================


class TestInit:
    """``__init__`` validates ``phase`` and locks thresholds."""

    @pytest.mark.parametrize(
        ("phase", "expected_min", "expected_max"),
        [
            (1, 50, 100),
            (2, 200, 500),
            (3, 200, 500),
        ],
    )
    def test_valid_phase_sets_thresholds(
        self,
        phase: int,
        expected_min: int,
        expected_max: int,
    ) -> None:
        collector = TrajectoryCollector(
            api_client=MagicMock(),
            obs_parser=MagicMock(spec=ObservationParser),
            resp_parser=ResponseParser(),
            phase=phase,
        )
        assert collector.phase == phase
        assert collector.min_valid_samples == expected_min
        assert collector.max_requests == expected_max

    @pytest.mark.parametrize("invalid_phase", [0, 4, -1, 99])
    def test_invalid_phase_raises_value_error(self, invalid_phase: int) -> None:
        with pytest.raises(ValueError, match=r"phase must be 1, 2, or 3"):
            TrajectoryCollector(
                api_client=MagicMock(),
                obs_parser=MagicMock(spec=ObservationParser),
                resp_parser=ResponseParser(),
                phase=invalid_phase,
            )

    @pytest.mark.parametrize("bad_type", [1.0, "1", None, True, False])
    def test_non_int_phase_raises_value_error(self, bad_type: Any) -> None:
        with pytest.raises(ValueError, match=r"phase must be int"):
            TrajectoryCollector(
                api_client=MagicMock(),
                obs_parser=MagicMock(spec=ObservationParser),
                resp_parser=ResponseParser(),
                phase=bad_type,
            )


# =========================================================================
# Happy path — collect returns list[(prompt, response)]
# =========================================================================


class TestCollectHappyPath:
    """Mock engine + scripted API → collector returns valid pairs."""

    def test_returns_pairs_phase1(self) -> None:
        # 50 valid responses; collector should stop at the requested
        # ``num_samples`` (= 50) without hitting max_requests.
        engine = _FakeEngine()
        api_client = _ScriptedAPIClient(
            [_signal("ETWT")] * 60  # extra cushion
        )
        collector = TrajectoryCollector(
            api_client=api_client,
            obs_parser=ObservationParser(),
            resp_parser=ResponseParser(),
            phase=1,
        )

        samples = collector.collect(engine, num_samples=50)

        assert len(samples) == 50
        assert all(isinstance(p, str) and p for p, _ in samples)
        assert all(_signal("ETWT") == r for _, r in samples)
        # Each successful pair caused one API request → ≤ 50 calls + 0 skips.
        assert api_client.call_count == 50
        # Engine.set_phase called once per accepted sample.
        assert len(engine.set_phase_calls) == 50

    def test_pairs_use_real_observation_parser_output(self) -> None:
        engine = _FakeEngine(lane_counts={"lane_x": 5})
        api_client = _ScriptedAPIClient([_signal("NTST")] * 60)
        obs = ObservationParser()
        collector = TrajectoryCollector(
            api_client=api_client,
            obs_parser=obs,
            resp_parser=ResponseParser(),
            phase=1,
        )

        samples = collector.collect(engine, num_samples=50)

        # Prompt of first sample must equal what the parser produces for
        # the initial state (current_phase=ETWT, current_phase_time=0).
        expected_first_prompt = obs.parse(
            {
                "lane_vehicle_count": {"lane_x": 5},
                "current_phase": "ETWT",
                "current_phase_time": 0,
            }
        )
        assert samples[0][0] == expected_first_prompt
        assert "## Observation" in samples[0][0]
        assert "## Output Format" in samples[0][0]

    def test_phase_index_resolver_called_per_sample(self) -> None:
        engine = _FakeEngine()
        api_client = _ScriptedAPIClient([_signal("ELWL")] * 60)
        # Custom resolver that uses a non-default mapping. If the collector
        # truly delegates to it, set_phase will receive index 7 for ELWL.
        custom_mapping = {"ETWT": 5, "NTST": 6, "ELWL": 7, "NLSL": 8}

        def resolver(_intersection_id: str, phase_name: str) -> int:
            return custom_mapping[phase_name]

        collector = TrajectoryCollector(
            api_client=api_client,
            obs_parser=ObservationParser(),
            resp_parser=ResponseParser(),
            phase=1,
        )
        samples = collector.collect(
            engine,
            num_samples=50,
            intersection_id="int_42",
            phase_index_resolver=resolver,
        )

        assert len(samples) == 50
        # Every set_phase call routed ELWL → 7 via custom resolver.
        assert all(idx == 7 for _, idx in engine.set_phase_calls)
        assert all(iid == "int_42" for iid, _ in engine.set_phase_calls)


# =========================================================================
# Insufficient data — phase 1 / phase 2
# =========================================================================


class TestInsufficientData:
    """Collector raises ``InsufficientDataError`` when below threshold."""

    def test_phase1_below_50_raises(self) -> None:
        # All responses are invalid (no <signal> tag) → no valid samples.
        # Budget is exhausted (100 requests) and we have 0 < 50 → raise.
        engine = _FakeEngine()
        api_client = _ScriptedAPIClient(
            ["I don't know"] * TrajectoryCollector.MAX_REQUESTS_PHASE1
        )
        collector = TrajectoryCollector(
            api_client=api_client,
            obs_parser=ObservationParser(),
            resp_parser=ResponseParser(),
            phase=1,
        )

        with pytest.raises(InsufficientDataError) as exc_info:
            collector.collect(engine, num_samples=50)

        msg = str(exc_info.value)
        assert "phase=1" in msg
        assert ">= 50" in msg
        assert "0 valid samples" in msg
        # All 100 requests were spent.
        assert api_client.call_count == TrajectoryCollector.MAX_REQUESTS_PHASE1

    def test_phase2_below_200_raises(self) -> None:
        # Pre-populate 199 valid responses + flood with invalid → after
        # 500 requests we have 199 < 200 → raise. Collector must NOT use
        # phase-1 threshold of 50.
        valid = [_signal("ETWT")] * 199
        invalid = ["garbage"] * (TrajectoryCollector.MAX_REQUESTS_FULL - 199)
        engine = _FakeEngine()
        api_client = _ScriptedAPIClient(valid + invalid)
        collector = TrajectoryCollector(
            api_client=api_client,
            obs_parser=ObservationParser(),
            resp_parser=ResponseParser(),
            phase=2,
        )

        with pytest.raises(InsufficientDataError) as exc_info:
            collector.collect(engine, num_samples=200)

        msg = str(exc_info.value)
        assert "phase=2" in msg
        assert ">= 200" in msg
        assert "199 valid samples" in msg

    def test_phase3_below_200_raises(self) -> None:
        engine = _FakeEngine()
        api_client = _ScriptedAPIClient(
            ["nope"] * TrajectoryCollector.MAX_REQUESTS_FULL
        )
        collector = TrajectoryCollector(
            api_client=api_client,
            obs_parser=ObservationParser(),
            resp_parser=ResponseParser(),
            phase=3,
        )
        with pytest.raises(InsufficientDataError, match="phase=3"):
            collector.collect(engine, num_samples=200)

    def test_num_samples_below_threshold_raises_value_error(self) -> None:
        """num_samples must be >= phase threshold (else collect always fails)."""
        engine = _FakeEngine()
        api_client = _ScriptedAPIClient([])
        collector = TrajectoryCollector(
            api_client=api_client,
            obs_parser=ObservationParser(),
            resp_parser=ResponseParser(),
            phase=1,
        )
        with pytest.raises(ValueError, match="num_samples"):
            collector.collect(engine, num_samples=10)


# =========================================================================
# Invalid response handling
# =========================================================================


class TestInvalidResponseSkipping:
    """Responses without ``<signal>`` are skipped, not counted."""

    def test_no_signal_tag_skipped(self) -> None:
        # First 50 are missing the tag; next 50 are valid. Collector
        # must consume both groups → 100 requests, 50 valid samples.
        responses = (
            ["plain text without tag"] * 50 + [_signal("ETWT")] * 50
        )
        engine = _FakeEngine()
        api_client = _ScriptedAPIClient(responses)
        collector = TrajectoryCollector(
            api_client=api_client,
            obs_parser=ObservationParser(),
            resp_parser=ResponseParser(),
            phase=1,
        )

        samples = collector.collect(engine, num_samples=50)

        assert len(samples) == 50
        assert api_client.call_count == 100
        # All saved samples are the valid ones (with the tag).
        assert all(_signal("ETWT") == r for _, r in samples)

    def test_signal_with_invalid_phase_value_skipped(self) -> None:
        # ``<signal>FOO</signal>`` is malformed (FOO ∉ VALID_PHASES) →
        # skipped; ``<signal>NLSL</signal>`` is valid.
        responses = ["<signal>FOO</signal>"] * 50 + [_signal("NLSL")] * 50
        engine = _FakeEngine()
        api_client = _ScriptedAPIClient(responses)
        collector = TrajectoryCollector(
            api_client=api_client,
            obs_parser=ObservationParser(),
            resp_parser=ResponseParser(),
            phase=1,
        )

        samples = collector.collect(engine, num_samples=50)

        assert len(samples) == 50
        assert all(_signal("NLSL") == r for _, r in samples)

    def test_empty_response_skipped(self) -> None:
        # Empty content (e.g. API timeout fallback) must NOT count.
        responses = [""] * 50 + [_signal("ETWT")] * 50
        engine = _FakeEngine()
        api_client = _ScriptedAPIClient(responses)
        collector = TrajectoryCollector(
            api_client=api_client,
            obs_parser=ObservationParser(),
            resp_parser=ResponseParser(),
            phase=1,
        )
        samples = collector.collect(engine, num_samples=50)
        assert len(samples) == 50


# =========================================================================
# Request budget exhaustion
# =========================================================================


class TestBudgetExhaustion:
    """Collector stops at ``max_requests`` even if threshold not reached."""

    def test_request_limit_exceeded_propagates_break(self) -> None:
        """Puter quota hit (RequestLimitExceeded) ends the loop early."""
        # 30 valid responses, then RequestLimitExceeded. Phase 1 needs ≥50
        # → InsufficientDataError, but ``request_count`` reflects only
        # the 30 successful calls (the failed call is not counted because
        # the exception is raised before counting).
        responses: list[Any] = [_signal("ETWT")] * 30 + [
            RequestLimitExceeded("Puter quota hit"),
        ]
        engine = _FakeEngine()
        api_client = _ScriptedAPIClient(responses)
        collector = TrajectoryCollector(
            api_client=api_client,
            obs_parser=ObservationParser(),
            resp_parser=ResponseParser(),
            phase=1,
        )

        with pytest.raises(InsufficientDataError) as exc_info:
            collector.collect(engine, num_samples=50)

        msg = str(exc_info.value)
        assert "30 valid samples" in msg
        assert ">= 50" in msg

    def test_max_requests_phase1_caps_at_100(self) -> None:
        # Mixed 30% valid / 70% invalid → at 100 requests, ~30 valid.
        # Should stop at exactly 100 calls and raise InsufficientDataError.
        responses: list[Any] = []
        for _ in range(50):
            responses.extend(
                [_signal("ETWT")] * 1 + ["nope"] * 9
            )  # 10% valid
        # → 5 valid in first 50 calls, 10 valid in 100 calls. Below 50.
        engine = _FakeEngine()
        api_client = _ScriptedAPIClient(responses)
        collector = TrajectoryCollector(
            api_client=api_client,
            obs_parser=ObservationParser(),
            resp_parser=ResponseParser(),
            phase=1,
        )

        with pytest.raises(InsufficientDataError):
            collector.collect(engine, num_samples=50)

        # Hard cap = MAX_REQUESTS_PHASE1 = 100.
        assert api_client.call_count == TrajectoryCollector.MAX_REQUESTS_PHASE1


# =========================================================================
# Logging progress
# =========================================================================


class TestLogging:
    """Progress is logged at the configured cadence (every 10 samples)."""

    def test_progress_logged_every_10_valid_samples(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        engine = _FakeEngine()
        api_client = _ScriptedAPIClient([_signal("ETWT")] * 60)
        collector = TrajectoryCollector(
            api_client=api_client,
            obs_parser=ObservationParser(),
            resp_parser=ResponseParser(),
            phase=1,
        )

        with caplog.at_level(
            logging.INFO, logger="src.training.trajectory_collector"
        ):
            collector.collect(engine, num_samples=50)

        progress_msgs = [
            rec.message
            for rec in caplog.records
            if "collected" in rec.message and "valid samples" in rec.message
        ]
        # We expect progress at 10, 20, 30, 40, 50 → 5 messages.
        assert len(progress_msgs) == 5
        # First message reports 10/50, last reports 50/50.
        assert "collected 10/50" in progress_msgs[0]
        assert "collected 50/50" in progress_msgs[-1]

    def test_initial_log_includes_phase_and_thresholds(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        engine = _FakeEngine()
        api_client = _ScriptedAPIClient([_signal("ETWT")] * 60)
        collector = TrajectoryCollector(
            api_client=api_client,
            obs_parser=ObservationParser(),
            resp_parser=ResponseParser(),
            phase=2,
        )
        with caplog.at_level(
            logging.INFO, logger="src.training.trajectory_collector"
        ):
            try:
                collector.collect(engine, num_samples=200)
            except InsufficientDataError:
                pass  # Not enough scripted responses; we only need the
                # initial log line for this assertion.

        init_msgs = [
            rec.message
            for rec in caplog.records
            if "phase=2" in rec.message and "min_valid=200" in rec.message
        ]
        assert init_msgs, "expected initial info log with phase + thresholds"


# =========================================================================
# Integration: real ResponseParser agrees with collector's regex check
# =========================================================================


class TestResponseValidityAgreement:
    """``_is_response_valid`` must agree with ``ResponseParser`` semantics
    on the parts that matter for IFT data: a response with a valid
    ``<signal>`` tag is *both* valid (collector keeps it) AND parsed to
    the corresponding phase (downstream simulation advances)."""

    @pytest.mark.parametrize(
        "phase",
        ["ETWT", "NTST", "ELWL", "NLSL"],
    )
    def test_valid_signal_keeps_sample_and_advances_simulation(
        self, phase: str
    ) -> None:
        engine = _FakeEngine()
        api_client = _ScriptedAPIClient([_signal(phase)] * 60)
        collector = TrajectoryCollector(
            api_client=api_client,
            obs_parser=ObservationParser(),
            resp_parser=ResponseParser(),
            phase=1,
        )
        samples = collector.collect(engine, num_samples=50)
        assert len(samples) == 50
        # The injected default resolver maps ETWT=0 / NTST=1 / ELWL=2 / NLSL=3.
        expected_index = {"ETWT": 0, "NTST": 1, "ELWL": 2, "NLSL": 3}[phase]
        assert all(
            idx == expected_index for _, idx in engine.set_phase_calls
        )

    def test_signal_is_case_insensitive_and_whitespace_robust(self) -> None:
        # Collector treats ``<signal> etwt \n</signal>`` as valid because
        # the regex + strip + uppercase check matches ETWT.
        responses = ["<signal>  etwt  \n</signal>"] * 60
        engine = _FakeEngine()
        api_client = _ScriptedAPIClient(responses)
        collector = TrajectoryCollector(
            api_client=api_client,
            obs_parser=ObservationParser(),
            resp_parser=ResponseParser(),
            phase=1,
        )
        samples = collector.collect(engine, num_samples=50)
        assert len(samples) == 50
