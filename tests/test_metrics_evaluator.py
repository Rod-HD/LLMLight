"""Unit tests for ``src.metrics_evaluator``.

Validates:
    - Requirement 9.1: ATT = mean travel time của xe đã hoàn thành, round 2.
    - Requirement 9.2: AQL = trung bình per-lane qua tất cả lane và timesteps
      (KHÔNG phải tổng số xe).
    - Requirement 9.3: AWT = mean cumulative wait time (speed < 0.1 m/s).
    - Requirement 9.4: ATT/AWT ∈ [0, total_timesteps]; AQL ∈
      [0, max_lane_capacity_observed].
    - Requirement 9.5: empty completed vehicles → ATT = total_timesteps + log
      warning.
    - Requirement 9.6: comparison table với mean ± std, columns Method | ATT
      | AQL | AWT | Runs | Backend.
    - Requirement 12.6: bound validation enforced.

Property test cho Property 6 (Metrics Bounds, ≥100 hypothesis iterations) sống
ở subtask 8.2 trong file riêng — file này CHỈ chứa unit tests điểm/edge case.
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

from src.metrics_evaluator import MetricsEvaluator  # noqa: E402
from src.sim_config import MetricsResult  # noqa: E402


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def evaluator_demo() -> MetricsEvaluator:
    """Demo mode evaluator (250 timesteps)."""
    return MetricsEvaluator(total_timesteps=250)


@pytest.fixture
def evaluator_full() -> MetricsEvaluator:
    """Full mode evaluator (3600 timesteps)."""
    return MetricsEvaluator(total_timesteps=3600)


# =========================================================================
# __init__ — total_timesteps validation
# =========================================================================


class TestInit:
    """Validates Requirement 10 AC 5 / Req 9 AC 4 (total_timesteps domain)."""

    def test_accepts_250(self):
        ev = MetricsEvaluator(total_timesteps=250)
        assert ev.total_timesteps == 250

    def test_accepts_3600(self):
        ev = MetricsEvaluator(total_timesteps=3600)
        assert ev.total_timesteps == 3600

    @pytest.mark.parametrize("bad", [0, 1, 100, 251, 3599, 3601, 7200, -1])
    def test_rejects_other_ints(self, bad):
        with pytest.raises(ValueError, match="total_timesteps"):
            MetricsEvaluator(total_timesteps=bad)

    def test_rejects_float(self):
        with pytest.raises(ValueError, match="total_timesteps"):
            MetricsEvaluator(total_timesteps=250.0)  # type: ignore[arg-type]

    def test_rejects_bool(self):
        # bool là subclass int, phải bị reject để tránh silent bug.
        with pytest.raises(ValueError, match="total_timesteps"):
            MetricsEvaluator(total_timesteps=True)  # type: ignore[arg-type]

    def test_rejects_string(self):
        with pytest.raises(ValueError, match="total_timesteps"):
            MetricsEvaluator(total_timesteps="250")  # type: ignore[arg-type]

    def test_speed_threshold_class_constant(self):
        # Requirement 9 AC 2/3: 0.1 m/s.
        assert MetricsEvaluator.SPEED_THRESHOLD == 0.1


# =========================================================================
# compute_att
# =========================================================================


class TestComputeATT:
    """Validates Requirement 9 AC 1 + AC 4 + AC 5."""

    def test_simple_mean(self, evaluator_demo):
        # mean([10, 20, 30]) = 20.0
        assert evaluator_demo.compute_att([10.0, 20.0, 30.0]) == 20.0

    def test_rounds_to_two_decimals(self, evaluator_demo):
        # mean([1, 2, 3, 4, 5, 6, 7]) = 28/7 = 4.0 — pick a non-trivial case.
        # mean([1, 2]) = 1.5 → rounds to 1.5 (already 1 decimal).
        # Use a mean that needs rounding: [10, 20, 31] -> 20.333... -> 20.33.
        assert evaluator_demo.compute_att([10.0, 20.0, 31.0]) == 20.33

    def test_single_completed_vehicle(self, evaluator_demo):
        assert evaluator_demo.compute_att([42.5]) == 42.5

    def test_empty_returns_total_timesteps_demo(self, evaluator_demo, caplog):
        """Requirement 9 AC 5: empty → fallback to total_timesteps + warning."""
        with caplog.at_level(logging.WARNING, logger="src.metrics_evaluator"):
            att = evaluator_demo.compute_att([])
        assert att == 250.0
        assert isinstance(att, float)
        # Verify warning logged with the fallback value.
        warnings = [
            r for r in caplog.records if r.levelno == logging.WARNING
        ]
        assert any("250" in r.getMessage() for r in warnings), (
            f"expected warning mentioning 250; got {[r.getMessage() for r in warnings]}"
        )

    def test_empty_returns_total_timesteps_full(self, evaluator_full, caplog):
        with caplog.at_level(logging.WARNING, logger="src.metrics_evaluator"):
            att = evaluator_full.compute_att([])
        assert att == 3600.0

    def test_int_inputs_accepted(self, evaluator_demo):
        # ints are valid (will be coerced to float).
        assert evaluator_demo.compute_att([10, 20, 30]) == 20.0

    def test_zero_travel_times(self, evaluator_demo):
        # Edge case: all 0 — mean 0, in bound.
        assert evaluator_demo.compute_att([0.0, 0.0, 0.0]) == 0.0

    def test_at_upper_bound(self, evaluator_demo):
        # ATT exactly == total_timesteps (250) is allowed.
        assert evaluator_demo.compute_att([250.0]) == 250.0

    def test_negative_value_rejected(self, evaluator_demo):
        with pytest.raises(ValueError, match=">= 0"):
            evaluator_demo.compute_att([10.0, -5.0])

    def test_bool_rejected(self, evaluator_demo):
        with pytest.raises(ValueError, match="bool"):
            evaluator_demo.compute_att([True, False])  # type: ignore[list-item]

    def test_non_numeric_rejected(self, evaluator_demo):
        with pytest.raises(ValueError, match="number"):
            evaluator_demo.compute_att(["10", "20"])  # type: ignore[list-item]

    def test_nan_rejected(self, evaluator_demo):
        with pytest.raises(ValueError, match="finite"):
            evaluator_demo.compute_att([float("nan"), 10.0])

    def test_inf_rejected(self, evaluator_demo):
        with pytest.raises(ValueError, match="finite"):
            evaluator_demo.compute_att([float("inf")])

    def test_exceeds_bound_raises(self, evaluator_demo):
        # Single value > total_timesteps means caller has corrupt data.
        with pytest.raises(ValueError, match="exceeds total_timesteps"):
            evaluator_demo.compute_att([300.0])

    def test_exceeds_bound_full(self, evaluator_full):
        with pytest.raises(ValueError, match="exceeds total_timesteps"):
            evaluator_full.compute_att([3601.0])


# =========================================================================
# compute_aql
# =========================================================================


class TestComputeAQL:
    """Validates Requirement 9 AC 2 + AC 4."""

    def test_per_lane_mean(self, evaluator_demo):
        # 2 timesteps × 2 lanes = 4 (lane, timestep) cells.
        # Sum queues = 1 + 3 + 2 + 4 = 10. Cells = 4. AQL = 2.5.
        steps = [
            {"lane_a": 1, "lane_b": 3},
            {"lane_a": 2, "lane_b": 4},
        ]
        assert evaluator_demo.compute_aql(steps) == 2.5

    def test_aql_is_per_lane_not_total_vehicles(self, evaluator_demo):
        # 1 timestep × 4 lanes, queues [1, 1, 1, 1].
        # Per-lane AQL = mean = 1.0 (NOT 4 = total vehicles).
        steps = [{"l1": 1, "l2": 1, "l3": 1, "l4": 1}]
        assert evaluator_demo.compute_aql(steps) == 1.0

    def test_empty_returns_zero(self, evaluator_demo):
        assert evaluator_demo.compute_aql([]) == 0.0

    def test_all_empty_step_dicts_returns_zero(self, evaluator_demo):
        assert evaluator_demo.compute_aql([{}, {}, {}]) == 0.0

    def test_rounds_two_decimals(self, evaluator_demo):
        # 3 lanes, 1 timestep, queues [1, 2, 4]. Mean = 7/3 = 2.333...
        steps = [{"a": 1, "b": 2, "c": 4}]
        assert evaluator_demo.compute_aql(steps) == 2.33

    def test_single_lane_single_step(self, evaluator_demo):
        assert evaluator_demo.compute_aql([{"lane_a": 7}]) == 7.0

    def test_zero_queues(self, evaluator_demo):
        steps = [{"a": 0, "b": 0}, {"a": 0, "b": 0}]
        assert evaluator_demo.compute_aql(steps) == 0.0

    def test_negative_queue_rejected(self, evaluator_demo):
        with pytest.raises(ValueError, match=">= 0"):
            evaluator_demo.compute_aql([{"lane_a": -1}])

    def test_float_queue_rejected(self, evaluator_demo):
        # queue counts are integer (number of vehicles per lane).
        with pytest.raises(ValueError, match="int"):
            evaluator_demo.compute_aql(
                [{"lane_a": 1.5}]  # type: ignore[dict-item]
            )

    def test_bool_queue_rejected(self, evaluator_demo):
        with pytest.raises(ValueError, match="int"):
            evaluator_demo.compute_aql(
                [{"lane_a": True}]  # type: ignore[dict-item]
            )

    def test_non_dict_step_rejected(self, evaluator_demo):
        with pytest.raises(ValueError, match="dict"):
            evaluator_demo.compute_aql(
                [["lane_a", 1]]  # type: ignore[list-item]
            )

    def test_non_list_input_rejected(self, evaluator_demo):
        with pytest.raises(ValueError, match="list/tuple"):
            evaluator_demo.compute_aql(
                {"step_0": {"lane_a": 1}}  # type: ignore[arg-type]
            )

    def test_aql_bound_max_lane_capacity_observed(self, evaluator_demo):
        # AQL must be ≤ max queue observed in any (lane, step) pair.
        # Test 5 timesteps × 2 lanes; max queue = 50; AQL ≤ 50.
        steps = [
            {"lane_a": 5, "lane_b": 50},
            {"lane_a": 0, "lane_b": 10},
            {"lane_a": 1, "lane_b": 1},
            {"lane_a": 0, "lane_b": 0},
            {"lane_a": 0, "lane_b": 0},
        ]
        aql = evaluator_demo.compute_aql(steps)
        # Sum = 67. Cells = 10. Mean = 6.7.
        assert aql == 6.7
        # Bound: ≤ 50 (max observed).
        assert aql <= 50


# =========================================================================
# compute_awt
# =========================================================================


class TestComputeAWT:
    """Validates Requirement 9 AC 3 + AC 4."""

    def test_simple_mean(self, evaluator_demo):
        assert evaluator_demo.compute_awt([5.0, 10.0, 15.0]) == 10.0

    def test_rounds_two_decimals(self, evaluator_demo):
        # mean([1, 2, 4]) = 7/3 = 2.333... → 2.33
        assert evaluator_demo.compute_awt([1.0, 2.0, 4.0]) == 2.33

    def test_empty_returns_zero(self, evaluator_demo):
        # AWT = 0 (no vehicles waited) is meaningful; not a fallback case.
        assert evaluator_demo.compute_awt([]) == 0.0

    def test_zero_wait_times(self, evaluator_demo):
        # Edge: vehicles existed but never below speed threshold.
        assert evaluator_demo.compute_awt([0.0, 0.0, 0.0]) == 0.0

    def test_at_upper_bound(self, evaluator_demo):
        # AWT == total_timesteps is allowed (vehicle waits the entire sim).
        assert evaluator_demo.compute_awt([250.0]) == 250.0

    def test_negative_rejected(self, evaluator_demo):
        with pytest.raises(ValueError, match=">= 0"):
            evaluator_demo.compute_awt([5.0, -1.0])

    def test_bool_rejected(self, evaluator_demo):
        with pytest.raises(ValueError, match="bool"):
            evaluator_demo.compute_awt([True])  # type: ignore[list-item]

    def test_exceeds_bound_raises(self, evaluator_demo):
        with pytest.raises(ValueError, match="exceeds total_timesteps"):
            evaluator_demo.compute_awt([260.0])

    def test_int_inputs_accepted(self, evaluator_full):
        assert evaluator_full.compute_awt([100, 200, 300]) == 200.0


# =========================================================================
# evaluate
# =========================================================================


class TestEvaluate:
    """Validates Requirement 9 AC 1-3 (composition into MetricsResult)."""

    def test_kwargs_path_returns_metrics_result(self, evaluator_demo):
        result = evaluator_demo.evaluate(
            vehicle_travel_times=[10.0, 20.0, 30.0],
            lane_queues_per_step=[{"l1": 1, "l2": 2}],
            vehicle_wait_times=[2.0, 4.0, 6.0],
        )
        assert isinstance(result, MetricsResult)
        assert result.att == 20.0
        assert result.aql == 1.5
        assert result.awt == 4.0

    def test_engine_path_uses_get_average_travel_time(self, evaluator_demo):
        class FakeEngine:
            def get_average_travel_time(self):
                return 42.0

        result = evaluator_demo.evaluate(
            engine=FakeEngine(),
            lane_queues_per_step=[{"l1": 0}],
            vehicle_wait_times=[],
        )
        assert result.att == 42.0
        assert result.aql == 0.0
        assert result.awt == 0.0

    def test_engine_no_travel_time_falls_back(self, evaluator_demo, caplog):
        class FakeEngine:
            pass

        with caplog.at_level(logging.WARNING, logger="src.metrics_evaluator"):
            result = evaluator_demo.evaluate(engine=FakeEngine())
        # Falls back to total_timesteps when nothing usable.
        assert result.att == 250.0
        assert result.aql == 0.0
        assert result.awt == 0.0

    def test_no_data_no_engine_warns_and_returns_default(
        self, evaluator_demo, caplog
    ):
        with caplog.at_level(logging.WARNING, logger="src.metrics_evaluator"):
            result = evaluator_demo.evaluate()
        assert result.att == 250.0
        assert result.aql == 0.0
        assert result.awt == 0.0


# =========================================================================
# generate_comparison_table
# =========================================================================


class TestGenerateComparisonTable:
    """Validates Requirement 9 AC 6 + comparison table format."""

    def test_columns_present_in_header(self, evaluator_demo):
        table = evaluator_demo.generate_comparison_table(
            [("maxpressure", MetricsResult(att=120.0, aql=2.5, awt=15.0))]
        )
        first_line = table.splitlines()[0]
        # All 6 required columns must be present.
        for col in [
            "Method",
            "ATT (s)",
            "AQL (vehicles/lane)",
            "AWT (s)",
            "Runs",
            "Backend",
        ]:
            assert col in first_line, f"Column {col!r} missing from header: {first_line}"

    def test_single_run_zero_std(self, evaluator_demo):
        table = evaluator_demo.generate_comparison_table(
            [("maxpressure", MetricsResult(att=100.0, aql=2.0, awt=10.0))]
        )
        lines = table.splitlines()
        # 3 lines: header + separator + 1 data row.
        assert len(lines) == 3
        data_row = lines[2]
        # Single run → std = 0.00.
        assert "100.00 ± 0.00" in data_row
        assert "2.00 ± 0.00" in data_row
        assert "10.00 ± 0.00" in data_row
        # Runs column = 1; backend = "-" for maxpressure.
        assert "| 1 |" in data_row
        assert "| - |" in data_row

    def test_multiple_runs_groups_and_computes_mean_std(self, evaluator_demo):
        # 3 runs with att = 100, 110, 120 → mean=110, sample-std≈10.0.
        results = [
            ("lightgpt_hf", MetricsResult(att=100.0, aql=2.0, awt=5.0)),
            ("lightgpt_hf", MetricsResult(att=110.0, aql=3.0, awt=10.0)),
            ("lightgpt_hf", MetricsResult(att=120.0, aql=4.0, awt=15.0)),
        ]
        table = evaluator_demo.generate_comparison_table(results)
        lines = table.splitlines()
        assert len(lines) == 3  # header + sep + 1 grouped row.
        data_row = lines[2]
        # mean(100, 110, 120) = 110.00; sample stdev = 10.00.
        assert "110.00 ± 10.00" in data_row
        # AQL mean(2, 3, 4) = 3.00; sample stdev = 1.00.
        assert "3.00 ± 1.00" in data_row
        # AWT mean(5, 10, 15) = 10.00; sample stdev = 5.00.
        assert "10.00 ± 5.00" in data_row
        # Runs = 3; backend = "local" for lightgpt_*.
        assert "| 3 |" in data_row
        assert "| local |" in data_row

    def test_multiple_methods_preserve_first_seen_order(self, evaluator_demo):
        results = [
            ("gpt4o_groq", MetricsResult(att=200.0, aql=5.0, awt=20.0)),
            ("maxpressure", MetricsResult(att=180.0, aql=4.0, awt=18.0)),
            ("gpt4o_groq", MetricsResult(att=210.0, aql=5.5, awt=22.0)),
        ]
        table = evaluator_demo.generate_comparison_table(results)
        lines = table.splitlines()
        # 4 lines: header + sep + 2 method rows.
        assert len(lines) == 4
        # First method row → gpt4o_groq (appeared first), second → maxpressure.
        assert "gpt4o_groq" in lines[2]
        assert "maxpressure" in lines[3]

    def test_backend_inference(self, evaluator_demo):
        results = [
            ("gpt4o_puter", MetricsResult(att=100.0, aql=1.0, awt=5.0)),
            ("gpt4o_groq", MetricsResult(att=100.0, aql=1.0, awt=5.0)),
            ("gpt4o_openai", MetricsResult(att=100.0, aql=1.0, awt=5.0)),
            ("lightgpt_hf", MetricsResult(att=100.0, aql=1.0, awt=5.0)),
            ("lightgpt_mine", MetricsResult(att=100.0, aql=1.0, awt=5.0)),
            ("maxpressure", MetricsResult(att=100.0, aql=1.0, awt=5.0)),
            ("advanced_maxpressure", MetricsResult(att=100.0, aql=1.0, awt=5.0)),
            ("advanced_colight", MetricsResult(att=100.0, aql=1.0, awt=5.0)),
        ]
        table = evaluator_demo.generate_comparison_table(results)
        # Find each row and check backend column matches expectation.
        rows_by_method = {
            line.split("|")[1].strip(): line
            for line in table.splitlines()
            if line.startswith("|") and not line.startswith("| ---")
            and not line.startswith("| Method")
        }
        # gpt4o_<X> → X
        assert "| puter |" in rows_by_method["gpt4o_puter"]
        assert "| groq |" in rows_by_method["gpt4o_groq"]
        assert "| openai |" in rows_by_method["gpt4o_openai"]
        # lightgpt_* → "local"
        assert "| local |" in rows_by_method["lightgpt_hf"]
        assert "| local |" in rows_by_method["lightgpt_mine"]
        # baselines → "-"
        assert "| - |" in rows_by_method["maxpressure"]
        assert "| - |" in rows_by_method["advanced_maxpressure"]
        assert "| - |" in rows_by_method["advanced_colight"]

    def test_empty_results_returns_header_only(self, evaluator_demo):
        table = evaluator_demo.generate_comparison_table([])
        lines = table.splitlines()
        assert len(lines) == 2  # header + separator only.

    def test_rejects_non_tuple_entry(self, evaluator_demo):
        with pytest.raises(ValueError, match="tuple"):
            evaluator_demo.generate_comparison_table(
                ["not a tuple"]  # type: ignore[list-item]
            )

    def test_rejects_non_metrics_value(self, evaluator_demo):
        with pytest.raises(ValueError, match="MetricsResult"):
            evaluator_demo.generate_comparison_table(
                [("method", {"att": 1.0})]  # type: ignore[list-item]
            )


# =========================================================================
# Bound validation invariants — Req 9 AC 4 / Req 12 AC 6
# =========================================================================


class TestBoundsInvariants:
    """Cross-checks that ATT/AWT ∈ [0, total_timesteps] and AQL ≥ 0."""

    def test_att_bound_demo(self, evaluator_demo):
        # ATT bounded at 250.
        att = evaluator_demo.compute_att([250.0, 200.0, 150.0])
        assert 0 <= att <= 250

    def test_att_bound_full(self, evaluator_full):
        att = evaluator_full.compute_att([3000.0, 3500.0, 3600.0])
        assert 0 <= att <= 3600

    def test_awt_bound_demo(self, evaluator_demo):
        awt = evaluator_demo.compute_awt([100.0, 150.0, 200.0, 249.99])
        assert 0 <= awt <= 250

    def test_awt_bound_full(self, evaluator_full):
        awt = evaluator_full.compute_awt([3599.0, 3600.0, 3500.0])
        assert 0 <= awt <= 3600

    def test_aql_nonneg(self, evaluator_demo):
        aql = evaluator_demo.compute_aql(
            [{"l1": 5, "l2": 10}, {"l1": 3, "l2": 6}]
        )
        assert aql >= 0

    def test_aql_bounded_by_max_lane_capacity(self, evaluator_full):
        # AQL ≤ max queue observed in any (lane, step).
        steps = [
            {"l1": 0, "l2": 100},
            {"l1": 50, "l2": 25},
        ]
        aql = evaluator_full.compute_aql(steps)
        max_capacity = 100
        assert 0 <= aql <= max_capacity
