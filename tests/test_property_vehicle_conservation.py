"""Property-Based Test: Vehicle Conservation Invariant (Property 5).

**Validates: Requirements 12.5**

For every randomly-generated simulation run on the ``CityFlowEngine``
wrapper, the following invariant MUST hold at every sampled timestep
``N``::

    engine.get_vehicle_count() == (
        engine.get_vehicles_spawned_total()
        - engine.get_vehicles_completed_total()
    )

This is the conservation law that validates the wrapper's spawn/complete
tracking logic (Component 1 in the design document).

Implementation note
-------------------
The real CityFlow binding is not available on the Windows dev box (it is
built from source via ``scripts/setup_env.sh`` in WSL2). This test uses a
mock CityFlow engine — analogous to the ``FakeCityFlowEngine`` in
``tests/test_cityflow_engine.py`` — parameterised by hypothesis. The mock
generates deterministic-but-arbitrary vehicle id sequences from a seeded
``random.Random`` to simulate spawn/complete patterns. The wrapper's
tracking logic is what we are validating: the invariant must hold for ALL
inputs.

Strategies
----------
- ``seed``: ``st.integers(0, 2**32 - 1)``
- ``num_steps``: ``st.integers(10, 200)``
- ``spawn_rate``: ``st.floats(0.0, 1.0)``
- ``complete_rate``: ``st.floats(0.0, 1.0)``

Settings: ``max_examples=100, deadline=None``.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cityflow_engine import CityFlowEngine  # noqa: E402


# ---------------------------------------------------------------------------
# Mock CityFlow engine
# ---------------------------------------------------------------------------


class FakeCityFlowEngine:
    """Mock that mimics the subset of ``cityflow.Engine`` API used by the
    wrapper for vehicle tracking.

    Parameterised by a ``vehicle_script``: an ordered list of vehicle id
    sets, one entry per timestep (index 0 = pre any ``next_step()``).
    The cursor advances on each ``next_step()`` call and is clamped to
    the last entry once exhausted.
    """

    def __init__(
        self,
        config_path: str,
        thread_num: int = 1,
        seed: int | None = None,
        *,
        vehicle_script: list[set[str]],
    ) -> None:
        self.config_path = config_path
        self.thread_num = thread_num
        self.seed = seed
        self._cursor = 0
        self._vehicle_script = vehicle_script

    # -- API mirroring cityflow.Engine ------------------------------------

    def next_step(self) -> None:
        if self._cursor < len(self._vehicle_script) - 1:
            self._cursor += 1

    def set_tl_phase(self, intersection_id: str, phase_index: int) -> None:
        # Not exercised by this property test, but required by the wrapper
        # contract for completeness.
        pass

    def get_lane_vehicle_count(self) -> dict[str, int]:
        return {}

    def get_vehicles(self, include_waiting: bool = True) -> list[str]:
        return list(self._vehicle_script[self._cursor])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_minimal_config(tmp_path: Path) -> Path:
    """Write a minimal CityFlow config JSON sufficient for wrapper init."""
    cfg = {
        "interval": 1.0,
        "seed": 0,
        "dir": str(tmp_path) + "/",
        "roadnetFile": "roadnet.json",
        "flowFile": "flow.json",
        "rlTrafficLight": True,
        "saveReplay": False,
        "roadnetLogFile": "frontend/web/roadnetLogFile.json",
        "replayLogFile": "frontend/web/replayLogFile.txt",
        "laneChange": False,
    }
    p = tmp_path / "config.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    return p


def _generate_vehicle_script(
    seed: int,
    num_steps: int,
    spawn_rate: float,
    complete_rate: float,
) -> list[set[str]]:
    """Generate a deterministic but arbitrary vehicle id sequence.

    For each timestep ``t`` ∈ ``[1, num_steps]`` (with ``t=0`` being the
    empty initial state):

    - With probability ``spawn_rate``, spawn 1..3 new vehicles with
      unique fresh ids (``v0``, ``v1``, …).
    - With probability ``complete_rate``, complete 1..min(3, |current|)
      currently-active vehicles (chosen by the seeded RNG).

    Returns:
        A list of length ``num_steps + 1`` where ``script[0] = set()``
        is the initial state and ``script[t]`` is the active vehicle id
        set after timestep ``t``. The set semantics intentionally mirror
        what ``cityflow.Engine.get_vehicles()`` returns.
    """
    rng = random.Random(seed)
    script: list[set[str]] = [set()]
    next_id = 0
    current: set[str] = set()

    for _ in range(num_steps):
        # Spawn phase.
        if rng.random() < spawn_rate:
            n_spawn = rng.randint(1, 3)
            for _ in range(n_spawn):
                current.add(f"v{next_id}")
                next_id += 1
        # Complete phase.
        if current and rng.random() < complete_rate:
            n_complete = rng.randint(1, min(3, len(current)))
            # ``rng.sample`` requires a sequence; sort for determinism.
            to_remove = rng.sample(sorted(current), n_complete)
            for vid in to_remove:
                current.discard(vid)
        script.append(set(current))

    return script


# ---------------------------------------------------------------------------
# Module-scoped config fixture (avoid recreating tmp dir for each example)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def shared_config(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Single CityFlow config JSON shared across all hypothesis examples."""
    tmp = tmp_path_factory.mktemp("vehicle_conservation")
    return _write_minimal_config(tmp)


# ---------------------------------------------------------------------------
# Property test
# ---------------------------------------------------------------------------


@given(
    seed=st.integers(min_value=0, max_value=2**32 - 1),
    num_steps=st.integers(min_value=10, max_value=200),
    spawn_rate=st.floats(
        min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
    ),
    complete_rate=st.floats(
        min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
    ),
)
@settings(
    max_examples=100,
    deadline=None,
    # ``shared_config`` is module-scoped; suppress just in case.
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_vehicle_conservation_invariant(
    shared_config: Path,
    seed: int,
    num_steps: int,
    spawn_rate: float,
    complete_rate: float,
) -> None:
    """**Validates: Requirements 12.5**

    For a randomly-generated vehicle script driven by ``seed``,
    ``num_steps``, ``spawn_rate``, and ``complete_rate``, the wrapper
    must satisfy::

        get_vehicle_count() == get_vehicles_spawned_total()
                              - get_vehicles_completed_total()

    at every timestep ``N`` from ``t=0`` through ``t=num_steps``.
    """
    script = _generate_vehicle_script(
        seed=seed,
        num_steps=num_steps,
        spawn_rate=spawn_rate,
        complete_rate=complete_rate,
    )

    def factory(*args: Any, **kwargs: Any) -> FakeCityFlowEngine:
        return FakeCityFlowEngine(*args, **kwargs, vehicle_script=script)

    # CityFlowEngine.seed must fit Python's int (any int OK); use seed
    # directly. The wrapper only stores and forwards it.
    engine = CityFlowEngine(
        str(shared_config),
        seed=seed,
        engine_factory=factory,
    )

    # Invariant at t=0 (initial state, no next_step called yet).
    cur = engine.get_vehicle_count()
    spawned = engine.get_vehicles_spawned_total()
    completed = engine.get_vehicles_completed_total()
    assert cur == spawned - completed, (
        f"Invariant violated at t=0: current={cur}, spawned={spawned}, "
        f"completed={completed}, diff={spawned - completed}"
    )
    assert cur >= 0
    assert spawned >= 0
    assert completed >= 0
    assert spawned >= completed

    # Sample EVERY timestep N ∈ [1, num_steps] — strongest form of the
    # property (covers the multi-timestep sampling requirement).
    for n in range(1, num_steps + 1):
        engine.next_step()

        cur = engine.get_vehicle_count()
        spawned = engine.get_vehicles_spawned_total()
        completed = engine.get_vehicles_completed_total()

        assert cur == spawned - completed, (
            f"Invariant violated at t={n}: current={cur}, "
            f"spawned={spawned}, completed={completed}, "
            f"diff={spawned - completed}, "
            f"params(seed={seed}, num_steps={num_steps}, "
            f"spawn_rate={spawn_rate}, complete_rate={complete_rate})"
        )
        # Conservation also implies these non-negativity bounds.
        assert cur >= 0, f"current count negative at t={n}"
        assert spawned >= 0, f"spawned total negative at t={n}"
        assert completed >= 0, f"completed total negative at t={n}"
        assert spawned >= completed, (
            f"spawned ({spawned}) < completed ({completed}) at t={n}"
        )
