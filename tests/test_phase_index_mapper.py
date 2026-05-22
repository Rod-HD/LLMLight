"""Unit tests for ``src.phase_index_mapper``.

Validates:
    - Requirement 2.5 (build mapping ``phase_name → phase_index`` per intersection)
    - Requirement 2.8 (KeyError on missing/invalid phase or unknown intersection)
    - Requirement 10.6 (phase set fixed at {ETWT, NTST, ELWL, NLSL})
    - Spec rationale: phase index order in roadnet.json is NOT fixed across
      datasets (Jinan 1 may set ``ETWT = 0`` while Hangzhou 1 may set
      ``ETWT = 2``).

Tests use synthetic CityFlow-style roadnet JSON files written to ``tmp_path``
because real Jinan/Hangzhou datasets are not bundled with this repo (data
lives in the LLMTSCS clone). The synthetic roadnets reproduce the relevant
schema fields: ``intersections[].id``, ``intersections[].virtual``,
``intersections[].point``, ``intersections[].trafficLight.lightphases``,
``intersections[].roadLinks``, and ``roads[].id`` / ``roads[].points``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.phase_index_mapper import PhaseIndexMapper  # noqa: E402


# =========================================================================
# Synthetic roadnet builder
# =========================================================================


def _make_road(road_id: str, *, from_x: float, from_y: float,
               to_x: float, to_y: float) -> dict:
    """Build a road dict where ``points[0]`` is far end and ``points[-1]`` is
    intersection end (per CityFlow convention)."""
    return {
        "id": road_id,
        "points": [
            {"x": from_x, "y": from_y},
            {"x": to_x, "y": to_y},
        ],
    }


def _make_link(start_road: str, end_road: str, *, link_type: str) -> dict:
    return {
        "startRoad": start_road,
        "endRoad": end_road,
        "type": link_type,
    }


def _make_phase(road_link_indices: list[int]) -> dict:
    return {"time": 30.0, "availableRoadLinks": road_link_indices}


def _build_standard_intersection(
    *,
    inter_id: str = "intersection_1_1",
    phase_order: tuple[str, str, str, str] = ("ETWT", "NTST", "ELWL", "NLSL"),
) -> dict:
    """Build a 4-approach intersection (E/W/N/S) at origin with the four
    canonical phases. ``phase_order`` controls the ORDER of phases in
    ``lightphases``, which is exactly the spec rationale we are validating
    (different datasets use different orders).
    """
    # Roads: incoming from each compass direction toward the intersection
    # (intersection center at origin).
    #   East approach  → from (+100, 0) toward (0, 0)
    #   West approach  → from (-100, 0) toward (0, 0)
    #   North approach → from (0, +100) toward (0, 0)
    #   South approach → from (0, -100) toward (0, 0)
    # For outgoing edges (endRoad) we keep their geometry identical (we only
    # use startRoad geometry to classify).
    roads = [
        _make_road("road_E_in", from_x=100, from_y=0, to_x=0, to_y=0),
        _make_road("road_W_in", from_x=-100, from_y=0, to_x=0, to_y=0),
        _make_road("road_N_in", from_x=0, from_y=100, to_x=0, to_y=0),
        _make_road("road_S_in", from_x=0, from_y=-100, to_x=0, to_y=0),
        # Outgoing roads (geometry irrelevant for classification).
        _make_road("road_E_out", from_x=0, from_y=0, to_x=100, to_y=0),
        _make_road("road_W_out", from_x=0, from_y=0, to_x=-100, to_y=0),
        _make_road("road_N_out", from_x=0, from_y=0, to_x=0, to_y=100),
        _make_road("road_S_out", from_x=0, from_y=0, to_x=0, to_y=-100),
    ]

    # Road links indexed in this fixed order:
    #   0: E approach → W out (straight)
    #   1: W approach → E out (straight)
    #   2: N approach → S out (straight)
    #   3: S approach → N out (straight)
    #   4: E approach → S out (left)
    #   5: W approach → N out (left)
    #   6: N approach → E out (left)
    #   7: S approach → W out (left)
    #   8: E approach → N out (right)
    #   9: W approach → S out (right)
    #  10: N approach → W out (right)
    #  11: S approach → E out (right)
    road_links = [
        _make_link("road_E_in", "road_W_out", link_type="go_straight"),  # 0
        _make_link("road_W_in", "road_E_out", link_type="go_straight"),  # 1
        _make_link("road_N_in", "road_S_out", link_type="go_straight"),  # 2
        _make_link("road_S_in", "road_N_out", link_type="go_straight"),  # 3
        _make_link("road_E_in", "road_S_out", link_type="turn_left"),    # 4
        _make_link("road_W_in", "road_N_out", link_type="turn_left"),    # 5
        _make_link("road_N_in", "road_E_out", link_type="turn_left"),    # 6
        _make_link("road_S_in", "road_W_out", link_type="turn_left"),    # 7
        _make_link("road_E_in", "road_N_out", link_type="turn_right"),   # 8
        _make_link("road_W_in", "road_S_out", link_type="turn_right"),   # 9
        _make_link("road_N_in", "road_W_out", link_type="turn_right"),   # 10
        _make_link("road_S_in", "road_E_out", link_type="turn_right"),   # 11
    ]

    # Phase definitions by name.
    phase_links_by_name = {
        # ETWT: East-West Through (straight on E and W)
        "ETWT": [0, 1, 8, 9, 10, 11],  # include right-on always; ignored
        # NTST: North-South Through
        "NTST": [2, 3, 8, 9, 10, 11],
        # ELWL: East-West Left
        "ELWL": [4, 5, 8, 9, 10, 11],
        # NLSL: North-South Left
        "NLSL": [6, 7, 8, 9, 10, 11],
    }

    # Build lightphases in the requested order.
    light_phases = [_make_phase(phase_links_by_name[name]) for name in phase_order]

    intersection = {
        "id": inter_id,
        "point": {"x": 0.0, "y": 0.0},
        "virtual": False,
        "roads": ["road_E_in", "road_W_in", "road_N_in", "road_S_in",
                  "road_E_out", "road_W_out", "road_N_out", "road_S_out"],
        "roadLinks": road_links,
        "trafficLight": {
            "roadLinkIndices": list(range(len(road_links))),
            "lightphases": light_phases,
        },
    }
    return {
        "intersections": [intersection],
        "roads": roads,
    }


def _build_multi_intersection_roadnet() -> dict:
    """Build a 2-intersection roadnet where each intersection has a
    DIFFERENT phase order in ``lightphases``. This validates the core
    spec rationale: phase index is NOT fixed across datasets/intersections.
    """
    inter1 = _build_standard_intersection(
        inter_id="inter_A",
        phase_order=("ETWT", "NTST", "ELWL", "NLSL"),  # ETWT = 0
    )
    inter2 = _build_standard_intersection(
        inter_id="inter_B",
        phase_order=("ELWL", "NLSL", "ETWT", "NTST"),  # ETWT = 2
    )
    # Rename each one's roads to avoid id collisions.
    def _prefix(roadnet: dict, prefix: str) -> dict:
        rename: dict[str, str] = {}
        for road in roadnet["roads"]:
            old = road["id"]
            new = f"{prefix}_{old}"
            rename[old] = new
            road["id"] = new
        for inter in roadnet["intersections"]:
            inter["roads"] = [rename[r] for r in inter["roads"]]
            for link in inter["roadLinks"]:
                link["startRoad"] = rename[link["startRoad"]]
                link["endRoad"] = rename[link["endRoad"]]
        return roadnet

    inter1 = _prefix(inter1, "A")
    inter2 = _prefix(inter2, "B")

    return {
        "intersections": inter1["intersections"] + inter2["intersections"],
        "roads": inter1["roads"] + inter2["roads"],
    }


def _write_roadnet(tmp_path: Path, data: dict, name: str = "roadnet.json") -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


# =========================================================================
# Tests
# =========================================================================


class TestPhaseIndexMapperBasic:
    """Synthetic 4-phase roadnet, canonical order ETWT=0/NTST=1/ELWL=2/NLSL=3."""

    def test_resolves_all_four_phases_in_canonical_order(self, tmp_path: Path):
        data = _build_standard_intersection()
        path = _write_roadnet(tmp_path, data)
        mapper = PhaseIndexMapper(str(path))

        assert mapper.get_index("intersection_1_1", "ETWT") == 0
        assert mapper.get_index("intersection_1_1", "NTST") == 1
        assert mapper.get_index("intersection_1_1", "ELWL") == 2
        assert mapper.get_index("intersection_1_1", "NLSL") == 3

    def test_all_intersections_returns_resolved_ids(self, tmp_path: Path):
        data = _build_standard_intersection()
        path = _write_roadnet(tmp_path, data)
        mapper = PhaseIndexMapper(str(path))

        result = mapper.all_intersections()
        assert isinstance(result, list)
        assert result == ["intersection_1_1"]


class TestPhaseIndexMapperVariableOrder:
    """Validate spec rationale: phase index differs between intersections /
    datasets, so hard-coding is wrong."""

    def test_phase_index_differs_between_two_intersections(self, tmp_path: Path):
        data = _build_multi_intersection_roadnet()
        path = _write_roadnet(tmp_path, data)
        mapper = PhaseIndexMapper(str(path))

        # inter_A used canonical order  ETWT=0, NTST=1, ELWL=2, NLSL=3
        assert mapper.get_index("inter_A", "ETWT") == 0
        assert mapper.get_index("inter_A", "NTST") == 1
        assert mapper.get_index("inter_A", "ELWL") == 2
        assert mapper.get_index("inter_A", "NLSL") == 3

        # inter_B used  ELWL=0, NLSL=1, ETWT=2, NTST=3
        assert mapper.get_index("inter_B", "ELWL") == 0
        assert mapper.get_index("inter_B", "NLSL") == 1
        assert mapper.get_index("inter_B", "ETWT") == 2
        assert mapper.get_index("inter_B", "NTST") == 3

    def test_all_intersections_lists_both(self, tmp_path: Path):
        data = _build_multi_intersection_roadnet()
        path = _write_roadnet(tmp_path, data)
        mapper = PhaseIndexMapper(str(path))
        assert sorted(mapper.all_intersections()) == ["inter_A", "inter_B"]


class TestPhaseIndexMapperErrors:
    """KeyError on missing intersection / unknown phase / phase not present."""

    def test_keyerror_on_unknown_intersection(self, tmp_path: Path):
        data = _build_standard_intersection()
        path = _write_roadnet(tmp_path, data)
        mapper = PhaseIndexMapper(str(path))

        with pytest.raises(KeyError) as excinfo:
            mapper.get_index("does_not_exist", "ETWT")
        # KeyError repr uses repr() of the message, but content should still mention id.
        assert "does_not_exist" in str(excinfo.value)

    def test_keyerror_on_invalid_phase_name(self, tmp_path: Path):
        data = _build_standard_intersection()
        path = _write_roadnet(tmp_path, data)
        mapper = PhaseIndexMapper(str(path))

        with pytest.raises(KeyError) as excinfo:
            mapper.get_index("intersection_1_1", "GREEN")
        assert "GREEN" in str(excinfo.value)

    def test_keyerror_on_missing_phase_at_intersection(self, tmp_path: Path):
        # Build intersection with ONLY ETWT and NTST (missing ELWL/NLSL).
        data = _build_standard_intersection(
            phase_order=("ETWT", "NTST", "ETWT", "NTST")  # duplicates → only ETWT/NTST resolve
        )
        path = _write_roadnet(tmp_path, data)
        mapper = PhaseIndexMapper(str(path))

        with pytest.raises(KeyError) as excinfo:
            mapper.get_index("intersection_1_1", "ELWL")
        msg = str(excinfo.value)
        assert "ELWL" in msg
        assert "intersection_1_1" in msg

    def test_filenotfound_when_path_missing(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            PhaseIndexMapper(str(tmp_path / "missing.json"))

    def test_valueerror_on_invalid_json(self, tmp_path: Path):
        bad = tmp_path / "bad.json"
        bad.write_text("{ not json", encoding="utf-8")
        with pytest.raises(ValueError):
            PhaseIndexMapper(str(bad))


class TestPhaseIndexMapperEdgeCases:
    """Skips for virtual intersections + handles missing trafficLight."""

    def test_skips_virtual_intersection(self, tmp_path: Path):
        data = _build_standard_intersection(inter_id="real_inter")
        # Add a virtual intersection (boundary node).
        data["intersections"].append({
            "id": "virtual_inter",
            "point": {"x": 200.0, "y": 0.0},
            "virtual": True,
            "roads": [],
            "roadLinks": [],
        })
        path = _write_roadnet(tmp_path, data)
        mapper = PhaseIndexMapper(str(path))

        assert mapper.all_intersections() == ["real_inter"]
        with pytest.raises(KeyError):
            mapper.get_index("virtual_inter", "ETWT")

    def test_skips_intersection_without_traffic_light(self, tmp_path: Path):
        data = _build_standard_intersection(inter_id="signaled")
        data["intersections"].append({
            "id": "unsignalized",
            "point": {"x": 200.0, "y": 0.0},
            "virtual": False,
            "roads": [],
            "roadLinks": [],
            # No trafficLight key
        })
        path = _write_roadnet(tmp_path, data)
        mapper = PhaseIndexMapper(str(path))

        assert mapper.all_intersections() == ["signaled"]

    def test_first_matching_phase_wins_when_duplicates_exist(self, tmp_path: Path):
        # Phase ETWT appears at index 0 AND index 4. The mapper keeps the FIRST.
        data = _build_standard_intersection(
            phase_order=("ETWT", "NTST", "ELWL", "NLSL"),
        )
        # Append a duplicate ETWT phase at index 4.
        # availableRoadLinks for ETWT = [0, 1, 8, 9, 10, 11]
        data["intersections"][0]["trafficLight"]["lightphases"].append({
            "time": 30.0,
            "availableRoadLinks": [0, 1, 8, 9, 10, 11],
        })
        path = _write_roadnet(tmp_path, data)
        mapper = PhaseIndexMapper(str(path))

        assert mapper.get_index("intersection_1_1", "ETWT") == 0  # not 4
