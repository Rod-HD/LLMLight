"""Phase Index Mapper (Component 2).

Parse ``roadnet.json`` của CityFlow và xây dựng mapping
``intersection_id → {phase_name → phase_index}`` nơi ``phase_name`` thuộc
``{ETWT, NTST, ELWL, NLSL}``.

Trong CityFlow, thứ tự các phase trong ``trafficLight.lightphases``
KHÔNG cố định giữa các dataset (ví dụ Jinan 1 có thể đặt ``ETWT = 0``
trong khi Hangzhou 1 có thể đặt ``ETWT = 2``), nên việc hard-code
``phase_index`` sẽ gây hành vi sai. Component này resolve mapping ngay
tại ``__init__`` (cache, không re-parse mỗi timestep).

Algorithm
---------
1. Đọc ``roadnet.json`` (đã được CityFlow build).
2. Với mỗi ``intersection`` không phải ``virtual`` và có ``trafficLight``:

   a. Tính ``approach_direction`` (E/W/N/S) cho mỗi ``roadLink`` từ
      hình học của ``startRoad`` (vector ``first_point - last_point``;
      trục có |Δ| lớn hơn xác định compass).
   b. Mỗi ``roadLink`` được biểu diễn bởi ``(approach_direction, type)``
      với ``type ∈ {go_straight, turn_left, turn_right}``.
   c. Với mỗi phase ``i`` trong ``lightphases``: lấy danh sách
      ``availableRoadLinks`` → tập hợp các ``(approach_direction, type)``,
      bỏ qua ``turn_right`` (thường được bật trong tất cả pha).
   d. Phân loại phase:

      * ``ETWT`` ↔ tất cả movement còn lại đều là ``(E|W, go_straight)``
      * ``NTST`` ↔ tất cả movement còn lại đều là ``(N|S, go_straight)``
      * ``ELWL`` ↔ tất cả movement còn lại đều là ``(E|W, turn_left)``
      * ``NLSL`` ↔ tất cả movement còn lại đều là ``(N|S, turn_left)``
      * Khác → bỏ qua (ví dụ all-red, yellow, hoặc không phân loại được).

3. Lưu mapping dạng ``intersection_id → {phase_name → first_matching_index}``.

_Requirements_: 2.5, 2.8, 10.6
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Final

logger = logging.getLogger(__name__)


class PhaseIndexMapper:
    """Map ``phase_name`` (ETWT/NTST/ELWL/NLSL) → ``phase_index`` cho mỗi
    intersection của một dataset CityFlow."""

    VALID_PHASES: Final[tuple[str, ...]] = ("ETWT", "NTST", "ELWL", "NLSL")

    # Type literals dùng trong CityFlow roadnet schema.
    _TYPE_STRAIGHT: Final[str] = "go_straight"
    _TYPE_LEFT: Final[str] = "turn_left"
    _TYPE_RIGHT: Final[str] = "turn_right"

    def __init__(self, roadnet_json_path: str) -> None:
        """Parse roadnet và build mapping ngay tại ``__init__``.

        Args:
            roadnet_json_path: Đường dẫn tuyệt đối tới ``roadnet.json``
                của dataset CityFlow.

        Raises:
            FileNotFoundError: Nếu file không tồn tại.
            ValueError: Nếu file không phải JSON hợp lệ.
        """
        path = Path(roadnet_json_path)
        if not path.exists():
            raise FileNotFoundError(
                f"roadnet.json not found: {roadnet_json_path}"
            )
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON in roadnet file {roadnet_json_path}: {exc}"
            ) from exc

        self._roadnet_path = roadnet_json_path
        self._mapping: dict[str, dict[str, int]] = {}
        self._build_mapping(data)

        if not self._mapping:
            logger.warning(
                "PhaseIndexMapper: no controllable intersections found in %s",
                roadnet_json_path,
            )

    # ------------------------------------------------------------ Public --

    def get_index(self, intersection_id: str, phase_name: str) -> int:
        """Trả về ``phase_index`` (CityFlow) cho ``phase_name`` của
        ``intersection_id``.

        Raises:
            KeyError: Nếu ``intersection_id`` không có mapping, hoặc
                ``phase_name`` không thuộc ``VALID_PHASES``, hoặc
                ``phase_name`` chưa được resolve cho intersection đó.
        """
        if intersection_id not in self._mapping:
            available = sorted(self._mapping.keys())
            preview = available[:5]
            tail = "..." if len(available) > 5 else ""
            raise KeyError(
                f"Intersection {intersection_id!r} not found in roadnet "
                f"{self._roadnet_path!r}. "
                f"Available intersections (first 5): {preview}{tail}"
            )

        if phase_name not in self.VALID_PHASES:
            raise KeyError(
                f"Phase name {phase_name!r} not in valid set "
                f"{self.VALID_PHASES}"
            )

        inter_map = self._mapping[intersection_id]
        if phase_name not in inter_map:
            raise KeyError(
                f"Phase {phase_name!r} missing for intersection "
                f"{intersection_id!r} in roadnet {self._roadnet_path!r}. "
                f"Resolved phases at this intersection: "
                f"{sorted(inter_map.keys())}"
            )
        return inter_map[phase_name]

    def all_intersections(self) -> list[str]:
        """Trả về danh sách ``intersection_id`` đã được resolve mapping."""
        return list(self._mapping.keys())

    # ----------------------------------------------------------- Internal --

    def _build_mapping(self, data: dict) -> None:
        """Duyệt mọi intersection của roadnet và resolve phase mapping."""
        roads = data.get("roads", []) or []
        road_by_id: dict[str, dict] = {
            r["id"]: r for r in roads if isinstance(r, dict) and "id" in r
        }

        intersections = data.get("intersections", []) or []
        for inter in intersections:
            if not isinstance(inter, dict):
                continue
            inter_id = inter.get("id")
            if not inter_id:
                continue
            # CityFlow đánh dấu các nút biên (không có đèn) là virtual.
            if inter.get("virtual", False):
                continue
            traffic_light = inter.get("trafficLight")
            if not isinstance(traffic_light, dict):
                continue

            phase_to_idx = self._map_intersection(inter, road_by_id)
            if phase_to_idx:
                self._mapping[inter_id] = phase_to_idx
            else:
                logger.warning(
                    "PhaseIndexMapper: intersection %r had no recognizable "
                    "ETWT/NTST/ELWL/NLSL phase; skipping.",
                    inter_id,
                )

    def _map_intersection(
        self, inter: dict, road_by_id: dict[str, dict]
    ) -> dict[str, int]:
        """Resolve mapping cho MỘT intersection."""
        traffic_light = inter["trafficLight"]
        light_phases = traffic_light.get("lightphases", []) or []
        road_links = inter.get("roadLinks", []) or []

        # Tính (approach_dir, type) cho mỗi roadLink. None → roadLink
        # invalid (thiếu startRoad hoặc road không tồn tại).
        link_signatures: list[tuple[str, str] | None] = []
        for link in road_links:
            if not isinstance(link, dict):
                link_signatures.append(None)
                continue
            start_road_id = link.get("startRoad")
            link_type = link.get("type")
            if not start_road_id or not link_type:
                link_signatures.append(None)
                continue
            road = road_by_id.get(start_road_id)
            if not road:
                link_signatures.append(None)
                continue
            approach_dir = self._compute_approach_direction(road)
            if approach_dir is None:
                link_signatures.append(None)
                continue
            link_signatures.append((approach_dir, link_type))

        phase_to_idx: dict[str, int] = {}
        for idx, phase in enumerate(light_phases):
            if not isinstance(phase, dict):
                continue
            available = phase.get("availableRoadLinks", []) or []
            sigs: list[tuple[str, str]] = []
            for li in available:
                if not isinstance(li, int):
                    continue
                if 0 <= li < len(link_signatures):
                    sig = link_signatures[li]
                    if sig is not None:
                        sigs.append(sig)

            # Bỏ qua turn_right (CityFlow thường bật right-on-red mọi pha;
            # right turns không phân biệt ETWT/NTST/ELWL/NLSL).
            relevant = [s for s in sigs if s[1] != self._TYPE_RIGHT]
            if not relevant:
                continue

            phase_name = self._classify_phase(relevant)
            if phase_name is None:
                continue
            # Pha đầu tiên match thắng (các pha trùng phía sau bị bỏ qua).
            if phase_name not in phase_to_idx:
                phase_to_idx[phase_name] = idx

        return phase_to_idx

    @staticmethod
    def _compute_approach_direction(road: dict) -> str | None:
        """Tính compass direction (E/W/N/S) của ``road`` khi đi vào
        intersection.

        Quy ước:
          - ``road.points[0]`` là điểm xa intersection (start).
          - ``road.points[-1]`` là điểm tại intersection (end).
          - Vector ``approach_from = first_point - last_point`` chỉ về
            phía road đến (compass direction).

        Trả về ``None`` nếu road có < 2 điểm hoặc điểm thiếu trường
        ``x``/``y``.
        """
        points = road.get("points", []) or []
        if len(points) < 2:
            return None
        first = points[0]
        last = points[-1]
        if not isinstance(first, dict) or not isinstance(last, dict):
            return None
        try:
            dx = float(first["x"]) - float(last["x"])
            dy = float(first["y"]) - float(last["y"])
        except (KeyError, TypeError, ValueError):
            return None

        # Cùng điểm hoặc gần như cùng điểm — không xác định được hướng.
        if dx == 0.0 and dy == 0.0:
            return None

        if abs(dx) >= abs(dy):
            return "E" if dx > 0 else "W"
        return "N" if dy > 0 else "S"

    @classmethod
    def _classify_phase(
        cls, movements: list[tuple[str, str]]
    ) -> str | None:
        """Phân loại phase từ tập (approach_direction, type).

        Args:
            movements: List các ``(approach_dir, type)`` của các roadLink
                được phép trong phase, đã LOẠI ``turn_right``.

        Returns:
            ``"ETWT"``, ``"NTST"``, ``"ELWL"``, ``"NLSL"``, hoặc ``None``
            nếu không khớp với 4 pha chuẩn của LLMLight.
        """
        if not movements:
            return None

        dirs = {m[0] for m in movements}
        types = {m[1] for m in movements}

        # Chỉ chấp nhận pha thuần (chỉ một loại type, chỉ một trục).
        if types == {cls._TYPE_STRAIGHT}:
            if dirs and dirs.issubset({"E", "W"}):
                return "ETWT"
            if dirs and dirs.issubset({"N", "S"}):
                return "NTST"
            return None
        if types == {cls._TYPE_LEFT}:
            if dirs and dirs.issubset({"E", "W"}):
                return "ELWL"
            if dirs and dirs.issubset({"N", "S"}):
                return "NLSL"
            return None
        return None
