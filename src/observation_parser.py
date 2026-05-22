"""Observation Parser (Component 3).

Chuyển đổi trạng thái CityFlow tại một timestep thành prompt text có cấu
trúc gồm 3 section (observation / instruction / output format) để
LLM agent đưa ra quyết định pha tín hiệu giao thông.

Thiết kế tuân thủ:

- Output xác định (deterministic, byte-for-byte cho cùng input). Để bảo
  đảm tính tất định, các lane được sort theo thứ tự **alphabetical** trước
  khi format — không phụ thuộc thứ tự insertion của ``dict``.
- Chấp nhận state ở 1 trong 2 dạng:

  * ``dict`` — thường do CityFlow Engine sản xuất tại runtime.
  * :class:`src.sim_config.IntersectionState` — dataclass đã được validate.

  Cả hai đều được normalize thành cùng cấu trúc nội bộ trước khi format
  để output không phụ thuộc kiểu input.

- Validate state đầu vào:

  * Field bắt buộc: ``lane_vehicle_count`` (dict[str, int]),
    ``current_phase`` (str ∈ ``VALID_PHASES``), ``current_phase_time``
    (int >= 0).
  * Queue length là ``int >= 0``; **KHÔNG** áp đặt giới hạn trên cố định
    (sức chứa lane phụ thuộc dataset — Requirement 3 AC 4).
  * ``ValueError`` chỉ rõ tên field bị thiếu (Requirement 3 AC 5) hoặc
    giá trị phase không hợp lệ (Requirement 3 AC 6).

- 3 section header dùng prefix ``## `` để Property 3 (Prompt Template
  Structure) có thể detect bằng substring matching:

  * ``## Observation``
  * ``## Instruction``
  * ``## Output Format``

_Requirements_: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6
"""

from __future__ import annotations

from typing import Any, Final

from src.sim_config import VALID_PHASES as _SIM_VALID_PHASES
from src.sim_config import IntersectionState

__all__ = ["ObservationParser"]


class ObservationParser:
    """Format CityFlow intersection state thành prompt text 3-section.

    Class không có internal state biến đổi — :meth:`parse` là pure function
    của ``state``. Có thể gọi từ nhiều thread / nhiều intersection mà không
    cần đồng bộ hóa.
    """

    #: Tập pha hợp lệ. Đồng nhất với ``src.sim_config.VALID_PHASES`` để tránh
    #: drift giữa hai module (cả hai cùng phải khớp Requirement 10 AC 6).
    VALID_PHASES: Final[frozenset[str]] = frozenset(_SIM_VALID_PHASES)

    # Tên 3 section header — dùng làm marker cho Property 3.
    _HEADER_OBSERVATION: Final[str] = "## Observation"
    _HEADER_INSTRUCTION: Final[str] = "## Instruction"
    _HEADER_OUTPUT_FORMAT: Final[str] = "## Output Format"

    # Required fields (dict variant) — used in error messages.
    _REQUIRED_FIELDS: Final[tuple[str, ...]] = (
        "lane_vehicle_count",
        "current_phase",
        "current_phase_time",
    )

    # ------------------------------------------------------------------ API

    def parse(self, state: dict[str, Any] | IntersectionState) -> str:
        """Chuyển CityFlow state thành prompt text 3-section.

        Args:
            state: ``dict`` chứa các key
                ``{lane_vehicle_count, current_phase, current_phase_time}``,
                hoặc :class:`IntersectionState` đã validate.

        Returns:
            Prompt text gồm đúng 3 section được phân tách bằng header
            ``## Observation`` / ``## Instruction`` / ``## Output Format``.

        Raises:
            ValueError: Nếu thiếu field bắt buộc, kiểu dữ liệu sai, queue
                length âm, hoặc ``current_phase`` không thuộc
                :attr:`VALID_PHASES`. Message luôn nêu tên field vi phạm.
        """
        lane_counts, phase, phase_time = self._extract_and_validate(state)

        observation = self._format_observation(lane_counts, phase, phase_time)
        instruction = self._format_instruction()
        output_spec = self._format_output_spec()

        # Hai newline giữa các section để header đứng riêng dòng + dễ đọc.
        # Output không có trailing newline để tránh ambiguity khi tokenize.
        return f"{observation}\n\n{instruction}\n\n{output_spec}"

    # ------------------------------------------------------ Section formatters

    def _format_observation(
        self,
        lane_counts: dict[str, int],
        phase: str,
        phase_time: int,
    ) -> str:
        """Format section observation: per-lane queue + pha hiện tại + phase time.

        Lane được liệt kê theo thứ tự alphabetical để đảm bảo determinism
        bất kể thứ tự insertion vào dict gốc.
        """
        # Header + danh sách lane theo thứ tự alphabetical.
        lines: list[str] = [self._HEADER_OBSERVATION, ""]

        if lane_counts:
            lines.append("Queue length per lane (number of waiting vehicles):")
            for lane_id in sorted(lane_counts):
                count = lane_counts[lane_id]
                lines.append(f"- {lane_id}: {count}")
        else:
            # Edge case: intersection không có lane nào được report — vẫn
            # phải tạo prompt deterministic để pipeline không crash.
            lines.append("Queue length per lane (number of waiting vehicles):")
            lines.append("- (no lanes reported)")

        lines.append("")
        lines.append(f"Current phase: {phase}")
        lines.append(f"Current phase time: {phase_time} seconds")

        return "\n".join(lines)

    def _format_instruction(self) -> str:
        """Section mô tả nhiệm vụ chọn pha tín hiệu (Requirement 3 AC 3.b)."""
        valid_phases_sorted = sorted(self.VALID_PHASES)
        valid_phases_str = ", ".join(valid_phases_sorted)
        return (
            f"{self._HEADER_INSTRUCTION}\n"
            "\n"
            "You are controlling a single traffic signal at one intersection. "
            "Based on the queue lengths, current phase, and current phase time "
            "above, choose ONE phase from the allowed set to activate next so "
            "as to minimize the average waiting time of vehicles at this "
            "intersection.\n"
            "\n"
            f"Allowed phases: {valid_phases_str}.\n"
            "\n"
            "- ETWT: East-West Through (straight movements on the East and "
            "West approaches).\n"
            "- NTST: North-South Through (straight movements on the North "
            "and South approaches).\n"
            "- ELWL: East-West Left (left-turn movements on the East and "
            "West approaches).\n"
            "- NLSL: North-South Left (left-turn movements on the North and "
            "South approaches)."
        )

    def _format_output_spec(self) -> str:
        """Section chỉ định output format (Requirement 3 AC 3.c)."""
        return (
            f"{self._HEADER_OUTPUT_FORMAT}\n"
            "\n"
            "Return your decision wrapped in a <signal> tag, with no "
            "additional commentary outside the tag. The content inside the "
            "tag MUST be exactly one of the allowed phase names "
            "(ETWT, NTST, ELWL, NLSL).\n"
            "\n"
            "Example: <signal>ETWT</signal>"
        )

    # --------------------------------------------------- Validation helpers

    def _extract_and_validate(
        self,
        state: dict[str, Any] | IntersectionState,
    ) -> tuple[dict[str, int], str, int]:
        """Normalize + validate state. Trả về (lane_counts, phase, phase_time).

        Hỗ trợ cả ``IntersectionState`` (đã được dataclass validate) và
        ``dict`` (chưa validate). Validation lặp lại đầy đủ cho dict nhằm
        bảo đảm thông báo lỗi đến từ chính ``ObservationParser`` —
        Requirement 3 AC 5/6 yêu cầu parser tự raise ``ValueError`` chỉ rõ
        field/phase vi phạm.
        """
        if isinstance(state, IntersectionState):
            # Dataclass đã validate trong ``__post_init__``; clone dict để
            # downstream code không phụ thuộc vào reference của caller.
            return (
                dict(state.lane_vehicle_count),
                state.current_phase,
                state.current_phase_time,
            )

        if not isinstance(state, dict):
            raise ValueError(
                "ObservationParser.parse: 'state' must be a dict or "
                f"IntersectionState; got {type(state).__name__}"
            )

        # Check missing fields first — message phải chỉ rõ tên field
        # (Requirement 3 AC 5).
        missing = [f for f in self._REQUIRED_FIELDS if f not in state]
        if missing:
            raise ValueError(
                "ObservationParser.parse: state is missing required "
                f"field(s): {missing}"
            )

        lane_counts_raw = state["lane_vehicle_count"]
        phase_raw = state["current_phase"]
        phase_time_raw = state["current_phase_time"]

        # --- lane_vehicle_count -------------------------------------------
        if not isinstance(lane_counts_raw, dict):
            raise ValueError(
                "ObservationParser.parse: field 'lane_vehicle_count' must "
                f"be a dict; got {type(lane_counts_raw).__name__}"
            )
        lane_counts: dict[str, int] = {}
        for lane_id, count in lane_counts_raw.items():
            if not isinstance(lane_id, str) or not lane_id:
                raise ValueError(
                    "ObservationParser.parse: keys of 'lane_vehicle_count' "
                    f"must be non-empty str; got {lane_id!r}"
                )
            # ``bool`` is a subclass of ``int`` — reject explicitly so
            # ``True``/``False`` don't slip through as 1/0 silently.
            if isinstance(count, bool) or not isinstance(count, int):
                raise ValueError(
                    "ObservationParser.parse: field "
                    f"'lane_vehicle_count[{lane_id!r}]' must be int; got "
                    f"{type(count).__name__} ({count!r})"
                )
            if count < 0:
                raise ValueError(
                    "ObservationParser.parse: field "
                    f"'lane_vehicle_count[{lane_id!r}]' must be >= 0; "
                    f"got {count}"
                )
            lane_counts[lane_id] = count

        # --- current_phase ------------------------------------------------
        if not isinstance(phase_raw, str):
            raise ValueError(
                "ObservationParser.parse: field 'current_phase' must be a "
                f"str; got {type(phase_raw).__name__} ({phase_raw!r})"
            )
        if phase_raw not in self.VALID_PHASES:
            raise ValueError(
                "ObservationParser.parse: field 'current_phase' has invalid "
                f"value {phase_raw!r}; expected one of "
                f"{sorted(self.VALID_PHASES)}"
            )

        # --- current_phase_time -------------------------------------------
        if isinstance(phase_time_raw, bool) or not isinstance(
            phase_time_raw, int
        ):
            raise ValueError(
                "ObservationParser.parse: field 'current_phase_time' must "
                f"be int; got {type(phase_time_raw).__name__} "
                f"({phase_time_raw!r})"
            )
        if phase_time_raw < 0:
            raise ValueError(
                "ObservationParser.parse: field 'current_phase_time' must "
                f"be >= 0; got {phase_time_raw}"
            )

        return lane_counts, phase_raw, phase_time_raw
