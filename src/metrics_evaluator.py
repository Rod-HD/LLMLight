"""Metrics Evaluator (Component 9).

Tính 3 chỉ số của paper LLMLight (Lai et al., KDD 2025) sau khi simulation
CityFlow kết thúc:

* **ATT** — *Average Travel Time* (giây): trung bình cộng thời gian di
  chuyển của xe **đã hoàn thành** hành trình.
* **AQL** — *Average Queue Length* (xe / lane): trung bình **per-lane**
  số xe đang chờ (tốc độ ``< SPEED_THRESHOLD``), lấy trung bình qua tất
  cả lane và tất cả timesteps. **KHÔNG** phải tổng số xe trong mạng.
* **AWT** — *Average Waiting Time* (giây): trung bình thời gian tích lũy
  một xe có tốc độ ``< SPEED_THRESHOLD``, tính trên tất cả xe.

Bound (Requirement 9 AC 4 / Requirement 12 AC 6):

* ``ATT ∈ [0, total_timesteps]``
* ``AWT ∈ [0, total_timesteps]``
* ``AQL ∈ [0, max_lane_capacity_observed]`` (KHÔNG phải
  ``[0, total_vehicles]``).

Edge case (Requirement 9 AC 5): nếu không có xe nào hoàn thành hành trình
trong simulation, ``compute_att`` trả về ``total_timesteps`` (cast về
``float``) và log cảnh báo.

``MetricsResult`` được reuse từ :mod:`src.sim_config` (đã định nghĩa
trong Task 1.6) để đảm bảo schema thống nhất giữa runner / evaluator
/ UI Demo.

_Requirements_: 9.1-9.6, 12.6
"""

from __future__ import annotations

import logging
import math
import statistics
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from src.sim_config import MetricsResult

logger = logging.getLogger(__name__)


__all__ = ["MetricsEvaluator"]


#: Tập ``total_timesteps`` được hỗ trợ — phải khớp
#: ``src.sim_config.VALID_TOTAL_TIMESTEPS`` (250 = Demo, 3600 = Full).
_VALID_TOTAL_TIMESTEPS: tuple[int, ...] = (250, 3600)


# --------------------------------------------------------------------------
# Backend inference từ method name (cho generate_comparison_table)
# --------------------------------------------------------------------------


def _infer_backend(method: str) -> str:
    """Suy ra tên backend hiển thị trong cột ``Backend`` của comparison table.

    Method LLM API (``gpt4o_<backend>``) → tên backend tương ứng.
    Method local (``lightgpt_*``) → ``"local"``.
    Baselines RL → ``"-"`` (rule-based / RL không gọi LLM).
    """
    if method.startswith("gpt4o_"):
        return method.split("_", 1)[1]
    if method.startswith("lightgpt_"):
        return "local"
    return "-"


# --------------------------------------------------------------------------
# Aggregate helper cho mean ± std
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _MeanStd:
    """Tóm tắt mean + std cho một series. ``std == 0`` khi chỉ có 1 sample."""

    mean: float
    std: float

    def format(self) -> str:
        """Format ``"mean ± std"`` rounded 2 decimals."""
        return f"{round(self.mean, 2):.2f} ± {round(self.std, 2):.2f}"


def _mean_std(values: Sequence[float]) -> _MeanStd:
    """Compute mean và sample std (n-1, fallback 0 khi n=1)."""
    if not values:
        return _MeanStd(mean=0.0, std=0.0)
    mean = float(statistics.fmean(values))
    if len(values) == 1:
        return _MeanStd(mean=mean, std=0.0)
    # statistics.stdev dùng mẫu (n-1), khớp convention "mean ± std" của paper.
    return _MeanStd(mean=mean, std=float(statistics.stdev(values)))


# --------------------------------------------------------------------------
# MetricsEvaluator
# --------------------------------------------------------------------------


class MetricsEvaluator:
    """Tính ATT / AQL / AWT từ data thu thập trong simulation.

    Workflow điển hình (runner):

    1. Trong simulation loop: tích lũy ``lane_queues_per_step``,
       ``vehicle_travel_times`` (xe completed), ``vehicle_wait_times``
       (cumulative low-speed time per vehicle) — thường lấy từ
       :class:`src.cityflow_engine.CityFlowEngine`.
    2. Sau simulation: gọi ``compute_att`` / ``compute_aql`` /
       ``compute_awt`` riêng, hoặc gọi ``evaluate(...)`` để build
       :class:`MetricsResult` một lần.
    3. Khi đã có nhiều runs (mean ± std): gọi ``generate_comparison_table``
       với ``list[(method, MetricsResult)]``.

    Attributes:
        total_timesteps: 250 (Demo) hoặc 3600 (Full) — bound trên cho ATT
            và AWT, và là giá trị fallback ATT khi không có xe hoàn thành.
    """

    #: Ngưỡng tốc độ (m/s) phân biệt "đang chờ" và "đang di chuyển" — khớp
    #: định nghĩa AQL/AWT trong paper LLMLight (Requirement 9 AC 2/3).
    SPEED_THRESHOLD: float = 0.1

    #: Số chữ số thập phân được giữ lại cho mọi metric output
    #: (Requirement 9 AC 1-3).
    DECIMAL_PLACES: int = 2

    def __init__(self, total_timesteps: int) -> None:
        """Khởi tạo evaluator với upper-bound ``total_timesteps``.

        Args:
            total_timesteps: Số timesteps simulation. Chỉ chấp nhận giá
                trị trong :data:`_VALID_TOTAL_TIMESTEPS` ``= (250, 3600)``.

        Raises:
            ValueError: ``total_timesteps`` không phải int hoặc không
                thuộc tập hợp lệ.
        """
        if isinstance(total_timesteps, bool) or not isinstance(
            total_timesteps, int
        ):
            raise ValueError(
                "MetricsEvaluator.total_timesteps must be int; "
                f"got {type(total_timesteps).__name__} ({total_timesteps!r})"
            )
        if total_timesteps not in _VALID_TOTAL_TIMESTEPS:
            raise ValueError(
                "MetricsEvaluator.total_timesteps must be one of "
                f"{list(_VALID_TOTAL_TIMESTEPS)}; got {total_timesteps}"
            )
        self.total_timesteps: int = total_timesteps

    # ============================ ATT =====================================

    def compute_att(self, vehicle_travel_times: Sequence[float]) -> float:
        """Tính Average Travel Time.

        Args:
            vehicle_travel_times: Travel time (giây) của các xe **đã hoàn
                thành** hành trình. Mỗi giá trị phải là số ``>= 0``.

        Returns:
            ``float`` rounded 2 decimal places, bounded
            ``[0, total_timesteps]``. Nếu input rỗng → trả về
            ``float(total_timesteps)`` và log warning (Requirement 9 AC 5).

        Raises:
            ValueError: Nếu kết quả vượt ``total_timesteps`` (gợi ý bug
                trong dataset / runner — Requirement 12 AC 6).
        """
        values = self._coerce_nonneg_floats(
            vehicle_travel_times, name="vehicle_travel_times"
        )
        if not values:
            logger.warning(
                "MetricsEvaluator.compute_att: no completed vehicles; "
                "returning total_timesteps=%d as ATT (Requirement 9 AC 5).",
                self.total_timesteps,
            )
            return float(self.total_timesteps)

        att = sum(values) / len(values)
        att_rounded = round(att, self.DECIMAL_PLACES)

        # Bound check: ATT ∈ [0, total_timesteps] (Req 9 AC 4 / Req 12 AC 6).
        if att_rounded < 0:
            raise ValueError(
                f"MetricsEvaluator.compute_att: ATT < 0 ({att_rounded}); "
                "all travel times must be >= 0."
            )
        if att_rounded > self.total_timesteps:
            raise ValueError(
                f"MetricsEvaluator.compute_att: ATT={att_rounded} exceeds "
                f"total_timesteps={self.total_timesteps}; "
                "any travel time > total_timesteps is impossible "
                "(Requirement 9 AC 4)."
            )
        return att_rounded

    # ============================ AQL =====================================

    def compute_aql(
        self, lane_queues_per_step: Sequence[dict[str, int]]
    ) -> float:
        """Tính Average Queue Length **per-lane**.

        AQL = trung bình ``(số xe queue trên 1 lane)`` lấy đồng thời qua
        tất cả lane và tất cả timesteps. Đây **KHÔNG** phải trung bình
        tổng số xe trong mạng (Requirement 9 AC 2).

        Args:
            lane_queues_per_step: List có ``len = num_timesteps``; mỗi
                phần tử là dict ``lane_id -> queue_count`` (xe có tốc độ
                ``< SPEED_THRESHOLD`` trên lane đó tại timestep ấy). Mỗi
                queue count phải là int ``>= 0``.

        Returns:
            ``float`` rounded 2 decimal places, bounded
            ``[0, max_lane_capacity_observed]`` (xe/lane). Nếu input
            rỗng → trả về ``0.0``.

        Raises:
            ValueError: queue count âm, hoặc kết quả vượt
                ``max_lane_capacity_observed`` (gợi ý bug — Req 12 AC 6).
        """
        if not isinstance(lane_queues_per_step, (list, tuple)):
            raise ValueError(
                "MetricsEvaluator.compute_aql: lane_queues_per_step must be "
                "a list/tuple of dict; "
                f"got {type(lane_queues_per_step).__name__}"
            )

        if len(lane_queues_per_step) == 0:
            return 0.0

        # Iterate timesteps, accumulate (sum_queues, lane_count).
        total_queue = 0.0
        total_lanes = 0
        max_lane_capacity = 0  # observed upper bound per-lane per-timestep.

        for step_idx, step_queues in enumerate(lane_queues_per_step):
            if not isinstance(step_queues, dict):
                raise ValueError(
                    "MetricsEvaluator.compute_aql: each entry in "
                    "lane_queues_per_step must be dict; entry "
                    f"{step_idx} is {type(step_queues).__name__}"
                )
            for lane_id, queue_count in step_queues.items():
                if not isinstance(lane_id, str) or not lane_id:
                    raise ValueError(
                        "MetricsEvaluator.compute_aql: lane_id must be "
                        f"non-empty str; got {lane_id!r} in step {step_idx}"
                    )
                if isinstance(queue_count, bool) or not isinstance(
                    queue_count, int
                ):
                    raise ValueError(
                        "MetricsEvaluator.compute_aql: queue count must be "
                        f"int; got {type(queue_count).__name__} "
                        f"({queue_count!r}) for lane {lane_id!r} at step "
                        f"{step_idx}"
                    )
                if queue_count < 0:
                    raise ValueError(
                        "MetricsEvaluator.compute_aql: queue count must be "
                        f">= 0; got {queue_count} for lane {lane_id!r} at "
                        f"step {step_idx}"
                    )
                total_queue += float(queue_count)
                total_lanes += 1
                if queue_count > max_lane_capacity:
                    max_lane_capacity = queue_count

        if total_lanes == 0:
            # All step dicts empty — defensible 0.0 per spec defaults.
            return 0.0

        aql = total_queue / total_lanes
        aql_rounded = round(aql, self.DECIMAL_PLACES)

        # Bound check (Req 9 AC 4 / Req 12 AC 6):
        # AQL ∈ [0, max_lane_capacity_observed], NOT [0, total_vehicles].
        if aql_rounded < 0:
            raise ValueError(
                f"MetricsEvaluator.compute_aql: AQL < 0 ({aql_rounded})."
            )
        if aql_rounded > max_lane_capacity:
            # Mathematically impossible since max ≥ mean for non-negative
            # values. Defensive only.
            raise ValueError(
                f"MetricsEvaluator.compute_aql: AQL={aql_rounded} exceeds "
                f"max_lane_capacity_observed={max_lane_capacity} "
                "(Requirement 9 AC 4)."
            )
        return aql_rounded

    # ============================ AWT =====================================

    def compute_awt(self, vehicle_wait_times: Sequence[float]) -> float:
        """Tính Average Waiting Time.

        Args:
            vehicle_wait_times: Cumulative wait time (giây) của mỗi xe —
                tổng thời gian xe đó có tốc độ ``< SPEED_THRESHOLD`` trong
                suốt simulation. Mỗi giá trị phải là số ``>= 0``.

        Returns:
            ``float`` rounded 2 decimal places, bounded
            ``[0, total_timesteps]``. Nếu input rỗng → trả về ``0.0``
            (không có xe nào có wait time để trung bình).

        Raises:
            ValueError: Nếu kết quả vượt ``total_timesteps`` (Req 12 AC 6).
        """
        values = self._coerce_nonneg_floats(
            vehicle_wait_times, name="vehicle_wait_times"
        )
        if not values:
            return 0.0

        awt = sum(values) / len(values)
        awt_rounded = round(awt, self.DECIMAL_PLACES)

        if awt_rounded < 0:
            raise ValueError(
                f"MetricsEvaluator.compute_awt: AWT < 0 ({awt_rounded})."
            )
        if awt_rounded > self.total_timesteps:
            raise ValueError(
                f"MetricsEvaluator.compute_awt: AWT={awt_rounded} exceeds "
                f"total_timesteps={self.total_timesteps} (Req 9 AC 4)."
            )
        return awt_rounded

    # ============================ evaluate ================================

    def evaluate(
        self,
        engine: Any | None = None,
        *,
        vehicle_travel_times: Sequence[float] | None = None,
        lane_queues_per_step: Sequence[dict[str, int]] | None = None,
        vehicle_wait_times: Sequence[float] | None = None,
    ) -> MetricsResult:
        """Tính tất cả metrics và trả về :class:`MetricsResult`.

        Có hai cách dùng:

        1. **Truyền data trực tiếp** (preferred — runner đã track sẵn):
           cung cấp ba kwargs ``vehicle_travel_times``,
           ``lane_queues_per_step``, ``vehicle_wait_times``.
        2. **Truyền engine** (convenience): nếu chỉ có ``engine`` mà không
           cung cấp data, evaluator sẽ thử lấy ATT từ
           ``engine.get_average_travel_time()`` (CityFlow native API),
           đặt AQL/AWT = 0 nếu không có data — chế độ này chủ yếu cho
           smoke test và **KHÔNG khuyến nghị** cho production runs vì
           thiếu AQL/AWT đầy đủ.

        Args:
            engine: Optional CityFlow engine (với
                ``get_average_travel_time()``). Bỏ qua nếu kwargs có
                ``vehicle_travel_times``.
            vehicle_travel_times: Travel times của xe đã hoàn thành.
            lane_queues_per_step: Lane queue dict per timestep.
            vehicle_wait_times: Cumulative wait time per vehicle.

        Returns:
            :class:`MetricsResult` với 3 metric đã round 2 decimal places.
        """
        # ATT
        if vehicle_travel_times is not None:
            att = self.compute_att(vehicle_travel_times)
        elif engine is not None:
            att = self._att_from_engine(engine)
        else:
            logger.warning(
                "MetricsEvaluator.evaluate: no vehicle_travel_times and no "
                "engine; defaulting ATT to total_timesteps."
            )
            att = float(self.total_timesteps)

        # AQL
        if lane_queues_per_step is not None:
            aql = self.compute_aql(lane_queues_per_step)
        else:
            logger.warning(
                "MetricsEvaluator.evaluate: lane_queues_per_step not "
                "provided; defaulting AQL=0.0 (smoke test mode)."
            )
            aql = 0.0

        # AWT
        if vehicle_wait_times is not None:
            awt = self.compute_awt(vehicle_wait_times)
        else:
            logger.warning(
                "MetricsEvaluator.evaluate: vehicle_wait_times not "
                "provided; defaulting AWT=0.0 (smoke test mode)."
            )
            awt = 0.0

        return MetricsResult(att=att, aql=aql, awt=awt)

    def _att_from_engine(self, engine: Any) -> float:
        """Best-effort: lấy ATT từ CityFlow native API."""
        getter = getattr(engine, "get_average_travel_time", None)
        if getter is None:
            # Fallback: try the underlying ``cityflow.Engine`` accessor.
            inner = getattr(engine, "_engine", None)
            if inner is not None:
                getter = getattr(inner, "get_average_travel_time", None)

        if getter is None:
            logger.warning(
                "MetricsEvaluator.evaluate: engine has no "
                "get_average_travel_time(); defaulting ATT to "
                "total_timesteps."
            )
            return float(self.total_timesteps)

        try:
            value = float(getter())
        except Exception as exc:  # noqa: BLE001 - defensive
            logger.warning(
                "MetricsEvaluator.evaluate: engine.get_average_travel_time() "
                "raised %s; defaulting ATT to total_timesteps.",
                exc,
            )
            return float(self.total_timesteps)

        if value <= 0 or math.isnan(value):
            logger.warning(
                "MetricsEvaluator.evaluate: engine reported ATT=%.2f (no "
                "vehicles completed); defaulting to total_timesteps.",
                value,
            )
            return float(self.total_timesteps)

        # Clamp at the upper bound for safety (CityFlow returns non-clamped).
        if value > self.total_timesteps:
            return float(self.total_timesteps)

        return round(value, self.DECIMAL_PLACES)

    # ===================== generate_comparison_table ======================

    def generate_comparison_table(
        self,
        results: Sequence[tuple[str, MetricsResult]],
    ) -> str:
        """Build Markdown comparison table.

        Columns:

        * ``Method`` — tên phương pháp (group key).
        * ``ATT (s)`` — ``mean ± std`` ATT.
        * ``AQL (vehicles/lane)`` — ``mean ± std`` AQL.
        * ``AWT (s)`` — ``mean ± std`` AWT.
        * ``Runs`` — số runs aggregated cho method này.
        * ``Backend`` — suy ra từ method name (``gpt4o_<X>`` → ``X``;
          ``lightgpt_*`` → ``"local"``; baselines → ``"-"``).

        Multiple entries cùng method được group lại và tính mean ± std
        sample (n-1 stdev; fallback ``std=0`` khi chỉ có 1 run).

        Args:
            results: Iterable of ``(method, MetricsResult)`` tuples.

        Returns:
            Markdown string. Header + separator + 1 row mỗi method. Thứ
            tự rows giữ theo lần xuất hiện đầu tiên của method trong
            ``results``.
        """
        if not isinstance(results, (list, tuple)):
            # Accept any iterable but materialize to allow re-iteration.
            results = list(results)

        # Group preserving first-seen order of method.
        groups: dict[str, list[MetricsResult]] = {}
        method_order: list[str] = []
        for item in results:
            if not isinstance(item, tuple) or len(item) != 2:
                raise ValueError(
                    "MetricsEvaluator.generate_comparison_table: each entry "
                    f"must be a (method, MetricsResult) tuple; got {item!r}"
                )
            method, metrics = item
            if not isinstance(method, str) or not method:
                raise ValueError(
                    "MetricsEvaluator.generate_comparison_table: method "
                    f"must be a non-empty str; got {method!r}"
                )
            if not isinstance(metrics, MetricsResult):
                raise ValueError(
                    "MetricsEvaluator.generate_comparison_table: metrics "
                    f"must be MetricsResult; got {type(metrics).__name__}"
                )
            if method not in groups:
                groups[method] = []
                method_order.append(method)
            groups[method].append(metrics)

        header = (
            "| Method | ATT (s) | AQL (vehicles/lane) | AWT (s) "
            "| Runs | Backend |"
        )
        separator = (
            "| --- | --- | --- | --- | --- | --- |"
        )
        lines: list[str] = [header, separator]

        for method in method_order:
            metrics_list = groups[method]
            atts = [m.att for m in metrics_list]
            aqls = [m.aql for m in metrics_list]
            awts = [m.awt for m in metrics_list]
            att_stat = _mean_std(atts).format()
            aql_stat = _mean_std(aqls).format()
            awt_stat = _mean_std(awts).format()
            runs = len(metrics_list)
            backend = _infer_backend(method)
            lines.append(
                f"| {method} | {att_stat} | {aql_stat} | {awt_stat} "
                f"| {runs} | {backend} |"
            )

        return "\n".join(lines)

    # ============================ helpers ================================

    @staticmethod
    def _coerce_nonneg_floats(
        values: Iterable[Any], *, name: str
    ) -> list[float]:
        """Coerce iterable thành ``list[float]``, validate ``>= 0``.

        ``bool`` bị reject vì là subclass của ``int`` — tránh bug
        ``True`` được hiểu là 1 giây.
        """
        if values is None:
            raise ValueError(
                f"MetricsEvaluator: {name} must not be None"
            )
        if isinstance(values, (str, bytes)):
            raise ValueError(
                f"MetricsEvaluator: {name} must be a sequence of numbers, "
                f"not {type(values).__name__}"
            )
        out: list[float] = []
        for idx, v in enumerate(values):
            if isinstance(v, bool):
                raise ValueError(
                    f"MetricsEvaluator: {name}[{idx}] must be a number, "
                    f"not bool ({v!r})"
                )
            if not isinstance(v, (int, float)):
                raise ValueError(
                    f"MetricsEvaluator: {name}[{idx}] must be a number; "
                    f"got {type(v).__name__} ({v!r})"
                )
            fv = float(v)
            if math.isnan(fv) or math.isinf(fv):
                raise ValueError(
                    f"MetricsEvaluator: {name}[{idx}] must be finite; "
                    f"got {v!r}"
                )
            if fv < 0:
                raise ValueError(
                    f"MetricsEvaluator: {name}[{idx}] must be >= 0; "
                    f"got {v!r}"
                )
            out.append(fv)
        return out
