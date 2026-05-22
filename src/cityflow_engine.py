"""CityFlow Engine wrapper (Component 1).

Cung cấp một class :class:`CityFlowEngine` đóng gói thư viện CityFlow
binding (``cityflow.Engine``) để:

* Khởi tạo engine từ file CityFlow config JSON với ``seed`` (do
  :class:`SeedManager` cấp).
* Tùy chọn bật chế độ ghi replay (``save_replay=True``) — sửa CityFlow
  config (file copy trong thư mục ``results/replays/``) thêm
  ``"saveReplay": true`` và đặt ``"roadnetLogFile"`` / ``"replayLogFile"``
  trỏ về ``results/replays/{dataset}_{method}_{run_id}.txt``. Mặc định
  ``save_replay=False`` để tiết kiệm I/O ở Phase 2/3 (replay file có thể
  vài trăm MB) — Requirement 14 AC 11.
* Tiến simulation 1 timestep (:meth:`next_step`).
* Cung cấp queue length per-lane (:meth:`get_lane_vehicle_count`),
  tổng số xe đang trong mạng (:meth:`get_vehicle_count`), tổng số xe
  đã spawn (:meth:`get_vehicles_spawned_total`), và tổng số xe đã rời
  mạng (:meth:`get_vehicles_completed_total`) — cần cho Property 5
  Vehicle Conservation Invariant (Requirement 12.5).
* Đặt pha tín hiệu (:meth:`set_phase`) với timing logic:

  - Khi pha mới KHÁC pha hiện tại: chèn yellow 3s + all-red 2s + green
    30s (tổng 35s).
  - Khi pha mới == pha hiện tại (giữ nguyên): chỉ green 30s (KHÔNG chèn
    yellow/all-red).

CityFlow tự quản lý chuyển pha (yellow/all-red phases) khi :meth:`set_phase`
được gọi với một ``phase_index`` mới. Trong wrapper này chúng tôi KHÔNG
cố gắng can thiệp vào phase index bằng cách "chèn" pha vàng/all-red
nhân tạo (CityFlow không có "phase index" cố định cho yellow/all-red);
thay vào đó wrapper chỉ điều khiển SỐ TIMESTEP được cấp cho mỗi giai
đoạn:

  * Đổi pha → gọi ``eng.set_tl_phase(intersection, phase_index)`` rồi
    advance ``yellow_duration + all_red_duration + green_duration``
    timesteps. CityFlow native interphase model (nếu được cấu hình
    trong roadnet) sẽ tự xử lý transition; nếu không, đây tương đương
    một pha xanh kéo dài hơn — đây là sự đánh đổi đơn giản hóa được
    document trong design.
  * Giữ pha → KHÔNG gọi ``set_tl_phase`` lại; chỉ advance ``green_duration``
    timesteps.

NHẬN ``phase_index`` ĐÃ ĐƯỢC RESOLVE bởi :class:`PhaseIndexMapper`.
KHÔNG hard-code mapping ``phase_name → phase_index`` trong engine
(vì thứ tự pha trong roadnet.json không cố định giữa các dataset).

Module import sạch ngay cả khi ``cityflow`` Python binding chưa được cài
(môi trường dev Windows trước khi setup_env.sh chạy trong WSL2). Tuy
nhiên :meth:`CityFlowEngine.__init__` sẽ raise :class:`ImportError` rõ
ràng nếu binding không khả dụng tại runtime. Tham số ``engine_factory``
cho phép inject một mock factory trong unit test.

_Requirements_: 1.3, 2.1, 2.4, 2.6, 2.7, 10.1-10.5, 14.11
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Iterable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Optional binding import — guarded so the module imports cleanly on dev
# environments where ``cityflow`` (built from source via setup_env.sh in
# WSL2) is not yet available. Tests must inject ``engine_factory`` to
# bypass the binding entirely.
# ---------------------------------------------------------------------------

_cityflow: Any | None
try:
    import cityflow as _cityflow  # type: ignore[import-not-found]
except Exception as _cityflow_exc:  # pragma: no cover - depends on env
    _cityflow = None
    logger.debug("cityflow binding not importable: %s", _cityflow_exc)


# Default timing values — CAN BE OVERRIDDEN per-instance qua __init__ kwargs
# nếu runner cần test với timing khác. Mặc định khớp config/simulation.json
# (Requirement 10.1-10.3).
_DEFAULT_GREEN_DURATION = 30
_DEFAULT_YELLOW_DURATION = 3
_DEFAULT_ALL_RED_DURATION = 2


#: Đường dẫn relative tới project root nơi ghi replay file (ổ D mounted
#: tại /mnt/d/ trong WSL2). Resolve relative tới CWD tại runtime.
_REPLAY_DIR = Path("results") / "replays"


class CityFlowEngine:
    """Wrapper quanh ``cityflow.Engine`` với timing + replay management.

    Attributes:
        config_path: Đường dẫn ABSOLUTE tới file CityFlow config JSON
            đang được engine sử dụng (có thể là ``effective_config_path``
            khi ``save_replay=True``).
        seed: Seed đã truyền cho CityFlow.
        green_duration: Thời lượng pha xanh tối thiểu (mặc định 30s).
        yellow_duration: Thời lượng pha vàng (mặc định 3s).
        all_red_duration: Thời lượng all-red (mặc định 2s).
        save_replay: Có bật chế độ ghi replay hay không.
        replay_file: Đường dẫn replay.txt (chỉ set khi ``save_replay=True``).
    """

    def __init__(
        self,
        config_path: str,
        seed: int,
        save_replay: bool = False,
        *,
        dataset: str | None = None,
        method: str | None = None,
        run_id: int | None = None,
        green_duration: int = _DEFAULT_GREEN_DURATION,
        yellow_duration: int = _DEFAULT_YELLOW_DURATION,
        all_red_duration: int = _DEFAULT_ALL_RED_DURATION,
        engine_factory: Callable[..., Any] | None = None,
    ) -> None:
        """Khởi tạo CityFlow engine từ file config JSON.

        Args:
            config_path: Đường dẫn tuyệt đối đến file CityFlow config JSON.
            seed: Seed cho CityFlow (do :class:`SeedManager` cấp).
            save_replay: Nếu ``True``, sửa CityFlow config thêm
                ``"saveReplay": true`` và đặt ``"roadnetLogFile"`` /
                ``"replayLogFile"`` trỏ về
                ``results/replays/{dataset}_{method}_{run_id}.txt``.
                Mặc định ``False`` để tiết kiệm I/O.
            dataset: Tên dataset (vd. ``"jinan_1"``). REQUIRED khi
                ``save_replay=True``.
            method: Tên phương pháp (vd. ``"lightgpt_hf"``). REQUIRED khi
                ``save_replay=True``.
            run_id: Index run (0, 1, 2). REQUIRED khi ``save_replay=True``.
            green_duration: Thời lượng pha xanh tối thiểu (mặc định 30s).
            yellow_duration: Thời lượng pha vàng (mặc định 3s).
            all_red_duration: Thời lượng all-red (mặc định 2s).
            engine_factory: Optional callable dùng thay cho
                ``cityflow.Engine``. Hữu ích cho unit test trên môi trường
                không cài CityFlow. Nếu ``None``, dùng
                ``cityflow.Engine`` trực tiếp.

        Raises:
            FileNotFoundError: Nếu ``config_path`` không tồn tại.
            json.JSONDecodeError: Nếu config không phải JSON hợp lệ.
            ValueError: Nếu ``save_replay=True`` mà thiếu dataset/method/run_id,
                hoặc timing values không hợp lệ (<= 0).
            ImportError: Nếu ``engine_factory=None`` và ``cityflow`` binding
                không khả dụng tại runtime.
        """
        # --- Validate timing config ----------------------------------------
        for tname, tval in (
            ("green_duration", green_duration),
            ("yellow_duration", yellow_duration),
            ("all_red_duration", all_red_duration),
        ):
            if isinstance(tval, bool) or not isinstance(tval, int):
                raise ValueError(
                    f"CityFlowEngine: {tname} must be int; "
                    f"got {type(tval).__name__} ({tval!r})"
                )
            if tval <= 0:
                raise ValueError(
                    f"CityFlowEngine: {tname} must be > 0; got {tval}"
                )

        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(
                "CityFlowEngine: seed must be int; "
                f"got {type(seed).__name__} ({seed!r})"
            )

        # --- Validate config_path exists -----------------------------------
        cfg_path = Path(config_path)
        if not cfg_path.exists():
            raise FileNotFoundError(
                f"CityFlowEngine: config_path not found: {config_path}"
            )
        if not cfg_path.is_file():
            raise FileNotFoundError(
                f"CityFlowEngine: config_path is not a file: {config_path}"
            )

        # --- Parse config JSON (raises json.JSONDecodeError) ---------------
        with cfg_path.open("r", encoding="utf-8") as f:
            config_data = json.load(f)
        if not isinstance(config_data, dict):
            raise ValueError(
                "CityFlowEngine: top-level config JSON must be an object; "
                f"got {type(config_data).__name__}"
            )

        # --- Resolve save_replay --------------------------------------------
        effective_config_path = str(cfg_path.resolve())
        replay_file: str | None = None
        if save_replay:
            missing = [
                name
                for name, val in (
                    ("dataset", dataset),
                    ("method", method),
                    ("run_id", run_id),
                )
                if val is None
            ]
            if missing:
                raise ValueError(
                    "CityFlowEngine: save_replay=True requires "
                    f"{missing} to be provided (got None)"
                )
            assert dataset is not None and method is not None and run_id is not None
            effective_config_path, replay_file = self._prepare_replay_config(
                cfg_path=cfg_path,
                config_data=config_data,
                dataset=dataset,
                method=method,
                run_id=int(run_id),
            )

        # --- Resolve engine factory -----------------------------------------
        if engine_factory is None:
            if _cityflow is None:
                raise ImportError(
                    "CityFlowEngine: cityflow Python binding is not "
                    "available. Install via setup_env.sh in WSL2 "
                    "(builds CityFlow from source) or pass an "
                    "engine_factory argument for unit tests."
                )
            engine_factory = _cityflow.Engine  # type: ignore[attr-defined]

        # --- Instantiate underlying engine ----------------------------------
        # CityFlow signature: cityflow.Engine(config_path: str, thread_num: int = 1)
        # Seed is read from config JSON ("seed" key) — but we also try to
        # propagate it through the constructor for engines that accept it.
        try:
            engine = engine_factory(effective_config_path, thread_num=1, seed=seed)
        except TypeError:
            # Real CityFlow binding does not accept ``seed`` kwarg; fallback.
            engine = engine_factory(effective_config_path, thread_num=1)

        # --- Persist state ---------------------------------------------------
        self.config_path: str = effective_config_path
        self.original_config_path: str = str(cfg_path.resolve())
        self.seed: int = int(seed)
        self.green_duration: int = int(green_duration)
        self.yellow_duration: int = int(yellow_duration)
        self.all_red_duration: int = int(all_red_duration)
        self.save_replay: bool = bool(save_replay)
        self.replay_file: str | None = replay_file

        self._engine: Any = engine

        # Track current phase per intersection (for change-vs-hold logic).
        self._current_phase_index: dict[str, int] = {}
        # Track all vehicle ids seen so far, to compute spawned/completed
        # counters (Vehicle Conservation Invariant — Property 5).
        self._seen_vehicle_ids: set[str] = set()
        # Track current vehicle ids (last polled). Refreshed on every
        # ``next_step()`` and on the ``get_*`` accessors so callers get
        # consistent values regardless of which method they call first.
        self._current_vehicle_ids: set[str] = set()
        # Eagerly initialize counters with whatever vehicles are present at
        # t=0 (typically empty, but CityFlow doesn't forbid pre-spawned
        # vehicles in custom flow files).
        self._refresh_vehicle_tracking()

    # -------------------------------------------------------------- Public --

    def next_step(self) -> None:
        """Advance simulation by 1 timestep và cập nhật vehicle tracking."""
        self._engine.next_step()
        self._refresh_vehicle_tracking()

    def get_lane_vehicle_count(self) -> dict[str, int]:
        """Return số xe trên mỗi lane.

        Returns:
            Mapping ``lane_id -> queue_count`` (>= 0).
        """
        raw = self._engine.get_lane_vehicle_count()
        if not isinstance(raw, dict):
            raise TypeError(
                "CityFlowEngine: cityflow.get_lane_vehicle_count() returned "
                f"non-dict ({type(raw).__name__})"
            )
        # Coerce to plain dict[str, int] (CityFlow returns python dict already
        # but defensive copy avoids accidental external mutation).
        return {str(lane): int(count) for lane, count in raw.items()}

    def get_vehicle_count(self) -> int:
        """Tổng số xe đang trong mạng tại timestep hiện tại (>= 0)."""
        # Prefer the internal counter (refreshed on every next_step) over
        # the binding's get_vehicle_count() to keep all three accessors
        # consistent (avoid race when caller mixes them).
        self._refresh_vehicle_tracking()
        return len(self._current_vehicle_ids)

    def get_vehicles_spawned_total(self) -> int:
        """Tổng số xe đã spawn từ timestep 0 đến hiện tại (>= 0).

        Cần cho Property 5 — Vehicle Conservation Invariant (Requirement 12.5):
        ``current = spawned - completed``.
        """
        self._refresh_vehicle_tracking()
        return len(self._seen_vehicle_ids)

    def get_vehicles_completed_total(self) -> int:
        """Tổng số xe đã rời mạng từ timestep 0 đến hiện tại (>= 0)."""
        self._refresh_vehicle_tracking()
        return len(self._seen_vehicle_ids) - len(self._current_vehicle_ids)

    def set_phase(self, intersection_id: str, phase_index: int) -> None:
        """Đặt pha tín hiệu cho một intersection và advance simulation.

        Logic timing (Requirement 10.1-10.4):

        * Khi pha mới KHÁC pha hiện tại (hoặc lần đầu set cho intersection):
          gọi ``cityflow.set_tl_phase(intersection, phase_index)`` và
          advance ``yellow_duration + all_red_duration + green_duration``
          timesteps (mặc định 3 + 2 + 30 = 35).
        * Khi pha mới == pha hiện tại (giữ nguyên): KHÔNG gọi
          ``set_tl_phase`` lại; advance ``green_duration`` timesteps
          (mặc định 30).

        Note:
            CityFlow tự xử lý "interphase" (yellow/all-red transitions) ở
            backend nếu roadnet được build với ``setIntervalPhase``. Trong
            wrapper này chúng tôi điều khiển TIMING (số timestep được
            advance), không cố gắng can thiệp vào phase index nội bộ. Đây
            là hợp đồng đã document trong design.

        Args:
            intersection_id: ID của intersection (nguyên gốc trong roadnet).
            phase_index: Phase index ĐÃ được :class:`PhaseIndexMapper`
                resolve. KHÔNG hard-code mapping ``phase_name -> index``
                trong engine.

        Raises:
            ValueError: Nếu ``phase_index`` không phải int hoặc < 0.
        """
        if not isinstance(intersection_id, str) or not intersection_id:
            raise ValueError(
                "CityFlowEngine.set_phase: intersection_id must be a "
                f"non-empty str; got {intersection_id!r}"
            )
        if isinstance(phase_index, bool) or not isinstance(phase_index, int):
            raise ValueError(
                "CityFlowEngine.set_phase: phase_index must be int; "
                f"got {type(phase_index).__name__} ({phase_index!r})"
            )
        if phase_index < 0:
            raise ValueError(
                "CityFlowEngine.set_phase: phase_index must be >= 0; "
                f"got {phase_index}"
            )

        previous = self._current_phase_index.get(intersection_id)
        is_change = previous is None or previous != phase_index

        if is_change:
            self._engine.set_tl_phase(intersection_id, phase_index)
            self._current_phase_index[intersection_id] = phase_index
            steps = (
                self.yellow_duration
                + self.all_red_duration
                + self.green_duration
            )
        else:
            steps = self.green_duration

        for _ in range(steps):
            self.next_step()

    # ------------------------------------------------------------- Internal --

    def _refresh_vehicle_tracking(self) -> None:
        """Re-poll ``cityflow.get_vehicles()`` và cập nhật internal counters.

        Idempotent — gọi nhiều lần trong cùng một timestep không sai (chỉ
        thêm các id mới vào ``_seen_vehicle_ids`` nếu CityFlow report
        thêm xe; không bao giờ remove khỏi tập hợp đã thấy).
        """
        try:
            raw = self._engine.get_vehicles(include_waiting=True)
        except TypeError:
            # Older CityFlow bindings không có ``include_waiting`` kwarg.
            raw = self._engine.get_vehicles()

        ids = self._normalize_vehicle_ids(raw)
        self._current_vehicle_ids = ids
        self._seen_vehicle_ids.update(ids)

    @staticmethod
    def _normalize_vehicle_ids(raw: Any) -> set[str]:
        """Coerce CityFlow vehicle id collection thành ``set[str]``.

        CityFlow native binding trả về ``list[str]``; some mock factories
        may return any iterable.
        """
        if raw is None:
            return set()
        if isinstance(raw, set):
            return {str(v) for v in raw}
        if isinstance(raw, (list, tuple)):
            return {str(v) for v in raw}
        if isinstance(raw, Iterable):
            return {str(v) for v in raw}
        raise TypeError(
            "CityFlowEngine: cityflow.get_vehicles() returned "
            f"non-iterable ({type(raw).__name__})"
        )

    def _prepare_replay_config(
        self,
        *,
        cfg_path: Path,
        config_data: dict,
        dataset: str,
        method: str,
        run_id: int,
    ) -> tuple[str, str]:
        """Build a replay-enabled copy of the CityFlow config.

        Returns:
            Tuple ``(effective_config_path, replay_file_absolute_path)``.

        - ``effective_config_path``: đường dẫn tuyệt đối tới file config
          mới (bản copy của ``cfg_path`` đã sửa thêm ``saveReplay`` +
          ``replayLogFile`` + ``roadnetLogFile``).
        - ``replay_file_absolute_path``: đường dẫn tuyệt đối tới file
          replay.txt CityFlow sẽ ghi vào.

        File config copy được đặt cùng thư mục với replay file để CityFlow
        resolve relative paths đúng (``dir`` field trong CityFlow config).
        """
        replay_dir = (Path.cwd() / _REPLAY_DIR).resolve()
        replay_dir.mkdir(parents=True, exist_ok=True)

        replay_basename = f"{dataset}_{method}_run{run_id}"
        replay_log = replay_dir / f"{replay_basename}.txt"
        roadnet_log = replay_dir / f"{replay_basename}_roadnet.json"
        config_copy_path = replay_dir / f"{replay_basename}_config.json"

        # CityFlow uses ``dir`` as base prefix and resolves
        # ``roadnetLogFile`` / ``replayLogFile`` relative to it. Keep
        # original ``dir`` (it points at the dataset folder) and use
        # absolute paths for the log files via overriding to absolute.
        # Concretely: we set the log filenames to be absolute paths
        # (CityFlow accepts absolute paths in these fields) so the
        # ``dir`` + filename concatenation still resolves correctly even
        # if dir has trailing slash.
        new_config = dict(config_data)
        new_config["saveReplay"] = True
        new_config["roadnetLogFile"] = str(roadnet_log)
        new_config["replayLogFile"] = str(replay_log)

        # Persist the copy.
        config_copy_path.write_text(
            json.dumps(new_config, indent=2),
            encoding="utf-8",
        )

        logger.info(
            "CityFlowEngine: save_replay=True; replay file will be "
            "written to %s (config copy at %s)",
            replay_log,
            config_copy_path,
        )

        return str(config_copy_path), str(replay_log)


__all__ = ["CityFlowEngine"]
