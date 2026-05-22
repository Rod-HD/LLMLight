"""Data models và simulation config (Component data layer).

Module này định nghĩa các dataclass dùng chung trong toàn bộ pipeline:

- :class:`IntersectionState`  — trạng thái 1 nút giao tại 1 timestep (input
  cho ``ObservationParser``).
- :class:`SimulationConfig`   — cấu hình simulation thống nhất cho mọi
  phương pháp (timing + phase set + total_timesteps), được load từ
  ``config/simulation.json``.
- :class:`MetricsResult`      — output 3 chỉ số ATT/AQL/AWT của
  ``MetricsEvaluator``.
- :class:`TokenUsageLog`      — lightweight stub cho LLM API token tracking
  (được ``MultiBackendAPIClient`` sản xuất chi tiết hơn ở Task 7.1).
- :class:`ExperimentResult`   — kết quả 1 lần chạy thí nghiệm, được
  evaluator (Task 13.7) load từ ``results/metrics/``.

Validation chạy trong ``__post_init__`` và raise :class:`ValueError` chỉ rõ
field bị vi phạm.

Module này giữ ZERO third-party dependencies (không pydantic, không numpy)
để có thể được mọi test (kể cả CI tối thiểu) import trực tiếp.

_Requirements_: 5.9, 5.10, 10.1-10.8, 11.5, 12.6, 14.5, 14.11
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, ClassVar, Iterable, Literal

# --------------------------------------------------------------------------
# Type aliases
# --------------------------------------------------------------------------

#: Tập pha tín hiệu cố định cho mọi intersection (Requirement 10 AC 6).
VALID_PHASES: tuple[str, ...] = ("ETWT", "NTST", "ELWL", "NLSL")

#: Hai mode timestep được hỗ trợ (Requirement 10 AC 5).
VALID_TOTAL_TIMESTEPS: tuple[int, ...] = (250, 3600)

#: Tên label phase được ghi vào ``ExperimentResult.phase_label``.
PhaseLabel = Literal["Phase1", "Phase2", "Phase3"]

#: Backend lựa chọn cho ``MultiBackendAPIClient`` (Requirement 7).
APIBackend = Literal["puter", "groq", "openai", "none"]

#: Liệt kê các method được ghi vào ``ExperimentResult.method``.
#: ``lightgpt_hf`` = HuggingFace pre-trained, ``lightgpt_mine`` = self-finetuned
#: (xem Requirement 5 AC 9-12). Hai giá trị này thay thế giá trị cũ
#: ``lightgpt_llama3``.
Method = Literal[
    "lightgpt_hf",
    "lightgpt_mine",
    "gpt4o_puter",
    "gpt4o_groq",
    "gpt4o_openai",
    "maxpressure",
    "advanced_maxpressure",
    "advanced_colight",
]

#: Giá trị method hợp lệ — runtime check (Literal không enforce ở runtime).
VALID_METHODS: frozenset[str] = frozenset(
    [
        "lightgpt_hf",
        "lightgpt_mine",
        "gpt4o_puter",
        "gpt4o_groq",
        "gpt4o_openai",
        "maxpressure",
        "advanced_maxpressure",
        "advanced_colight",
    ]
)

#: Phase label hợp lệ — runtime check.
VALID_PHASE_LABELS: frozenset[str] = frozenset(["Phase1", "Phase2", "Phase3"])


# --------------------------------------------------------------------------
# IntersectionState
# --------------------------------------------------------------------------


@dataclass
class IntersectionState:
    """Trạng thái một nút giao tại một timestep.

    Là input cho :class:`ObservationParser` (Component 3). CityFlow Engine
    (Component 1) sản xuất ra dict này tại mỗi điểm quyết định.

    Attributes:
        intersection_id: ID intersection trong ``roadnet.json``.
        lane_vehicle_count: Mapping ``lane_id -> queue length``. Mỗi giá trị
            phải là ``int >= 0``. KHÔNG có giới hạn trên cố định — sức chứa
            lane phụ thuộc dataset.
        current_phase: Pha hiện tại, phải thuộc ``VALID_PHASES``.
        current_phase_time: Số giây đã trôi qua kể từ khi pha bắt đầu, ``>= 0``.
    """

    intersection_id: str
    lane_vehicle_count: dict[str, int]
    current_phase: str
    current_phase_time: int

    def __post_init__(self) -> None:
        if not isinstance(self.intersection_id, str) or not self.intersection_id:
            raise ValueError(
                "IntersectionState.intersection_id must be a non-empty str; "
                f"got {self.intersection_id!r}"
            )

        if not isinstance(self.lane_vehicle_count, dict):
            raise ValueError(
                "IntersectionState.lane_vehicle_count must be a dict; "
                f"got {type(self.lane_vehicle_count).__name__}"
            )
        for lane_id, count in self.lane_vehicle_count.items():
            if not isinstance(lane_id, str) or not lane_id:
                raise ValueError(
                    "IntersectionState.lane_vehicle_count keys must be "
                    f"non-empty str; got {lane_id!r}"
                )
            # bool is subclass of int — reject explicitly to avoid silent bug.
            if isinstance(count, bool) or not isinstance(count, int):
                raise ValueError(
                    f"IntersectionState.lane_vehicle_count[{lane_id!r}] must "
                    f"be int; got {type(count).__name__} ({count!r})"
                )
            if count < 0:
                raise ValueError(
                    f"IntersectionState.lane_vehicle_count[{lane_id!r}] must "
                    f"be >= 0; got {count}"
                )

        if self.current_phase not in VALID_PHASES:
            raise ValueError(
                "IntersectionState.current_phase must be one of "
                f"{sorted(VALID_PHASES)}; got {self.current_phase!r}"
            )

        if isinstance(self.current_phase_time, bool) or not isinstance(
            self.current_phase_time, int
        ):
            raise ValueError(
                "IntersectionState.current_phase_time must be int; "
                f"got {type(self.current_phase_time).__name__} "
                f"({self.current_phase_time!r})"
            )
        if self.current_phase_time < 0:
            raise ValueError(
                "IntersectionState.current_phase_time must be >= 0; "
                f"got {self.current_phase_time}"
            )


# --------------------------------------------------------------------------
# SimulationConfig
# --------------------------------------------------------------------------


@dataclass
class SimulationConfig:
    """Cấu hình simulation thống nhất cho tất cả phương pháp.

    Tham chiếu đến ``config/simulation.json`` (Requirement 10 AC 7). Mọi
    runner script phải load file này qua :meth:`from_json` và truyền xuống
    ``CityFlowEngine`` để đảm bảo timing + phase set đồng nhất giữa các
    phương pháp.

    Attributes:
        roadnet_file: Tên file roadnet (key dataset trong simulation.json,
            ví dụ ``"anon_3_4_jinan_real"``).
        flow_file: Tên file flow (vehicle arrival pattern). Có thể trùng
            với ``roadnet_file`` cho các dataset standard.
        green_duration: Thời lượng pha xanh tối thiểu, mặc định 30s
            (Requirement 10 AC 1). Phải ``> 0``.
        yellow_duration: Thời lượng pha vàng, mặc định 3s
            (Requirement 10 AC 2). Phải ``> 0``.
        all_red_duration: Thời lượng all-red, mặc định 2s
            (Requirement 10 AC 3). Phải ``> 0``.
        total_timesteps: Tổng số timestep chạy simulation, phải ∈
            ``{250, 3600}`` (Requirement 10 AC 5).
        phase_set: Tập pha hợp lệ, mặc định ``VALID_PHASES``
            (Requirement 10 AC 6).
    """

    roadnet_file: str
    flow_file: str
    green_duration: int = 30
    yellow_duration: int = 3
    all_red_duration: int = 2
    total_timesteps: int = 3600
    phase_set: tuple[str, ...] = VALID_PHASES

    def __post_init__(self) -> None:
        if not isinstance(self.roadnet_file, str) or not self.roadnet_file:
            raise ValueError(
                "SimulationConfig.roadnet_file must be a non-empty str; "
                f"got {self.roadnet_file!r}"
            )
        if not isinstance(self.flow_file, str) or not self.flow_file:
            raise ValueError(
                "SimulationConfig.flow_file must be a non-empty str; "
                f"got {self.flow_file!r}"
            )

        for fname in ("green_duration", "yellow_duration", "all_red_duration"):
            value = getattr(self, fname)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(
                    f"SimulationConfig.{fname} must be int; "
                    f"got {type(value).__name__} ({value!r})"
                )
            if value <= 0:
                raise ValueError(
                    f"SimulationConfig.{fname} must be > 0; got {value}"
                )

        if isinstance(self.total_timesteps, bool) or not isinstance(
            self.total_timesteps, int
        ):
            raise ValueError(
                "SimulationConfig.total_timesteps must be int; "
                f"got {type(self.total_timesteps).__name__} "
                f"({self.total_timesteps!r})"
            )
        if self.total_timesteps not in VALID_TOTAL_TIMESTEPS:
            raise ValueError(
                "SimulationConfig.total_timesteps must be one of "
                f"{list(VALID_TOTAL_TIMESTEPS)}; got {self.total_timesteps}"
            )

        # phase_set: chấp nhận tuple/list, normalize về tuple, kiểm tra phase
        # nằm trong VALID_PHASES và không trùng lặp.
        if isinstance(self.phase_set, list):
            self.phase_set = tuple(self.phase_set)
        if not isinstance(self.phase_set, tuple):
            raise ValueError(
                "SimulationConfig.phase_set must be a tuple/list of str; "
                f"got {type(self.phase_set).__name__}"
            )
        if not self.phase_set:
            raise ValueError(
                "SimulationConfig.phase_set must not be empty"
            )
        seen: set[str] = set()
        for phase in self.phase_set:
            if not isinstance(phase, str):
                raise ValueError(
                    "SimulationConfig.phase_set entries must be str; "
                    f"got {type(phase).__name__} ({phase!r})"
                )
            if phase not in VALID_PHASES:
                raise ValueError(
                    "SimulationConfig.phase_set entries must be in "
                    f"{sorted(VALID_PHASES)}; got {phase!r}"
                )
            if phase in seen:
                raise ValueError(
                    "SimulationConfig.phase_set must not contain duplicates; "
                    f"{phase!r} appears more than once"
                )
            seen.add(phase)

    # ----------------------------------------------------------------- I/O --

    @classmethod
    def from_json(
        cls,
        path: str | Path,
        *,
        dataset: str,
        mode: Literal["demo", "full"] = "full",
    ) -> "SimulationConfig":
        """Load ``SimulationConfig`` từ ``config/simulation.json``.

        File ``simulation.json`` chứa tham số timing chung + section
        ``datasets`` (Phase 1/2) và ``phase3_datasets`` (Phase 3). Method
        này resolve dataset theo cả hai section, chọn ``total_timesteps``
        theo ``mode`` (``"demo"`` → 250, ``"full"`` → 3600), và normalize
        ``flow_file`` (mặc định bằng ``roadnet_file`` nếu file không
        khai báo riêng).

        Args:
            path: Đường dẫn tới ``simulation.json``.
            dataset: Key dataset (ví dụ ``"jinan_1"``, ``"hangzhou_1"``,
                ``"newyork_1"``).
            mode: ``"demo"`` cho Phase 1 (250 timesteps), ``"full"`` cho
                Phase 2 / Phase 3 (3600 timesteps).

        Returns:
            ``SimulationConfig`` với validation đã chạy.

        Raises:
            FileNotFoundError: ``path`` không tồn tại.
            ValueError: JSON malformed, dataset không khai báo, mode không
                hợp lệ, hoặc trường timing thiếu / sai kiểu.
        """
        if mode not in ("demo", "full"):
            raise ValueError(
                f"SimulationConfig.from_json: mode must be 'demo' or 'full'; "
                f"got {mode!r}"
            )

        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(
                f"SimulationConfig.from_json: config file not found: "
                f"{config_path}"
            )

        try:
            raw = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"SimulationConfig.from_json: failed to parse JSON at "
                f"{config_path}: {exc}"
            ) from exc

        if not isinstance(raw, dict):
            raise ValueError(
                f"SimulationConfig.from_json: top-level JSON must be an "
                f"object; got {type(raw).__name__}"
            )

        # Tìm dataset block ở 1 trong 2 section.
        datasets_block = raw.get("datasets") or {}
        phase3_block = raw.get("phase3_datasets") or {}
        dataset_entry: dict[str, Any] | None = None
        if isinstance(datasets_block, dict) and dataset in datasets_block:
            dataset_entry = datasets_block[dataset]
        elif isinstance(phase3_block, dict) and dataset in phase3_block:
            dataset_entry = phase3_block[dataset]

        if dataset_entry is None:
            available = sorted(
                [k for k in datasets_block if not k.startswith("_")]
                + [k for k in phase3_block if not k.startswith("_")]
            )
            raise ValueError(
                f"SimulationConfig.from_json: dataset {dataset!r} not "
                f"declared in {config_path}. Available: {available}"
            )

        if not isinstance(dataset_entry, dict):
            raise ValueError(
                f"SimulationConfig.from_json: dataset {dataset!r} entry must "
                f"be an object; got {type(dataset_entry).__name__}"
            )

        roadnet_file = dataset_entry.get("roadnet_file")
        if not isinstance(roadnet_file, str) or not roadnet_file:
            raise ValueError(
                f"SimulationConfig.from_json: dataset {dataset!r} missing "
                f"'roadnet_file' (must be non-empty str)"
            )
        # ``flow_file`` mặc định bằng ``roadnet_file`` (CityFlow convention
        # cho các dataset standard).
        flow_file = dataset_entry.get("flow_file", roadnet_file)
        if not isinstance(flow_file, str) or not flow_file:
            raise ValueError(
                f"SimulationConfig.from_json: dataset {dataset!r} 'flow_file' "
                f"must be a non-empty str if provided"
            )

        # Timing block — sử dụng default trong dataclass nếu thiếu, nhưng kiểu
        # phải đúng nếu được khai báo.
        def _get_int(key: str, default: int) -> int:
            value = raw.get(key, default)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(
                    f"SimulationConfig.from_json: top-level field {key!r} "
                    f"must be int; got {type(value).__name__} ({value!r})"
                )
            return value

        green_duration = _get_int("green_duration", 30)
        yellow_duration = _get_int("yellow_duration", 3)
        all_red_duration = _get_int("all_red_duration", 2)

        total_timesteps_demo = _get_int("total_timesteps_demo", 250)
        total_timesteps_full = _get_int("total_timesteps_full", 3600)
        total_timesteps = (
            total_timesteps_demo if mode == "demo" else total_timesteps_full
        )

        phase_set_raw = raw.get("phase_set", list(VALID_PHASES))
        if not isinstance(phase_set_raw, (list, tuple)):
            raise ValueError(
                "SimulationConfig.from_json: 'phase_set' must be a list/tuple; "
                f"got {type(phase_set_raw).__name__}"
            )
        phase_set = tuple(phase_set_raw)

        return cls(
            roadnet_file=roadnet_file,
            flow_file=flow_file,
            green_duration=green_duration,
            yellow_duration=yellow_duration,
            all_red_duration=all_red_duration,
            total_timesteps=total_timesteps,
            phase_set=phase_set,
        )

    # ---------------------------------------------------------- Compare ----

    #: Các trường mặc định bị so sánh khi runner enforce
    #: Requirement 10 AC 8 (tham số phương pháp phải khớp config).
    DEFAULT_COMPARE_FIELDS: tuple[str, ...] = (
        "green_duration",
        "yellow_duration",
        "all_red_duration",
        "total_timesteps",
        "phase_set",
    )

    def compare(
        self,
        other: "SimulationConfig",
        fields: Iterable[str] | None = None,
    ) -> None:
        """Verify rằng ``other`` khớp ``self`` ở các trường được chỉ định.

        Được runner script gọi sau khi build ``SimulationConfig`` từ CLI
        flags hoặc từ override (nếu có) để enforce Requirement 10 AC 7-8:
        nếu phương pháp chạy với tham số khác ``config/simulation.json``,
        raise ``ValueError`` chỉ rõ tham số không khớp.

        Args:
            other: Config khác cần so sánh (thường là config build từ CLI).
            fields: Danh sách field để so sánh. Mặc định
                :attr:`DEFAULT_COMPARE_FIELDS` (timing + phase set + total
                timesteps; KHÔNG so sánh ``roadnet_file`` / ``flow_file``
                vì hai runner cùng phương pháp có thể khác dataset).

        Raises:
            ValueError: Nếu có ít nhất một field khác nhau. Message liệt kê
                tên field, giá trị self, và giá trị other.
        """
        if not isinstance(other, SimulationConfig):
            raise ValueError(
                "SimulationConfig.compare: 'other' must be SimulationConfig; "
                f"got {type(other).__name__}"
            )

        compare_fields = (
            tuple(fields) if fields is not None else self.DEFAULT_COMPARE_FIELDS
        )

        valid_field_names = {f.name for f in self_dataclass_fields(self)}
        unknown = [f for f in compare_fields if f not in valid_field_names]
        if unknown:
            raise ValueError(
                f"SimulationConfig.compare: unknown field(s) {unknown}. "
                f"Available: {sorted(valid_field_names)}"
            )

        mismatches: list[str] = []
        for fname in compare_fields:
            self_val = getattr(self, fname)
            other_val = getattr(other, fname)
            if self_val != other_val:
                mismatches.append(
                    f"{fname}: config={self_val!r} runner={other_val!r}"
                )

        if mismatches:
            raise ValueError(
                "SimulationConfig.compare: parameter mismatch with "
                "config/simulation.json (Requirement 10 AC 7-8). "
                "Mismatched fields:\n  - "
                + "\n  - ".join(mismatches)
            )


def self_dataclass_fields(obj: Any) -> tuple:
    """Helper: trả về :func:`dataclasses.fields` cho một dataclass instance.

    Tách ra để mock dễ trong test (không bắt buộc), và vì gọi trực tiếp
    ``fields(self)`` gây nhầm lẫn với param ``fields`` của ``compare``.
    """
    return fields(obj)


# --------------------------------------------------------------------------
# MetricsResult
# --------------------------------------------------------------------------


@dataclass
class MetricsResult:
    """Output 3 chỉ số ATT/AQL/AWT của ``MetricsEvaluator`` (Component 8).

    Bound (Requirement 9 AC 4 / 12 AC 6):
      - ``att`` ∈ ``[0, total_timesteps]``
      - ``aql`` ∈ ``[0, max_lane_capacity_observed]``
      - ``awt`` ∈ ``[0, total_timesteps]``

    Validation ở đây CHỈ kiểm tra ``>= 0`` (bound trên phụ thuộc dataset
    + total_timesteps nên được kiểm tra ở evaluator, không ở dataclass).

    Attributes:
        att: Average Travel Time (giây), 2 decimal places.
        aql: Average Queue Length (xe / lane), 2 decimal places.
        awt: Average Waiting Time (giây), 2 decimal places.
    """

    att: float
    aql: float
    awt: float

    def __post_init__(self) -> None:
        for fname in ("att", "aql", "awt"):
            value = getattr(self, fname)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    f"MetricsResult.{fname} must be a number (int/float); "
                    f"got {type(value).__name__} ({value!r})"
                )
            # Cho phép int → cast thành float để ổn định serialization.
            setattr(self, fname, float(value))
            if getattr(self, fname) < 0:
                raise ValueError(
                    f"MetricsResult.{fname} must be >= 0; got {value}"
                )


# --------------------------------------------------------------------------
# TokenUsageLog (lightweight stub)
# --------------------------------------------------------------------------


@dataclass
class TokenUsageLog:
    """Tóm tắt token usage cho 1 run LLM API.

    ``MultiBackendAPIClient`` (Task 7.1) sẽ ghi đè / mở rộng cấu trúc này
    với chi tiết per-request (errors, retries). Đây là stub tối thiểu để
    ``ExperimentResult.token_usage`` type-check.
    """

    backend: APIBackend = "none"
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_requests: int = 0

    def __post_init__(self) -> None:
        if self.backend not in ("puter", "groq", "openai", "none"):
            raise ValueError(
                "TokenUsageLog.backend must be one of "
                "{'puter','groq','openai','none'}; "
                f"got {self.backend!r}"
            )
        for fname in ("total_input_tokens", "total_output_tokens", "total_requests"):
            value = getattr(self, fname)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(
                    f"TokenUsageLog.{fname} must be int; "
                    f"got {type(value).__name__} ({value!r})"
                )
            if value < 0:
                raise ValueError(
                    f"TokenUsageLog.{fname} must be >= 0; got {value}"
                )


# --------------------------------------------------------------------------
# ExperimentResult
# --------------------------------------------------------------------------


@dataclass
class ExperimentResult:
    """Kết quả một lần chạy thí nghiệm (1 run).

    Được runner ghi vào ``results/metrics/<dataset>_<method>_<phase>_run<N>.json``;
    evaluator (Task 13.7) load tất cả các file và sản xuất bảng comparison.

    Attributes:
        method: Tên phương pháp; phải thuộc :data:`VALID_METHODS`.
            Lưu ý hai biến thể LightGPT 0.5B chạy song song
            (``lightgpt_hf`` / ``lightgpt_mine``) thay thế giá trị cũ
            ``lightgpt_llama3`` (Requirement 5 AC 9).
        dataset: Dataset key; phải ∈ ``{"jinan_1", "hangzhou_1", "newyork_1"}``.
        run_id: Index run trong batch (Phase 1 LLM API = 1 run; baseline +
            LightGPT local = 3 runs; Phase 2/3 = 3 runs cho mọi method).
        seed: Seed thực tế đã apply (= ``RANDOM_SEED + run_id``).
        metrics: ``MetricsResult`` đã compute.
        token_usage: ``TokenUsageLog`` cho LLM API methods, ``None`` cho
            LightGPT local + baselines.
        duration_seconds: Wall-clock time của run.
        timestamp: ISO 8601 string (UTC khuyến khích).
        phase_label: ``"Phase1"`` / ``"Phase2"`` / ``"Phase3"`` — thay thế
            kiểu cũ ``Literal["Phase1", "Phase2"]`` (Requirement 13 AC 6).
        replay_file: Đường dẫn ``replay.txt`` do CityFlow ghi khi chạy với
            ``save_replay=True``; ``None`` nếu không save (Phase 2/3 mặc định
            ``None`` để tiết kiệm I/O). UI Demo (Component 13, Task 18) đọc
            field này để xác định replay file nào tương ứng với run.
    """

    method: str
    dataset: str
    run_id: int
    seed: int
    metrics: MetricsResult
    token_usage: TokenUsageLog | None
    duration_seconds: float
    timestamp: str
    phase_label: PhaseLabel
    replay_file: str | None = None
    api_cache_manifest: str | None = None
    run_fingerprint: str | None = None

    # Dataset hợp lệ — class-level constant, KHÔNG phải dataclass field.
    _VALID_DATASETS: ClassVar[frozenset[str]] = frozenset(
        ["jinan_1", "hangzhou_1", "newyork_1"]
    )

    def __post_init__(self) -> None:
        if self.method not in VALID_METHODS:
            raise ValueError(
                "ExperimentResult.method must be one of "
                f"{sorted(VALID_METHODS)}; got {self.method!r}"
            )

        if self.dataset not in self._VALID_DATASETS:
            raise ValueError(
                "ExperimentResult.dataset must be one of "
                f"{sorted(self._VALID_DATASETS)}; got {self.dataset!r}"
            )

        if isinstance(self.run_id, bool) or not isinstance(self.run_id, int):
            raise ValueError(
                "ExperimentResult.run_id must be int; "
                f"got {type(self.run_id).__name__} ({self.run_id!r})"
            )
        if self.run_id < 0:
            raise ValueError(
                f"ExperimentResult.run_id must be >= 0; got {self.run_id}"
            )

        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError(
                "ExperimentResult.seed must be int; "
                f"got {type(self.seed).__name__} ({self.seed!r})"
            )

        if not isinstance(self.metrics, MetricsResult):
            raise ValueError(
                "ExperimentResult.metrics must be MetricsResult; "
                f"got {type(self.metrics).__name__}"
            )

        if self.token_usage is not None and not isinstance(
            self.token_usage, TokenUsageLog
        ):
            raise ValueError(
                "ExperimentResult.token_usage must be TokenUsageLog or None; "
                f"got {type(self.token_usage).__name__}"
            )

        if isinstance(self.duration_seconds, bool) or not isinstance(
            self.duration_seconds, (int, float)
        ):
            raise ValueError(
                "ExperimentResult.duration_seconds must be a number; "
                f"got {type(self.duration_seconds).__name__} "
                f"({self.duration_seconds!r})"
            )
        if self.duration_seconds < 0:
            raise ValueError(
                "ExperimentResult.duration_seconds must be >= 0; "
                f"got {self.duration_seconds}"
            )
        # Normalize int → float để consistent JSON output.
        self.duration_seconds = float(self.duration_seconds)

        if not isinstance(self.timestamp, str) or not self.timestamp:
            raise ValueError(
                "ExperimentResult.timestamp must be a non-empty str; "
                f"got {self.timestamp!r}"
            )

        if self.phase_label not in VALID_PHASE_LABELS:
            raise ValueError(
                "ExperimentResult.phase_label must be one of "
                f"{sorted(VALID_PHASE_LABELS)}; got {self.phase_label!r}"
            )

        if self.replay_file is not None and (
            not isinstance(self.replay_file, str) or not self.replay_file
        ):
            raise ValueError(
                "ExperimentResult.replay_file must be a non-empty str or None; "
                f"got {self.replay_file!r}"
            )

        if self.api_cache_manifest is not None and (
            not isinstance(self.api_cache_manifest, str)
            or not self.api_cache_manifest
        ):
            raise ValueError(
                "ExperimentResult.api_cache_manifest must be a non-empty "
                f"str or None; got {self.api_cache_manifest!r}"
            )

        if self.run_fingerprint is not None and (
            not isinstance(self.run_fingerprint, str)
            or not self.run_fingerprint
        ):
            raise ValueError(
                "ExperimentResult.run_fingerprint must be a non-empty str "
                f"or None; got {self.run_fingerprint!r}"
            )


# --------------------------------------------------------------------------
# BaselineResult — output cho RL baselines (Component 8)
# --------------------------------------------------------------------------


#: Method hợp lệ cho ``BaselineResult`` (Requirement 8).
VALID_BASELINE_METHODS: frozenset[str] = frozenset(
    [
        "maxpressure",
        "advanced_maxpressure",
        "advanced_colight",
    ]
)

#: Dataset hợp lệ cho ``BaselineResult`` — chia sẻ với ``ExperimentResult``.
_VALID_BASELINE_DATASETS: frozenset[str] = frozenset(
    ["jinan_1", "hangzhou_1", "newyork_1"]
)


@dataclass
class BaselineResult:
    """Kết quả 1 run của RL baseline (Component 8 — RLBaselinesRunner).

    Là payload return của ``RLBaselinesRunner.run_*`` (Task 11.1-11.3).
    Sau đó runner chính (Task 13) wrap thành :class:`ExperimentResult`
    để ghi vào ``results/metrics/`` và đưa vào bảng comparison.

    Attributes:
        method: Tên baseline; phải thuộc :data:`VALID_BASELINE_METHODS`
            (``"maxpressure"`` | ``"advanced_maxpressure"`` |
            ``"advanced_colight"``).
        dataset: Dataset key; phải thuộc
            ``{"jinan_1", "hangzhou_1", "newyork_1"}``.
        att: Average Travel Time (giây) — ``>= 0``, làm tròn 2 chữ số
            thập phân.
        aql: Average Queue Length (xe / lane) — ``>= 0``, 2 chữ số
            thập phân.
        awt: Average Waiting Time (giây) — ``>= 0``, 2 chữ số thập phân.
        run_id: Index run trong batch (0, 1, 2 — Requirement 8 AC 8).
        seed: Seed thực tế đã apply cho run này
            (= ``RANDOM_SEED + run_id`` — Requirement 8 AC 9).
    """

    method: str
    dataset: str
    att: float
    aql: float
    awt: float
    run_id: int
    seed: int

    def __post_init__(self) -> None:
        if self.method not in VALID_BASELINE_METHODS:
            raise ValueError(
                "BaselineResult.method must be one of "
                f"{sorted(VALID_BASELINE_METHODS)}; got {self.method!r}"
            )

        if self.dataset not in _VALID_BASELINE_DATASETS:
            raise ValueError(
                "BaselineResult.dataset must be one of "
                f"{sorted(_VALID_BASELINE_DATASETS)}; got {self.dataset!r}"
            )

        for fname in ("att", "aql", "awt"):
            value = getattr(self, fname)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    f"BaselineResult.{fname} must be a number (int/float); "
                    f"got {type(value).__name__} ({value!r})"
                )
            fv = float(value)
            if fv < 0:
                raise ValueError(
                    f"BaselineResult.{fname} must be >= 0; got {value}"
                )
            setattr(self, fname, fv)

        if isinstance(self.run_id, bool) or not isinstance(self.run_id, int):
            raise ValueError(
                "BaselineResult.run_id must be int; "
                f"got {type(self.run_id).__name__} ({self.run_id!r})"
            )
        if self.run_id < 0:
            raise ValueError(
                f"BaselineResult.run_id must be >= 0; got {self.run_id}"
            )

        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError(
                "BaselineResult.seed must be int; "
                f"got {type(self.seed).__name__} ({self.seed!r})"
            )
