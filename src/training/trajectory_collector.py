"""Trajectory Collector (Component 7 — Task 10.1).

Thu thập trajectory ``(prompt, raw_response)`` từ một LLM "teacher" (GPT-4o
hoặc Llama-3.3-70B) để dùng làm dữ liệu IFT cho Qwen2-0.5B.

Phụ thuộc theo Component 7 trong design:

* :class:`~src.training.multi_backend_api_client.MultiBackendAPIClient` —
  gọi LLM remote qua một trong 3 backend (priority OPENAI > GROQ > PUTER).
* :class:`~src.observation_parser.ObservationParser` — chuyển trạng thái
  CityFlow tại 1 timestep thành prompt text 3-section.
* :class:`~src.response_parser.ResponseParser` — verify response chứa
  ``<signal>...</signal>`` với value ∈ ``{ETWT, NTST, ELWL, NLSL}``.

Hành vi theo phase (Requirement 6 AC 8 + Glossary Phase1/Phase2/Phase3):

============= ============================== ==============================
phase         Min valid samples              Max requests
============= ============================== ==============================
1 (Demo)      50                             100  (PUTER quota / debug)
2 (Full)      200                            500  (Groq/OpenAI budget)
3 (Full)      200                            500
============= ============================== ==============================

* Tại mỗi vòng lặp:

  1. Đọc trạng thái từ engine (``get_lane_vehicle_count`` + tracking nội
     bộ cho ``current_phase`` / ``current_phase_time``).
  2. Tạo prompt qua ``ObservationParser.parse`` (deterministic — Property 2).
  3. Gọi ``api_client.chat_completion(prompt)`` → ``APIResponse``.
  4. Nếu raw response chứa ``<signal>`` với value hợp lệ: lưu cặp
     ``(prompt, raw_response)`` vào trajectory (KHÔNG strip / normalize
     response — IFT trainer cần raw text để học format).
  5. Áp dụng phase qua ``engine.set_phase(intersection_id, phase_index)``
     để simulation tiến tiếp realistically (CityFlow tự xử lý timing
     yellow/all-red/green; xem ``CityFlowEngine.set_phase`` docstring).

* Vòng lặp dừng khi:

  * Đã đạt ``num_samples`` valid samples; HOẶC
  * Đã dùng ``max_requests`` requests (theo phase); HOẶC
  * ``MultiBackendAPIClient.chat_completion`` raise
    :class:`~src.training.multi_backend_api_client.RequestLimitExceeded`
    (Puter quota cứng 100 req/run).

* Sau khi dừng, nếu ``len(samples) < min_valid_samples`` → raise
  :class:`InsufficientDataError`.

Module này KHÔNG tự khởi chạy CityFlow simulation; engine phải đã được
khởi tạo và ở trạng thái sẵn sàng (``next_step()`` callable). Trong unit
tests, engine + api_client + obs_parser được mock; chỉ ``ResponseParser``
là thật để verify signal extraction (Property 1 / 4).

_Requirements_: 6.7, 6.8
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Final, Protocol

from src.observation_parser import ObservationParser
from src.response_parser import ResponseParser
from src.training.multi_backend_api_client import (
    MultiBackendAPIClient,
    RequestLimitExceeded,
)

logger = logging.getLogger(__name__)

__all__ = [
    "InsufficientDataError",
    "PhaseIndexResolver",
    "TrajectoryCollector",
]


class InsufficientDataError(RuntimeError):
    """Raised khi :meth:`TrajectoryCollector.collect` không thu thập được
    đủ valid samples theo ngưỡng phase (50 ở Phase 1, 200 ở Phase 2/3)
    sau khi đã dùng hết request budget hoặc backend đã raise
    :class:`RequestLimitExceeded`.

    Subclass của :class:`RuntimeError` để runner có thể catch ở mức
    rộng nếu cần fallback (ví dụ chuyển sang backend khác hoặc dataset
    khác).
    """


class PhaseIndexResolver(Protocol):
    """Callable resolve ``(intersection_id, phase_name)`` → ``phase_index``.

    Trong production, instance của
    :class:`~src.phase_index_mapper.PhaseIndexMapper` cung cấp method
    :meth:`PhaseIndexMapper.get_index` đúng signature này. Trong tests,
    có thể inject một lambda đơn giản.

    Method này được gọi với:

    * ``intersection_id``: ID intersection trong roadnet (str).
    * ``phase_name``: Một trong ``{"ETWT", "NTST", "ELWL", "NLSL"}``.

    Kỳ vọng raise ``KeyError`` nếu intersection / phase không có mapping.
    """

    def __call__(self, intersection_id: str, phase_name: str) -> int: ...


# Default mapping — chỉ dùng làm fallback khi caller KHÔNG inject
# ``phase_index_resolver``. KHÔNG nên dùng trong production vì thứ tự pha
# trong CityFlow ``roadnet.json`` không cố định giữa các dataset
# (Requirement 2 AC 5).
_DEFAULT_PHASE_INDEX_MAPPING: Final[dict[str, int]] = {
    "ETWT": 0,
    "NTST": 1,
    "ELWL": 2,
    "NLSL": 3,
}


def _default_phase_index_resolver(
    intersection_id: str, phase_name: str
) -> int:
    """Fallback resolver dùng mapping cố định ``ETWT=0, NTST=1, ELWL=2, NLSL=3``.

    Chỉ phù hợp cho unit tests — production phải truyền
    :meth:`PhaseIndexMapper.get_index` qua param ``phase_index_resolver``.
    """
    if phase_name not in _DEFAULT_PHASE_INDEX_MAPPING:
        raise KeyError(
            f"Unknown phase {phase_name!r} for intersection "
            f"{intersection_id!r}; expected one of "
            f"{sorted(_DEFAULT_PHASE_INDEX_MAPPING)}"
        )
    return _DEFAULT_PHASE_INDEX_MAPPING[phase_name]


class TrajectoryCollector:
    """Thu thập ``(prompt, raw_response)`` pairs từ LLM teacher.

    Class state hoàn toàn xác định bởi 4 dependency truyền qua constructor:
    api_client, obs_parser, resp_parser, phase. ``collect`` không thay
    đổi state nội bộ (tạo trajectory mới mỗi lần gọi) — có thể gọi nhiều
    lần với engine khác nhau.
    """

    #: Min valid samples cho Phase 1 (Demo) — dùng cho debug / pipeline
    #: validate trong free quota.
    MIN_VALID_SAMPLES_PHASE1: Final[int] = 50

    #: Min valid samples cho Phase 2 — fine-tuning Qwen2-0.5B có ý nghĩa.
    MIN_VALID_SAMPLES_PHASE2: Final[int] = 200

    #: Min valid samples cho Phase 3 — đồng nhất với Phase 2.
    MIN_VALID_SAMPLES_PHASE3: Final[int] = 200

    #: Max requests cho Phase 1 — match Puter quota cứng (100 req/run).
    MAX_REQUESTS_PHASE1: Final[int] = 100

    #: Max requests cho Phase 2/3 — budget cap khi dùng Groq / OpenAI;
    #: ngăn collector chạy vô hạn nếu hầu hết responses đều invalid.
    MAX_REQUESTS_FULL: Final[int] = 500

    #: Tần suất log progress (mỗi N valid samples).
    _LOG_EVERY: Final[int] = 10

    #: Default intersection ID khi caller không truyền (chủ yếu cho test).
    _DEFAULT_INTERSECTION_ID: Final[str] = "intersection_0"

    #: Phase mặc định để khởi tạo prompt đầu tiên.
    _INITIAL_PHASE: Final[str] = "ETWT"

    def __init__(
        self,
        api_client: MultiBackendAPIClient,
        obs_parser: ObservationParser,
        resp_parser: ResponseParser,
        phase: int,
    ) -> None:
        """Khởi tạo collector.

        Args:
            api_client: ``MultiBackendAPIClient`` đã chọn xong backend.
                Trong Phase 1, có thể là Puter (≤100 req/run); Phase 2/3
                phải là Groq hoặc OpenAI (Puter sẽ bị
                ``MultiBackendAPIClient`` reject ở mode=full).
            obs_parser: ``ObservationParser`` (Component 3).
            resp_parser: ``ResponseParser`` (Component 4) dùng cho fallback
                logging — collector tự verify validity bằng regex riêng
                để có thể distinguish "valid signal" vs "fallback ETWT".
            phase: ``1``, ``2``, hoặc ``3`` — quyết định ``min_valid_samples``
                và ``max_requests``.

        Raises:
            ValueError: Nếu ``phase`` không thuộc ``{1, 2, 3}``.
        """
        if isinstance(phase, bool) or not isinstance(phase, int):
            raise ValueError(
                "TrajectoryCollector: phase must be int in {1, 2, 3}; "
                f"got {type(phase).__name__} ({phase!r})"
            )
        if phase not in (1, 2, 3):
            raise ValueError(
                "TrajectoryCollector: phase must be 1, 2, or 3; "
                f"got {phase}"
            )

        self._api_client = api_client
        self._obs_parser = obs_parser
        self._resp_parser = resp_parser
        self._phase = phase

        if phase == 1:
            self._min_valid_samples = self.MIN_VALID_SAMPLES_PHASE1
            self._max_requests = self.MAX_REQUESTS_PHASE1
        elif phase == 2:
            self._min_valid_samples = self.MIN_VALID_SAMPLES_PHASE2
            self._max_requests = self.MAX_REQUESTS_FULL
        else:  # phase == 3
            self._min_valid_samples = self.MIN_VALID_SAMPLES_PHASE3
            self._max_requests = self.MAX_REQUESTS_FULL

    # --------------------------------------------------------- Properties --

    @property
    def phase(self) -> int:
        """Phase đã được constructor lock (1, 2, hoặc 3)."""
        return self._phase

    @property
    def min_valid_samples(self) -> int:
        """Ngưỡng tối thiểu (50 ở Phase 1, 200 ở Phase 2/3)."""
        return self._min_valid_samples

    @property
    def max_requests(self) -> int:
        """Budget tối đa requests gửi đi (100 ở Phase 1, 500 ở Phase 2/3)."""
        return self._max_requests

    # -------------------------------------------------------------- collect --

    def collect(
        self,
        engine: Any,
        num_samples: int,
        *,
        intersection_id: str | None = None,
        phase_index_resolver: Callable[[str, str], int] | None = None,
    ) -> list[tuple[str, str]]:
        """Thu thập trajectory ``(prompt, raw_response)`` pairs.

        Args:
            engine: Đối tượng giống ``CityFlowEngine`` cung cấp
                ``get_lane_vehicle_count() -> dict[str, int]``,
                ``set_phase(intersection_id, phase_index)``, và (tùy chọn)
                ``next_step()``. Trong test, engine được mock đầy đủ.
            num_samples: Số valid samples mong muốn (cap trên).
            intersection_id: ID intersection để gọi ``set_phase``. Mặc định
                ``"intersection_0"`` (sufficient cho unit tests; production
                runner phải truyền ID thật từ roadnet).
            phase_index_resolver: Callable resolve
                ``(intersection_id, phase_name) -> phase_index``. Mặc định
                dùng mapping cố định ``ETWT=0/NTST=1/ELWL=2/NLSL=3`` —
                CHỈ phù hợp cho test; production phải truyền
                :meth:`PhaseIndexMapper.get_index`.

        Returns:
            ``list[tuple[str, str]]`` — mỗi cặp gồm prompt (output của
            ``ObservationParser.parse``) và raw response (giữ nguyên text
            từ LLM, KHÔNG strip / normalize). Độ dài luôn ``>=
            min_valid_samples``; trường hợp khác → raise
            :class:`InsufficientDataError`.

        Raises:
            InsufficientDataError: Khi ``len(samples) < min_valid_samples``
                sau khi đã dùng hết ``max_requests`` hoặc backend
                ``RequestLimitExceeded``.
            ValueError: Nếu ``num_samples < min_valid_samples`` (caller
                yêu cầu ít hơn ngưỡng phase — không hợp lý, sẽ luôn fail).
        """
        if isinstance(num_samples, bool) or not isinstance(num_samples, int):
            raise ValueError(
                "TrajectoryCollector.collect: num_samples must be int; "
                f"got {type(num_samples).__name__} ({num_samples!r})"
            )
        if num_samples < self._min_valid_samples:
            raise ValueError(
                f"TrajectoryCollector.collect: num_samples ({num_samples}) "
                f"must be >= min_valid_samples for phase {self._phase} "
                f"({self._min_valid_samples}); otherwise the call would "
                "always raise InsufficientDataError."
            )

        if intersection_id is None or not intersection_id:
            intersection_id = self._DEFAULT_INTERSECTION_ID
        if phase_index_resolver is None:
            phase_index_resolver = _default_phase_index_resolver

        samples: list[tuple[str, str]] = []
        request_count = 0

        # Internal phase tracking — feeds into ``ObservationParser.parse``.
        # CityFlow doesn't expose current_phase directly via the wrapper,
        # so we mirror it ourselves. Phase time is approximated by adding
        # ``_PHASE_TIME_TICK`` after each decision; ObservationParser only
        # requires ``int >= 0``, so the exact value doesn't affect IFT
        # data quality (the LLM teacher decides regardless).
        current_phase = self._INITIAL_PHASE
        current_phase_time = 0

        logger.info(
            "TrajectoryCollector.collect: phase=%d, num_samples=%d, "
            "min_valid=%d, max_requests=%d, intersection_id=%r",
            self._phase,
            num_samples,
            self._min_valid_samples,
            self._max_requests,
            intersection_id,
        )

        while len(samples) < num_samples and request_count < self._max_requests:
            # 1. Build state from engine.
            try:
                lane_vehicle_count = engine.get_lane_vehicle_count()
            except Exception as exc:  # noqa: BLE001 - defensive
                logger.warning(
                    "TrajectoryCollector: engine.get_lane_vehicle_count() "
                    "raised %s: %s; aborting collection loop",
                    type(exc).__name__,
                    exc,
                )
                break

            state = {
                "lane_vehicle_count": lane_vehicle_count,
                "current_phase": current_phase,
                "current_phase_time": current_phase_time,
            }

            # 2. Build prompt. If parser rejects state, advance and retry.
            try:
                prompt = self._obs_parser.parse(state)
            except ValueError as exc:
                logger.warning(
                    "TrajectoryCollector: ObservationParser rejected state: "
                    "%s; advancing engine one step",
                    exc,
                )
                self._safe_next_step(engine)
                continue

            # 3. Call API. RequestLimitExceeded means Puter quota hit at
            # the client level — bail out (we've used everything we can).
            try:
                api_response = self._api_client.chat_completion(prompt)
            except RequestLimitExceeded as exc:
                logger.warning(
                    "TrajectoryCollector: API request limit exceeded after "
                    "%d requests (%d valid samples collected): %s",
                    request_count,
                    len(samples),
                    exc,
                )
                break
            request_count += 1

            raw_response = (
                api_response.content if api_response is not None else ""
            )

            # 4. Verify validity. Save only when raw response actually
            # contains <signal> with a phase ∈ VALID_PHASES (so we don't
            # capture the parser's silent fallback to ETWT).
            if self._is_response_valid(raw_response):
                samples.append((prompt, raw_response))
                if len(samples) % self._LOG_EVERY == 0:
                    logger.info(
                        "TrajectoryCollector: collected %d/%d valid samples "
                        "(requests used: %d/%d)",
                        len(samples),
                        num_samples,
                        request_count,
                        self._max_requests,
                    )
            else:
                logger.debug(
                    "TrajectoryCollector: skipped invalid response "
                    "(no <signal> tag or phase ∉ VALID_PHASES); "
                    "request %d, samples %d/%d",
                    request_count,
                    len(samples),
                    num_samples,
                )

            # 5. Advance simulation by applying the parsed phase. Use
            # ResponseParser.parse for the side-effect of falling back to
            # ETWT (and warning) on invalid responses — this keeps the
            # simulation moving regardless of API quality.
            phase_name = self._resp_parser.parse(raw_response)
            try:
                phase_index = phase_index_resolver(intersection_id, phase_name)
                engine.set_phase(intersection_id, phase_index)
            except Exception as exc:  # noqa: BLE001 - defensive
                logger.warning(
                    "TrajectoryCollector: engine.set_phase(%r, %s) failed: "
                    "%s; falling back to single next_step()",
                    intersection_id,
                    phase_name,
                    exc,
                )
                if not self._safe_next_step(engine):
                    break

            # 6. Update internal phase tracking.
            if phase_name != current_phase:
                current_phase = phase_name
                current_phase_time = 0
            else:
                # Approximate phase time accumulation. Exact value
                # doesn't affect prompt validity; ObservationParser only
                # requires int >= 0.
                current_phase_time += 1

        # Final validation against the phase threshold.
        if len(samples) < self._min_valid_samples:
            raise InsufficientDataError(
                f"TrajectoryCollector (phase={self._phase}): collected only "
                f"{len(samples)} valid samples; required >= "
                f"{self._min_valid_samples} (requests used: "
                f"{request_count}/{self._max_requests}). Verify backend "
                "quota and prompt quality, or rerun with a larger budget."
            )

        logger.info(
            "TrajectoryCollector.collect: completed with %d valid samples "
            "(requests used: %d/%d)",
            len(samples),
            request_count,
            self._max_requests,
        )
        return samples

    # ----------------------------------------------------------- internals --

    @staticmethod
    def _is_response_valid(response: object) -> bool:
        """Return ``True`` iff ``response`` chứa ``<signal>...</signal>``
        với value ∈ ``ResponseParser.VALID_PHASES``.

        Sử dụng cùng regex với :class:`ResponseParser` để consistent với
        parser logic; KHÔNG gọi ``ResponseParser.parse`` vì nó luôn
        fallback về ``ETWT`` nên không phân biệt được "thực sự là ETWT"
        và "không có tag → fallback ETWT".
        """
        if not isinstance(response, str) or not response:
            return False
        match = ResponseParser._SIGNAL_RE.search(response)
        if match is None:
            return False
        candidate = match.group(1).strip().upper()
        return candidate in ResponseParser.VALID_PHASES

    @staticmethod
    def _safe_next_step(engine: Any) -> bool:
        """Gọi ``engine.next_step()`` nếu có, swallow lỗi.

        Returns ``False`` khi engine không có ``next_step`` hoặc nó raise —
        caller dùng để decide có break loop hay không.
        """
        next_step = getattr(engine, "next_step", None)
        if not callable(next_step):
            return False
        try:
            next_step()
            return True
        except Exception as exc:  # noqa: BLE001 - defensive
            logger.warning(
                "TrajectoryCollector: engine.next_step() raised %s: %s",
                type(exc).__name__,
                exc,
            )
            return False
