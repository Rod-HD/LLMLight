"""Phase Approval Gate (Component 12).

Enforce thứ tự thực thi 3 pha (Phase 1 → Phase 2 → Phase 3) và yêu cầu xác
nhận thủ công trước Phase 3.

Module này được TẤT CẢ runner scripts gọi NGAY SAU ``SeedManager().apply()``
và TRƯỚC bất kỳ khởi tạo client/engine nào khác (CityFlowEngine,
MultiBackendAPIClient, LightGPTInference, …) — tránh tốn tài nguyên rồi mới
abort.

Trách nhiệm:
    1. ``validate_phase(phase, dataset, mode)``: kiểm tra phase hợp lệ +
       dataset hợp lệ với phase + cảnh báo khi mode override không nhất quán.
    2. ``check_prerequisite(phase)``: Phase 2 cần file Phase 1 (cảnh báo +
       confirm); Phase 3 HARD BLOCK nếu thiếu CẢ HAI file Phase 2.
    3. ``request_manual_approval(phase, ...)``: Phase 3 yêu cầu user gõ
       chính xác ``"yes"`` (case-insensitive, strip whitespace).
    4. ``phase_label(phase)``: trả về ``"Phase1"`` / ``"Phase2"`` /
       ``"Phase3"`` để runner gắn vào ``ExperimentResult``.

Override cho automation / CI:
    - Constructor parameter ``input_fn`` cho phép inject hàm nhận input
      tuỳ ý (mặc định ``builtins.input``).
    - Env var ``LLMLIGHT_AUTO_APPROVE=yes`` (case-insensitive) buộc gate
      tự động phê duyệt mọi prompt — chỉ nên đặt trong CI / unit test.

_Requirements_: 13.1, 13.2, 13.3, 13.4, 13.5, 13.6, 13.7
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Callable, FrozenSet, Tuple

logger = logging.getLogger(__name__)


class PhaseApprovalGate:
    """Component 12 — enforce phase ordering + manual approval."""

    VALID_PHASES: Tuple[int, ...] = (1, 2, 3)
    PHASE12_DATASETS: FrozenSet[str] = frozenset({"jinan_1", "hangzhou_1"})
    PHASE3_DATASETS: FrozenSet[str] = frozenset({"newyork_1"})
    VALID_MODES: FrozenSet[str] = frozenset({"demo", "full"})

    DEFAULT_MODE_BY_PHASE: dict = {1: "demo", 2: "full", 3: "full"}

    AUTO_APPROVE_ENV_VAR: str = "LLMLIGHT_AUTO_APPROVE"

    def __init__(
        self,
        results_dir: str | os.PathLike[str] = "results/metrics",
        *,
        input_fn: Callable[[str], str] | None = None,
    ) -> None:
        """Khởi tạo gate.

        Args:
            results_dir: Thư mục chứa ``comparison_{dataset}_phase{N}.md``
                để kiểm tra prerequisite. Mặc định ``"results/metrics"``.
            input_fn: Hàm nhận input của user (mặc định ``builtins.input``).
                Cho phép unit test inject ``input_fn`` thay vì
                ``monkeypatch`` builtin.
        """
        self.results_dir: Path = Path(results_dir)
        self._input_fn: Callable[[str], str] = input_fn if input_fn is not None else input

    # ----------------------------------------------------------------- API --

    def validate_phase(self, phase: int, dataset: str, mode: str) -> None:
        """Validate phase + dataset + mode tương thích.

        Quy tắc:
            - ``phase`` phải thuộc ``{1, 2, 3}``.
            - Phase 1 / Phase 2: dataset phải thuộc ``PHASE12_DATASETS``;
              raise ``ValueError`` nếu nhận ``newyork_1``.
            - Phase 3: dataset phải thuộc ``PHASE3_DATASETS``;
              raise ``ValueError`` nếu nhận ``jinan_1`` hoặc ``hangzhou_1``.
            - Mode mặc định: Phase 1 → ``"demo"``, Phase 2/3 → ``"full"``.
              Nếu user override không nhất quán (Phase 1 + ``"full"`` hoặc
              Phase 2/3 + ``"demo"``) thì chỉ log cảnh báo, KHÔNG raise.

        Raises:
            ValueError: Nếu ``phase`` không thuộc ``{1,2,3}``, dataset không
                hợp lệ với phase, hoặc mode không thuộc ``{"demo", "full"}``.
        """
        if phase not in self.VALID_PHASES:
            raise ValueError(
                f"Invalid phase {phase!r}: must be one of {self.VALID_PHASES}."
            )

        if mode not in self.VALID_MODES:
            raise ValueError(
                f"Invalid mode {mode!r}: must be one of "
                f"{sorted(self.VALID_MODES)}."
            )

        # ----- Dataset compatibility ------------------------------------
        if phase in (1, 2):
            if dataset not in self.PHASE12_DATASETS:
                raise ValueError(
                    f"Phase {phase} chỉ chấp nhận dataset trong "
                    f"{sorted(self.PHASE12_DATASETS)}; nhận được "
                    f"{dataset!r}. (New York 1 chỉ dành cho Phase 3.)"
                )
        else:  # phase == 3
            if dataset not in self.PHASE3_DATASETS:
                raise ValueError(
                    f"Phase 3 chỉ chấp nhận dataset trong "
                    f"{sorted(self.PHASE3_DATASETS)}; nhận được "
                    f"{dataset!r}. (Jinan 1 / Hangzhou 1 dành cho Phase 1/2.)"
                )

        # ----- Mode override warning (Requirement 13.7) -----------------
        default_mode = self.DEFAULT_MODE_BY_PHASE[phase]
        if mode != default_mode:
            logger.warning(
                "Mode override không nhất quán: Phase %d mặc định mode %r "
                "nhưng user truyền mode %r. Tiếp tục với mode user-supplied; "
                "đảm bảo bạn hiểu chi phí + thời gian chạy thay đổi.",
                phase,
                default_mode,
                mode,
            )

    # ----------------------------------------------------------------------

    def check_prerequisite(self, phase: int) -> None:
        """Kiểm tra prerequisite của phase trên ``results_dir``.

        - Phase 1: no-op.
        - Phase 2: cảnh báo nếu thiếu ``comparison_jinan_1_phase1.md`` HOẶC
          ``comparison_hangzhou_1_phase1.md``; yêu cầu user xác nhận trước
          khi tiếp tục. Nếu user không xác nhận → raise ``RuntimeError``.
        - Phase 3: HARD BLOCK với ``RuntimeError`` nếu thiếu CẢ HAI file
          ``comparison_jinan_1_phase2.md`` VÀ ``comparison_hangzhou_1_phase2.md``.

        Raises:
            ValueError: Nếu ``phase`` không thuộc ``{1, 2, 3}``.
            RuntimeError: Phase 2 mà user không xác nhận tiếp tục, hoặc
                Phase 3 mà thiếu cả hai file Phase 2.
        """
        if phase not in self.VALID_PHASES:
            raise ValueError(
                f"Invalid phase {phase!r}: must be one of {self.VALID_PHASES}."
            )

        if phase == 1:
            logger.info("Phase 1: no prerequisite check required.")
            return

        if phase == 2:
            self._check_phase2_prerequisite()
            return

        # phase == 3
        self._check_phase3_prerequisite()

    def _check_phase2_prerequisite(self) -> None:
        missing = [
            f"comparison_{ds}_phase1.md"
            for ds in ("jinan_1", "hangzhou_1")
            if not (self.results_dir / f"comparison_{ds}_phase1.md").exists()
        ]
        if not missing:
            logger.info(
                "Phase 2 prerequisite OK: cả hai file Phase 1 tồn tại trong %s.",
                self.results_dir,
            )
            return

        # Nếu LLMLIGHT_SKIP_PHASE1=yes → bypass toàn bộ Phase 1 check (user đã
        # xác nhận skip Phase 1 một lần, không hỏi lại mỗi run).
        skip_env = os.environ.get("LLMLIGHT_SKIP_PHASE1", "").strip().lower()
        if skip_env == "yes":
            logger.warning(
                "LLMLIGHT_SKIP_PHASE1=yes: bỏ qua prerequisite Phase 1. "
                "Thiếu: %s. Tiếp tục Phase 2.",
                missing,
            )
            return

        warning_msg = (
            "Phase 1 chưa hoàn tất: thiếu file Phase 1 trong "
            f"{self.results_dir}: {missing}. "
            "Phase 2 nên chạy SAU khi Phase 1 đã có kết quả ổn định."
        )
        logger.warning(warning_msg)

        if not self._confirm_yes(
            f"\n[PhaseApprovalGate] {warning_msg}\n"
            "Tiếp tục Phase 2 dù chưa có kết quả Phase 1? Gõ 'yes' để tiếp tục: "
        ):
            raise RuntimeError(
                "Phase 2 bị huỷ vì thiếu kết quả Phase 1 và user không xác "
                f"nhận tiếp tục. Thiếu: {missing}"
            )
        logger.info("Phase 2 user-confirmed continuation despite missing Phase 1 files.")

    def _check_phase3_prerequisite(self) -> None:
        required = [
            self.results_dir / f"comparison_{ds}_phase2.md"
            for ds in ("jinan_1", "hangzhou_1")
        ]
        missing = [str(p) for p in required if not p.exists()]

        # HARD BLOCK chỉ khi thiếu CẢ HAI file (theo task spec).
        if len(missing) == len(required):
            raise RuntimeError(
                "Phase 3 yêu cầu Phase 2 hoàn tất trước. Thiếu cả hai file "
                f"Phase 2 trong {self.results_dir}: {missing}. "
                "Chạy `scripts/evaluate.py --phase 2 --dataset jinan_1` và "
                "`--dataset hangzhou_1` trước khi mở Phase 3."
            )

        # Thiếu một file → cảnh báo (không hard block, vì task spec yêu cầu
        # hard block chỉ khi thiếu cả hai).
        if missing:
            logger.warning(
                "Phase 3 prerequisite cảnh báo: chỉ tồn tại một phần kết quả "
                "Phase 2 trong %s. Thiếu: %s. Pipeline tiếp tục, nhưng so "
                "sánh với Phase 2 sẽ không đầy đủ.",
                self.results_dir,
                missing,
            )
        else:
            logger.info(
                "Phase 3 prerequisite OK: cả hai file Phase 2 tồn tại trong %s.",
                self.results_dir,
            )

    # ----------------------------------------------------------------------

    def request_manual_approval(
        self,
        phase: int,
        estimated_cost_usd: float | None = None,
        estimated_time_hours: float | None = None,
    ) -> bool:
        """Yêu cầu user xác nhận thủ công trước Phase 3.

        - Phase 1 / Phase 2: trả về ``True`` ngay (không cần approval).
        - Phase 3: in cảnh báo về chi phí + thời gian ước tính, yêu cầu user
          gõ chính xác ``"yes"`` (case-insensitive, strip whitespace) để
          tiếp tục. Bất kỳ giá trị khác (kể cả Enter rỗng) → trả về
          ``False`` để runner abort.

        Args:
            phase: Phase ID (1, 2, hoặc 3).
            estimated_cost_usd: Token cost ước tính (USD) — hiển thị nếu
                không ``None`` (chỉ Phase 3).
            estimated_time_hours: Thời gian chạy ước tính (giờ) — hiển thị
                nếu không ``None`` (chỉ Phase 3).

        Returns:
            ``True`` nếu user xác nhận hoặc phase < 3; ``False`` nếu Phase 3
            mà user không gõ ``"yes"``.

        Raises:
            ValueError: Nếu ``phase`` không thuộc ``{1, 2, 3}``.
        """
        if phase not in self.VALID_PHASES:
            raise ValueError(
                f"Invalid phase {phase!r}: must be one of {self.VALID_PHASES}."
            )

        if phase in (1, 2):
            logger.info(
                "Phase %d: manual approval not required, continuing.", phase
            )
            return True

        # phase == 3
        cost_str = (
            f"${estimated_cost_usd:.2f}"
            if estimated_cost_usd is not None
            else "không ước tính (chưa rõ backend)"
        )
        time_str = (
            f"{estimated_time_hours:.2f} giờ"
            if estimated_time_hours is not None
            else "không ước tính"
        )

        warning = (
            "\n"
            "============================================================\n"
            "  PHASE 3 — MANUAL APPROVAL REQUIRED\n"
            "============================================================\n"
            "  Bạn đang chuẩn bị chạy Phase 3 (New York 1, 28x7 = 196\n"
            "  intersections, 3600 timesteps, 3 runs cho TẤT CẢ phương pháp).\n"
            "\n"
            f"  Token cost ước tính (nếu dùng OpenAI): {cost_str}\n"
            f"  Thời gian chạy ước tính: {time_str}\n"
            "\n"
            "  Phase 3 KHÔNG nằm trong quota miễn phí và có thể tốn nhiều\n"
            "  giờ + chi phí token đáng kể. Hãy chắc chắn rằng:\n"
            "    1. Phase 2 đã có kết quả ổn định trên Jinan 1 + Hangzhou 1.\n"
            "    2. Bạn có đủ ngân sách OpenAI / quota Groq cho Phase 3.\n"
            "    3. Bạn có thể giám sát run trong vài giờ tới.\n"
            "\n"
            "  Gõ chính xác 'yes' (case-insensitive) để tiếp tục.\n"
            "  Bất kỳ giá trị khác (kể cả Enter rỗng) sẽ HUỶ Phase 3.\n"
            "============================================================\n"
        )
        # In ra stderr để user thấy ngay cả khi stdout đang được pipe.
        print(warning)
        logger.warning("Phase 3 manual approval prompt displayed to user.")

        confirmed = self._confirm_yes("Continue Phase 3? Type 'yes' to proceed: ")
        if confirmed:
            logger.info("Phase 3 manual approval GRANTED by user.")
        else:
            logger.warning(
                "Phase 3 manual approval DENIED by user; runner should abort."
            )
        return confirmed

    # ----------------------------------------------------------------------

    def phase_label(self, phase: int) -> str:
        """Trả về label dạng ``"Phase{N}"`` để gắn vào ``ExperimentResult``.

        Args:
            phase: 1, 2, hoặc 3.

        Returns:
            ``"Phase1"``, ``"Phase2"``, hoặc ``"Phase3"``.

        Raises:
            ValueError: Nếu ``phase`` không thuộc ``{1, 2, 3}``.
        """
        if phase not in self.VALID_PHASES:
            raise ValueError(
                f"Invalid phase {phase!r}: must be one of {self.VALID_PHASES}."
            )
        return f"Phase{phase}"

    # ------------------------------------------------------------- helpers --

    def _confirm_yes(self, prompt: str) -> bool:
        """Hỏi user (qua ``self._input_fn``) và trả về ``True`` nếu trả lời
        chính xác ``"yes"`` (case-insensitive, strip whitespace).

        Nếu env var ``LLMLIGHT_AUTO_APPROVE`` được set với giá trị ``"yes"``
        (case-insensitive, strip whitespace) thì auto-confirm — phục vụ CI
        / unit test.
        """
        auto = os.environ.get(self.AUTO_APPROVE_ENV_VAR, "").strip().lower()
        if auto == "yes":
            logger.info(
                "Auto-approve via env var %s=yes; skipping interactive prompt.",
                self.AUTO_APPROVE_ENV_VAR,
            )
            return True

        try:
            answer = self._input_fn(prompt)
        except EOFError:
            # Không có stdin (ví dụ chạy non-interactive) → từ chối an toàn.
            logger.warning(
                "EOFError while reading user confirmation; treating as DENY."
            )
            return False

        return (answer or "").strip().lower() == "yes"
