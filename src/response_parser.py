"""Response Parser (Component 4).

Trích xuất pha tín hiệu từ output thô của LLM (LightGPT local hoặc remote
GPT-4o / Llama-3.3 qua ``MultiBackendAPIClient``).

Hành vi:
  * Tìm tag ``<signal>...</signal>`` ĐẦU TIÊN trong response.
  * Loại bỏ whitespace (space / tab / newline / carriage return) ở **đầu và
    cuối** nội dung tag. KHÔNG collapse whitespace bên trong (hành vi này
    không có ý nghĩa với một phase token đơn lẻ, nhưng giữ nguyên để khớp
    đúng yêu cầu spec).
  * So sánh không phân biệt hoa thường với tập ``VALID_PHASES``; trả về dạng
    chữ in hoa.
  * Fallback về ``DEFAULT_PHASE = "ETWT"`` khi parse thất bại; log warning
    chứa tối đa ``MAX_LOG_LENGTH = 500`` ký tự đầu của response.
  * Nếu phát hiện nhiều hơn 1 tag ``<signal>``: log warning riêng và tiếp
    tục dùng tag đầu tiên (không phải lỗi parse).

_Requirements_: 4.1, 4.2, 4.3, 4.4, 4.5, 12.1, 12.4
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)


class ResponseParser:
    """Trích xuất pha tín hiệu từ LLM response.

    Class constants:
        VALID_PHASES: Tập 4 pha hợp lệ trong LLMLight (paper KDD 2025).
        DEFAULT_PHASE: Pha fallback khi parse thất bại (theo spec).
        MAX_LOG_LENGTH: Giới hạn ký tự khi log raw response trong cảnh báo.

    Class attributes:
        _SIGNAL_RE: Regex compiled một lần (module load time) để tìm TẤT CẢ
            tag ``<signal>...</signal>``. Cờ ``DOTALL`` cho phép ``.`` match
            newline; cờ ``IGNORECASE`` cho phép tag ``<SIGNAL>`` hoặc
            ``<Signal>``. Nhóm ``[^>]*`` cho phép thuộc tính bất kỳ
            (hiếm gặp trong thực tế nhưng tương thích với output LLM
            không kiểm soát).
    """

    VALID_PHASES: frozenset[str] = frozenset({"ETWT", "NTST", "ELWL", "NLSL"})
    DEFAULT_PHASE: str = "ETWT"
    MAX_LOG_LENGTH: int = 500

    # Regex tìm tag <signal>...</signal>; nội dung capture ở group 1.
    # ``.*?`` non-greedy → mỗi match là một tag riêng biệt.
    _SIGNAL_RE: re.Pattern[str] = re.compile(
        r"<signal\b[^>]*>(.*?)</signal>",
        flags=re.DOTALL | re.IGNORECASE,
    )

    def __init__(self, logger_: logging.Logger | None = None) -> None:
        """Khởi tạo parser.

        Args:
            logger_: Logger tuỳ chọn để inject (hữu ích khi runner muốn log
                chung qua một logger nhất định, hoặc khi test bắt log).
                Mặc định dùng ``logging.getLogger("src.response_parser")``.
        """
        self._logger = logger_ if logger_ is not None else logger

    # ------------------------------------------------------------------ API --

    def parse(self, response: str) -> str:
        """Trích xuất pha tín hiệu từ LLM response.

        Quy trình:
          1. Validate input là ``str`` (non-string → fallback + warning).
          2. Tìm tất cả tag ``<signal>`` trong response.
          3. Nếu không có tag → fallback ``DEFAULT_PHASE`` + log warning với
             tối đa 500 ký tự đầu.
          4. Nếu có ≥ 2 tag → log warning riêng về việc phát hiện nhiều tag,
             vẫn tiếp tục với tag đầu tiên.
          5. Strip whitespace (``\\s+`` → space/tab/newline/CR) ở đầu/cuối
             nội dung tag đầu tiên.
          6. Uppercase + validate trong ``VALID_PHASES``.
          7. Nếu hợp lệ → trả về uppercase. Nếu không → fallback + warning.

        Args:
            response: Raw response text từ LLM.

        Returns:
            Một trong ``{"ETWT", "NTST", "ELWL", "NLSL"}``. KHÔNG bao giờ
            raise; mọi đường dẫn lỗi đều fallback về ``DEFAULT_PHASE`` và
            log warning.
        """
        if not isinstance(response, str):
            self._warn_fallback(
                response,
                reason=(
                    f"response is not a str (got {type(response).__name__}); "
                    "falling back to DEFAULT_PHASE"
                ),
            )
            return self.DEFAULT_PHASE

        raw_inner = self._extract_signal_tag(response)
        if raw_inner is None:
            self._warn_fallback(
                response,
                reason="no <signal> tag found; falling back to DEFAULT_PHASE",
            )
            return self.DEFAULT_PHASE

        # Strip leading/trailing whitespace (space/tab/newline/CR) per AC 4.
        # Python's str.strip() with no arg strips exactly these characters
        # (and a few related ones like form feed / vertical tab) — acceptable
        # because spec lists these as whitespace categories and never excludes
        # form feed / vt. KHÔNG collapse internal whitespace.
        normalized = raw_inner.strip()
        candidate = normalized.upper()

        if candidate in self.VALID_PHASES:
            return candidate

        self._warn_fallback(
            response,
            reason=(
                "<signal> content "
                f"{normalized!r} not in VALID_PHASES; "
                "falling back to DEFAULT_PHASE"
            ),
        )
        return self.DEFAULT_PHASE

    # ----------------------------------------------------------- Internals --

    def _extract_signal_tag(self, response: str) -> str | None:
        """Trích xuất nội dung của tag ``<signal>`` đầu tiên.

        Sử dụng ``finditer`` thay vì ``findall`` để tránh duyệt lại chuỗi:
        chúng ta chỉ cần biết "có ≥ 2 match hay không" để bật cảnh báo.

        Args:
            response: Raw response (đã được caller validate là ``str``).

        Returns:
            Chuỗi capture group 1 (nội dung giữa ``<signal>`` và
            ``</signal>``) của match ĐẦU TIÊN; hoặc ``None`` nếu không có
            tag nào.

        Side effects:
            Log một warning ở mức ``WARNING`` nếu phát hiện nhiều hơn 1 tag.
        """
        iterator = self._SIGNAL_RE.finditer(response)
        first = next(iterator, None)
        if first is None:
            return None

        # Kiểm tra xem có match thứ hai không để phát warning. ``next`` với
        # default ``None`` an toàn ngay cả khi iterator đã cạn.
        second = next(iterator, None)
        if second is not None:
            # Đếm tổng số match cho thông điệp warning rõ ràng hơn — chấp
            # nhận chi phí O(n) một lần vì đường dẫn này hiếm.
            total = 2 + sum(1 for _ in iterator)
            self._logger.warning(
                "Multiple <signal> tags detected (count=%d); using the first "
                "tag value. Response prefix: %r",
                total,
                self._truncate(response),
            )

        return first.group(1)

    def _warn_fallback(self, response: object, *, reason: str) -> None:
        """Log warning về fallback về ``DEFAULT_PHASE`` kèm prefix response.

        Args:
            response: Raw response (có thể không phải ``str`` nếu caller
                gặp input không hợp lệ).
            reason: Mô tả ngắn gọn nguyên nhân fallback.
        """
        self._logger.warning(
            "ResponseParser fallback to %s: %s. Response prefix: %r",
            self.DEFAULT_PHASE,
            reason,
            self._truncate(response),
        )

    @classmethod
    def _truncate(cls, response: object) -> str:
        """Cắt response còn tối đa ``MAX_LOG_LENGTH`` ký tự (đầu chuỗi).

        Coerce non-str về ``repr`` để tránh ``TypeError`` khi log.
        """
        text = response if isinstance(response, str) else repr(response)
        if len(text) > cls.MAX_LOG_LENGTH:
            return text[: cls.MAX_LOG_LENGTH]
        return text
