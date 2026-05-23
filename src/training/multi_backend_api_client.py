"""Multi-Backend API Client (Component 6).

Gọi LLM remote qua **một trong bốn backend** với priority
``CODEXHUB > OPENAI > GROQ > PUTER``:

* ``CodexHub`` (proxy endpoint hỗ trợ cx/gpt-5.4, cx/gpt-5.5, kr/* models) —
  endpoint từ ``CODEXHUB_BASE_URL`` env var (mặc định ``https://api.codexhub.click/v1``).
  Được chọn khi ``CODEXHUB_API_KEY`` set; ưu tiên cao nhất.
* ``OpenAI`` (paid, ``gpt-4o``) — endpoint ``https://api.openai.com/v1/``.
  Được chọn khi ``OPENAI_API_KEY`` set; không giới hạn requests.
* ``Groq`` (free, ``llama-3.3-70b-versatile``) — endpoint
  ``https://api.groq.com/openai/v1/``. Được chọn khi ``OPENAI_API_KEY``
  *không* set nhưng ``GROQ_API_KEY`` set; ~14k requests/ngày miễn phí.
* ``Puter`` (demo, ``gpt-4o``) — endpoint ``https://api.puter.com/puterai/openai/v1/``.
  Được chọn khi cả ``CODEXHUB_API_KEY``, ``OPENAI_API_KEY`` và ``GROQ_API_KEY``
  *đều không* set, ``PUTER_AUTH_TOKEN`` set; ≤100 requests/run, chỉ ``Demo_mode``.

Hành vi:

* Block ``--mode full`` khi backend được chọn là Puter (``ValueError``).
* Raise ``ValueError`` khi cả 3 biến môi trường đều không được set.
* ``chat_completion(prompt)`` dùng OpenAI-compatible format (``messages``,
  ``temperature=0``).
* HTTP 429: chờ 60s rồi retry, tối đa 3 lần; nếu vẫn fail → empty content
  (caller dùng default phase ``ETWT``).
* Timeout 30s/request → empty content + log warning.
* HTTP error khác → empty content + log warning + mã HTTP.
* Puter request limit (100/run) → ``RequestLimitExceeded`` khi vượt.

Thiết kế testable:

* Constructor nhận optional ``sleep_fn`` (mặc định ``time.sleep``) để test
  inject no-op cho retry path nhanh.
* Constructor nhận optional ``http_post`` (mặc định wrapper quanh
  ``requests.post``) để test mock toàn bộ HTTP layer mà không cần
  ``requests`` được cài.
* ``python-dotenv`` dùng nếu có; fallback parse ``.env`` thủ công nếu thiếu.

_Requirements_: 7.1-7.12
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Optional ``requests`` import — guarded so test suites có thể chạy mà không
# cần cài ``requests`` (tests luôn inject ``http_post`` qua constructor).
# Ở runtime production (WSL2 venv), ``requests`` đã được setup_env.sh cài.
# ---------------------------------------------------------------------------

try:  # pragma: no cover - depends on environment
    import requests as _requests  # type: ignore[import-not-found]
except Exception as _req_exc:  # pragma: no cover
    _requests = None
    logger.debug("requests not importable: %s", _req_exc)


class RequestLimitExceeded(Exception):
    """Raised khi Puter backend vượt quá ``PUTER_MAX_REQUESTS`` trong 1 run."""


class APIBackend(Enum):
    """Backend đang được sử dụng. Giá trị string match key trong
    ``TokenUsageLog.backend`` để serialization dễ đọc."""

    PUTER = "puter"
    GROQ = "groq"
    OPENAI = "openai"
    CODEXHUB = "codexhub"


@dataclass(frozen=True)
class APIResponse:
    """Kết quả gọi ``chat_completion``.

    ``content`` là rỗng (``""``) khi gặp timeout, HTTP error khác 429, hoặc
    429 vẫn fail sau 3 lần retry. Caller phải fallback về default phase
    ``ETWT`` khi thấy content rỗng (xem Response Parser).
    """

    content: str
    input_tokens: int
    output_tokens: int
    backend: APIBackend


@dataclass
class TokenUsageLog:
    """Tổng kết token usage và request count cho một run.

    Track riêng cho từng backend (mỗi instance ``MultiBackendAPIClient``
    chỉ gắn với 1 backend). Được trả về qua ``get_usage_log()``.
    """

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_requests: int = 0
    backend: APIBackend = APIBackend.PUTER


# ---------------------------------------------------------------------------
# Default HTTP layer (uses ``requests`` if available)
# ---------------------------------------------------------------------------


@dataclass
class _HttpResponse:
    """Lightweight response object thay thế ``requests.Response`` để dễ test.

    Tests có thể trả về instance này từ ``http_post`` callable mà không cần
    cài ``requests``.
    """

    status_code: int
    json_body: Any = None
    text: str = ""

    def json(self) -> Any:
        return self.json_body


class _HttpTimeout(Exception):
    """Raised by default ``http_post`` wrapper khi ``requests`` ném timeout.

    Tests inject custom ``http_post`` có thể raise class này để simulate
    timeout mà không cần import ``requests``.
    """


def _parse_sse_to_json(text: str) -> dict[str, Any] | None:
    """Assemble a synthetic non-streaming JSON body from an SSE stream.

    CodexHub always returns SSE regardless of ``stream=false``. This function
    reconstructs a ``chat.completions`` JSON object so the rest of the client
    can treat it uniformly.
    """
    content_parts: list[str] = []
    finish_reason: str | None = None
    meta: dict[str, Any] = {}
    for line in text.splitlines():
        if not line.startswith("data: "):
            continue
        chunk = line[6:].strip()
        if chunk == "[DONE]":
            break
        try:
            obj: dict[str, Any] = json.loads(chunk)
        except Exception:
            continue
        if not meta:
            meta = {k: v for k, v in obj.items() if k != "choices"}
        choices = obj.get("choices") or []
        if choices:
            delta = (choices[0] or {}).get("delta") or {}
            content_parts.append(delta.get("content") or "")
            fr = (choices[0] or {}).get("finish_reason")
            if fr:
                finish_reason = fr
    if not meta and not content_parts:
        return None
    return {
        **meta,
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "".join(content_parts)},
                "finish_reason": finish_reason,
            }
        ],
        "usage": meta.get("usage") or {},
    }


def _default_http_post(
    url: str,
    *,
    headers: dict[str, str],
    json_body: dict[str, Any],
    timeout: float,
) -> _HttpResponse:
    """Wrapper mặc định quanh ``requests.post``.

    Raises:
        _HttpTimeout: Khi ``requests`` ném timeout exception.
        RuntimeError: Khi ``requests`` không được cài.
    """
    if _requests is None:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "Package 'requests' is not installed; cannot make HTTP calls. "
            "Install it via setup_env.sh, or pass http_post=... to "
            "MultiBackendAPIClient for testing."
        )
    try:
        resp = _requests.post(  # type: ignore[union-attr]
            url, headers=headers, json=json_body, timeout=timeout
        )
    except _requests.exceptions.Timeout as exc:  # type: ignore[union-attr]
        raise _HttpTimeout(str(exc)) from exc

    body: Any = None
    try:
        body = resp.json()
    except Exception:  # noqa: BLE001 - response may not be JSON
        # Fallback: parse Server-Sent Events (SSE) stream.
        # Some proxies (e.g. CodexHub) always return SSE regardless of
        # the `stream` field in the request body.
        body = _parse_sse_to_json(resp.text or "")
    return _HttpResponse(
        status_code=resp.status_code,
        json_body=body,
        text=resp.text or "",
    )


# ---------------------------------------------------------------------------
# .env loading (python-dotenv if available, else manual parse)
# ---------------------------------------------------------------------------


def _load_env_file(env_path: str) -> None:
    """Load biến môi trường từ ``env_path`` vào ``os.environ`` mà KHÔNG
    overwrite biến đã được set sẵn (ưu tiên env vars truyền qua shell hoặc
    ``monkeypatch.setenv``).

    Sử dụng ``python-dotenv`` nếu được cài; fallback parser thủ công nếu
    không. Nếu file không tồn tại → no-op (caller fallback về env vars).
    """
    path = Path(env_path)
    if not path.exists():
        logger.debug("env file %s not found; using process env vars only.", env_path)
        return

    # Try python-dotenv first.
    try:  # pragma: no cover - exercised when dotenv installed
        from dotenv import load_dotenv  # type: ignore[import-not-found]

        load_dotenv(dotenv_path=str(path), override=False)
        return
    except ImportError:
        logger.debug("python-dotenv not installed; using manual .env parser.")

    # Manual fallback: parse KEY=VALUE lines, ignore comments and blanks.
    try:
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            # Strip optional surrounding quotes.
            if (
                len(value) >= 2
                and value[0] == value[-1]
                and value[0] in ("'", '"')
            ):
                value = value[1:-1]
            if key and key not in os.environ:
                os.environ[key] = value
    except OSError as exc:
        logger.warning("Failed to read env file %s: %s", env_path, exc)


# ---------------------------------------------------------------------------
# MultiBackendAPIClient
# ---------------------------------------------------------------------------


class MultiBackendAPIClient:
    """Client gọi GPT-4o (Puter/OpenAI) hoặc Llama-3.3-70B (Groq)
    qua endpoint OpenAI-compatible.

    Backend được chọn 1 lần tại constructor và không đổi trong vòng đời
    instance. Mỗi instance tracking ``TokenUsageLog`` riêng cho backend đó.
    """

    PUTER_URL = "https://api.puter.com/puterai/openai/v1/"
    GROQ_URL = "https://api.groq.com/openai/v1/"
    OPENAI_URL = "https://api.openai.com/v1/"
    CODEXHUB_DEFAULT_URL = "https://api.codexhub.click/v1"
    CODEXHUB_DEFAULT_MODEL = "cx/gpt-5.5"

    PUTER_MODEL = "gpt-4o"
    GROQ_MODEL = "llama-3.3-70b-versatile"
    OPENAI_MODEL = "gpt-4o"

    PUTER_MAX_REQUESTS = 100
    TIMEOUT_SECONDS = 30
    RETRY_WAIT_SECONDS = 60
    MAX_RETRIES = 3

    # Lỗi cấu hình môi trường (Requirement 7.2).
    ERR_NO_BACKEND = "Phải cấu hình ít nhất một trong bốn backend API"

    # Lỗi mode-block với Puter (Requirement 7.4).
    ERR_FULL_MODE_PUTER = (
        "--mode full không tương thích với backend Puter; "
        "cấu hình CODEXHUB_API_KEY, GROQ_API_KEY hoặc OPENAI_API_KEY"
    )

    def __init__(
        self,
        env_path: str = ".env",
        mode: str = "demo",
        *,
        sleep_fn: Callable[[float], None] | None = None,
        http_post: Callable[..., _HttpResponse] | None = None,
    ) -> None:
        """Khởi tạo client và resolve backend.

        Args:
            env_path: Đường dẫn tới file ``.env``. Nếu file không tồn tại,
                fallback dùng env vars trong process trực tiếp.
            mode: ``"demo"`` hoặc ``"full"``. Chỉ ảnh hưởng tới backend
                Puter (``"full"`` + Puter → ``ValueError``).
            sleep_fn: Hàm sleep dùng cho retry path (mặc định
                ``time.sleep``). Test inject no-op để chạy nhanh.
            http_post: Hàm HTTP POST tuỳ chỉnh, signature
                ``(url, *, headers, json_body, timeout) -> _HttpResponse``.
                Mặc định wrapper quanh ``requests.post``. Test inject để
                mock toàn bộ HTTP layer.

        Raises:
            ValueError: Nếu cả 3 biến môi trường đều không set, hoặc
                ``mode == "full"`` mà backend được chọn là PUTER.
        """
        # Load .env (silent no-op nếu file thiếu) — không overwrite env vars
        # đã được set qua monkeypatch hoặc shell.
        _load_env_file(env_path)

        self._mode = (mode or "demo").strip().lower()
        self._sleep_fn = sleep_fn if sleep_fn is not None else time.sleep
        self._http_post = http_post if http_post is not None else _default_http_post

        # ---- Backend selection (priority CODEXHUB > OPENAI > GROQ > PUTER) -
        codexhub_key = (os.environ.get("CODEXHUB_API_KEY") or "").strip()
        openai_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
        groq_key = (os.environ.get("GROQ_API_KEY") or "").strip()
        puter_token = (os.environ.get("PUTER_AUTH_TOKEN") or "").strip()

        if codexhub_key:
            self._backend = APIBackend.CODEXHUB
            self._api_key = codexhub_key
            self._base_url = (
                os.environ.get("CODEXHUB_BASE_URL") or self.CODEXHUB_DEFAULT_URL
            ).strip().rstrip("/")
            self._model = (
                os.environ.get("CODEXHUB_MODEL") or self.CODEXHUB_DEFAULT_MODEL
            ).strip()
        elif openai_key:
            self._backend = APIBackend.OPENAI
            self._api_key = openai_key
            self._base_url = self.OPENAI_URL
            self._model = self.OPENAI_MODEL
        elif groq_key:
            self._backend = APIBackend.GROQ
            self._api_key = groq_key
            self._base_url = self.GROQ_URL
            self._model = self.GROQ_MODEL
        elif puter_token:
            self._backend = APIBackend.PUTER
            self._api_key = puter_token
            self._base_url = self.PUTER_URL
            self._model = self.PUTER_MODEL
        else:
            raise ValueError(self.ERR_NO_BACKEND)

        # ---- Mode-block: --mode full không tương thích với Puter --------
        if self._mode == "full" and self._backend is APIBackend.PUTER:
            raise ValueError(self.ERR_FULL_MODE_PUTER)

        # ---- Usage tracking --------------------------------------------
        self._usage = TokenUsageLog(backend=self._backend)
        self._request_count = 0  # increments on EVERY attempt (including 429)

        logger.info(
            "MultiBackendAPIClient initialised: backend=%s, model=%s, mode=%s.",
            self._backend.value,
            self._model,
            self._mode,
        )

    # ---------------------------------------------------------------- API --

    @property
    def backend(self) -> APIBackend:
        """Backend đang dùng (immutable sau __init__)."""
        return self._backend

    @property
    def model(self) -> str:
        """Tên model được gửi trong field ``model`` của OpenAI request."""
        return self._model

    @property
    def mode(self) -> str:
        """``"demo"`` hoặc ``"full"`` (lower-case)."""
        return self._mode

    @property
    def requests_remaining(self) -> int | None:
        """Số requests còn lại trong run.

        Returns:
            * ``int`` (≥ 0) khi backend là Puter.
            * ``None`` khi backend là Groq hoặc OpenAI (no limit).
        """
        if self._backend is APIBackend.PUTER:
            remaining = self.PUTER_MAX_REQUESTS - self._request_count
            return max(0, remaining)
        return None  # CODEXHUB / OPENAI / GROQ: no request limit

    def get_usage_log(self) -> TokenUsageLog:
        """Trả về snapshot copy của token usage log."""
        return TokenUsageLog(
            total_input_tokens=self._usage.total_input_tokens,
            total_output_tokens=self._usage.total_output_tokens,
            total_requests=self._usage.total_requests,
            backend=self._usage.backend,
        )

    # ---------------------------------------------------------- chat call --

    def chat_completion(self, prompt: str) -> APIResponse:
        """Gửi prompt và trả về ``APIResponse``.

        Hành vi lỗi:
          * HTTP 429: chờ ``RETRY_WAIT_SECONDS`` rồi retry, tối đa
            ``MAX_RETRIES`` lần. Nếu vẫn fail → ``content=""``.
          * Timeout sau ``TIMEOUT_SECONDS``: ``content=""`` + log warning.
          * HTTP error khác: ``content=""`` + log warning với mã HTTP.

        Args:
            prompt: Text được tạo bởi ``ObservationParser``.

        Returns:
            ``APIResponse``. ``content`` rỗng khi gặp lỗi không recover được;
            caller phải fallback về default phase ``ETWT``.

        Raises:
            RequestLimitExceeded: Nếu Puter backend đã hit
                ``PUTER_MAX_REQUESTS`` requests trước khi gọi.
        """
        # Enforce Puter request limit BEFORE bumping counter (không áp dụng CODEXHUB).
        if self._backend is APIBackend.PUTER:
            if self._request_count >= self.PUTER_MAX_REQUESTS:
                raise RequestLimitExceeded(
                    "Puter backend đã hit giới hạn "
                    f"{self.PUTER_MAX_REQUESTS} requests/run; "
                    f"đã dùng {self._request_count}."
                )

        url = self._base_url.rstrip("/") + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        body: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        }

        for attempt in range(1, self.MAX_RETRIES + 1):
            # Count this attempt (Puter limit applies to attempts, not just
            # successful responses; safest interpretation of "≤100 req/run").
            self._request_count += 1

            try:
                resp = self._http_post(
                    url,
                    headers=headers,
                    json_body=body,
                    timeout=float(self.TIMEOUT_SECONDS),
                )
            except _HttpTimeout as exc:
                logger.warning(
                    "API timeout after %ds on attempt %d/%d (backend=%s): %s",
                    self.TIMEOUT_SECONDS,
                    attempt,
                    self.MAX_RETRIES,
                    self._backend.value,
                    exc,
                )
                return self._empty_response()
            except Exception as exc:  # noqa: BLE001 - defensive
                # Non-timeout transport errors (DNS, connection reset, etc.)
                # are treated like other HTTP errors per spec.
                logger.warning(
                    "API transport error on attempt %d/%d (backend=%s): %s",
                    attempt,
                    self.MAX_RETRIES,
                    self._backend.value,
                    exc,
                )
                return self._empty_response()

            status = resp.status_code
            if status == 200:
                return self._parse_success(resp)

            if status == 429:
                # Rate-limited → wait and retry, unless this is the last attempt.
                logger.warning(
                    "Rate-limited (HTTP 429) on attempt %d/%d (backend=%s); "
                    "waiting %ds before retry.",
                    attempt,
                    self.MAX_RETRIES,
                    self._backend.value,
                    self.RETRY_WAIT_SECONDS,
                )
                if attempt < self.MAX_RETRIES:
                    self._sleep_fn(float(self.RETRY_WAIT_SECONDS))
                    continue
                # Exhausted retries.
                logger.warning(
                    "Rate-limit retries exhausted (%d attempts) on backend=%s; "
                    "returning empty content.",
                    self.MAX_RETRIES,
                    self._backend.value,
                )
                return self._empty_response()

            # Other HTTP error → empty + warn + status code, no retry.
            logger.warning(
                "API HTTP error %d on backend=%s (attempt %d): %s",
                status,
                self._backend.value,
                attempt,
                resp.text[:200] if isinstance(resp.text, str) else "",
            )
            return self._empty_response()

        # Should be unreachable (loop returns in every branch), but keep a
        # defensive fallback to satisfy type checkers.
        return self._empty_response()  # pragma: no cover

    # ----------------------------------------------------------- internals --

    def _parse_success(self, resp: _HttpResponse) -> APIResponse:
        """Extract content + tokens từ một 200 OK response."""
        body = resp.json_body if resp.json_body is not None else {}
        content = ""
        input_tokens = 0
        output_tokens = 0

        try:
            choices = body.get("choices") or []
            if choices:
                msg = (choices[0] or {}).get("message") or {}
                content = msg.get("content") or ""
            usage = body.get("usage") or {}
            input_tokens = int(usage.get("prompt_tokens") or 0)
            output_tokens = int(usage.get("completion_tokens") or 0)
        except (AttributeError, TypeError, ValueError) as exc:
            logger.warning(
                "Malformed 200 OK response on backend=%s: %s",
                self._backend.value,
                exc,
            )
            content = ""

        # Update usage tracking.
        self._usage.total_requests += 1
        self._usage.total_input_tokens += max(0, input_tokens)
        self._usage.total_output_tokens += max(0, output_tokens)

        return APIResponse(
            content=content,
            input_tokens=max(0, input_tokens),
            output_tokens=max(0, output_tokens),
            backend=self._backend,
        )

    def _empty_response(self) -> APIResponse:
        """APIResponse với content rỗng (caller fallback ETWT)."""
        return APIResponse(
            content="",
            input_tokens=0,
            output_tokens=0,
            backend=self._backend,
        )

