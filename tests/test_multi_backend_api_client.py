"""Unit tests for ``src.training.multi_backend_api_client`` (Task 7.1).

Validates Requirement 7 (AC 1-12) — Multi-Backend API Client supporting
Puter, Groq, OpenAI with priority ``OPENAI > GROQ > PUTER``.

Strategy:

* ``http_post`` được inject qua constructor để tránh phụ thuộc ``requests``
  thực và để mock toàn bộ HTTP layer.
* ``sleep_fn`` được inject thành no-op để retry path không thực sự chờ
  60 giây trong tests.
* ``env_path`` trỏ tới file không tồn tại để tránh load ``.env`` thật của
  project (có thể chứa keys); env vars được control hoàn toàn bởi
  ``monkeypatch``.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import pytest

# Make ``src`` importable when running ``pytest`` from project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.multi_backend_api_client import (  # noqa: E402
    APIBackend,
    APIResponse,
    MultiBackendAPIClient,
    RequestLimitExceeded,
    TokenUsageLog,
    _HttpResponse,
    _HttpTimeout,
)


# =========================================================================
# Helpers
# =========================================================================


NONEXISTENT_ENV = "/tmp/__definitely_not_a_real_dotenv_file__.env"


def _ok_body(content: str = "answer", prompt_tokens: int = 12,
             completion_tokens: int = 7) -> dict:
    """Build a 200 OK OpenAI-compatible body."""
    return {
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        },
    }


def _ok_response(**kwargs) -> _HttpResponse:
    return _HttpResponse(status_code=200, json_body=_ok_body(**kwargs), text="")


def _err_response(status: int, text: str = "") -> _HttpResponse:
    return _HttpResponse(status_code=status, json_body=None, text=text)


class _Recorder:
    """Callable that records each call and returns scripted responses.

    Useful for verifying exact request body / headers / URL across attempts.
    """

    def __init__(self, responses: list):
        # ``responses`` may contain ``_HttpResponse`` or ``Exception`` instances.
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def __call__(self, url, *, headers, json_body, timeout):
        self.calls.append(
            {
                "url": url,
                "headers": dict(headers),
                "json_body": dict(json_body),
                "timeout": timeout,
            }
        )
        if not self._responses:
            raise AssertionError(
                "Recorder ran out of scripted responses; received unexpected "
                f"call to {url}"
            )
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture(autouse=True)
def _clear_backend_env_vars(monkeypatch):
    """Ensure no leakage of real ``OPENAI_API_KEY`` / ``GROQ_API_KEY`` /
    ``PUTER_AUTH_TOKEN`` from the developer's environment into tests.
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("PUTER_AUTH_TOKEN", raising=False)


@pytest.fixture
def noop_sleep():
    """A ``time.sleep``-compatible no-op + call counter."""
    calls: list[float] = []

    def _sleep(seconds: float) -> None:
        calls.append(seconds)

    _sleep.calls = calls  # type: ignore[attr-defined]
    return _sleep


# =========================================================================
# Backend selection priority (AC 1, AC 2)
# =========================================================================


class TestBackendSelectionPriority:
    """AC 1: priority OPENAI > GROQ > PUTER. AC 2: error if none set."""

    def test_all_three_keys_set_picks_openai(self, monkeypatch, noop_sleep):
        monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
        monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
        monkeypatch.setenv("PUTER_AUTH_TOKEN", "puter-tok")

        client = MultiBackendAPIClient(
            env_path=NONEXISTENT_ENV,
            mode="demo",
            sleep_fn=noop_sleep,
            http_post=_Recorder([]),
        )
        assert client.backend is APIBackend.OPENAI
        assert client.model == MultiBackendAPIClient.OPENAI_MODEL == "gpt-4o"

    def test_groq_and_puter_set_picks_groq(self, monkeypatch, noop_sleep):
        monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
        monkeypatch.setenv("PUTER_AUTH_TOKEN", "puter-tok")

        client = MultiBackendAPIClient(
            env_path=NONEXISTENT_ENV,
            mode="demo",
            sleep_fn=noop_sleep,
            http_post=_Recorder([]),
        )
        assert client.backend is APIBackend.GROQ
        assert client.model == "llama-3.3-70b-versatile"

    def test_only_puter_set_picks_puter(self, monkeypatch, noop_sleep):
        monkeypatch.setenv("PUTER_AUTH_TOKEN", "puter-tok")

        client = MultiBackendAPIClient(
            env_path=NONEXISTENT_ENV,
            mode="demo",
            sleep_fn=noop_sleep,
            http_post=_Recorder([]),
        )
        assert client.backend is APIBackend.PUTER
        assert client.model == "gpt-4o"

    def test_openai_only_picks_openai(self, monkeypatch, noop_sleep):
        monkeypatch.setenv("OPENAI_API_KEY", "test-openai-only")

        client = MultiBackendAPIClient(
            env_path=NONEXISTENT_ENV,
            mode="full",
            sleep_fn=noop_sleep,
            http_post=_Recorder([]),
        )
        assert client.backend is APIBackend.OPENAI

    def test_no_backend_raises_value_error(self, noop_sleep):
        """AC 2: if none of the three env vars are set → ValueError."""
        with pytest.raises(ValueError, match="ít nhất một trong bốn backend"):
            MultiBackendAPIClient(
                env_path=NONEXISTENT_ENV,
                mode="demo",
                sleep_fn=noop_sleep,
                http_post=_Recorder([]),
            )

    def test_empty_string_env_var_treated_as_unset(self, monkeypatch, noop_sleep):
        """A whitespace-only or empty key should not select that backend."""
        monkeypatch.setenv("OPENAI_API_KEY", "   ")  # whitespace only
        monkeypatch.setenv("GROQ_API_KEY", "")  # empty
        monkeypatch.setenv("PUTER_AUTH_TOKEN", "real-token")

        client = MultiBackendAPIClient(
            env_path=NONEXISTENT_ENV,
            mode="demo",
            sleep_fn=noop_sleep,
            http_post=_Recorder([]),
        )
        assert client.backend is APIBackend.PUTER


# =========================================================================
# Mode-block: --mode full + PUTER → ValueError (AC 4)
# =========================================================================


class TestModeBlock:
    """AC 4: ``mode=full`` with PUTER backend must abort."""

    def test_full_mode_with_puter_raises(self, monkeypatch, noop_sleep):
        monkeypatch.setenv("PUTER_AUTH_TOKEN", "puter-tok")
        with pytest.raises(ValueError) as exc_info:
            MultiBackendAPIClient(
                env_path=NONEXISTENT_ENV,
                mode="full",
                sleep_fn=noop_sleep,
                http_post=_Recorder([]),
            )
        msg = str(exc_info.value)
        assert "GROQ_API_KEY" in msg
        assert "OPENAI_API_KEY" in msg

    def test_full_mode_with_groq_ok(self, monkeypatch, noop_sleep):
        monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
        client = MultiBackendAPIClient(
            env_path=NONEXISTENT_ENV,
            mode="full",
            sleep_fn=noop_sleep,
            http_post=_Recorder([]),
        )
        assert client.backend is APIBackend.GROQ
        assert client.mode == "full"

    def test_full_mode_with_openai_ok(self, monkeypatch, noop_sleep):
        monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
        client = MultiBackendAPIClient(
            env_path=NONEXISTENT_ENV,
            mode="full",
            sleep_fn=noop_sleep,
            http_post=_Recorder([]),
        )
        assert client.backend is APIBackend.OPENAI

    def test_demo_mode_with_puter_ok(self, monkeypatch, noop_sleep):
        monkeypatch.setenv("PUTER_AUTH_TOKEN", "puter-tok")
        client = MultiBackendAPIClient(
            env_path=NONEXISTENT_ENV,
            mode="demo",
            sleep_fn=noop_sleep,
            http_post=_Recorder([]),
        )
        assert client.backend is APIBackend.PUTER
        assert client.mode == "demo"

    def test_mode_case_insensitive(self, monkeypatch, noop_sleep):
        monkeypatch.setenv("PUTER_AUTH_TOKEN", "puter-tok")
        with pytest.raises(ValueError):
            MultiBackendAPIClient(
                env_path=NONEXISTENT_ENV,
                mode="FULL",  # uppercase still blocked
                sleep_fn=noop_sleep,
                http_post=_Recorder([]),
            )


# =========================================================================
# chat_completion: success path (AC 6, 7, 11)
# =========================================================================


class TestChatCompletionSuccess:
    """200 OK responses are parsed correctly across backends."""

    @pytest.mark.parametrize(
        "env_var,expected_backend,expected_model,expected_url_prefix",
        [
            ("OPENAI_API_KEY", APIBackend.OPENAI, "gpt-4o",
             "https://api.openai.com/v1/"),
            ("GROQ_API_KEY", APIBackend.GROQ, "llama-3.3-70b-versatile",
             "https://api.groq.com/openai/v1/"),
            ("PUTER_AUTH_TOKEN", APIBackend.PUTER, "gpt-4o",
             "https://api.puter.com/puterai/openai/v1/"),
        ],
    )
    def test_each_backend_uses_correct_model_and_url(
        self, monkeypatch, noop_sleep, env_var, expected_backend,
        expected_model, expected_url_prefix,
    ):
        monkeypatch.setenv(env_var, "fake-key-123")
        recorder = _Recorder([_ok_response()])
        client = MultiBackendAPIClient(
            env_path=NONEXISTENT_ENV,
            mode="demo",
            sleep_fn=noop_sleep,
            http_post=recorder,
        )

        resp = client.chat_completion("hello world")

        assert resp.backend is expected_backend
        # Single call recorded.
        assert len(recorder.calls) == 1
        call = recorder.calls[0]
        assert call["url"].startswith(expected_url_prefix)
        assert call["url"].endswith("/chat/completions")
        assert call["json_body"]["model"] == expected_model

    def test_successful_response_returns_content_and_tokens(
        self, monkeypatch, noop_sleep,
    ):
        monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
        recorder = _Recorder([
            _HttpResponse(
                status_code=200,
                json_body=_ok_body(content="<signal>NTST</signal>",
                                   prompt_tokens=42, completion_tokens=9),
                text="",
            )
        ])
        client = MultiBackendAPIClient(
            env_path=NONEXISTENT_ENV,
            mode="full",
            sleep_fn=noop_sleep,
            http_post=recorder,
        )

        resp = client.chat_completion("prompt")

        assert isinstance(resp, APIResponse)
        assert resp.content == "<signal>NTST</signal>"
        assert resp.input_tokens == 42
        assert resp.output_tokens == 9
        assert resp.backend is APIBackend.OPENAI

    def test_temperature_zero_in_request_body(self, monkeypatch, noop_sleep):
        """AC 6: requests must use ``temperature=0`` for determinism."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk")
        recorder = _Recorder([_ok_response()])
        client = MultiBackendAPIClient(
            env_path=NONEXISTENT_ENV,
            mode="demo",
            sleep_fn=noop_sleep,
            http_post=recorder,
        )

        client.chat_completion("anything")

        body = recorder.calls[0]["json_body"]
        assert body["temperature"] == 0

    def test_authorization_header_uses_bearer(self, monkeypatch, noop_sleep):
        monkeypatch.setenv("GROQ_API_KEY", "test-groq-token")
        recorder = _Recorder([_ok_response()])
        client = MultiBackendAPIClient(
            env_path=NONEXISTENT_ENV,
            mode="full",
            sleep_fn=noop_sleep,
            http_post=recorder,
        )

        client.chat_completion("prompt")

        headers = recorder.calls[0]["headers"]
        assert headers["Authorization"] == "Bearer test-groq-token"
        assert headers["Content-Type"] == "application/json"

    def test_messages_format_is_openai_compatible(
        self, monkeypatch, noop_sleep,
    ):
        monkeypatch.setenv("OPENAI_API_KEY", "sk")
        recorder = _Recorder([_ok_response()])
        client = MultiBackendAPIClient(
            env_path=NONEXISTENT_ENV,
            mode="demo",
            sleep_fn=noop_sleep,
            http_post=recorder,
        )

        client.chat_completion("Hello, GPT!")

        body = recorder.calls[0]["json_body"]
        assert body["messages"] == [{"role": "user", "content": "Hello, GPT!"}]


# =========================================================================
# Retry logic on HTTP 429 (AC 8)
# =========================================================================


class TestRetryOn429:
    """AC 8: HTTP 429 → wait 60s, retry up to 3 times."""

    def test_429_then_success_returns_content(
        self, monkeypatch, noop_sleep,
    ):
        monkeypatch.setenv("OPENAI_API_KEY", "sk")
        recorder = _Recorder([
            _err_response(429, "rate limited"),
            _ok_response(content="ok-after-retry"),
        ])
        client = MultiBackendAPIClient(
            env_path=NONEXISTENT_ENV,
            mode="demo",
            sleep_fn=noop_sleep,
            http_post=recorder,
        )

        resp = client.chat_completion("prompt")

        assert resp.content == "ok-after-retry"
        # One retry path → exactly one sleep call of 60s.
        assert noop_sleep.calls == [60.0]
        # Two HTTP attempts.
        assert len(recorder.calls) == 2

    def test_429_three_times_returns_empty(self, monkeypatch, noop_sleep, caplog):
        monkeypatch.setenv("OPENAI_API_KEY", "sk")
        recorder = _Recorder([
            _err_response(429),
            _err_response(429),
            _err_response(429),
        ])
        client = MultiBackendAPIClient(
            env_path=NONEXISTENT_ENV,
            mode="demo",
            sleep_fn=noop_sleep,
            http_post=recorder,
        )

        with caplog.at_level(
            logging.WARNING, logger="src.training.multi_backend_api_client",
        ):
            resp = client.chat_completion("prompt")

        assert resp.content == ""
        assert resp.input_tokens == 0
        assert resp.output_tokens == 0
        assert resp.backend is APIBackend.OPENAI
        # 3 attempts, only 2 sleeps in between (no sleep after final attempt).
        assert noop_sleep.calls == [60.0, 60.0]
        assert len(recorder.calls) == 3
        # Warning logs mention rate-limit / HTTP 429.
        warnings = [r.getMessage() for r in caplog.records
                    if r.levelno == logging.WARNING]
        assert any("429" in m or "Rate-limit" in m for m in warnings)

    def test_429_does_not_count_failed_attempt_as_successful_request(
        self, monkeypatch, noop_sleep,
    ):
        """``total_requests`` counts only successful 200 OK responses."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk")
        recorder = _Recorder([
            _err_response(429),
            _err_response(429),
            _err_response(429),
        ])
        client = MultiBackendAPIClient(
            env_path=NONEXISTENT_ENV,
            mode="demo",
            sleep_fn=noop_sleep,
            http_post=recorder,
        )

        client.chat_completion("prompt")

        log = client.get_usage_log()
        assert log.total_requests == 0
        assert log.total_input_tokens == 0
        assert log.total_output_tokens == 0


# =========================================================================
# Timeout (AC 9)
# =========================================================================


class TestTimeout:
    """AC 9: timeout → empty content + log warning."""

    def test_timeout_returns_empty_content(self, monkeypatch, noop_sleep, caplog):
        monkeypatch.setenv("OPENAI_API_KEY", "sk")
        recorder = _Recorder([_HttpTimeout("request timed out after 30s")])
        client = MultiBackendAPIClient(
            env_path=NONEXISTENT_ENV,
            mode="demo",
            sleep_fn=noop_sleep,
            http_post=recorder,
        )

        with caplog.at_level(
            logging.WARNING, logger="src.training.multi_backend_api_client",
        ):
            resp = client.chat_completion("prompt")

        assert resp.content == ""
        assert resp.backend is APIBackend.OPENAI
        warnings = [r.getMessage() for r in caplog.records
                    if r.levelno == logging.WARNING]
        assert any("timeout" in m.lower() for m in warnings)


# =========================================================================
# Other HTTP errors (AC 10)
# =========================================================================


class TestOtherHttpErrors:
    """AC 10: HTTP error other than 429 → empty + warn + log status code."""

    @pytest.mark.parametrize("status", [500, 502, 503, 400, 401, 403, 404])
    def test_http_error_returns_empty(self, monkeypatch, noop_sleep, caplog, status):
        monkeypatch.setenv("OPENAI_API_KEY", "sk")
        recorder = _Recorder([_err_response(status, f"err {status}")])
        client = MultiBackendAPIClient(
            env_path=NONEXISTENT_ENV,
            mode="demo",
            sleep_fn=noop_sleep,
            http_post=recorder,
        )

        with caplog.at_level(
            logging.WARNING, logger="src.training.multi_backend_api_client",
        ):
            resp = client.chat_completion("prompt")

        assert resp.content == ""
        # Single attempt, no retry on non-429 error.
        assert len(recorder.calls) == 1
        assert noop_sleep.calls == []
        warnings = [r.getMessage() for r in caplog.records
                    if r.levelno == logging.WARNING]
        assert any(str(status) in m for m in warnings)


# =========================================================================
# Puter request limit (AC 5, AC 12)
# =========================================================================


class TestPuterRequestLimit:
    """AC 5: Puter limited to 100 requests/run."""

    def test_first_100_requests_succeed_then_101_raises(
        self, monkeypatch, noop_sleep,
    ):
        monkeypatch.setenv("PUTER_AUTH_TOKEN", "puter-tok")
        # Provide 100 successful responses; 101st must NOT issue an HTTP call.
        recorder = _Recorder([_ok_response() for _ in range(100)])
        client = MultiBackendAPIClient(
            env_path=NONEXISTENT_ENV,
            mode="demo",
            sleep_fn=noop_sleep,
            http_post=recorder,
        )

        for i in range(100):
            resp = client.chat_completion(f"prompt-{i}")
            assert resp.content == "answer"

        # remaining is now 0
        assert client.requests_remaining == 0

        with pytest.raises(RequestLimitExceeded):
            client.chat_completion("over-the-limit")

        # Recorder should only have been called 100 times.
        assert len(recorder.calls) == 100

    def test_requests_remaining_int_for_puter(self, monkeypatch, noop_sleep):
        monkeypatch.setenv("PUTER_AUTH_TOKEN", "puter-tok")
        client = MultiBackendAPIClient(
            env_path=NONEXISTENT_ENV,
            mode="demo",
            sleep_fn=noop_sleep,
            http_post=_Recorder([_ok_response(), _ok_response()]),
        )
        assert client.requests_remaining == 100
        client.chat_completion("p1")
        assert client.requests_remaining == 99
        client.chat_completion("p2")
        assert client.requests_remaining == 98

    def test_requests_remaining_none_for_groq(self, monkeypatch, noop_sleep):
        monkeypatch.setenv("GROQ_API_KEY", "gsk")
        client = MultiBackendAPIClient(
            env_path=NONEXISTENT_ENV,
            mode="full",
            sleep_fn=noop_sleep,
            http_post=_Recorder([]),
        )
        assert client.requests_remaining is None

    def test_requests_remaining_none_for_openai(self, monkeypatch, noop_sleep):
        monkeypatch.setenv("OPENAI_API_KEY", "sk")
        client = MultiBackendAPIClient(
            env_path=NONEXISTENT_ENV,
            mode="full",
            sleep_fn=noop_sleep,
            http_post=_Recorder([]),
        )
        assert client.requests_remaining is None


# =========================================================================
# Token usage tracking (AC 11)
# =========================================================================


class TestTokenUsageTracking:
    """AC 11: ``get_usage_log()`` reports total tokens + backend."""

    def test_usage_log_initial_state(self, monkeypatch, noop_sleep):
        monkeypatch.setenv("OPENAI_API_KEY", "sk")
        client = MultiBackendAPIClient(
            env_path=NONEXISTENT_ENV,
            mode="full",
            sleep_fn=noop_sleep,
            http_post=_Recorder([]),
        )

        log = client.get_usage_log()
        assert isinstance(log, TokenUsageLog)
        assert log.total_input_tokens == 0
        assert log.total_output_tokens == 0
        assert log.total_requests == 0
        assert log.backend is APIBackend.OPENAI

    def test_usage_accumulates_over_multiple_requests(
        self, monkeypatch, noop_sleep,
    ):
        monkeypatch.setenv("OPENAI_API_KEY", "sk")
        recorder = _Recorder([
            _HttpResponse(200, _ok_body(prompt_tokens=10, completion_tokens=5), ""),
            _HttpResponse(200, _ok_body(prompt_tokens=20, completion_tokens=15), ""),
            _HttpResponse(200, _ok_body(prompt_tokens=7, completion_tokens=3), ""),
        ])
        client = MultiBackendAPIClient(
            env_path=NONEXISTENT_ENV,
            mode="full",
            sleep_fn=noop_sleep,
            http_post=recorder,
        )

        client.chat_completion("p1")
        client.chat_completion("p2")
        client.chat_completion("p3")

        log = client.get_usage_log()
        assert log.total_input_tokens == 37
        assert log.total_output_tokens == 23
        assert log.total_requests == 3
        assert log.backend is APIBackend.OPENAI

    def test_usage_log_returns_snapshot_copy(self, monkeypatch, noop_sleep):
        """Mutating the returned log must not affect internal state."""
        monkeypatch.setenv("GROQ_API_KEY", "gsk")
        recorder = _Recorder([_ok_response(prompt_tokens=5, completion_tokens=2)])
        client = MultiBackendAPIClient(
            env_path=NONEXISTENT_ENV,
            mode="full",
            sleep_fn=noop_sleep,
            http_post=recorder,
        )

        client.chat_completion("p")
        log1 = client.get_usage_log()
        log1.total_input_tokens = 9999

        log2 = client.get_usage_log()
        assert log2.total_input_tokens == 5  # unaffected

    def test_failed_requests_do_not_increment_token_counters(
        self, monkeypatch, noop_sleep,
    ):
        monkeypatch.setenv("OPENAI_API_KEY", "sk")
        recorder = _Recorder([
            _err_response(500),
            _ok_response(prompt_tokens=4, completion_tokens=6),
        ])
        client = MultiBackendAPIClient(
            env_path=NONEXISTENT_ENV,
            mode="full",
            sleep_fn=noop_sleep,
            http_post=recorder,
        )

        client.chat_completion("fail")
        client.chat_completion("ok")

        log = client.get_usage_log()
        # Only the second request is counted in totals.
        assert log.total_input_tokens == 4
        assert log.total_output_tokens == 6
        assert log.total_requests == 1


# =========================================================================
# Read-only properties
# =========================================================================


class TestProperties:
    def test_backend_immutable_after_init(self, monkeypatch, noop_sleep):
        monkeypatch.setenv("OPENAI_API_KEY", "sk")
        client = MultiBackendAPIClient(
            env_path=NONEXISTENT_ENV,
            mode="full",
            sleep_fn=noop_sleep,
            http_post=_Recorder([]),
        )
        assert client.backend is APIBackend.OPENAI
        # ``backend`` is exposed as a read-only property; setting must fail.
        with pytest.raises(AttributeError):
            client.backend = APIBackend.PUTER  # type: ignore[misc]

    def test_model_property_returns_correct_string(
        self, monkeypatch, noop_sleep,
    ):
        monkeypatch.setenv("GROQ_API_KEY", "gsk")
        client = MultiBackendAPIClient(
            env_path=NONEXISTENT_ENV,
            mode="full",
            sleep_fn=noop_sleep,
            http_post=_Recorder([]),
        )
        assert client.model == "llama-3.3-70b-versatile"

    def test_mode_property_normalised_lowercase(self, monkeypatch, noop_sleep):
        monkeypatch.setenv("OPENAI_API_KEY", "sk")
        client = MultiBackendAPIClient(
            env_path=NONEXISTENT_ENV,
            mode="DEMO",
            sleep_fn=noop_sleep,
            http_post=_Recorder([]),
        )
        assert client.mode == "demo"

