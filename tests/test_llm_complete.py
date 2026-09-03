"""``complete()``: one call, retried, throttled, and mapped to the taxonomy.

Only the socket is fake here. The real openai SDK builds the request, and the
real SDK exception classes come back for each status, so error mapping and retry
classification are exercised against the objects they will actually meet.

**Why not respx**, which the plan names: respx patches ``httpx``, and the
installed ``openai`` 3.1.0 moved its HTTP stack to ``httpx2``. respx never sees
the SDK's requests -- they go to the network and time out. ``FakeEndpoint`` below
is the part of respx these tests need, patched onto the transport the SDK really
uses. ``test_the_sdk_still_uses_httpx2`` is the tripwire: if the openai pin ever
moves back to the ``httpx`` line, that one test fails loudly instead of thirty
tests quietly reaching for the network.

One behavior tested here is a change rather than harvested, and it has a
control test that fails against the unchanged version: exhaustion raising the
mapped ``LLMError`` instead of tenacity's ``RetryError``. Retry itself is the
harvested shape -- ``max_retries``, ``retry_delay`` and ``exponential_backoff``
as named parameters over module-level defaults.
"""

import asyncio
import inspect
import json
from collections.abc import Callable, Iterator
from dataclasses import FrozenInstanceError, replace
from typing import Any

import httpx2
import pytest
import tenacity
from openai import AsyncOpenAI

from agent_toolkit.embed import vectors
from agent_toolkit.llm import (
    Completion,
    DictConfigResolver,
    LLMConfig,
    complete,
    complete_with_reasoning,
    factory,
    get_traffic_controller,
    set_config_resolver,
)
from agent_toolkit.llm.exceptions import (
    LLMAPIError,
    LLMAuthenticationError,
    LLMRateLimitError,
)
from agent_toolkit.llm.executors import (
    extract_reasoning_content,
    extract_response_content,
)

BASE_URL = "https://api.test/v1"
MODEL = "test-model"

# Qwen3 matches the pattern the harvested code used to force ``enable_thinking``
# off for. Nothing keys on the model name any more; it is here to prove that.
THINKING_MODEL = "qwen3-8b"

REPLY = "Xin chào, tôi có thể giúp gì cho bạn?"

# Two retries and no delay, passed explicitly wherever a test makes the retry
# actually fire. The module defaults are eight retries five seconds apart, which
# would make one exhaustion test take eight minutes.
FAST = {"max_retries": 2, "retry_delay": 0.0}

Handler = Callable[[httpx2.Request], httpx2.Response]


# --- the provider, minus the socket -----------------------------------------


class FakeEndpoint:
    """A queue of replies and a record of the requests that got them.

    The last reply repeats once the queue runs out, so ``api(fail(500))`` fails
    every attempt and ``api(fail(500), ok())`` fails once then succeeds.
    """

    def __init__(self, replies: tuple[Handler | BaseException, ...]) -> None:
        self._replies = replies
        self.requests: list[httpx2.Request] = []

    @property
    def call_count(self) -> int:
        return len(self.requests)

    def body(self) -> dict[str, Any]:
        payload: dict[str, Any] = json.loads(self.requests[-1].content)
        return payload

    async def handle(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        reply = self._replies[min(self.call_count, len(self._replies)) - 1]
        if isinstance(reply, BaseException):
            raise reply
        return reply(request)


def _completion(content: str | None = REPLY) -> dict[str, Any]:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": MODEL,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }


def ok(content: str | None = REPLY) -> Handler:
    """A 200 carrying ``content``, built fresh for each request."""
    return lambda request: httpx2.Response(200, json=_completion(content))


def fail(status: int, message: str = "server exploded", **headers: str) -> Handler:
    return lambda request: httpx2.Response(
        status, json={"error": {"message": message}}, headers=headers or None
    )


def _config(**overrides: Any) -> LLMConfig:
    base = LLMConfig(model=MODEL, api_key="test-key", base_url=BASE_URL)
    return replace(base, **overrides)


def _install(**overrides: Any) -> None:
    base = _config(**overrides)
    set_config_resolver(
        DictConfigResolver(
            {MODEL: base, THINKING_MODEL: replace(base, model=THINKING_MODEL)}
        )
    )


@pytest.fixture(autouse=True)
def isolated() -> Iterator[None]:
    _install()
    yield
    set_config_resolver(None)


@pytest.fixture
def api(monkeypatch: pytest.MonkeyPatch) -> Callable[..., FakeEndpoint]:
    def install(*replies: Handler | BaseException) -> FakeEndpoint:
        endpoint = FakeEndpoint(replies)
        monkeypatch.setattr(
            httpx2.AsyncHTTPTransport, "handle_async_request", endpoint.handle
        )
        return endpoint

    return install


def test_the_sdk_still_uses_httpx2() -> None:
    """The assumption every fake in this file rests on.

    If this fails the openai pin moved off ``httpx2``, the patched transport is
    one the SDK no longer calls, and respx is usable again -- switch back to it.
    """
    client = AsyncOpenAI(api_key="unused", base_url=BASE_URL)
    assert type(client._client._transport).__module__.startswith("httpx2")


# --- the call itself ---------------------------------------------------------


class TestOneCall:
    async def test_it_returns_the_message_content(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        api(ok())
        assert await complete(model=MODEL, prompt="Chào bạn") == REPLY

    async def test_the_request_carries_the_resolved_model_and_key(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        endpoint = api(ok())
        await complete(model=MODEL, prompt="Chào bạn")
        assert endpoint.body()["model"] == MODEL
        assert endpoint.requests[-1].headers["authorization"] == "Bearer test-key"
        assert str(endpoint.requests[-1].url) == f"{BASE_URL}/chat/completions"

    async def test_the_prompt_becomes_a_user_message(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        endpoint = api(ok())
        await complete(model=MODEL, prompt="Chào bạn", system_prompt="Bạn là trợ lý.")
        assert endpoint.body()["messages"] == [
            {"role": "system", "content": "Bạn là trợ lý."},
            {"role": "user", "content": "Chào bạn"},
        ]

    async def test_messages_replace_the_prompt_entirely(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        endpoint = api(ok())
        await complete(
            model=MODEL,
            prompt="ignored",
            system_prompt="also ignored",
            messages=[{"role": "user", "content": "Đặt lịch hẹn"}],
        )
        assert endpoint.body()["messages"] == [
            {"role": "user", "content": "Đặt lịch hẹn"}
        ]

    async def test_a_response_with_no_choices_is_an_empty_string(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        api(lambda request: httpx2.Response(200, json={**_completion(), "choices": []}))
        assert await complete(model=MODEL, prompt="Chào bạn") == ""

    async def test_a_null_content_is_an_empty_string_not_the_word_none(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        api(ok(content=None))
        assert await complete(model=MODEL, prompt="Chào bạn") == ""


# --- retry ------------------------------------------------------------------


class TestRetry:
    async def test_a_server_error_is_retried_then_succeeds(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        endpoint = api(fail(500), ok())
        assert await complete(model=MODEL, prompt="Chào bạn", **FAST) == REPLY
        assert endpoint.call_count == 2

    async def test_a_rate_limit_is_retried(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        endpoint = api(fail(429, "rate limit exceeded"), ok())
        assert await complete(model=MODEL, prompt="Chào bạn", **FAST) == REPLY
        assert endpoint.call_count == 2

    async def test_exhaustion_raises_the_mapped_error_not_tenacitys(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        """The taxonomy has to survive the last attempt.

        ``reraise=True`` is what makes that true. Without it tenacity raises its
        own ``RetryError`` and the mapped exception is only reachable through
        ``RetryError.last_attempt`` -- see the control below.
        """
        endpoint = api(fail(500))
        with pytest.raises(LLMAPIError) as caught:
            await complete(model=MODEL, prompt="Chào bạn", **FAST)
        assert caught.value.status_code == 500
        assert not isinstance(caught.value, tenacity.RetryError)
        assert endpoint.call_count == FAST["max_retries"] + 1

    async def test_without_reraise_tenacity_would_hide_the_mapped_error(self) -> None:
        """Control for the test above: prove ``reraise=True`` is load-bearing."""

        @tenacity.retry(
            retry=tenacity.retry_if_exception(lambda exc: True),
            wait=tenacity.wait_fixed(0),
            stop=tenacity.stop_after_attempt(2),
        )
        async def always_fails() -> str:
            raise LLMAPIError("server exploded", status_code=500)

        with pytest.raises(tenacity.RetryError):
            await always_fails()

    async def test_an_authentication_error_fails_fast(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        endpoint = api(fail(401, "invalid api key"))
        with pytest.raises(LLMAuthenticationError):
            await complete(model=MODEL, prompt="Chào bạn")
        assert endpoint.call_count == 1

    async def test_a_not_found_fails_fast(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        endpoint = api(fail(404, "unknown model"))
        with pytest.raises(LLMAPIError) as caught:
            await complete(model=MODEL, prompt="Chào bạn")
        assert caught.value.status_code == 404
        assert endpoint.call_count == 1

    @pytest.mark.parametrize(
        "error",
        [asyncio.CancelledError(), KeyboardInterrupt(), GeneratorExit()],
        ids=lambda e: type(e).__name__,
    )
    async def test_a_base_exception_is_never_retried(
        self, api: Callable[..., FakeEndpoint], error: BaseException
    ) -> None:
        """Cancellation must reach the caller on the first attempt.

        These three are ``BaseException``, so the executor's ``except Exception``
        never sees them and they are never mapped. The guard that matters is
        ``is_retriable`` returning False, because tenacity itself catches
        ``BaseException`` and would otherwise retry them.
        """
        endpoint = api(error)
        with pytest.raises(type(error)):
            await complete(model=MODEL, prompt="Chào bạn")
        assert endpoint.call_count == 1

    async def test_a_per_call_policy_overrides_the_default(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        endpoint = api(fail(500))
        with pytest.raises(LLMAPIError):
            await complete(model=MODEL, prompt="Chào bạn", max_retries=0)
        assert endpoint.call_count == 1


class TestRetryAfterIsNotPopulatedYet:
    async def test_the_header_is_ignored_by_the_mapper(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        """``map_error`` never reads ``Retry-After``.

        It builds ``LLMRateLimitError(str(exc), provider=...)`` and nothing
        else, in the harvested code too. This test fails the day someone
        populates it, which is the point: closing the gap should be a visible
        change rather than a silent one.
        """
        api(fail(429, "rate limit exceeded", **{"Retry-After": "30"}))
        with pytest.raises(LLMRateLimitError) as caught:
            await complete(model=MODEL, prompt="Chào bạn", max_retries=0)
        assert caught.value.retry_after is None


# --- the retry keywords, as named parameters --------------------------------


class TestRetryKeywords:
    """``max_retries``, ``retry_delay`` and ``exponential_backoff``.

    The three names the harvested ``complete()`` took. They are named
    parameters here too, so they configure the retry and never fall through
    ``**kwargs`` into the request body.
    """

    async def test_max_retries_caps_the_attempts(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        endpoint = api(fail(500))
        with pytest.raises(LLMAPIError):
            await complete(model=MODEL, prompt="Chào bạn", max_retries=0)
        assert endpoint.call_count == 1

    async def test_they_override_the_default_policy(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        endpoint = api(fail(500))
        with pytest.raises(LLMAPIError):
            await complete(
                model=MODEL,
                prompt="Chào bạn",
                max_retries=1,
                retry_delay=0.0,
                exponential_backoff=False,
            )
        assert endpoint.call_count == 2

    @pytest.mark.parametrize(
        ("name", "value"),
        [("max_retries", 0), ("retry_delay", 0.0), ("exponential_backoff", False)],
    )
    async def test_they_never_reach_the_request_body(
        self, api: Callable[..., FakeEndpoint], name: str, value: object
    ) -> None:
        endpoint = api(ok())
        await complete(model=MODEL, prompt="Chào bạn", **{name: value})
        assert name not in endpoint.body()

    async def test_an_unknown_keyword_really_does_reach_the_provider(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        """``**kwargs`` is the chat-completion passthrough.

        A keyword this function does not name lands in the request body, which
        is why the three above are named rather than left to fall through.
        """
        endpoint = api(ok())
        await complete(model=MODEL, prompt="Chào bạn", top_p=0.25)
        assert endpoint.body()["top_p"] == 0.25


# --- traffic control --------------------------------------------------------


class TestTrafficControl:
    async def test_every_attempt_takes_a_slot(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        controller = get_traffic_controller(MODEL)
        seen: list[int] = []

        def record(request: httpx2.Request) -> httpx2.Response:
            seen.append(controller.active_requests)
            return fail(500)(request)

        api(record)
        with pytest.raises(LLMAPIError):
            await complete(model=MODEL, prompt="Chào bạn", **FAST)
        assert seen == [1, 1, 1]

    async def test_a_failed_attempt_gives_its_slot_back(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        _install(max_concurrency=1)
        api(fail(500), fail(500), fail(500), ok())
        with pytest.raises(LLMAPIError):
            await complete(model=MODEL, prompt="Chào bạn", **FAST)
        controller = get_traffic_controller(MODEL)
        assert controller.active_requests == 0
        # With one slot in total, a slot leaked by the three failures above would
        # block this call forever rather than fail it.
        assert (
            await asyncio.wait_for(
                complete(model=MODEL, prompt="Chào bạn"), timeout=1.0
            )
            == REPLY
        )

    async def test_a_held_slot_really_would_be_caught(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        """Control for the timeout in the test above."""
        api(ok())
        controller = get_traffic_controller(MODEL, max_concurrency=1)
        await controller.__aenter__()  # never exited
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(
                complete(model=MODEL, prompt="Chào bạn"), timeout=0.5
            )

    async def test_the_controller_takes_its_limits_from_the_config(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        """The two ``LLMConfig`` fields nothing read in the harvested code."""
        _install(max_concurrency=3, requests_per_minute=61)
        api(ok())
        await complete(model=MODEL, prompt="Chào bạn")
        controller = get_traffic_controller(MODEL)
        assert (controller.max_concurrency, controller.rpm) == (3, 61)


# --- what the resolved config contributes ------------------------------------


class TestResolvedConfig:
    async def test_max_tokens_and_temperature_come_from_the_config(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        _install(max_tokens=64, temperature=0.1)
        endpoint = api(ok())
        await complete(model=MODEL, prompt="Chào bạn")
        assert (endpoint.body()["max_tokens"], endpoint.body()["temperature"]) == (
            64,
            0.1,
        )

    async def test_an_explicit_argument_wins_over_the_config(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        _install(max_tokens=64, temperature=0.1)
        endpoint = api(ok())
        await complete(model=MODEL, prompt="Chào bạn", max_tokens=8, temperature=0.9)
        assert (endpoint.body()["max_tokens"], endpoint.body()["temperature"]) == (
            8,
            0.9,
        )

    async def test_top_p_comes_from_the_config(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        _install(top_p=0.8)
        endpoint = api(ok())
        await complete(model=MODEL, prompt="Chào bạn")
        assert endpoint.body()["top_p"] == 0.8

    async def test_top_p_is_sent_only_when_something_sets_it(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        """Unset it stays out of the body: not every endpoint accepts the field."""
        endpoint = api(ok())
        await complete(model=MODEL, prompt="Chào bạn")
        assert "top_p" not in endpoint.body()
        await complete(model=MODEL, prompt="Chào bạn", top_p=0.5)
        assert endpoint.body()["top_p"] == 0.5

    async def test_an_explicit_top_p_wins_over_the_config(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        _install(top_p=0.8)
        endpoint = api(ok())
        await complete(model=MODEL, prompt="Chào bạn", top_p=0.5)
        assert endpoint.body()["top_p"] == 0.5

    async def test_enable_thinking_comes_from_the_config(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        _install(enable_thinking=False)
        endpoint = api(ok())
        await complete(model=MODEL, prompt="Chào bạn")
        assert endpoint.body()["chat_template_kwargs"] == {"enable_thinking": False}

    async def test_an_explicit_enable_thinking_wins_over_the_config(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        _install(enable_thinking=False)
        endpoint = api(ok())
        await complete(model=MODEL, prompt="Chào bạn", enable_thinking=True)
        assert endpoint.body()["chat_template_kwargs"] == {"enable_thinking": True}

    async def test_enable_thinking_is_sent_only_when_something_sets_it(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        endpoint = api(ok())
        await complete(model=MODEL, prompt="Chào bạn")
        assert "chat_template_kwargs" not in endpoint.body()

    async def test_the_timeout_comes_from_the_config(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        """It configures the client, so it is read off the request, not the body."""
        _install(timeout=7.0)
        endpoint = api(ok())
        await complete(model=MODEL, prompt="Chào bạn")
        assert endpoint.requests[-1].extensions["timeout"]["read"] == 7.0
        assert "timeout" not in endpoint.body()

    async def test_the_default_timeout_is_the_harvested_two_minutes(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        endpoint = api(ok())
        await complete(model=MODEL, prompt="Chào bạn")
        assert endpoint.requests[-1].extensions["timeout"]["read"] == 120.0

    async def test_an_explicit_timeout_wins_over_the_config(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        """``timeout`` is a named parameter of ``sdk_complete``, so the config's
        value goes through ``kwargs`` rather than alongside it. Passed both ways
        it is "got multiple values for keyword argument" -- raised inside the
        retried block, so it would be mapped and retried nine times before a
        caller saw the TypeError."""
        _install(timeout=99.0)
        endpoint = api(ok())
        await complete(model=MODEL, prompt="Chào bạn", timeout=3.0)
        assert endpoint.requests[-1].extensions["timeout"]["read"] == 3.0
        assert "timeout" not in endpoint.body()

    async def test_extra_headers_are_merged_over_the_configured_ones(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        _install(extra_headers={"x-tenant": "a", "x-keep": "yes"})
        endpoint = api(ok())
        await complete(model=MODEL, prompt="Chào bạn", extra_headers={"x-tenant": "b"})
        headers = endpoint.requests[-1].headers
        assert (headers["x-tenant"], headers["x-keep"]) == ("b", "yes")

    async def test_reasoning_effort_is_sent_only_when_set(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        endpoint = api(ok())
        await complete(model=MODEL, prompt="Chào bạn")
        assert "reasoning_effort" not in endpoint.body()
        await complete(model=MODEL, prompt="Chào bạn", reasoning_effort="low")
        assert endpoint.body()["reasoning_effort"] == "low"

    async def test_api_version_is_accepted_and_never_sent(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        """It configures the Azure client, which v0.1 does not build.

        Accepted because every current call site passes it; named explicitly in
        the signature so it cannot fall through ``**kwargs`` into the body.
        """
        endpoint = api(ok())
        await complete(model=MODEL, prompt="Chào bạn", api_version="2024-06-01")
        assert "api-version" not in str(endpoint.requests[-1].url)
        assert "api_version" not in endpoint.body()


# --- extract_response_content ------------------------------------------------


class _NoContent:
    """An SDK-ish object whose text fields are all empty."""

    content = None
    provider_specific_fields = {"a": 1}


class TestExtractResponseContent:
    @pytest.mark.parametrize(
        ("message", "expected"),
        [
            (None, ""),
            ("Đặt lịch hẹn", "Đặt lịch hẹn"),
            ({"content": "Đặt lịch hẹn"}, "Đặt lịch hẹn"),
            ({"content": None}, ""),
            ({"content": None, "text": "fallback"}, "fallback"),
            ({"content": [{"text": "Đặt "}, {"text": "lịch"}]}, "Đặt lịch"),
            ({"content": ["a", "b"]}, "ab"),
            (7, "7"),
        ],
        ids=[
            "none",
            "str",
            "mapping",
            "null-content",
            "text-fallback",
            "text-parts",
            "str-parts",
            "int",
        ],
    )
    def test_the_shapes_it_reads(self, message: object, expected: str) -> None:
        assert extract_response_content(message) == expected

    def test_a_complex_object_yields_empty_string_not_its_repr(self) -> None:
        """The reason this helper exists rather than ``str(message)``.

        A repr would inject ``"{'provider_specific_fields': ...}"`` into the
        response and corrupt whatever parses it downstream.
        """
        result = extract_response_content(_NoContent())
        assert result == ""
        assert "provider_specific_fields" not in result


# --- the retry defaults ------------------------------------------------------


class TestRetryDefaults:
    def test_they_are_the_hosts_settings(self) -> None:
        """The numbers ``factory.py:35`` used to import from ``src.dependencies``."""
        assert (
            factory.DEFAULT_MAX_RETRIES,
            factory.DEFAULT_RETRY_DELAY,
            factory.DEFAULT_EXPONENTIAL_BACKOFF,
        ) == (8, 5.0, True)

    def test_the_backoff_is_capped_at_two_minutes(self) -> None:
        assert factory.MAX_BACKOFF == 120.0

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("max_retries", 8),
            ("retry_delay", 5.0),
            ("exponential_backoff", True),
        ],
    )
    def test_an_unset_keyword_takes_the_module_default(
        self, name: str, expected: object
    ) -> None:
        """The defaults are bound into the signature, as in the harvested code."""
        for func in (complete, complete_with_reasoning, vectors):
            assert inspect.signature(func).parameters[name].default == expected, (
                func.__name__
            )


# --- thinking and reasoning --------------------------------------------------

REASONING = "Người dùng chào tôi, nên tôi chào lại."


def with_reasoning(field: str, reasoning: str, content: str = REPLY) -> Handler:
    """A 200 whose message carries reasoning in a separate ``field``."""

    def handler(request: httpx2.Request) -> httpx2.Response:
        payload = _completion(content)
        payload["choices"][0]["message"][field] = reasoning
        return httpx2.Response(200, json=payload)

    return handler


class TestThinkingSwitch:
    async def test_nothing_is_sent_when_it_is_not_asked_for(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        """The regression guard for removing the automatic opt-out.

        ``sdk_complete`` used to send ``enable_thinking: False`` for any ``qwen3*``
        model. The server's default decides now.
        """
        endpoint = api(ok())
        await complete(model=THINKING_MODEL, prompt="Chào bạn")
        assert "chat_template_kwargs" not in endpoint.body()
        assert "extra_body" not in endpoint.body()

    @pytest.mark.parametrize("enabled", [True, False])
    async def test_an_explicit_value_is_sent(
        self, api: Callable[..., FakeEndpoint], enabled: bool
    ) -> None:
        endpoint = api(ok())
        await complete(model=MODEL, prompt="Chào bạn", enable_thinking=enabled)
        assert endpoint.body()["chat_template_kwargs"] == {"enable_thinking": enabled}

    async def test_it_is_sent_for_any_model_not_just_the_qwen_line(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        """A caller asking is the reason to send it; the model name is not.

        A server that does not know the kwarg answers 400, which is a clearer
        outcome than silently discarding what the caller asked for.
        """
        endpoint = api(ok())
        await complete(model=MODEL, prompt="Chào bạn", enable_thinking=True)
        assert endpoint.body()["chat_template_kwargs"] == {"enable_thinking": True}


class TestReasoningFromAField:
    @pytest.mark.parametrize("field", ["reasoning_content", "reasoning"])
    async def test_both_spellings_are_read(
        self, api: Callable[..., FakeEndpoint], field: str
    ) -> None:
        api(with_reasoning(field, REASONING))
        result = await complete_with_reasoning(model=MODEL, prompt="Chào bạn")
        assert result.content == REPLY
        assert result.reasoning == REASONING

    async def test_reasoning_content_wins_when_both_are_present(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        def handler(request: httpx2.Request) -> httpx2.Response:
            payload = _completion(REPLY)
            payload["choices"][0]["message"]["reasoning_content"] = REASONING
            payload["choices"][0]["message"]["reasoning"] = "the other one"
            return httpx2.Response(200, json=payload)

        api(handler)
        result = await complete_with_reasoning(model=MODEL, prompt="Chào bạn")
        assert result.reasoning == REASONING

    async def test_no_reasoning_field_is_an_empty_string(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        api(ok())
        result = await complete_with_reasoning(model=MODEL, prompt="Chào bạn")
        assert result.reasoning == ""
        assert result.content == REPLY

    async def test_extract_reasoning_content_reads_a_mapping_too(self) -> None:
        assert extract_reasoning_content({"reasoning_content": REASONING}) == REASONING
        assert extract_reasoning_content({"content": REPLY}) == ""
        assert extract_reasoning_content(None) == ""


class TestReasoningInline:
    async def test_a_think_block_is_split_off_the_answer(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        api(ok(f"<think>{REASONING}</think>{REPLY}"))
        result = await complete_with_reasoning(model=THINKING_MODEL, prompt="Chào bạn")
        assert result.content == REPLY
        assert result.reasoning == f"<think>{REASONING}</think>"

    async def test_a_prefilled_opening_tag_is_split_too(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        """No opening tag in the response, which is what vLLM usually returns."""
        api(ok(f"{REASONING}</think>{REPLY}"))
        result = await complete_with_reasoning(model=THINKING_MODEL, prompt="Chào bạn")
        assert result.content == REPLY
        assert result.reasoning == f"{REASONING}</think>"

    async def test_a_truncated_block_leaves_the_content_empty(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        api(ok(f"<think>{REASONING}"))
        result = await complete_with_reasoning(model=THINKING_MODEL, prompt="Chào bạn")
        assert result.content == ""
        assert result.reasoning == f"<think>{REASONING}"

    async def test_a_field_wins_over_an_inline_block_but_content_is_still_split(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        """Some servers report the field *and* leave the block in the content.

        Concatenating would duplicate the text, so the field is what gets
        reported -- but the block still comes out of the answer.
        """
        api(
            with_reasoning("reasoning_content", REASONING, f"<think>dup</think>{REPLY}")
        )
        result = await complete_with_reasoning(model=MODEL, prompt="Chào bạn")
        assert result.content == REPLY
        assert result.reasoning == REASONING


class TestCompleteStillReturnsAString:
    async def test_it_returns_the_answer_alone(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        """The pinned signature: ``str`` in, ``str`` out, reasoning dropped."""
        api(ok(f"<think>{REASONING}</think>{REPLY}"))
        result = await complete(model=THINKING_MODEL, prompt="Chào bạn")
        assert result == REPLY
        assert isinstance(result, str)

    async def test_a_reasoning_field_does_not_leak_into_it(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        api(with_reasoning("reasoning_content", REASONING))
        assert await complete(model=MODEL, prompt="Chào bạn") == REPLY

    async def test_every_keyword_reaches_the_sibling(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        """``complete`` re-spells the signature by hand, so this checks the wiring."""
        endpoint = api(ok())
        await complete(
            prompt="Chào bạn",
            system_prompt="Bạn là trợ lý.",
            model=MODEL,
            enable_thinking=False,
            max_retries=0,
            top_p=0.25,
        )
        body = endpoint.body()
        assert body["messages"][0] == {"role": "system", "content": "Bạn là trợ lý."}
        assert body["model"] == MODEL
        assert body["top_p"] == 0.25
        assert body["chat_template_kwargs"] == {"enable_thinking": False}


class TestCompletionShape:
    def test_reasoning_defaults_to_empty(self) -> None:
        assert Completion(content=REPLY).reasoning == ""

    def test_it_is_frozen(self) -> None:
        completion = Completion(content=REPLY)
        with pytest.raises(FrozenInstanceError):
            completion.content = "other"  # type: ignore[misc]
