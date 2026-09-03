"""``vectors()``: one batch, retried, throttled, and mapped to the taxonomy.

Built the way ``test_llm_complete.py`` is and for the same reason: only the
socket is fake, so the real openai SDK builds the request and the real SDK
exception classes come back for each status. What that buys on this route is
that the response shape under test is the one the SDK parses into
``CreateEmbeddingResponse``, not a dict this file invented.

Three behaviors here are this route's alone and each has a test that fails
without the code that answers it: rows are paired to inputs by ``index`` and not
by arrival order, a batch that comes back short raises instead of returning a
shifted list, and chat parameters are never defaulted into an embeddings body.

Traffic control is this route's own registry, not the chat route's -- see
``TestTrafficControl``, which pins that the two budgets are separate.
"""

import asyncio
import dataclasses
import json
import logging
import pathlib
from collections.abc import Callable, Iterator
from dataclasses import replace
from typing import Any

import httpx2
import pytest

from agent_toolkit.embed import (
    DictConfigResolver,
    EmbedConfig,
    EnvConfigResolver,
    YamlConfigResolver,
    get_traffic_controller,
    resolve_config,
    set_config_resolver,
    vectors,
)
from agent_toolkit.embed import config as embed_config
from agent_toolkit.embed import traffic_control as embed_traffic
from agent_toolkit.embed.exceptions import (
    EmbedAPIError,
    EmbedAuthenticationError,
    EmbedConfigError,
    EmbedError,
    EmbedInputTooLargeError,
    EmbedProviderError,
)
from agent_toolkit.llm import config as llm_config
from agent_toolkit.llm import traffic_control as llm_traffic

BASE_URL = "https://api.test/v1"
MODEL = "test-embedder"

TEXTS = ("khách hàng muốn tra cứu sao kê", "cho tôi xin sao kê tài khoản")

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

    def body(self, at: int = -1) -> dict[str, Any]:
        payload: dict[str, Any] = json.loads(self.requests[at].content)
        return payload

    def headers(self, at: int = -1) -> httpx2.Headers:
        return self.requests[at].headers

    async def handle(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        reply = self._replies[min(self.call_count, len(self._replies)) - 1]
        if isinstance(reply, BaseException):
            raise reply
        return reply(request)


def _row(index: int, vector: list[float]) -> dict[str, Any]:
    return {"object": "embedding", "index": index, "embedding": vector}


def _response(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "object": "list",
        "model": MODEL,
        "data": rows,
        "usage": {"prompt_tokens": 8, "total_tokens": 8},
    }


def ok(*vectors: list[float]) -> Handler:
    """A 200 carrying one row per vector, numbered in the order given."""
    rows = [_row(index, vector) for index, vector in enumerate(vectors)]
    return lambda request: httpx2.Response(200, json=_response(rows))


def ok_shuffled(*vectors: list[float]) -> Handler:
    """A 200 whose rows arrive back to front, each still carrying its ``index``.

    What a provider is allowed to do and what nothing downstream could detect,
    which is why the pairing is taken from ``index``.
    """
    rows = [_row(index, vector) for index, vector in enumerate(vectors)]
    return lambda request: httpx2.Response(200, json=_response(rows[::-1]))


def ok_echo(request: httpx2.Request) -> httpx2.Response:
    """A 200 with one unit vector per input, so a batch's size is visible."""
    payload = json.loads(request.content)
    rows = [_row(index, [float(index)]) for index in range(len(payload["input"]))]
    return httpx2.Response(200, json=_response(rows))


def fail(status: int, message: str = "server exploded", **headers: str) -> Handler:
    return lambda request: httpx2.Response(
        status, json={"error": {"message": message}}, headers=headers or None
    )


def _install(**overrides: Any) -> None:
    base = EmbedConfig(model=MODEL, api_key="test-key", base_url=BASE_URL)
    set_config_resolver(DictConfigResolver({MODEL: replace(base, **overrides)}))


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


# --- the call itself ---------------------------------------------------------


class TestOneCall:
    async def test_it_returns_one_vector_per_text(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        api(ok([1.0, 0.0], [0.0, 1.0]))
        assert await vectors(TEXTS, model=MODEL) == [[1.0, 0.0], [0.0, 1.0]]

    async def test_the_request_carries_the_resolved_model_key_and_route(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        endpoint = api(ok([1.0], [2.0]))
        await vectors(TEXTS, model=MODEL)
        assert endpoint.body()["model"] == MODEL
        assert endpoint.body()["input"] == list(TEXTS)
        assert endpoint.requests[-1].headers["authorization"] == "Bearer test-key"
        assert str(endpoint.requests[-1].url) == f"{BASE_URL}/embeddings"

    async def test_one_request_carries_every_text(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        """The batch is the point of the route: no ``batch_size``, one call."""
        endpoint = api(ok_echo)
        await vectors([f"text {n}" for n in range(20)], model=MODEL)
        assert endpoint.call_count == 1

    async def test_extra_parameters_reach_the_body(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        endpoint = api(ok([1.0], [2.0]))
        await vectors(TEXTS, model=MODEL, dimensions=256)
        assert endpoint.body()["dimensions"] == 256

    async def test_no_chat_parameter_is_defaulted_into_the_body(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        """``complete`` fills these from the config; this route must not.

        They are the body of a different call. An endpoint that rejects one
        answers 400, and one that ignores it took a number that meant nothing.
        Since ``EmbedConfig`` has no such field, there is nothing here to
        default from -- ``TestConfig`` pins that half; this pins the body.
        """
        endpoint = api(ok([1.0], [2.0]))
        await vectors(TEXTS, model=MODEL)
        assert not {"max_tokens", "temperature", "top_p"} & set(endpoint.body())

    async def test_a_chat_config_cannot_even_be_installed_on_this_route(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        """Control for the test above: the leak is unrepresentable now.

        The chat route defaults ``max_tokens`` off its config, so the way this
        route stays clean used to be that the factory declined to read the
        field. Now the field does not exist to read.
        """
        with pytest.raises(TypeError, match="max_tokens"):
            EmbedConfig(model=MODEL, max_tokens=512)  # type: ignore[call-arg]

    async def test_timeout_configures_the_client_and_not_the_body(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        endpoint = api(ok([1.0], [2.0]))
        await vectors(TEXTS, model=MODEL, timeout=7.0)
        assert "timeout" not in endpoint.body()

    async def test_an_empty_input_answers_without_a_call(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        """Embedding nothing needs no endpoint, and ``input: []`` is a 400."""
        endpoint = api(ok())
        assert await vectors([], model=MODEL) == []
        assert endpoint.call_count == 0


# --- pairing: the failure nothing downstream can see -------------------------


class TestPairing:
    async def test_rows_are_paired_by_index_and_not_by_arrival(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        """The provider may answer out of order; ``index`` says what answers what."""
        api(ok_shuffled([1.0, 0.0], [0.0, 1.0]))
        assert await vectors(TEXTS, model=MODEL) == [[1.0, 0.0], [0.0, 1.0]]

    async def test_a_short_answer_raises_rather_than_shifting_the_list(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        api(ok([1.0, 0.0]))
        with pytest.raises(EmbedProviderError) as raised:
            await vectors(TEXTS, model=MODEL)
        assert "2 texts" in str(raised.value)
        assert "1 vectors" in str(raised.value)

    async def test_a_short_answer_is_not_retried(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        """It is a response the provider stands behind, not a transport failure."""
        endpoint = api(ok([1.0, 0.0]))
        with pytest.raises(EmbedProviderError):
            await vectors(TEXTS, model=MODEL)
        assert endpoint.call_count == 1


# --- batching ----------------------------------------------------------------


class TestBatching:
    async def test_batch_size_splits_the_request(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        endpoint = api(ok_echo)
        await vectors([f"text {n}" for n in range(5)], model=MODEL, batch_size=2)
        assert endpoint.call_count == 3
        assert [len(endpoint.body(at)["input"]) for at in range(3)] == [2, 2, 1]

    async def test_the_pieces_come_back_in_the_order_the_texts_were_given(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        """Concurrent requests, so arrival order is not the answer's order."""
        api(ok_echo)
        texts = [f"text {n}" for n in range(5)]
        assert await vectors(texts, model=MODEL, batch_size=2) == [
            [0.0],
            [1.0],
            [0.0],
            [1.0],
            [0.0],
        ]

    async def test_a_batch_of_one_is_one_request_per_text(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        endpoint = api(ok_echo)
        await vectors(TEXTS, model=MODEL, batch_size=1)
        assert endpoint.call_count == 2

    async def test_a_batch_smaller_than_one_is_refused(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        api(ok_echo)
        with pytest.raises(ValueError, match="at least one text"):
            await vectors(TEXTS, model=MODEL, batch_size=0)


# --- retry, and the taxonomy -------------------------------------------------


class TestRetry:
    async def test_a_server_error_is_retried_and_then_succeeds(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        endpoint = api(fail(500), ok([1.0], [2.0]))
        assert await vectors(TEXTS, model=MODEL, **FAST) == [[1.0], [2.0]]
        assert endpoint.call_count == 2

    async def test_exhaustion_raises_the_mapped_error_and_not_a_retry_error(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        """``reraise=True``, the same choice ``complete`` makes and for its reason."""
        endpoint = api(fail(500))
        with pytest.raises(EmbedError):
            await vectors(TEXTS, model=MODEL, **FAST)
        assert endpoint.call_count == FAST["max_retries"] + 1

    async def test_a_text_over_the_models_limit_is_not_retried(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        """Where this route parts from the chat route's classification.

        ``ProviderContextWindowError`` sets no status code and so is retried by
        the chat route's catch-all -- a wart ``test_llm_errors.py`` records
        rather than fixes. A text over the model's token limit is over it on
        every attempt, so here it fails at once and the caller chunks it.
        """
        endpoint = api(fail(400, "This model's maximum context length is 8192 tokens"))
        with pytest.raises(EmbedInputTooLargeError):
            await vectors(TEXTS, model=MODEL, **FAST)
        assert endpoint.call_count == 1

    async def test_authentication_is_not_retried(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        endpoint = api(fail(401, "bad key"))
        with pytest.raises(EmbedAuthenticationError):
            await vectors(TEXTS, model=MODEL)
        assert endpoint.call_count == 1

    async def test_max_retries_caps_the_attempts(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        endpoint = api(fail(500))
        with pytest.raises(EmbedError):
            await vectors(TEXTS, model=MODEL, max_retries=0)
        assert endpoint.call_count == 1

    async def test_the_retry_keywords_override_the_process_wide_policy(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        endpoint = api(fail(500))
        with pytest.raises(EmbedError):
            await vectors(
                TEXTS,
                model=MODEL,
                max_retries=1,
                retry_delay=0.0,
                exponential_backoff=False,
            )
        assert endpoint.call_count == 2

    @pytest.mark.parametrize(
        ("name", "value"),
        [("max_retries", 0), ("retry_delay", 0.0), ("exponential_backoff", False)],
    )
    async def test_the_retry_keywords_never_reach_the_body(
        self, api: Callable[..., FakeEndpoint], name: str, value: object
    ) -> None:
        """``**kwargs`` is the request body here, so a stray one would ship."""
        endpoint = api(ok([1.0], [2.0]))
        await vectors(TEXTS, model=MODEL, **{name: value})
        assert name not in endpoint.body()

    async def test_a_per_call_keyword_overrides_the_module_default(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        endpoint = api(fail(500))
        with pytest.raises(EmbedError):
            await vectors(TEXTS, model=MODEL, max_retries=0)
        assert endpoint.call_count == 1


# --- traffic control ---------------------------------------------------------


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
        with pytest.raises(EmbedError):
            await vectors(TEXTS, model=MODEL, **FAST)
        assert seen == [1, 1, 1]

    async def test_a_failed_attempt_gives_its_slot_back(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        _install(max_concurrency=1)
        api(fail(500), fail(500), fail(500), ok([1.0], [2.0]))
        with pytest.raises(EmbedError):
            await vectors(TEXTS, model=MODEL, **FAST)
        controller = get_traffic_controller(MODEL)
        assert controller.active_requests == 0
        # With one slot in total, a slot leaked by the three failures above would
        # block this call forever rather than fail it.
        assert await asyncio.wait_for(vectors(TEXTS, model=MODEL), timeout=1.0) == [
            [1.0],
            [2.0],
        ]

    async def test_a_held_slot_really_would_be_caught(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        """Control for the timeout in the test above."""
        api(ok([1.0], [2.0]))
        controller = get_traffic_controller(MODEL, max_concurrency=1)
        await controller.__aenter__()  # never exited
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(vectors(TEXTS, model=MODEL), timeout=0.5)

    async def test_the_controller_takes_its_limits_from_the_config(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        _install(max_concurrency=3, requests_per_minute=61)
        api(ok([1.0], [2.0]))
        await vectors(TEXTS, model=MODEL)
        controller = get_traffic_controller(MODEL)
        assert (controller.max_concurrency, controller.rpm) == (3, 61)

    async def test_the_concurrency_cap_bounds_the_batch_fan_out(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        """Four batches, one slot, so they go out one at a time.

        The controller is looked up inside the handler rather than before the
        call: ``get_traffic_controller`` applies its limits only when it builds
        the controller, so touching it first would cache one with the default
        cap and the config's would never be read.
        """
        _install(max_concurrency=1)
        seen: list[int] = []

        def record(request: httpx2.Request) -> httpx2.Response:
            seen.append(get_traffic_controller(MODEL).active_requests)
            return ok_echo(request)

        api(record)
        await vectors(("a", "b", "c", "d"), model=MODEL, batch_size=1)
        assert seen == [1, 1, 1, 1]

    async def test_a_wider_cap_really_does_let_them_overlap(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        """Control for the test above: prove the cap is what serialized them."""
        _install(max_concurrency=4)
        seen: list[int] = []

        def record(request: httpx2.Request) -> httpx2.Response:
            seen.append(get_traffic_controller(MODEL).active_requests)
            return ok_echo(request)

        api(record)
        await vectors(("a", "b", "c", "d"), model=MODEL, batch_size=1)
        assert max(seen) > 1

    async def test_the_budget_is_this_routes_own_and_not_the_chat_routes(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        """``vectors`` keeps its own registry, so one model has two budgets.

        The cost of ``agent_toolkit.embed.traffic_control`` being its own module
        rather than the chat route's: a model called from both routes is
        throttled twice, once per route, not once in total.
        """
        api(ok([1.0], [2.0]))
        await vectors(TEXTS, model=MODEL)
        assert get_traffic_controller(MODEL) is not llm_traffic.get_traffic_controller(
            MODEL
        )
        assert get_traffic_controller(MODEL) is embed_traffic.get_traffic_controller(
            MODEL
        )


# --- what reaches the log ----------------------------------------------------


class TestLogging:
    async def test_a_retry_is_logged_as_a_warning(
        self, api: Callable[..., FakeEndpoint], caplog: pytest.LogCaptureFixture
    ) -> None:
        api(fail(500), ok([1.0], [2.0]))
        with caplog.at_level(logging.WARNING, logger="agent_toolkit.embed.factory"):
            await vectors(TEXTS, model=MODEL, **FAST)
        assert [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]

    async def test_a_failure_is_logged_as_an_error_naming_the_model(
        self, api: Callable[..., FakeEndpoint], caplog: pytest.LogCaptureFixture
    ) -> None:
        api(fail(500))
        with caplog.at_level(logging.ERROR, logger="agent_toolkit.embed.factory"):
            with pytest.raises(EmbedError):
                await vectors(TEXTS, model=MODEL, max_retries=0)
        errors = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
        assert len(errors) == 1
        assert MODEL in errors[0]
        assert "EmbedAPIError" in errors[0]

    async def test_a_call_logs_its_shape_and_its_result_at_debug(
        self, api: Callable[..., FakeEndpoint], caplog: pytest.LogCaptureFixture
    ) -> None:
        api(ok_echo)
        with caplog.at_level(logging.DEBUG, logger="agent_toolkit.embed.factory"):
            await vectors(("a", "b", "c", "d"), model=MODEL, batch_size=2)
        debug = [r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG]
        assert any("texts=4 batches=2" in message for message in debug)
        assert any("vectors=4" in message for message in debug)

    async def test_the_logger_is_named_for_its_module(self) -> None:
        assert embed_traffic.logger.name == "agent_toolkit.embed.traffic_control"


# --- an unused import guard --------------------------------------------------


def test_the_api_error_class_is_reachable() -> None:
    """``EmbedAPIError`` is the mapped type the 500s above arrive as."""
    assert issubclass(EmbedAPIError, EmbedError)


# --- configuration -----------------------------------------------------------


class TestConfig:
    """``agent_toolkit.embed`` resolves against its own installed resolver."""

    def test_the_config_carries_no_chat_parameters(self) -> None:
        """The fields ``EmbedConfig`` deliberately does not have.

        ``max_tokens``, ``temperature`` and ``top_p`` are the body of the chat
        route; ``reasoning_effort``, ``enable_thinking`` and ``api_version``
        mean nothing on this one. A field here would be a number a resolver
        could set and this route would never send.
        """
        present = {field.name for field in dataclasses.fields(EmbedConfig)}
        assert not present & {
            "api_version",
            "enable_thinking",
            "max_tokens",
            "reasoning_effort",
            "temperature",
            "top_p",
        }
        assert present == {
            "api_key",
            "base_url",
            "binding",
            "extra_headers",
            "max_concurrency",
            "model",
            "requests_per_minute",
            "timeout",
        }

    def test_the_resolver_is_not_the_chat_routes(self) -> None:
        """Installing one for ``llm`` does not configure ``embed``.

        The cost of ``agent_toolkit.embed.config`` being its own module: a host
        that wants both routes configured installs a resolver on each.
        """
        set_config_resolver(None)
        llm_config.set_config_resolver(
            llm_config.DictConfigResolver(
                {MODEL: llm_config.LLMConfig(model=MODEL, api_key="from-llm")}
            )
        )
        try:
            with pytest.raises(EmbedConfigError, match="EMBED_MODEL"):
                resolve_config()
        finally:
            llm_config.set_config_resolver(None)

    def test_an_explicit_argument_beats_the_resolver(self) -> None:
        _install()
        config = resolve_config(model=MODEL, api_key="mine", base_url="https://mine/v1")
        assert (config.api_key, config.base_url) == ("mine", "https://mine/v1")

    def test_extra_headers_merge_into_a_copy(self) -> None:
        set_config_resolver(
            DictConfigResolver(
                {MODEL: EmbedConfig(model=MODEL, extra_headers={"x-a": "1"})}
            )
        )
        first = resolve_config(model=MODEL, extra_headers={"x-b": "2"})
        second = resolve_config(model=MODEL)
        assert first.extra_headers == {"x-a": "1", "x-b": "2"}
        assert second.extra_headers == {"x-a": "1"}

    def test_the_env_resolver_reads_the_embed_variables(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        set_config_resolver(None)
        monkeypatch.setenv("EMBED_MODEL", "env-embedder")
        monkeypatch.setenv("EMBED_API_KEY", "env-key")
        monkeypatch.setenv("EMBED_BASE_URL", "https://env/v1")
        monkeypatch.setenv("LLM_MODEL", "env-chat-model")
        config = EnvConfigResolver().resolve(None)
        assert (config.model, config.api_key, config.base_url) == (
            "env-embedder",
            "env-key",
            "https://env/v1",
        )

    def test_no_model_anywhere_names_what_to_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        set_config_resolver(None)
        monkeypatch.delenv("EMBED_MODEL", raising=False)
        with pytest.raises(EmbedConfigError, match="EMBED_MODEL"):
            resolve_config()

    def test_a_yaml_file_holds_every_model(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "embedders.yaml"
        path.write_text(
            "defaults:\n"
            "  timeout: 30.0\n"
            "  max_concurrency: 4\n"
            "models:\n"
            "  bge-m3: {}\n"
            "  multilingual-e5:\n"
            "    max_concurrency: 1\n",
            encoding="utf-8",
        )
        resolver = YamlConfigResolver(path)
        assert resolver.resolve("bge-m3").max_concurrency == 4
        assert resolver.resolve("multilingual-e5").max_concurrency == 1
        assert resolver.resolve("multilingual-e5").timeout == 30.0

    @pytest.mark.parametrize("key", ["api_key", "base_url"])
    def test_a_yaml_file_may_not_say_where_it_runs(
        self, tmp_path: pathlib.Path, key: str
    ) -> None:
        path = tmp_path / "embedders.yaml"
        path.write_text(f"models:\n  bge-m3:\n    {key}: leaked\n", encoding="utf-8")
        with pytest.raises(EmbedConfigError, match="vectors\\(\\)"):
            YamlConfigResolver(path)

    def test_a_chat_only_setting_in_yaml_is_rejected(
        self, tmp_path: pathlib.Path
    ) -> None:
        """``temperature`` is a typo on this route, not a quietly ignored key."""
        path = tmp_path / "embedders.yaml"
        path.write_text("models:\n  bge-m3:\n    temperature: 0\n", encoding="utf-8")
        with pytest.raises(EmbedConfigError, match="unknown setting"):
            YamlConfigResolver(path)

    async def test_the_resolved_config_reaches_the_request(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        set_config_resolver(
            DictConfigResolver(
                {
                    MODEL: EmbedConfig(
                        model=MODEL,
                        api_key="test-key",
                        base_url=BASE_URL,
                        extra_headers={"x-tenant": "dataforce"},
                    )
                }
            )
        )
        endpoint = api(ok([1.0], [2.0]))
        await vectors(TEXTS, model=MODEL)
        assert endpoint.body()["model"] == MODEL
        assert endpoint.headers()["x-tenant"] == "dataforce"

    def test_the_logger_is_named_for_its_module(self) -> None:
        assert embed_config.logger.name == "agent_toolkit.embed.config"
