"""The twelve-class taxonomy, the provider mapping, and the retry decision.

The taxonomy tests are deliberately mechanical. They exist because
`agent-evaluation` catches these classes by name and by parent, so a rename or a
re-parenting here is a silent breakage there -- exactly the kind of change a
review reads straight past.
"""

import asyncio

import aiohttp
import httpx
import openai
import pytest

from agent_toolkit.llm import exceptions as ex
from agent_toolkit.llm.error_mapping import is_retriable, map_error

# class, parent, status_code (None where the class sets none)
TAXONOMY: list[tuple[type[ex.LLMError], type[Exception], int | None]] = [
    (ex.LLMError, Exception, None),
    (ex.LLMConfigError, ex.LLMError, None),
    (ex.LLMProviderError, ex.LLMError, None),
    (ex.LLMCircuitBreakerError, ex.LLMError, None),
    (ex.LLMParseError, ex.LLMError, None),
    (ex.LLMAPIError, ex.LLMError, None),
    (ex.LLMTimeoutError, ex.LLMAPIError, 408),
    (ex.LLMRateLimitError, ex.LLMAPIError, 429),
    (ex.LLMAuthenticationError, ex.LLMAPIError, 401),
    (ex.LLMModelNotFoundError, ex.LLMAPIError, 404),
    (ex.ProviderQuotaExceededError, ex.LLMRateLimitError, 429),
    (ex.ProviderContextWindowError, ex.LLMAPIError, None),
]


class TestTaxonomy:
    def test_there_are_exactly_twelve(self) -> None:
        assert len(TAXONOMY) == 12
        assert sorted(ex.__all__) == sorted(cls.__name__ for cls, _, _ in TAXONOMY)

    @pytest.mark.parametrize(
        ("cls", "parent"),
        [(cls, parent) for cls, parent, _ in TAXONOMY],
        ids=[cls.__name__ for cls, _, _ in TAXONOMY],
    )
    def test_parent(self, cls: type[ex.LLMError], parent: type[Exception]) -> None:
        assert cls.__mro__[1] is parent

    @pytest.mark.parametrize(
        ("cls", "status_code"),
        [(cls, code) for cls, _, code in TAXONOMY if code is not None],
        ids=[cls.__name__ for cls, _, code in TAXONOMY if code is not None],
    )
    def test_status_code(self, cls: type[ex.LLMAPIError], status_code: int) -> None:
        assert cls("boom").status_code == status_code

    def test_everything_is_catchable_as_llm_error(self) -> None:
        for cls, _, _ in TAXONOMY:
            with pytest.raises(ex.LLMError):
                raise cls("boom")

    def test_an_llm_error_is_not_a_toolkit_error(self) -> None:
        """The two hierarchies stay separate: one except clause, one concern."""
        from agent_toolkit import ToolkitError

        assert not issubclass(ex.LLMError, ToolkitError)
        assert not issubclass(ToolkitError, ex.LLMError)


class TestPayloads:
    def test_rate_limit_carries_retry_after(self) -> None:
        assert ex.LLMRateLimitError(retry_after=2.5).retry_after == 2.5

    def test_retry_after_defaults_to_none(self) -> None:
        assert ex.LLMRateLimitError().retry_after is None

    def test_timeout_carries_the_timeout(self) -> None:
        assert ex.LLMTimeoutError(timeout=30.0).timeout == 30.0

    def test_model_not_found_carries_the_model(self) -> None:
        assert ex.LLMModelNotFoundError(model="no-such-model").model == "no-such-model"

    def test_details_and_provider_reach_the_message(self) -> None:
        err = ex.LLMError("boom", details={"attempt": 3}, provider="local")
        assert str(err) == "[local] boom (details: {'attempt': 3})"

    def test_a_bare_message_stays_bare(self) -> None:
        assert str(ex.LLMError("boom")) == "boom"

    def test_api_error_shows_provider_and_status(self) -> None:
        err = ex.LLMAPIError("boom", status_code=503, provider="local")
        assert str(err) == "[local] HTTP 503 boom"


def _openai_status_error(
    cls: type[openai.APIStatusError], status_code: int
) -> openai.APIStatusError:
    request = httpx.Request("POST", "https://example.invalid/v1/chat/completions")
    response = httpx.Response(status_code, request=request)
    return cls("upstream said no", response=response, body=None)


class TestMapError:
    def test_status_401_becomes_an_authentication_error(self) -> None:
        assert isinstance(
            map_error(_openai_status_error(openai.AuthenticationError, 401)),
            ex.LLMAuthenticationError,
        )

    def test_status_429_becomes_a_rate_limit_error(self) -> None:
        assert isinstance(
            map_error(_openai_status_error(openai.RateLimitError, 429)),
            ex.LLMRateLimitError,
        )

    def test_a_rate_limit_message_without_a_status_code_still_maps(self) -> None:
        assert isinstance(
            map_error(RuntimeError("Rate limit reached for gpt-4o")),
            ex.LLMRateLimitError,
        )

    def test_a_quota_message_maps_to_rate_limit(self) -> None:
        assert isinstance(
            map_error(RuntimeError("insufficient_quota")), ex.LLMRateLimitError
        )

    def test_a_context_length_message_maps_to_the_context_window_error(self) -> None:
        assert isinstance(
            map_error(RuntimeError("maximum context length is 8192 tokens")),
            ex.ProviderContextWindowError,
        )

    def test_an_unrecognized_error_becomes_an_api_error_keeping_its_status(
        self,
    ) -> None:
        source = _openai_status_error(openai.APIStatusError, 503)
        mapped = map_error(source, provider="local")
        assert type(mapped) is ex.LLMAPIError
        assert mapped.status_code == 503
        assert mapped.provider == "local"

    def test_an_error_with_no_status_code_at_all_maps_with_none(self) -> None:
        mapped = map_error(RuntimeError("connection reset"))
        assert type(mapped) is ex.LLMAPIError
        assert mapped.status_code is None

    def test_the_provider_label_survives_every_branch(self) -> None:
        cases: list[Exception] = [
            _openai_status_error(openai.AuthenticationError, 401),
            _openai_status_error(openai.RateLimitError, 429),
            RuntimeError("rate limit"),
            RuntimeError("maximum context"),
            RuntimeError("something else"),
        ]
        assert [map_error(exc, provider="local").provider for exc in cases] == [
            "local"
        ] * 5

    def test_the_sdk_type_rules_are_shadowed_by_the_status_heuristic(self) -> None:
        """Documented, not asserted as good: the two typed rules cannot fire.

        `map_error` checks `status_code` before consulting its rule list, and
        every `openai.APIStatusError` carries the status code of the response it
        was built from -- 401 for AuthenticationError, 429 for RateLimitError.
        The rules are kept because they are the original's, and because they are
        the only branch left if a future SDK stops setting `status_code`.
        """
        assert _openai_status_error(openai.AuthenticationError, 401).status_code == 401
        assert _openai_status_error(openai.RateLimitError, 429).status_code == 429


class TestIsRetriable:
    @pytest.mark.parametrize(
        "error",
        [
            ex.LLMTimeoutError(),
            ex.LLMRateLimitError(),
            ex.ProviderQuotaExceededError("out of credit"),
            ex.LLMAPIError("boom", status_code=500),
            ex.LLMAPIError("boom", status_code=503),
            ex.LLMAPIError("boom", status_code=None),
            ex.LLMProviderError("boom"),
            asyncio.TimeoutError(),
            aiohttp.ClientError(),
            httpx.ConnectError("no route to host"),
            ConnectionResetError(),
        ],
        ids=lambda e: type(e).__name__ + getattr(e, "message", ""),
    )
    def test_retriable(self, error: BaseException) -> None:
        assert is_retriable(error) is True

    @pytest.mark.parametrize(
        "error",
        [
            asyncio.CancelledError(),
            KeyboardInterrupt(),
            GeneratorExit(),
            ex.LLMAuthenticationError(),
            ex.LLMConfigError("no model"),
            ex.LLMAPIError("boom", status_code=400),
            ex.LLMAPIError("boom", status_code=404),
            ex.LLMModelNotFoundError(),
            ex.LLMAPIError("boom", status_code=422),
        ],
        ids=lambda e: type(e).__name__ + getattr(e, "message", ""),
    )
    def test_not_retriable(self, error: BaseException) -> None:
        assert is_retriable(error) is False

    def test_an_httpx_connection_error_is_retried_by_the_catch_all(self) -> None:
        """The branch that matters in practice, and it is the unnamed one.

        The client calls through the openai SDK, which raises `httpx` errors;
        the explicit branch names `aiohttp`, which nothing on this path raises.
        Connection failures are retried only because an unrecognized exception
        is retriable.
        """
        assert not isinstance(httpx.ConnectError("x"), aiohttp.ClientError)
        assert is_retriable(httpx.ConnectError("x")) is True

    def test_a_context_window_error_is_currently_retried(self) -> None:
        """Carried over unchanged, and flagged.

        `ProviderContextWindowError` sets no status code, so it reaches the
        `status_code is None` branch and is treated as a transient failure. A
        prompt that overflows the context window will overflow it identically on
        every retry, so this spends the full backoff schedule on a certainty.
        The spec pins retry classification as carrying over unchanged, so this
        test records the behavior rather than correcting it.
        """
        assert ex.ProviderContextWindowError("too long").status_code is None
        assert is_retriable(ex.ProviderContextWindowError("too long")) is True
