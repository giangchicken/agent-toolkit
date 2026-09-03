"""The ten-class taxonomy this route owns, its provider mapping, and its retry.

Built the way ``test_llm_errors.py`` is and for its reason: the classes are
caught by name and by parent downstream, so a rename or a re-parenting is a
silent breakage a review reads straight past.

What this file pins that its chat-route sibling does not is where the two trees
part. ``EmbedError`` is not an ``LLMError``, so one ``except LLMError`` no
longer covers both routes; in exchange a text over the model's token limit is
its own class and is not retried, where the chat route's window error sets no
status code and spends the full backoff schedule on a certainty.
"""

import asyncio

import aiohttp
import httpx
import openai
import pytest

from agent_toolkit.embed import exceptions as ex
from agent_toolkit.embed.error_mapping import is_retriable, map_error
from agent_toolkit.llm import exceptions as chat

# class, parent, status_code (None where the class sets none)
TAXONOMY: list[tuple[type[ex.EmbedError], type[Exception], int | None]] = [
    (ex.EmbedError, Exception, None),
    (ex.EmbedConfigError, ex.EmbedError, None),
    (ex.EmbedProviderError, ex.EmbedError, None),
    (ex.EmbedAPIError, ex.EmbedError, None),
    (ex.EmbedTimeoutError, ex.EmbedAPIError, 408),
    (ex.EmbedRateLimitError, ex.EmbedAPIError, 429),
    (ex.EmbedAuthenticationError, ex.EmbedAPIError, 401),
    (ex.EmbedModelNotFoundError, ex.EmbedAPIError, 404),
    (ex.EmbedInputTooLargeError, ex.EmbedAPIError, None),
    (ex.EmbedQuotaExceededError, ex.EmbedRateLimitError, 429),
]


class TestTaxonomy:
    def test_there_are_exactly_ten(self) -> None:
        assert len(TAXONOMY) == 10
        assert sorted(ex.__all__) == sorted(cls.__name__ for cls, _, _ in TAXONOMY)

    @pytest.mark.parametrize(
        ("cls", "parent"),
        [(cls, parent) for cls, parent, _ in TAXONOMY],
        ids=[cls.__name__ for cls, _, _ in TAXONOMY],
    )
    def test_parent(self, cls: type[ex.EmbedError], parent: type[Exception]) -> None:
        assert cls.__mro__[1] is parent

    @pytest.mark.parametrize(
        ("cls", "status_code"),
        [(cls, code) for cls, _, code in TAXONOMY if code is not None],
        ids=[cls.__name__ for cls, _, code in TAXONOMY if code is not None],
    )
    def test_status_code(self, cls: type[ex.EmbedAPIError], status_code: int) -> None:
        assert cls("boom").status_code == status_code

    def test_everything_is_catchable_as_embed_error(self) -> None:
        for cls, _, _ in TAXONOMY:
            with pytest.raises(ex.EmbedError):
                raise cls("boom")

    def test_the_chat_routes_classes_are_not_in_this_tree(self) -> None:
        """The whole point of the split, and the migration cost of it.

        A host that wrote one ``except LLMError`` around both routes catches
        nothing from ``vectors`` after this; it writes two clauses, or catches
        ``Exception``.
        """
        assert not issubclass(ex.EmbedError, chat.LLMError)
        assert not issubclass(chat.LLMError, ex.EmbedError)

    def test_an_embed_error_is_not_a_toolkit_error(self) -> None:
        """Same choice the chat route made: one except clause, one concern."""
        from agent_toolkit import ToolkitError

        assert not issubclass(ex.EmbedError, ToolkitError)
        assert not issubclass(ToolkitError, ex.EmbedError)

    def test_it_carries_no_class_this_route_cannot_raise(self) -> None:
        """The four the chat route has and this one does not, and why.

        Nothing here parses a model's output or trips a circuit breaker, and the
        two ``Provider*`` names are spelled for this route's failures instead.
        """
        absent = {
            "LLMParseError",
            "LLMCircuitBreakerError",
            "ProviderQuotaExceededError",
            "ProviderContextWindowError",
        }
        assert absent <= set(chat.__all__)
        assert not absent & set(ex.__all__)


class TestPayloads:
    def test_rate_limit_carries_retry_after(self) -> None:
        assert ex.EmbedRateLimitError(retry_after=2.5).retry_after == 2.5

    def test_retry_after_defaults_to_none(self) -> None:
        assert ex.EmbedRateLimitError().retry_after is None

    def test_timeout_carries_the_timeout(self) -> None:
        assert ex.EmbedTimeoutError(timeout=30.0).timeout == 30.0

    def test_model_not_found_carries_the_model(self) -> None:
        assert ex.EmbedModelNotFoundError(model="no-such-embedder").model == (
            "no-such-embedder"
        )

    def test_details_and_provider_reach_the_message(self) -> None:
        err = ex.EmbedError("boom", details={"batch": 3}, provider="local")
        assert str(err) == "[local] boom (details: {'batch': 3})"

    def test_a_bare_message_stays_bare(self) -> None:
        assert str(ex.EmbedError("boom")) == "boom"

    def test_api_error_shows_provider_and_status(self) -> None:
        err = ex.EmbedAPIError("boom", status_code=503, provider="local")
        assert str(err) == "[local] HTTP 503 boom"


def _openai_status_error(
    cls: type[openai.APIStatusError], status_code: int
) -> openai.APIStatusError:
    request = httpx.Request("POST", "https://example.invalid/v1/embeddings")
    response = httpx.Response(status_code, request=request)
    return cls("upstream said no", response=response, body=None)


class TestMapError:
    def test_status_401_becomes_an_authentication_error(self) -> None:
        assert isinstance(
            map_error(_openai_status_error(openai.AuthenticationError, 401)),
            ex.EmbedAuthenticationError,
        )

    def test_status_429_becomes_a_rate_limit_error(self) -> None:
        assert isinstance(
            map_error(_openai_status_error(openai.RateLimitError, 429)),
            ex.EmbedRateLimitError,
        )

    def test_a_rate_limit_message_without_a_status_code_still_maps(self) -> None:
        assert isinstance(
            map_error(RuntimeError("Rate limit reached for text-embedding-3-small")),
            ex.EmbedRateLimitError,
        )

    def test_a_quota_message_maps_to_rate_limit(self) -> None:
        assert isinstance(
            map_error(RuntimeError("insufficient_quota")), ex.EmbedRateLimitError
        )

    @pytest.mark.parametrize(
        "message",
        [
            "maximum context length is 8192 tokens",
            "This model's context length is exceeded",
            "input is too large for this model",
            "requested 9000 tokens; the input is too long",
        ],
    )
    def test_the_ways_a_provider_says_one_text_is_too_long(self, message: str) -> None:
        """Four phrasings seen from OpenAI-compatible embedding endpoints."""
        assert isinstance(map_error(RuntimeError(message)), ex.EmbedInputTooLargeError)

    def test_an_unrecognized_error_becomes_an_api_error_keeping_its_status(
        self,
    ) -> None:
        source = _openai_status_error(openai.APIStatusError, 503)
        mapped = map_error(source, provider="local")
        assert type(mapped) is ex.EmbedAPIError
        assert mapped.status_code == 503
        assert mapped.provider == "local"

    def test_an_error_with_no_status_code_at_all_maps_with_none(self) -> None:
        mapped = map_error(RuntimeError("connection reset"))
        assert type(mapped) is ex.EmbedAPIError
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

    def test_it_never_answers_with_one_of_the_chat_routes_classes(self) -> None:
        cases: list[Exception] = [
            _openai_status_error(openai.AuthenticationError, 401),
            _openai_status_error(openai.RateLimitError, 429),
            _openai_status_error(openai.APIStatusError, 503),
            RuntimeError("rate limit"),
            RuntimeError("maximum context"),
            RuntimeError("something else"),
        ]
        for exc in cases:
            mapped = map_error(exc)
            assert isinstance(mapped, ex.EmbedError)
            assert not isinstance(mapped, chat.LLMError)


class TestIsRetriable:
    @pytest.mark.parametrize(
        "error",
        [
            ex.EmbedTimeoutError(),
            ex.EmbedRateLimitError(),
            ex.EmbedQuotaExceededError("out of credit"),
            ex.EmbedAPIError("boom", status_code=500),
            ex.EmbedAPIError("boom", status_code=503),
            ex.EmbedAPIError("boom", status_code=None),
            ex.EmbedProviderError("boom"),
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
            ex.EmbedAuthenticationError(),
            ex.EmbedConfigError("no model"),
            ex.EmbedAPIError("boom", status_code=400),
            ex.EmbedAPIError("boom", status_code=404),
            ex.EmbedModelNotFoundError(),
            ex.EmbedAPIError("boom", status_code=422),
        ],
        ids=lambda e: type(e).__name__ + getattr(e, "message", ""),
    )
    def test_not_retriable(self, error: BaseException) -> None:
        assert is_retriable(error) is False

    def test_a_text_too_long_is_not_retried_where_the_chat_route_retries_it(
        self,
    ) -> None:
        """The one classification this route corrects rather than carries over.

        ``ProviderContextWindowError`` sets no status code, so it reaches the
        chat route's ``status_code is None`` branch and is treated as transient;
        ``test_llm_errors.py`` records that as a known wart. The same failure
        here has its own branch: a text of a given length is over the limit on
        every attempt, so retrying it buys nothing.
        """
        assert ex.EmbedInputTooLargeError("too long").status_code is None
        assert is_retriable(ex.EmbedInputTooLargeError("too long")) is False
        assert is_retriable(chat.ProviderContextWindowError("too long")) is True

    def test_a_chat_route_error_still_falls_through_the_catch_all(self) -> None:
        """These branches name this route's classes, so nothing else is known.

        An ``LLMRateLimitError`` reaching here would be retried by the catch-all
        rather than by the rate-limit branch -- the same answer, but by
        accident. It cannot happen on this route; the assertion says why the
        branches had to be respelled rather than shared.
        """
        assert is_retriable(chat.LLMAuthenticationError()) is True
