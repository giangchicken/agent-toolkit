"""``complete_structured()``: parse, repair, validate, or abstain.

The plan points at ``$VAT/tests/test_structured_output.py`` as "the starting
corpus". It is not one -- 154 lines with no test function in it, a module-level
live call to an internal endpoint, and everything but the last block commented
out. Nothing there is portable, so these cases are written from the acceptance
criteria instead.

Two of them are checked against the harvested behavior rather than only against
the new behavior. ``_harvested_repair`` below is
``validate_and_fix_structured_output`` reduced to the branch that decides an
outcome, and the two ``test_the_harvested_repair_would_have_*`` tests show what
it returns for the same responses: a dict with ``None`` in it, and a vote naming
a tool outside the catalog. Both look like successful extractions downstream.
That is the defect this module exists to not have.

The fake transport is the one from ``test_llm_complete.py``, copied rather than
shared: extracting it into ``conftest.py`` would mean editing a passing test
file, which is a separate change to ask for.
"""

import json
from collections.abc import Callable, Iterator
from dataclasses import FrozenInstanceError, replace
from typing import Any

import httpx2
import pytest

from agent_toolkit.llm import (
    DictConfigResolver,
    LLMConfig,
    RetryPolicy,
    ValidationInfo,
    complete_structured,
    set_config_resolver,
    set_default_retry_policy,
)
from agent_toolkit.llm.exceptions import (
    LLMAPIError,
    LLMAuthenticationError,
    LLMConfigError,
)
from agent_toolkit.string_utils import extract_json_from_text

BASE_URL = "https://api.test/v1"
MODEL = "test-model"
QWEN_MODEL = "qwen3-8b"

# Two retries and no delay: one test lets a 500 exhaust the policy.
FAST = RetryPolicy(max_retries=2, base_delay=0.0)

# The record's tool catalog. Every juror vote must be a subset of it, which is
# what "structurally catalog-bounded" means in requirement 18.
CATALOG = ["get_weather", "book_flight", "send_email"]
VOTE = ["get_weather", "send_email"]

Handler = Callable[[httpx2.Request], httpx2.Response]


def vote_schema(catalog: list[str]) -> dict[str, Any]:
    """The juror-vote schema for one record: an array bounded by ``catalog``.

    Built per call from that record's tools, not from a fixed table -- so the
    same response is a valid vote for one record and an invalid one for another.
    """
    return {"type": "array", "items": {"type": "string", "enum": list(catalog)}}


OBJECT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"tools": vote_schema(CATALOG)},
    "required": ["tools"],
}


# --- the provider, minus the socket -----------------------------------------


class FakeEndpoint:
    """A queue of replies and a record of the requests that got them."""

    def __init__(self, replies: tuple[Handler | BaseException, ...]) -> None:
        self._replies = replies
        self.requests: list[httpx2.Request] = []

    @property
    def call_count(self) -> int:
        return len(self.requests)

    def body(self, index: int = -1) -> dict[str, Any]:
        payload: dict[str, Any] = json.loads(self.requests[index].content)
        return payload

    def prompt(self, index: int = -1) -> str:
        return str(self.body(index)["messages"][-1]["content"])

    async def handle(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        reply = self._replies[min(self.call_count, len(self._replies)) - 1]
        if isinstance(reply, BaseException):
            raise reply
        return reply(request)


def _completion(content: str) -> dict[str, Any]:
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


def says(content: str) -> Handler:
    """A 200 whose message content is exactly ``content``."""
    return lambda request: httpx2.Response(200, json=_completion(content))


def fail(status: int, message: str = "unsupported parameter") -> Handler:
    return lambda request: httpx2.Response(status, json={"error": {"message": message}})


@pytest.fixture(autouse=True)
def isolated() -> Iterator[None]:
    base = LLMConfig(model=MODEL, api_key="test-key", base_url=BASE_URL)
    set_config_resolver(
        DictConfigResolver(
            {MODEL: base, QWEN_MODEL: replace(base, model=QWEN_MODEL)},
        )
    )
    set_default_retry_policy(FAST)
    yield
    set_config_resolver(None)
    set_default_retry_policy(None)


@pytest.fixture
def api(monkeypatch: pytest.MonkeyPatch) -> Callable[..., FakeEndpoint]:
    def install(*replies: Handler | BaseException) -> FakeEndpoint:
        endpoint = FakeEndpoint(replies)
        monkeypatch.setattr(
            httpx2.AsyncHTTPTransport, "handle_async_request", endpoint.handle
        )
        return endpoint

    return install


# --- the harvested behavior, for the two controls ----------------------------

_TYPES: dict[str, type | tuple[type, ...]] = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "object": dict,
    "array": list,
}


def _harvested_repair(output: Any, schema: dict[str, Any]) -> dict[str, Any]:
    """``$VAT/llm/llm_utils.py:242``, reduced to the branch that decides.

    Every required key is emitted; a value of the wrong top-level type becomes
    ``None``; nothing else about the schema -- ``enum`` included -- is consulted.
    The real function also returns ``json.dumps(...)`` despite its ``-> Dict``
    annotation; that part is left out because it changes nothing here.
    """
    properties: dict[str, Any] = schema.get("properties", {})
    result: dict[str, Any] = {}
    for key in schema.get("required", list(properties)):
        expected = properties.get(key, {}).get("type", "string")
        value = output.get(key)
        allowed = _TYPES.get(expected)
        result[key] = value if allowed and isinstance(value, allowed) else None
    return result


# --- a clean response --------------------------------------------------------


class TestCleanResponse:
    async def test_a_clean_array_validates_unrepaired(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        api(says(json.dumps(VOTE)))
        value, info = await complete_structured(
            "which tools?", vote_schema(CATALOG), model=MODEL
        )
        assert value == VOTE
        assert info.ok
        assert not info.repaired
        assert info.error is None

    async def test_a_clean_object_validates_unrepaired(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        api(says(json.dumps({"tools": VOTE})))
        value, info = await complete_structured(
            "which tools?", OBJECT_SCHEMA, model=MODEL
        )
        assert value == {"tools": VOTE}
        assert info.ok
        assert not info.repaired

    async def test_an_empty_array_is_a_valid_vote(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        """An abstention the model chose, which is not the same as one we forced."""
        api(says("[]"))
        value, info = await complete_structured(
            "which tools?", vote_schema(CATALOG), model=MODEL
        )
        assert value == []
        assert info.ok

    async def test_the_raw_text_is_kept_on_success(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        api(says('["get_weather"]'))
        _, info = await complete_structured(
            "which tools?", vote_schema(CATALOG), model=MODEL
        )
        assert info.raw == '["get_weather"]'


# --- a response that needs repair --------------------------------------------


class TestRepair:
    @pytest.mark.parametrize(
        ("label", "content"),
        [
            ("fenced", '```json\n["get_weather", "send_email"]\n```'),
            (
                "prose-wrapped",
                'Dựa trên yêu cầu, các tool cần dùng là: ["get_weather", "send_email"]. '
                "Hy vọng câu trả lời này hữu ích.",
            ),
            ("trailing comma", '["get_weather", "send_email",]'),
            (
                "fenced and prose",
                'Kết quả:\n```json\n["get_weather","send_email"]\n```',
            ),
        ],
    )
    async def test_it_validates_and_says_it_was_repaired(
        self, api: Callable[..., FakeEndpoint], label: str, content: str
    ) -> None:
        api(says(content))
        value, info = await complete_structured(
            "which tools?", vote_schema(CATALOG), model=MODEL
        )
        assert value == VOTE, label
        assert info.ok, label
        assert info.repaired, label

    async def test_the_same_value_unwrapped_is_not_reported_as_repaired(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        """The control for the four above: ``repaired`` is not simply always True."""
        api(says('["get_weather", "send_email"]'))
        value, info = await complete_structured(
            "which tools?", vote_schema(CATALOG), model=MODEL
        )
        assert value == VOTE
        assert not info.repaired

    async def test_surrounding_whitespace_alone_is_not_a_repair(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        api(says('\n  ["get_weather"]\n'))
        _, info = await complete_structured(
            "which tools?", vote_schema(CATALOG), model=MODEL
        )
        assert info.ok
        assert not info.repaired


# --- a response that violates the schema ------------------------------------


class TestViolation:
    async def test_a_wrong_type_is_a_failure_not_a_value(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        api(says(json.dumps({"tools": "get_weather"})))
        value, info = await complete_structured(
            "which tools?", OBJECT_SCHEMA, model=MODEL
        )
        assert value is None
        assert not info.ok
        assert info.error is not None

    async def test_an_enum_value_outside_the_catalog_is_a_failure(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        api(says(json.dumps(["get_weather", "delete_database"])))
        value, info = await complete_structured(
            "which tools?", vote_schema(CATALOG), model=MODEL
        )
        assert value is None
        assert not info.ok
        assert "delete_database" in (info.error or "")

    async def test_the_valid_part_of_a_partly_valid_vote_is_not_returned(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        """Requirement 18's clause: an invalid vote is an abstention, not a subset.

        Two of these three tools are in the catalog. Returning those two would be
        a vote the model never cast.
        """
        api(says(json.dumps(["get_weather", "delete_database", "send_email"])))
        value, info = await complete_structured(
            "which tools?", vote_schema(CATALOG), model=MODEL
        )
        assert value is None
        assert not info.ok

    async def test_unparseable_text_is_a_failure_with_the_text_kept(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        api(says("Tôi không chắc nên dùng tool nào."))
        value, info = await complete_structured(
            "which tools?", vote_schema(CATALOG), model=MODEL
        )
        assert value is None
        assert not info.ok
        assert info.error == "no JSON object or array in the response"
        assert info.raw == "Tôi không chắc nên dùng tool nào."

    async def test_an_empty_response_is_a_failure(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        api(says(""))
        value, info = await complete_structured(
            "which tools?", vote_schema(CATALOG), model=MODEL
        )
        assert value is None
        assert not info.ok

    async def test_a_json_null_is_validated_rather_than_called_unparseable(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        """``null`` parses. It is a value the schema rejects, not a missing one."""
        api(says("null"))
        value, info = await complete_structured(
            "which tools?", vote_schema(CATALOG), model=MODEL
        )
        assert value is None
        assert not info.ok
        assert info.error != "no JSON object or array in the response"

    async def test_a_truncated_constrained_decode_is_a_failure_not_a_prefix(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        """The real shape of a guided-decode run that hit ``max_tokens``.

        Nothing in an ``enum`` grammar forbids repeating a branch, so an unbounded
        array can repeat until the token limit and arrive with no closing bracket.
        ``extract_json_from_text`` finds no array -- which is the right answer.
        Repairing this into the first N elements would invent a vote out of a
        decoder loop.
        """
        api(says('[\n  "send_sms",\n' + '  "send_sms",\n' * 20 + '  "s'))
        value, info = await complete_structured(
            "which tools?", vote_schema(CATALOG + ["send_sms"]), model=MODEL
        )
        assert value is None
        assert not info.ok

    async def test_a_duplicated_vote_is_valid_and_this_module_says_so(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        """The hazard the module docstring measures, pinned rather than hidden.

        ``maxItems`` bounds the decoder loop above but does not stop it repeating,
        and a schema without ``uniqueItems`` admits the result. A caller who needs
        distinct tools has to say ``uniqueItems`` -- and that endpoint answers 500
        to it.
        """
        api(says(json.dumps(["send_sms", "send_sms"])))
        value, info = await complete_structured(
            "which tools?", vote_schema(["send_sms"]), model=MODEL
        )
        assert value == ["send_sms", "send_sms"]
        assert info.ok

    async def test_uniqueitems_is_what_would_have_rejected_it(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        """The other half: the schema keyword works here, just not on that endpoint."""
        api(says(json.dumps(["send_sms", "send_sms"])))
        schema = {**vote_schema(["send_sms"]), "uniqueItems": True}
        value, info = await complete_structured("which tools?", schema, model=MODEL)
        assert value is None
        assert not info.ok

    async def test_the_harvested_repair_would_have_returned_a_null_vote(self) -> None:
        """The defect, for the wrong-type response above.

        ``{"tools": None}`` is a dict with the required key present. A caller
        writing it to a votes file records an abstention the model never cast --
        and cannot tell it apart from one it did.
        """
        assert _harvested_repair({"tools": "get_weather"}, OBJECT_SCHEMA) == {
            "tools": None
        }

    async def test_the_harvested_repair_would_have_kept_an_uncatalogued_tool(
        self,
    ) -> None:
        """The other defect: ``enum`` is never consulted, so the bound is not one.

        A list is a list, so this passes the type check and comes back whole --
        including a tool that is not in this record's catalog.
        """
        assert _harvested_repair({"tools": ["delete_database"]}, OBJECT_SCHEMA) == {
            "tools": ["delete_database"]
        }


# --- the catalog bound is per call -------------------------------------------


class TestPerCallCatalog:
    async def test_the_same_vote_is_valid_for_one_record_and_not_the_other(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        endpoint = api(says(json.dumps(["book_flight"])))

        value, info = await complete_structured(
            "which tools?", vote_schema(CATALOG), model=MODEL
        )
        assert value == ["book_flight"]
        assert info.ok

        narrow, info = await complete_structured(
            "which tools?", vote_schema(["get_weather"]), model=MODEL
        )
        assert narrow is None
        assert not info.ok
        assert endpoint.call_count == 2

    async def test_an_empty_catalog_admits_only_the_empty_vote(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        api(says("[]"))
        value, info = await complete_structured(
            "which tools?", vote_schema([]), model=MODEL
        )
        assert value == []
        assert info.ok


# --- what each strategy puts on the wire ------------------------------------


class TestStrategyPayloads:
    async def test_auto_sends_response_format_carrying_the_schema(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        endpoint = api(says(json.dumps(VOTE)))
        _, info = await complete_structured(
            "which tools?", vote_schema(CATALOG), model=MODEL
        )
        assert info.strategy == "native"
        response_format = endpoint.body()["response_format"]
        assert response_format["type"] == "json_schema"
        assert response_format["json_schema"]["schema"] == vote_schema(CATALOG)
        assert "guided_json" not in endpoint.body()

    async def test_auto_does_not_claim_strict(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        """OpenAI strict mode requires an object at the root, and a vote is an array.

        Setting ``strict`` would make the pipeline's own schema unusable on the
        one provider that implements strict mode.
        """
        endpoint = api(says(json.dumps(VOTE)))
        await complete_structured("which tools?", vote_schema(CATALOG), model=MODEL)
        assert "strict" not in endpoint.body()["response_format"]["json_schema"]

    async def test_grammar_sends_guided_json(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        endpoint = api(says(json.dumps(VOTE)))
        _, info = await complete_structured(
            "which tools?", vote_schema(CATALOG), model=MODEL, mode="grammar"
        )
        assert info.strategy == "grammar"
        assert endpoint.body()["guided_json"] == vote_schema(CATALOG)
        assert "response_format" not in endpoint.body()

    async def test_prompt_puts_the_schema_in_the_prompt_and_nothing_on_the_wire(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        endpoint = api(says(json.dumps(VOTE)))
        _, info = await complete_structured(
            "which tools?", vote_schema(CATALOG), model=MODEL, mode="prompt"
        )
        assert info.strategy == "prompt"
        assert "which tools?" in endpoint.prompt()
        assert "get_weather" in endpoint.prompt()
        assert "response_format" not in endpoint.body()
        assert "guided_json" not in endpoint.body()

    async def test_prompt_mode_appends_a_message_when_messages_was_given(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        """``complete`` ignores ``prompt`` when ``messages`` is set.

        Without this branch the schema would be appended to a prompt the provider
        never sees, and every response would fail validation for no visible
        reason.
        """
        endpoint = api(says(json.dumps(VOTE)))
        history = [
            {"role": "system", "content": "Bạn là juror."},
            {"role": "user", "content": "Record 41."},
        ]
        await complete_structured(
            "ignored",
            vote_schema(CATALOG),
            model=MODEL,
            mode="prompt",
            messages=history,
        )
        messages = endpoint.body()["messages"]
        assert len(messages) == 3
        assert messages[:2] == history
        assert "get_weather" in messages[-1]["content"]


# --- falling back ------------------------------------------------------------


class TestFallback:
    async def test_grammar_falls_back_when_the_endpoint_rejects_it(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        endpoint = api(
            fail(400, "guided_json is not supported"), says(json.dumps(VOTE))
        )
        value, info = await complete_structured(
            "which tools?", vote_schema(CATALOG), model=MODEL, mode="grammar"
        )
        assert value == VOTE
        assert info.ok
        assert info.strategy == "prompt"
        assert endpoint.call_count == 2
        assert "guided_json" in endpoint.body(0)
        assert "guided_json" not in endpoint.body(1)
        assert "get_weather" in endpoint.prompt(1)

    async def test_auto_falls_back_when_response_format_is_rejected(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        endpoint = api(fail(400, "response_format is not supported"), says("[]"))
        _, info = await complete_structured(
            "which tools?", vote_schema(CATALOG), model=MODEL
        )
        assert info.ok
        assert info.strategy == "prompt"
        assert endpoint.call_count == 2

    async def test_a_422_also_falls_back(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        api(fail(422, "unprocessable"), says("[]"))
        _, info = await complete_structured(
            "which tools?", vote_schema(CATALOG), model=MODEL, mode="grammar"
        )
        assert info.strategy == "prompt"

    async def test_an_auth_failure_is_not_a_fallback(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        """The harvest's bare ``except Exception`` retried this one unchanged.

        A second request with the same bad key fails the same way and reports the
        fallback's error, hiding the cause.
        """
        endpoint = api(fail(401, "invalid api key"), says("[]"))
        with pytest.raises(LLMAuthenticationError):
            await complete_structured(
                "which tools?", vote_schema(CATALOG), model=MODEL, mode="grammar"
            )
        assert endpoint.call_count == 1

    async def test_a_server_error_is_retried_not_fallen_back(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        endpoint = api(fail(500, "upstream exploded"))
        with pytest.raises(LLMAPIError) as caught:
            await complete_structured(
                "which tools?", vote_schema(CATALOG), model=MODEL, mode="grammar"
            )
        assert caught.value.status_code == 500
        # Three attempts from FAST, all of them still the grammar strategy.
        assert endpoint.call_count == 3
        assert all("guided_json" in endpoint.body(i) for i in range(3))

    async def test_prompt_mode_has_nothing_to_fall_back_to(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        endpoint = api(fail(400, "your prompt is too long"))
        with pytest.raises(LLMAPIError):
            await complete_structured(
                "which tools?", vote_schema(CATALOG), model=MODEL, mode="prompt"
            )
        assert endpoint.call_count == 1

    async def test_a_schema_violation_does_not_fall_back(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        """Only a rejected *request* falls back. A bad answer is a result.

        Re-asking is the caller's decision -- requirement 18 wants the abstention
        recorded, not another request spent on the same juror.
        """
        endpoint = api(says(json.dumps(["delete_database"])))
        value, info = await complete_structured(
            "which tools?", vote_schema(CATALOG), model=MODEL, mode="grammar"
        )
        assert value is None
        assert info.strategy == "grammar"
        assert endpoint.call_count == 1


# --- the extra_body merge in sdk_complete ------------------------------------


class TestExtraBodyMerge:
    async def test_an_explicit_thinking_switch_survives_guided_json(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        """Both keys travel in ``extra_body``, so assigning would drop one.

        ``guided_json`` and ``chat_template_kwargs`` share the same dict, and
        ``sdk_complete`` merges rather than assigns. Whichever the caller sets
        last, both arrive.
        """
        endpoint = api(says(json.dumps(VOTE)))
        _, info = await complete_structured(
            "which tools?",
            vote_schema(CATALOG),
            model=QWEN_MODEL,
            mode="grammar",
            enable_thinking=False,
        )
        assert info.ok
        body = endpoint.body()
        assert body["guided_json"] == vote_schema(CATALOG)
        assert body["chat_template_kwargs"] == {"enable_thinking": False}

    async def test_nothing_turns_thinking_off_on_its_own(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        """The Qwen3 auto-opt-out is gone; the server's default decides.

        This is the regression guard for it. A model name is not consent to
        suppress reasoning -- reasoning is what the pipeline labels *with*.
        """
        endpoint = api(says(json.dumps(VOTE)))
        await complete_structured(
            "which tools?", vote_schema(CATALOG), model=QWEN_MODEL, mode="grammar"
        )
        assert "chat_template_kwargs" not in endpoint.body()

    async def test_a_callers_own_extra_body_survives_alongside_guided_json(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        endpoint = api(says(json.dumps(VOTE)))
        await complete_structured(
            "which tools?",
            vote_schema(CATALOG),
            model=MODEL,
            mode="grammar",
            extra_body={"top_k": 20},
        )
        assert endpoint.body()["top_k"] == 20
        assert "guided_json" in endpoint.body()


# --- reasoning is not the answer ---------------------------------------------


class TestReasoning:
    async def test_a_vote_the_model_talked_itself_out_of_is_not_recorded(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        """The reason parsing reads the answer and not the whole response.

        ``extract_json_from_text`` returns the *first* structure it finds. A
        reasoning model weighing options writes JSON it then rejects, so parsing
        the response whole would record the rejected vote -- and it would validate,
        because both are catalog-bounded arrays.
        """
        api(
            says(
                '<think>Có thể là ["get_weather"]... không, người dùng muốn đặt lịch.'
                '</think>["book_flight"]'
            )
        )
        value, info = await complete_structured(
            "which tools?", vote_schema(CATALOG), model=MODEL
        )
        assert value == ["book_flight"]
        assert info.ok

    async def test_the_reasoning_would_have_been_parsed_without_the_split(self) -> None:
        """The control: the same text, parsed whole, yields the rejected vote."""
        whole = (
            '<think>Có thể là ["get_weather"]... không, người dùng muốn đặt lịch.'
            '</think>["book_flight"]'
        )
        assert extract_json_from_text(whole) == ["get_weather"]

    async def test_the_reasoning_is_kept_next_to_the_vote(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        """A juror record needs the argument, not just the conclusion."""
        api(says(f"<think>Người dùng muốn đặt lịch.</think>{json.dumps(VOTE)}"))
        value, info = await complete_structured(
            "which tools?", vote_schema(CATALOG), model=MODEL
        )
        assert value == VOTE
        assert info.reasoning == "<think>Người dùng muốn đặt lịch.</think>"
        assert info.raw == json.dumps(VOTE)

    async def test_reasoning_from_a_separate_field_is_kept_too(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        def handler(request: httpx2.Request) -> httpx2.Response:
            payload = _completion(json.dumps(VOTE))
            payload["choices"][0]["message"]["reasoning_content"] = "vì cần đặt lịch"
            return httpx2.Response(200, json=payload)

        api(handler)
        _, info = await complete_structured(
            "which tools?", vote_schema(CATALOG), model=MODEL
        )
        assert info.reasoning == "vì cần đặt lịch"
        assert info.ok

    async def test_reasoning_is_empty_when_the_model_emitted_none(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        api(says(json.dumps(VOTE)))
        _, info = await complete_structured(
            "which tools?", vote_schema(CATALOG), model=MODEL
        )
        assert info.reasoning == ""

    async def test_a_reasoning_block_alone_is_a_clean_abstention(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        """Cut off mid-thought: there is no vote, and the reasoning says why."""
        api(says("<think>Tôi đang cân nhắc giữa hai tool"))
        value, info = await complete_structured(
            "which tools?", vote_schema(CATALOG), model=MODEL
        )
        assert value is None
        assert not info.ok
        assert info.reasoning == "<think>Tôi đang cân nhắc giữa hai tool"
        assert info.raw == ""

    async def test_thinking_can_be_switched_off_through_this_function_too(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        endpoint = api(says(json.dumps(VOTE)))
        await complete_structured(
            "which tools?",
            vote_schema(CATALOG),
            model=MODEL,
            enable_thinking=False,
        )
        assert endpoint.body()["chat_template_kwargs"] == {"enable_thinking": False}


# --- what is checked before a request is spent -------------------------------


class TestChecksBeforeSpending:
    async def test_an_invalid_schema_raises_and_costs_nothing(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        endpoint = api(says("[]"))
        with pytest.raises(LLMConfigError, match="invalid JSON schema"):
            await complete_structured(
                "which tools?", {"type": "not-a-json-type"}, model=MODEL
            )
        assert endpoint.call_count == 0

    async def test_an_unknown_mode_raises_and_costs_nothing(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        endpoint = api(says("[]"))
        with pytest.raises(LLMConfigError, match="unknown mode"):
            await complete_structured(
                "which tools?",
                vote_schema(CATALOG),
                model=MODEL,
                mode="native",  # type: ignore[arg-type]
            )
        assert endpoint.call_count == 0

    async def test_a_missing_model_still_raises_from_complete(
        self, api: Callable[..., FakeEndpoint]
    ) -> None:
        endpoint = api(says("[]"))
        with pytest.raises(LLMConfigError):
            await complete_structured("which tools?", vote_schema(CATALOG))
        assert endpoint.call_count == 0


# --- the returned info -------------------------------------------------------


class TestValidationInfo:
    async def test_it_is_frozen(self, api: Callable[..., FakeEndpoint]) -> None:
        api(says("[]"))
        _, info = await complete_structured(
            "which tools?", vote_schema(CATALOG), model=MODEL
        )
        with pytest.raises(FrozenInstanceError):
            info.ok = False  # type: ignore[misc]

    def test_error_is_the_only_optional_field(self) -> None:
        """A caller building one by hand must state the four that decide meaning."""
        info = ValidationInfo(ok=True, strategy="prompt", repaired=False, raw="[]")
        assert info.error is None
