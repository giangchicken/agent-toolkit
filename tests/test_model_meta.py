"""Model metadata: the tables, the two staleness fixes, and the token estimate.

Every fix here has a control test that runs the harvested version of the same
table and shows it giving the wrong answer, because "we fixed a stale pattern"
is worth nothing if the pattern was never stale in the direction claimed. One of
the plan's three criteria turned out to be exactly that -- see
``TestNativeToolCalling`` -- and is documented rather than dressed up.

The token measurements are recorded from real ``gemma-4-31B-it`` responses.
"""

import re
from collections.abc import Mapping, Sequence
from typing import Any

import pytest
import tiktoken

from agent_toolkit.llm import model_meta
from agent_toolkit.llm.model_meta import (
    count_tokens,
    model_family,
    supports_native_tool_calling,
    supports_reasoning,
)

MODEL = "gemma-4-31B-it"


# --- the harvested versions, for control tests -------------------------------


def _harvested_family(llm_name: str) -> str:
    """``$VAT/llm/llm_utils.py:68``, verbatim minus the try/except."""
    llm_name = llm_name.lower()
    if "gpt" in llm_name:
        return "gpt"
    elif "qwen" in llm_name:
        return "qwen"
    elif "llama" in llm_name or "vicuna" in llm_name:
        return "llama"
    elif "gemini" in llm_name:
        return "gemini"
    else:
        return "unknown"


# ``$VAT/llm/constants.py``, verbatim.
_VAT_REASONING = ["^qwen3-.*", "^glm-4.*", "^gpt-oss-.*", "^gpt-5.*", "^deepseek-.*"]
_VAT_NATIVE_FC = ["^glm-4.*", "^gpt-4.*", "^gpt-5.*"]


def _matches(patterns: list[str], name: str) -> bool:
    return any(re.match(pattern, name.lower()) for pattern in patterns)


# --- model_family ------------------------------------------------------------


class TestModelFamily:
    @pytest.mark.parametrize(
        ("name", "family"),
        [
            ("gemma-4-31B-it", "gemma"),
            ("gemma-3-27b-it", "gemma"),
            ("glm-5.1", "glm"),
            ("glm-4.6", "glm"),
            ("gpt-5.1", "gpt"),
            ("gpt-oss-120b", "gpt"),
            ("gpt-4o", "gpt"),
            ("qwen3-32b", "qwen"),
            ("qwen2.5-7b-instruct", "qwen"),
            ("deepseek-v3", "deepseek"),
            ("llama-3.3-70b-instruct", "llama"),
            ("vicuna-13b", "llama"),
            ("gemini-2.5-pro", "gemini"),
            ("mystery-model-7b", "unknown"),
            ("", "unknown"),
        ],
    )
    def test_the_table(self, name: str, family: str) -> None:
        assert model_family(name) == family

    def test_it_is_case_insensitive(self) -> None:
        assert model_family("GEMMA-4-31B-IT") == "gemma"

    def test_the_five_lines_the_jury_needs_are_five_families(self) -> None:
        """Requirement 19 counts *distinct* families, so collapsing any two lies."""
        jury = ["gemma-4-31B-it", "glm-5.1", "gpt-5.1", "qwen3-32b", "deepseek-v3"]
        assert len({model_family(name) for name in jury}) == 5

    def test_the_harvested_function_collapsed_three_of_them(self) -> None:
        """Control: the defect this table fixes, and why it mattered.

        gemma, glm and deepseek all fell through to ``"unknown"``, so a panel
        drawn from those three read as one family and passed no diversity check.
        Requirement 20 is worse off still: it must exclude jurors from the family
        that labelled 67.3% of the corpus, and that family had no name.
        """
        collapsed = {
            _harvested_family(n) for n in ("gemma-4-31B-it", "glm-5.1", "deepseek-v3")
        }
        assert collapsed == {"unknown"}
        assert _harvested_family("gemma-4-31B-it") != "gemma"

    def test_a_distill_takes_the_first_matching_row(self) -> None:
        """Documents the one ambiguous name rather than leaving it to chance."""
        assert model_family("deepseek-r1-distill-qwen-32b") == "deepseek"

    def test_an_unknown_name_returns_rather_than_raises(self) -> None:
        assert model_family("!!!") == model_meta.UNKNOWN_FAMILY


# --- supports_reasoning ------------------------------------------------------


class TestReasoning:
    def test_glm_5_1_is_a_reasoning_model(self) -> None:
        """The plan's second criterion."""
        assert supports_reasoning("glm-5.1") is True

    def test_the_harvested_table_missed_glm_5(self) -> None:
        """Control: ``^glm-4.*`` does not match ``glm-5.1``.

        ``agent-evaluation``'s copy of this table had already been patched;
        ``voice-agent-toolkit``'s had not. Two copies, one patched.
        """
        assert _matches(_VAT_REASONING, "glm-4.6") is True
        assert _matches(_VAT_REASONING, "glm-5.1") is False

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("qwen3-32b", True),
            ("qwen3.5-14b", True),
            ("glm-4.6", True),
            ("gpt-5.1", True),
            ("gpt-oss-120b", True),
            ("deepseek-r1", True),
            ("gemma-4-31B-it", False),
            ("gpt-4o", False),
            ("qwen2.5-7b-instruct", False),
            ("llama-3.3-70b-instruct", False),
            ("mystery-model-7b", False),
        ],
    )
    def test_the_table(self, name: str, expected: bool) -> None:
        assert supports_reasoning(name) is expected


# --- supports_native_tool_calling -------------------------------------------


class TestNativeToolCalling:
    def test_gemma_4_has_no_native_tool_calling(self) -> None:
        """The plan's third criterion -- which passed before the fix too.

        The plan attributes this to ``NON_NATIVE_FC_LLMS_PATTERNS`` holding a
        stale ``^gemma-3-.*``, and says gemma-4 was therefore "treated as having
        native function calling". It was not: that table has no reader anywhere
        in ``voice-agent-toolkit``, and the function consults the *allowlist*
        only, which gemma has never been on. The answer was already correct, so
        this test guards it rather than fixing it.
        """
        assert supports_native_tool_calling("gemma-4-31B-it") is False
        assert _matches(_VAT_NATIVE_FC, "gemma-4-31B-it") is False

    def test_glm_5_1_has_native_tool_calling(self) -> None:
        """The stale pattern that did bite, in the table the plan did not name."""
        assert supports_native_tool_calling("glm-5.1") is True

    def test_the_harvested_table_denied_it(self) -> None:
        """Control: the same ``^glm-4.*`` hole, in the tool-calling table.

        Unpatched in both copies, so ``glm-5.1`` -- the pipeline's default
        generator -- was sent prompt-based tool calling instead of ``tools``.
        """
        assert _matches(_VAT_NATIVE_FC, "glm-4.6") is True
        assert _matches(_VAT_NATIVE_FC, "glm-5.1") is False

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("glm-4.6", True),
            ("glm-5.1", True),
            ("gpt-4o", True),
            ("gpt-5.1", True),
            ("gemma-4-31B-it", False),
            ("qwen3-32b", False),
            ("deepseek-v3", False),
            ("mystery-model-7b", False),
        ],
    )
    def test_the_table(self, name: str, expected: bool) -> None:
        assert supports_native_tool_calling(name) is expected

    def test_an_unlisted_model_is_false_not_true(self) -> None:
        """The allowlist's direction is the safe one and worth pinning.

        False costs a prompt-based fallback that works everywhere. True would
        send ``tools`` to a model that ignores it, then wait for a tool call that
        never arrives.
        """
        assert supports_native_tool_calling("brand-new-model-v1") is False


# --- count_tokens ------------------------------------------------------------

# Recorded from real ``gemma-4-31B-it`` responses: ``prompt_tokens`` as the
# provider reported it, ``estimated`` as this module computes it.
RECORDED: list[dict[str, Any]] = [
    {
        "label": "short-vi",
        "messages": [{"role": "user", "content": "Xin chào"}],
        "prompt_tokens": 15,
        "estimated": 10,
    },
    {
        "label": "medium-vi",
        "messages": [
            {
                "role": "system",
                "content": "Bạn là trợ lý đặt lịch hẹn, trả lời ngắn gọn.",
            },
            {
                "role": "user",
                "content": "Tôi muốn đặt lịch hẹn khám vào thứ ba tuần sau.",
            },
        ],
        "prompt_tokens": 43,
        "estimated": 64,
    },
    {
        "label": "long-vi",
        "messages": [
            {
                "role": "system",
                "content": "Bạn là trợ lý chăm sóc khách hàng của một phòng khám.",
            },
            {
                "role": "user",
                "content": "Chào bạn, tôi cần hỏi về lịch làm việc của bác sĩ.",
            },
            {
                "role": "assistant",
                "content": "Dạ, phòng khám mở cửa từ 7 giờ 30 đến 17 giờ, từ thứ hai đến thứ sáu.",
            },
            {
                "role": "user",
                "content": (
                    "Vậy tôi muốn đặt lịch hẹn với bác sĩ chuyên khoa tim mạch vào sáng thứ tư tuần sau. "
                    "Tôi có thẻ bảo hiểm y tế, và tôi cũng muốn hỏi thêm là chi phí khám tổng quát "
                    "khoảng bao nhiêu, có cần đặt cọc trước không, và nếu tôi đến muộn mười lăm phút "
                    "thì có bị mất lượt không?"
                ),
            },
        ],
        "prompt_tokens": 153,
        "estimated": 251,
    },
    {
        "label": "medium-en",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful appointment booking assistant.",
            },
            {
                "role": "user",
                "content": "I would like to book an appointment next Tuesday morning.",
            },
        ],
        "prompt_tokens": 37,
        "estimated": 29,
    },
]


class TestCountTokens:
    def test_it_counts_a_vietnamese_message_list(self) -> None:
        """The plan's fourth criterion, first half."""
        messages: Sequence[Mapping[str, Any]] = RECORDED[1]["messages"]
        result = count_tokens(messages, MODEL)
        assert isinstance(result, int)
        assert result > 0

    def test_it_grows_with_the_conversation(self) -> None:
        sizes = [count_tokens(case["messages"], MODEL) for case in RECORDED[:3]]
        assert sizes == sorted(sizes)

    @pytest.mark.parametrize("case", RECORDED, ids=[c["label"] for c in RECORDED])
    def test_the_estimate_still_matches_what_was_recorded(
        self, case: dict[str, Any]
    ) -> None:
        """Pins the estimator. Change the heuristic and these move."""
        assert count_tokens(case["messages"], MODEL) == case["estimated"]

    def test_the_estimate_is_not_within_ten_percent_of_the_provider(self) -> None:
        """The plan's fourth criterion, second half, measured and not met.

        Asserted in the failing direction on purpose: the number this records is
        the reason ``count_tokens`` is documented as an estimate. If someone
        improves the estimator this test fails, which is the prompt to update the
        drift table in the module docstring.
        """
        drifts = {
            case["label"]: count_tokens(case["messages"], MODEL) / case["prompt_tokens"]
            - 1
            for case in RECORDED
        }
        worst = max(abs(drift) for drift in drifts.values())
        assert worst > 0.10, f"the estimate got good: {drifts}"
        assert min(drifts.values()) < 0 < max(drifts.values()), (
            f"drift is one-sided, so a correction factor would work: {drifts}"
        )

    def test_tiktoken_has_no_encoding_for_gemma(self) -> None:
        """Why the drift exists: the fallback tokenizer is a GPT one."""
        with pytest.raises(KeyError):
            tiktoken.encoding_for_model(MODEL)

    def test_no_model_name_falls_back_rather_than_raising(self) -> None:
        assert count_tokens([{"role": "user", "content": "Xin chào"}]) > 0

    def test_a_known_model_uses_its_own_encoding(self) -> None:
        """Not the fallback: gpt-4o's encoding differs from cl100k_base."""
        messages = [{"role": "user", "content": "Xin chào các bạn"}]
        assert count_tokens(messages, "gpt-4o") != count_tokens(messages, MODEL)

    def test_an_empty_value_costs_nothing_beyond_the_overhead(self) -> None:
        assert count_tokens([{"role": "", "content": ""}]) == 4


# --- criterion 5: one table to edit -----------------------------------------


class TestOneTableToEdit:
    """Adding a model line means editing one table and nothing else.

    Each test adds a fabricated row at runtime and checks the function follows
    it, which proves the table is read on every call rather than baked into a
    compiled copy somewhere.
    """

    def test_a_new_family_needs_only_family_markers(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            model_meta,
            "FAMILY_MARKERS",
            [("mistral", "mistral"), *model_meta.FAMILY_MARKERS],
        )
        assert model_family("mistral-large-2") == "mistral"

    def test_a_new_reasoning_line_needs_only_reasoning_patterns(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert supports_reasoning("gemma-5-40b-it") is False
        monkeypatch.setattr(
            model_meta,
            "REASONING_PATTERNS",
            [r"^gemma-5.*", *model_meta.REASONING_PATTERNS],
        )
        assert supports_reasoning("gemma-5-40b-it") is True

    def test_a_new_tool_calling_line_needs_only_that_table(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert supports_native_tool_calling("qwen4-70b") is False
        monkeypatch.setattr(
            model_meta,
            "NATIVE_TOOL_CALLING_PATTERNS",
            [r"^qwen4-.*", *model_meta.NATIVE_TOOL_CALLING_PATTERNS],
        )
        assert supports_native_tool_calling("qwen4-70b") is True

    def test_the_three_tables_are_the_only_module_state(self) -> None:
        """No fourth table, and no per-name cache to invalidate."""
        containers = {
            name
            for name, value in vars(model_meta).items()
            if isinstance(value, (list, dict, set)) and not name.startswith("_")
        }
        assert containers == {
            "FAMILY_MARKERS",
            "REASONING_PATTERNS",
            "NATIVE_TOOL_CALLING_PATTERNS",
        }
