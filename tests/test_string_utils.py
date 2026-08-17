"""Behavior tests for string_utils.

The ``TestExtractJsonFromText`` cases are ported from the harvested toolkit's
own suite, which is the regression gate T3 exists to inherit. That suite tested
exactly one of this module's five functions -- there were no cases for
``slot_filling``, ``normalize_text``, or ``compute_hash`` -- so the rest are new.

The 17 ``extract_xml_from_text`` cases in the source suite are dropped with the
function itself; it is out of scope for v0.1.
"""

import logging
import subprocess
import sys
import textwrap

import pytest

from agent_toolkit.string_utils import (
    MAX_SLOT_FILLING_PASSES,
    clean_thinking_tags,
    compute_hash,
    extract_json_from_text,
    normalize_text,
    slot_filling,
)

# Defined once and referenced by both the input and the expected value. The
# source suite inlined this Vietnamese paragraph twice; transcribing a 400-
# character string twice invites a typo that would look like an extraction bug.
# The personal name in the original fixture is replaced by a role -- what this
# case tests is long non-ASCII prose inside a fenced block, not the name.
_VI_ASSESSMENT = (
    "Trong cuộc gọi, khách hàng đã đồng ý thanh toán khoản nợ sau khi được xác "
    "nhận lại. Nhân viên tư vấn đã thực hiện đúng các bước trong kịch bản, từ "
    "xác minh danh tính, thông báo khoản nợ, đến việc thuyết phục và xác nhận "
    "cam kết thanh toán."
)


class TestExtractJsonFromText:
    """Ported from the harvested toolkit's suite."""

    def test_single_json_object(self) -> None:
        text = 'Here is some data: {"name": "John", "age": 30}'
        result = extract_json_from_text(text, extract_all=True)
        assert len(result) == 1
        assert result[0] == {"name": "John", "age": 30}

    def test_single_json_array(self) -> None:
        text = "Here is a list: [1, 2, 3, 4, 5]"
        result = extract_json_from_text(text, extract_all=True)
        assert len(result) == 1
        assert result[0] == [1, 2, 3, 4, 5]

    def test_json_in_code_block(self) -> None:
        text = (
            "Here is the response:\n```json\n{\n"
            f'    "assessment": "{_VI_ASSESSMENT}",\n'
            '    "correctness": true,\n'
            '    "need_to_use_tool": true,\n'
            '    "violations": [1, 2, 3]\n'
            "}\n```\n"
        )
        result = extract_json_from_text(text, extract_all=True)
        assert len(result) == 1
        assert result[0] == {
            "assessment": _VI_ASSESSMENT,
            "correctness": True,
            "need_to_use_tool": True,
            "violations": [1, 2, 3],
        }

    def test_multiple_json_objects(self) -> None:
        text = textwrap.dedent("""
            First object: {"id": 1, "name": "Alice"}
            Second object: {"id": 2, "name": "Bob"}
        """)
        result = extract_json_from_text(text, extract_all=True)
        assert len(result) == 2
        assert {"id": 1, "name": "Alice"} in result
        assert {"id": 2, "name": "Bob"} in result

    def test_mixed_json_types(self) -> None:
        text = textwrap.dedent("""
            Object: {"key": "value"}
            Array: ["item1", "item2"]
        """)
        result = extract_json_from_text(text, extract_all=True)
        assert len(result) == 2
        assert {"key": "value"} in result
        assert ["item1", "item2"] in result

    def test_nested_json_only_top_level(self) -> None:
        text = textwrap.dedent("""
            {
                "outer": "data",
                "nested": {"inner": "value"}
            }
        """)
        result = extract_json_from_text(text, extract_all=True)
        assert len(result) == 1
        assert result[0] == {"outer": "data", "nested": {"inner": "value"}}

    def test_multiple_code_blocks(self) -> None:
        text = textwrap.dedent("""
            First block:
            ```json
            {"first": true}
            ```

            Second block:
            ```json
            {"second": true}
            ```
        """)
        result = extract_json_from_text(text, extract_all=True)
        assert len(result) == 2
        assert {"first": True} in result
        assert {"second": True} in result

    def test_invalid_json_ignored(self) -> None:
        text = textwrap.dedent("""
            Valid: {"valid": true}
            Invalid: {invalid json here}
        """)
        result = extract_json_from_text(text, extract_all=True)
        assert len(result) == 1
        assert result[0] == {"valid": True}

    def test_empty_text(self) -> None:
        assert extract_json_from_text("") is None

    def test_nested_json_structures(self) -> None:
        text = textwrap.dedent("""
            [
                {
                    "id": "123",
                    "result": [
                        {"data": "1"},
                        {"data": "2"}
                    ]
                }
            ]

            Here is a python dictionary:
            {
                "key": {
                    "list": [1, 2, 3]
                }
            }
        """)
        result = extract_json_from_text(text, extract_all=True)
        assert len(result) == 2
        assert any(
            isinstance(r, list)
            and r[0].get("id") == "123"
            and r[0].get("result") == [{"data": "1"}, {"data": "2"}]
            for r in result
        )
        assert any(
            isinstance(r, dict) and "key" in r and r["key"].get("list") == [1, 2, 3]
            for r in result
        )

    def test_json_with_special_characters(self) -> None:
        text = '{"message": "Hello\\nWorld", "path": "C:\\Users\\test"}'
        result = extract_json_from_text(text, extract_all=True)
        assert len(result) == 1
        assert "message" in result[0]

    def test_json_array_of_objects(self) -> None:
        text = textwrap.dedent("""
            [
                {"id": 1, "name": "Item 1"},
                {"id": 2, "name": "Item 2"}
            ]
        """)
        result = extract_json_from_text(text, extract_all=True)
        assert len(result) == 1
        assert len(result[0]) == 2
        assert result[0][0]["id"] == 1


class TestExtractJsonRecovery:
    """The five recovery paths and the one refusal named in the T3 criteria."""

    def test_recovers_from_a_fenced_block(self) -> None:
        assert extract_json_from_text('noise ```json\n{"k": 1}\n``` more') == {"k": 1}

    def test_recovers_a_bare_object(self) -> None:
        assert extract_json_from_text('{"k": 1}') == {"k": 1}

    def test_recovers_a_bare_array(self) -> None:
        assert extract_json_from_text('["a", "b"]') == ["a", "b"]

    def test_recovers_a_prose_wrapped_object(self) -> None:
        text = 'Sure! Here is what I found: {"tools": ["a"]} Let me know if that helps.'
        assert extract_json_from_text(text) == {"tools": ["a"]}

    def test_recovers_a_repairable_malformed_object(self) -> None:
        """A trailing comma is invalid JSON; json_repair recovers it."""
        assert extract_json_from_text('{"a": 1, "b": 2,}') == {"a": 1, "b": 2}

    def test_returns_none_on_garbage(self) -> None:
        assert extract_json_from_text("no json here") is None

    def test_never_returns_a_string(self) -> None:
        """Requirement: a parsed dict/list/None, never the matched text."""
        for text in ('{"k": 1}', "[1]", "garbage", "", '```json\n{"k":1}\n```'):
            result = extract_json_from_text(text)
            assert result is None or isinstance(result, (dict, list)), text

    def test_extract_all_returns_a_list_never_none(self) -> None:
        assert extract_json_from_text("no json here", extract_all=True) == []


class TestExtractJsonGuards:
    """The two guards added because json-repair got more aggressive.

    Both exist for the consumer named in the pipeline spec: a juror's invalid
    response must become a clean abstention, never a plausible-looking vote.
    """

    def test_garbage_opening_as_an_object_is_not_repaired_into_a_list(self) -> None:
        """json-repair alone yields ['invalid json here}'] -- a fabricated vote."""
        assert extract_json_from_text("{invalid json here}") is None

    def test_two_disjoint_objects_are_both_found(self) -> None:
        text = '{"a": 1} and then {"b": 2}'
        result = extract_json_from_text(text, extract_all=True)
        assert result == [{"a": 1}, {"b": 2}]

    def test_two_disjoint_arrays_are_both_found(self) -> None:
        text = 'first ["a"] second ["b"]'
        result = extract_json_from_text(text, extract_all=True)
        assert result == [["a"], ["b"]]

    def test_a_brace_inside_a_string_does_not_split_the_span(self) -> None:
        """The depth guard skips string contents, so this stays one object."""
        text = '{"template": "use {{slot}} here", "n": 1}'
        assert extract_json_from_text(text) == {
            "template": "use {{slot}} here",
            "n": 1,
        }

    def test_a_bracket_inside_a_string_does_not_split_the_span(self) -> None:
        text = '{"note": "an array looks like [1, 2]", "n": 1}'
        assert extract_json_from_text(text) == {
            "note": "an array looks like [1, 2]",
            "n": 1,
        }

    def test_an_escaped_quote_does_not_confuse_the_string_tracker(self) -> None:
        text = '{"quote": "she said \\"hi\\" loudly", "n": 1}'
        result = extract_json_from_text(text)
        assert isinstance(result, dict) and result["n"] == 1

    def test_a_deeply_nested_single_object_still_parses(self) -> None:
        """Three levels deep: past what the sweep regex can match on its own."""
        text = '{"a": {"b": {"c": [1, 2]}}}'
        assert extract_json_from_text(text) == {"a": {"b": {"c": [1, 2]}}}


class TestSlotFilling:
    def test_fills_a_flat_placeholder(self) -> None:
        assert slot_filling("hello {{name}}", {"name": "world"}) == "hello world"

    def test_resolves_a_nested_chain(self) -> None:
        """A value containing a placeholder resolves on a later pass."""
        assert slot_filling("{{a}}", {"a": "{{b}}", "b": "x"}) == "x"

    def test_resolves_a_three_deep_chain(self) -> None:
        assert (
            slot_filling("{{a}}", {"a": "{{b}}", "b": "{{c}}", "c": "done"}) == "done"
        )

    def test_leaves_unknown_placeholders_untouched(self) -> None:
        assert slot_filling("{{known}} {{unknown}}", {"known": "v"}) == "v {{unknown}}"

    def test_terminates_on_direct_self_reference(self) -> None:
        """The replacement is a no-op, so the fixpoint check fires immediately."""
        assert slot_filling("{{a}}", {"a": "{{a}}"}) == "{{a}}"

    def test_reads_object_dict_through_the_value_wrapper(self) -> None:
        assert slot_filling("{{a}}", None, {"a": {"value": "wrapped"}}) == "wrapped"

    def test_object_dict_takes_precedence_over_key_value_mapping(self) -> None:
        result = slot_filling("{{a}}", {"a": "from_kv"}, {"a": {"value": "from_obj"}})
        assert result == "from_obj"

    def test_stringifies_non_string_values(self) -> None:
        assert slot_filling("{{n}}", {"n": 42}) == "42"

    def test_returns_input_unchanged_when_no_mapping_is_given(self) -> None:
        assert slot_filling("{{a}}") == "{{a}}"

    def test_does_not_raise_on_a_malformed_mapping(self) -> None:
        """The never-raise contract: a bad value logs and falls back."""
        assert slot_filling("{{a}}", None, {"a": "not_a_value_wrapper"}) == "{{a}}"

    def test_logs_at_debug_rather_than_raising(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.DEBUG, logger="agent_toolkit.string_utils"):
            slot_filling("{{a}}", None, {"a": "not_a_value_wrapper"})
        assert any("slot_filling failed" in r.message for r in caplog.records)


# Run in a subprocess so that a regression fails the test instead of hanging the
# suite forever. This is the case the upstream fixpoint loop does not terminate
# on, and the reason MAX_SLOT_FILLING_PASSES exists.
_MUTUAL_REFERENCE_PROBE = """
from agent_toolkit.string_utils import slot_filling
print(repr(slot_filling("{{a}}", {"a": "{{b}}", "b": "{{a}}"})))
"""


class TestSlotFillingTermination:
    def test_mutual_reference_is_bounded(self) -> None:
        proc = subprocess.run(
            [sys.executable, "-c", _MUTUAL_REFERENCE_PROBE],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.strip() in ("'{{a}}'", "'{{b}}'")

    def test_the_pass_cap_is_generous_relative_to_real_nesting_depth(self) -> None:
        """A guard on the constant: the design it serves is two-level."""
        assert MAX_SLOT_FILLING_PASSES >= 50

    def test_a_deep_but_legitimate_chain_still_resolves(self) -> None:
        depth = 40
        mapping = {f"k{i}": f"{{{{k{i + 1}}}}}" for i in range(depth)}
        mapping[f"k{depth}"] = "end"
        assert slot_filling("{{k0}}", mapping) == "end"


class TestNormalizeText:
    def test_collapses_whitespace_and_strips(self) -> None:
        assert normalize_text("  a   b  ") == "a b"

    def test_joins_digits_split_by_a_narrow_no_break_space(self) -> None:
        assert normalize_text("1\u202f234") == "1234"

    def test_converts_a_no_break_space_between_words_to_a_plain_space(self) -> None:
        assert normalize_text("a\u00a0b") == "a b"

    def test_applies_nfkc_normalization(self) -> None:
        assert normalize_text("\ufb01") == "fi"

    def test_keeps_vietnamese_diacritics_by_default(self) -> None:
        assert normalize_text("Tiếng Việt") == "Tiếng Việt"

    def test_removes_tone_marks_when_asked(self) -> None:
        """The regression test for the flag that used to do nothing."""
        assert normalize_text("Tiếng Việt", remove_tone_marks=True) == "Tieng Viet"

    def test_removes_tone_marks_from_either_normalization_form(self) -> None:
        """Precomposed and decomposed input must fold identically."""
        import unicodedata

        for form in ("NFC", "NFD"):
            text = unicodedata.normalize(form, "khách hàng ưu tiên")
            assert normalize_text(text, remove_tone_marks=True) == "khach hang uu tien"

    def test_tone_mark_folding_is_a_stable_dedup_key(self) -> None:
        """Why the pipeline wants this: two spellings collapse to one key."""
        a = normalize_text("thanh toán", remove_tone_marks=True)
        b = normalize_text("thanh toan", remove_tone_marks=True)
        assert a == b == "thanh toan"

    def test_does_not_raise_on_a_non_string(self) -> None:
        assert normalize_text(None) is None  # type: ignore[arg-type]


class TestComputeHash:
    def test_defaults_to_sha256(self) -> None:
        assert compute_hash("abc") == compute_hash("abc", "sha256")
        assert len(compute_hash("abc")) == 64

    @pytest.mark.parametrize(
        ("hash_type", "length"),
        [("md5", 32), ("sha1", 40), ("sha256", 64), ("sha512", 128)],
    )
    def test_digest_lengths(self, hash_type: str, length: int) -> None:
        assert len(compute_hash("abc", hash_type)) == length

    def test_unknown_hash_type_falls_back_to_sha256(self) -> None:
        assert compute_hash("abc", "not-a-hash") == compute_hash("abc", "sha256")

    def test_is_stable_across_calls(self) -> None:
        assert compute_hash("Tiếng Việt") == compute_hash("Tiếng Việt")

    def test_distinguishes_different_content(self) -> None:
        assert compute_hash("a") != compute_hash("b")

    def test_handles_non_ascii(self) -> None:
        """The pipeline hashes Vietnamese conversation text to build record ids."""
        assert len(compute_hash("khách hàng đã đồng ý")) == 64


class TestCleanThinkingTags:
    def test_removes_a_think_block(self) -> None:
        assert clean_thinking_tags("<think>reasoning</think>answer") == "answer"

    def test_removes_a_multiline_think_block(self) -> None:
        assert clean_thinking_tags("<think>\na\nb\n</think>\nanswer") == "answer"

    def test_is_case_insensitive(self) -> None:
        assert clean_thinking_tags("<THINK>x</THINK>answer") == "answer"

    def test_returns_empty_string_for_empty_input(self) -> None:
        assert clean_thinking_tags("") == ""

    def test_leaves_untagged_content_alone(self) -> None:
        assert clean_thinking_tags("just an answer") == "just an answer"

    def test_accepts_and_ignores_binding_and_model(self) -> None:
        """Signature compatibility with the harvested call sites."""
        result = clean_thinking_tags("<think>x</think>a", binding="openai", model="glm")
        assert result == "a"
