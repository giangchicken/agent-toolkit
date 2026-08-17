"""Behavior tests for iter_json_array and its file wrapper.

The corpus test is skipped unless AGENT_TOOLKIT_CORPUS points at a top-level
JSON array; the file is 126 MB and lives outside this repository.
"""

import io
import json
import os
import pathlib
import tracemalloc
from typing import Any

import pytest

from agent_toolkit.errors import ToolkitError
from agent_toolkit.file_utils import iter_json_array_file
from agent_toolkit.json_utils import iter_json_array


def _elements(text: str, **kwargs: Any) -> list[Any]:
    return list(iter_json_array(io.StringIO(text), **kwargs))


# Shaped like the corpus: an array of objects, pretty-printed with one space of
# indent, each carrying a nested message list.
def _corpus_shaped(count: int) -> str:
    records = [
        {
            "idx": i,
            "messages": [
                {"role": "system", "content": f"prompt {i} with tiếng Việt"},
                {"role": "user", "content": f"question {i}"},
            ],
            "meta": {"label": ["tool_a", "tool_b"], "llm_model": "gemma-4-31B-it"},
        }
        for i in range(count)
    ]
    return json.dumps(records, indent=1, ensure_ascii=False)


class TestStructure:
    def test_yields_each_element_in_order(self) -> None:
        assert _elements("[1, 2, 3]") == [1, 2, 3]

    def test_empty_array_yields_nothing(self) -> None:
        assert _elements("[]") == []

    def test_empty_array_with_whitespace_yields_nothing(self) -> None:
        assert _elements("[   ]") == []
        assert _elements("[\n\n]") == []
        assert _elements("  \n [\n ] \n ") == []

    def test_single_element(self) -> None:
        assert _elements('[{"a": 1}]') == [{"a": 1}]

    def test_objects_arrays_and_scalars_mixed(self) -> None:
        text = '[{"a": 1}, [1, 2], "s", 3, true, null]'
        assert _elements(text) == [{"a": 1}, [1, 2], "s", 3, True, None]

    def test_nested_structures_are_returned_whole(self) -> None:
        text = '[{"a": {"b": [1, {"c": 2}]}}]'
        assert _elements(text) == [{"a": {"b": [1, {"c": 2}]}}]

    def test_brackets_inside_strings_do_not_confuse_the_scan(self) -> None:
        text = '["a]b", "c[d", "e},{f"]'
        assert _elements(text) == ["a]b", "c[d", "e},{f"]

    def test_escaped_quotes_inside_strings(self) -> None:
        text = '["she said \\"hi\\"", "next"]'
        assert _elements(text) == ['she said "hi"', "next"]

    @pytest.mark.parametrize(
        "text",
        [
            "[1,2,3]",
            "[ 1 , 2 , 3 ]",
            "[\n 1,\n 2,\n 3\n]",
            "[\r\n1,\r\n2,\r\n3\r\n]",
            "[\t1,\t2,\t3\t]",
            "   [1, 2, 3]   ",
        ],
    )
    def test_whitespace_variations_all_yield_the_same_elements(self, text: str) -> None:
        assert _elements(text) == [1, 2, 3]


class TestRejectsNonArrays:
    @pytest.mark.parametrize(
        "text", ['{"a": 1}', '"just a string"', "42", "true", "null"]
    )
    def test_non_array_top_level_raises(self, text: str) -> None:
        with pytest.raises(ToolkitError, match="not an array"):
            _elements(text)

    def test_empty_input_raises(self) -> None:
        with pytest.raises(ToolkitError, match="no content"):
            _elements("")

    def test_whitespace_only_input_raises(self) -> None:
        with pytest.raises(ToolkitError, match="no content"):
            _elements("   \n\t  ")

    def test_non_positive_buffer_size_raises(self) -> None:
        with pytest.raises(ToolkitError, match="buffer_size must be positive"):
            _elements("[1]", buffer_size=0)


class TestBufferBoundaries:
    """A one-byte buffer forces every boundary condition on every character."""

    @pytest.mark.parametrize("buffer_size", [1, 2, 3, 7, 64, 1 << 20])
    def test_result_is_independent_of_buffer_size(self, buffer_size: int) -> None:
        text = _corpus_shaped(20)
        expected = json.loads(text)
        assert _elements(text, buffer_size=buffer_size) == expected

    def test_element_much_larger_than_the_buffer(self) -> None:
        """The corpus's largest element is 18 KB; this proves the general case."""
        big = "x" * 100_000
        text = json.dumps([{"pad": big}, {"pad": big}])
        result = _elements(text, buffer_size=16)
        assert len(result) == 2
        assert result[0]["pad"] == big

    def test_element_straddling_the_buffer_boundary(self) -> None:
        text = json.dumps([{"a": "b" * 50}, {"c": "d" * 50}])
        for buffer_size in range(1, 40):
            assert _elements(text, buffer_size=buffer_size) == json.loads(text)


class TestFailsLoudly:
    """Criterion 3: a malformed array never yields a short iteration silently."""

    def test_truncation_at_any_point_raises_rather_than_yielding_fewer(self) -> None:
        text = _corpus_shaped(100)
        full = json.loads(text)
        assert len(full) == 100

        checked = 0
        for cut in range(1, len(text), max(1, len(text) // 200)):
            truncated = text[:cut]
            yielded: list[Any] = []
            with pytest.raises(ToolkitError):
                for element in iter_json_array(io.StringIO(truncated)):
                    yielded.append(element)
            # Whatever it managed to yield must be a correct prefix, and the
            # iteration must have ended in an error rather than in silence.
            assert yielded == full[: len(yielded)], f"bad prefix at cut={cut}"
            assert len(yielded) < 100, f"claimed completeness at cut={cut}"
            checked += 1
        assert checked >= 100, f"only {checked} truncation points exercised"

    def test_missing_closing_bracket_raises(self) -> None:
        with pytest.raises(ToolkitError, match="unterminated"):
            _elements("[1, 2, 3")

    def test_missing_separator_raises(self) -> None:
        with pytest.raises(ToolkitError, match="expected ',' or ']'"):
            _elements("[1 2]")

    def test_malformed_element_mid_array_raises_after_yielding_the_prefix(self) -> None:
        yielded: list[Any] = []
        with pytest.raises(ToolkitError, match="malformed or truncated"):
            for element in iter_json_array(io.StringIO('[{"a": 1}, {bad}, 3]')):
                yielded.append(element)
        assert yielded == [{"a": 1}]

    def test_garbage_after_a_valid_array_is_not_silently_ignored(self) -> None:
        """The closing bracket ends iteration; trailing junk is out of contract."""
        assert _elements("[1, 2] trailing junk") == [1, 2]

    def test_trailing_comma_is_tolerated_as_documented(self) -> None:
        """Invalid JSON, but it cannot cause a short iteration."""
        assert _elements("[1, 2,]") == [1, 2]

    def test_nothing_is_read_until_iteration_starts(self) -> None:
        """Generator semantics: constructing it on a non-array does not raise."""
        generator = iter_json_array(io.StringIO('{"not": "an array"}'))
        with pytest.raises(ToolkitError):
            next(generator)


class TestMatchesJsonLoad:
    def test_every_element_equals_json_load_on_the_same_text(self) -> None:
        text = _corpus_shaped(500)
        assert _elements(text) == json.loads(text)

    def test_compact_and_pretty_printed_forms_agree(self) -> None:
        records = json.loads(_corpus_shaped(50))
        compact = json.dumps(records, separators=(",", ":"), ensure_ascii=False)
        pretty = json.dumps(records, indent=4, ensure_ascii=False)
        assert _elements(compact) == _elements(pretty) == records

    def test_non_ascii_survives_the_round_trip(self) -> None:
        records = [{"vi": "khách hàng đã đồng ý thanh toán"}]
        assert _elements(json.dumps(records, ensure_ascii=False)) == records
        assert _elements(json.dumps(records, ensure_ascii=True)) == records


class TestMemoryIsBounded:
    def test_peak_allocation_does_not_scale_with_file_size(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Invariant 4 in the form that always runs: memory is O(1) in length.

        The fixtures go to disk rather than through io.StringIO. A StringIO holds
        the whole document in the process, so tracemalloc would measure the test's
        own source data -- which grows with the fixture -- and report it as the
        reader's footprint. Measured that way a correct implementation still looks
        like it scales.
        """
        buffer_size = 1 << 16
        small, large = 2_000, 50_000
        peaks = {}
        sizes = {}
        for count in (small, large):
            path = tmp_path / f"{count}.json"
            path.write_text(_corpus_shaped(count), encoding="utf-8")
            sizes[count] = path.stat().st_size

            tracemalloc.start()
            total = sum(1 for _ in iter_json_array_file(path, buffer_size=buffer_size))
            peaks[count] = tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()
            assert total == count

        # Both fixtures must exceed the buffer. A file that fits in one read never
        # refills, so it never holds two buffers at once and its peak sits at
        # roughly a third of the steady state; comparing against it reports growth
        # that is really the difference between one read and many.
        for count in (small, large):
            assert sizes[count] > 4 * buffer_size, (
                f"the {count}-element fixture is only {sizes[count]:,} bytes, "
                f"not enough to exceed the {buffer_size:,} byte buffer"
            )
        assert sizes[large] > 20 * sizes[small], f"fixtures too close: {sizes}"

        growth = peaks[large] / peaks[small]
        ratio = sizes[large] / sizes[small]
        assert growth < 1.5, (
            f"peak grew {growth:.2f}x for a {ratio:.0f}x larger file: {peaks}"
        )
        assert peaks[large] < 16 * buffer_size, (
            f"peak {peaks[large]:,} exceeds 16x the {buffer_size:,} byte buffer"
        )


class TestFileWrapper:
    def test_reads_a_file(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "a.json"
        records = json.loads(_corpus_shaped(10))
        path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
        assert list(iter_json_array_file(path)) == records

    def test_accepts_a_string_path(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "a.json"
        path.write_text("[1, 2]", encoding="utf-8")
        assert list(iter_json_array_file(str(path))) == [1, 2]

    def test_strips_a_byte_order_mark(self, tmp_path: pathlib.Path) -> None:
        """A BOM is not whitespace; under plain utf-8 it reaches the parser."""
        path = tmp_path / "bom.json"
        path.write_text("[1, 2]", encoding="utf-8-sig")
        assert path.read_bytes().startswith(b"\xef\xbb\xbf")
        assert list(iter_json_array_file(path)) == [1, 2]

    def test_a_bom_would_break_plain_utf8(self, tmp_path: pathlib.Path) -> None:
        """Proves the default is doing work, not decoration."""
        path = tmp_path / "bom.json"
        path.write_text("[1, 2]", encoding="utf-8-sig")
        with pytest.raises(ToolkitError, match="not an array"):
            list(iter_json_array_file(path, encoding="utf-8"))

    def test_reads_vietnamese_content_from_disk(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "vi.json"
        records = [{"content": "khách hàng ưu tiên"}]
        path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
        assert list(iter_json_array_file(path)) == records

    def test_closes_the_file_when_iteration_completes(
        self, tmp_path: pathlib.Path
    ) -> None:
        path = tmp_path / "a.json"
        path.write_text("[1, 2]", encoding="utf-8")
        generator = iter_json_array_file(path)
        assert list(generator) == [1, 2]
        # Exhausting the generator runs the `with` block's exit.
        assert generator.gi_frame is None


_CORPUS = os.environ.get("AGENT_TOOLKIT_CORPUS")


@pytest.mark.skipif(not _CORPUS, reason="AGENT_TOOLKIT_CORPUS is not set")
class TestRealCorpus:
    def test_yields_the_expected_element_count(self) -> None:
        assert _CORPUS is not None
        expected = int(os.environ.get("AGENT_TOOLKIT_CORPUS_COUNT", "21172"))
        assert sum(1 for _ in iter_json_array_file(_CORPUS)) == expected

    def test_peak_resident_memory_stays_under_100mb(self) -> None:
        """Invariant 4, measured as RSS rather than as traced allocations."""
        assert _CORPUS is not None
        status = pathlib.Path("/proc/self/status")
        if not status.exists():
            pytest.skip("no /proc/self/status on this platform")

        def peak_rss_kb() -> int:
            for line in status.read_text().splitlines():
                if line.startswith("VmHWM:"):
                    return int(line.split()[1])
            raise AssertionError("VmHWM not reported")

        before = peak_rss_kb()
        count = sum(1 for _ in iter_json_array_file(_CORPUS))
        grew_mb = (peak_rss_kb() - before) / 1024
        assert count > 0
        assert grew_mb < 100, f"peak RSS grew {grew_mb:.0f} MB while streaming"
