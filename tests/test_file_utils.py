"""Readers, writers, and the atomicity guarantee.

The four acceptance criteria of T5 are covered by ``TestAtomicWriteJson``,
``TestJsonRoundTrip``, ``TestJsonlinesRoundTrip``, and
``test_adds_no_dependency_beyond_pyyaml``. Several tests below exist only to
prove another one is not vacuous -- an atomicity test passes trivially against a
writer that never writes anything at all.
"""

import ast
import json
import pathlib
import sys
from typing import Any

import pytest
import yaml

import agent_toolkit.file_utils as fu

# A payload json.dump writes part of before it raises: the first key serializes,
# the second has no encoder. Dict order is insertion order, so this is stable.
_UNSERIALIZABLE: dict[str, Any] = {"kept": "x" * 500, "bad": object()}

# Nested, non-ASCII, and no personal data: the file this ends up in is public.
_VIETNAMESE: dict[str, Any] = {
    "tiêu_đề": "Hướng dẫn sử dụng",
    "mục": [
        {"tên": "Đặt lịch hẹn", "số_bước": 3},
        {"tên": "Huỷ lịch hẹn", "số_bước": 2},
    ],
}


class TestAtomicWriteJson:
    def test_serialization_failure_leaves_the_original_intact(
        self, tmp_path: pathlib.Path
    ) -> None:
        """T5's first criterion, and the reason write_json was rewritten."""
        target = tmp_path / "artifact.json"
        target.write_text('{"keep": 1}', encoding="utf-8")

        with pytest.raises(TypeError):
            fu.write_json(target, _UNSERIALIZABLE)

        assert target.read_text(encoding="utf-8") == '{"keep": 1}'

    def test_the_failing_payload_really_writes_before_it_raises(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Non-vacuity: a naive writer truncates on this payload, so the test above bites."""
        naive = tmp_path / "naive.json"
        naive.write_text('{"keep": 1}', encoding="utf-8")

        with pytest.raises(TypeError):
            with open(naive, "w", encoding="utf-8") as fp:
                json.dump(_UNSERIALIZABLE, fp, indent=2)

        surviving = naive.read_text(encoding="utf-8")
        assert surviving != '{"keep": 1}'
        assert "kept" in surviving

    def test_no_temp_file_is_left_behind_on_failure(
        self, tmp_path: pathlib.Path
    ) -> None:
        target = tmp_path / "artifact.json"
        target.write_text("{}", encoding="utf-8")

        with pytest.raises(TypeError):
            fu.write_json(target, _UNSERIALIZABLE)

        assert [p.name for p in tmp_path.iterdir()] == ["artifact.json"]

    def test_no_temp_file_is_left_behind_on_success(
        self, tmp_path: pathlib.Path
    ) -> None:
        fu.write_json(tmp_path / "artifact.json", {"a": 1})
        assert [p.name for p in tmp_path.iterdir()] == ["artifact.json"]

    def test_creates_missing_parent_directories(self, tmp_path: pathlib.Path) -> None:
        target = tmp_path / "05_jury" / "nested" / "votes.json"
        fu.write_json(target, {"a": 1})
        assert fu.read_json(target) == {"a": 1}

    def test_replaces_an_existing_file_completely(self, tmp_path: pathlib.Path) -> None:
        """os.replace, not a partial overwrite: no tail of the old file survives."""
        target = tmp_path / "artifact.json"
        fu.write_json(target, {"long": "y" * 5_000})
        fu.write_json(target, {"short": 1})
        assert fu.read_json(target) == {"short": 1}
        assert "y" not in target.read_text(encoding="utf-8")

    def test_a_relative_path_with_no_directory_component_works(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The parent of "artifact.json" is "", which makedirs would reject."""
        monkeypatch.chdir(tmp_path)
        fu.write_json("artifact.json", {"a": 1})
        assert fu.read_json("artifact.json") == {"a": 1}


class TestJsonRoundTrip:
    def test_round_trips_nested_vietnamese_unchanged(
        self, tmp_path: pathlib.Path
    ) -> None:
        """T5's second criterion."""
        target = tmp_path / "artifact.json"
        fu.write_json(target, _VIETNAMESE)
        assert fu.read_json(target) == _VIETNAMESE

    def test_vietnamese_is_written_as_text_not_escapes(
        self, tmp_path: pathlib.Path
    ) -> None:
        """ensure_ascii=False: an artifact a reviewer opens should be legible."""
        target = tmp_path / "artifact.json"
        fu.write_json(target, _VIETNAMESE)
        raw = target.read_text(encoding="utf-8")
        assert "Hướng dẫn sử dụng" in raw
        assert "\\u" not in raw

    def test_a_top_level_array_round_trips_too(self, tmp_path: pathlib.Path) -> None:
        """read_json returns what the document holds, which is why it is typed Any."""
        target = tmp_path / "artifact.json"
        fu.write_json(target, [1, {"a": 2}])
        assert fu.read_json(target) == [1, {"a": 2}]


class TestJsonlinesRoundTrip:
    rows: list[dict[str, Any]] = [
        {"id": 1, "label": "Đặt lịch hẹn"},
        {"id": 2, "label": "check_availability"},
    ]

    def test_round_trips_a_list_of_dicts(self, tmp_path: pathlib.Path) -> None:
        """T5's third criterion."""
        target = tmp_path / "votes.jsonl"
        fu.write_jsonlines(target, self.rows)
        assert fu.read_jsonlines(target) == self.rows

    def test_trailing_newline_and_no_trailing_newline_read_identically(
        self, tmp_path: pathlib.Path
    ) -> None:
        body = '{"id": 1}\n{"id": 2}'
        with_newline = tmp_path / "with.jsonl"
        without_newline = tmp_path / "without.jsonl"
        with_newline.write_text(body + "\n", encoding="utf-8")
        without_newline.write_text(body, encoding="utf-8")

        assert fu.read_jsonlines(with_newline) == fu.read_jsonlines(without_newline)
        assert fu.read_jsonlines(with_newline) == [{"id": 1}, {"id": 2}]

    def test_write_jsonlines_ends_with_a_newline(self, tmp_path: pathlib.Path) -> None:
        """So appending with `cat` or `>>` cannot join two records."""
        target = tmp_path / "votes.jsonl"
        fu.write_jsonlines(target, self.rows)
        assert target.read_text(encoding="utf-8").endswith("}\n")

    def test_blank_lines_are_skipped(self, tmp_path: pathlib.Path) -> None:
        target = tmp_path / "votes.jsonl"
        target.write_text('{"id": 1}\n\n   \n{"id": 2}\n', encoding="utf-8")
        assert fu.read_jsonlines(target) == [{"id": 1}, {"id": 2}]

    def test_a_generator_is_accepted(self, tmp_path: pathlib.Path) -> None:
        target = tmp_path / "votes.jsonl"
        fu.write_jsonlines(target, ({"id": n} for n in range(3)))
        assert fu.read_jsonlines(target) == [{"id": 0}, {"id": 1}, {"id": 2}]

    def test_a_failing_generator_leaves_the_original_intact(
        self, tmp_path: pathlib.Path
    ) -> None:
        """The same atomicity write_json has: the rows arrive all or not at all."""
        target = tmp_path / "votes.jsonl"
        fu.write_jsonlines(target, self.rows)

        def rows_then_boom() -> Any:
            yield {"id": 99}
            raise RuntimeError("upstream stage failed")

        with pytest.raises(RuntimeError):
            fu.write_jsonlines(target, rows_then_boom())

        assert fu.read_jsonlines(target) == self.rows
        assert [p.name for p in tmp_path.iterdir()] == ["votes.jsonl"]

    def test_one_malformed_line_discards_the_whole_read(
        self, tmp_path: pathlib.Path
    ) -> None:
        target = tmp_path / "votes.jsonl"
        target.write_text('{"id": 1}\nnot json\n{"id": 2}\n', encoding="utf-8")
        assert fu.read_jsonlines(target) == []


class TestReadersReturnDefaults:
    """The harvested contract: a default and a debug log, never an exception.

    `agent-evaluation`'s `get_llm_config` branches on `if not raw:` after
    `read_json`, and its `load_prompt` documents "or empty string on failure".
    """

    def test_missing_file(self, tmp_path: pathlib.Path) -> None:
        missing = tmp_path / "nope"
        assert fu.read_txt(missing) == ""
        assert fu.read_json(missing) == {}
        assert fu.read_yaml(missing) == {}
        assert fu.read_jsonlines(missing) == []

    def test_a_directory_where_a_file_was_expected(
        self, tmp_path: pathlib.Path
    ) -> None:
        assert fu.read_txt(tmp_path) == ""
        assert fu.read_json(tmp_path) == {}

    def test_malformed_json(self, tmp_path: pathlib.Path) -> None:
        target = tmp_path / "artifact.json"
        target.write_text('{"a": ', encoding="utf-8")
        assert fu.read_json(target) == {}

    def test_malformed_yaml(self, tmp_path: pathlib.Path) -> None:
        target = tmp_path / "params.yaml"
        target.write_text("a: [1, 2\nb: {", encoding="utf-8")
        assert fu.read_yaml(target) == {}

    def test_an_empty_yaml_file_reads_as_an_empty_mapping_not_none(
        self, tmp_path: pathlib.Path
    ) -> None:
        target = tmp_path / "params.yaml"
        target.write_text("", encoding="utf-8")
        assert fu.read_yaml(target) == {}

    def test_the_defaults_are_distinguishable_from_success(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Non-vacuity: the readers do return content when there is content."""
        (tmp_path / "a.txt").write_text("hello", encoding="utf-8")
        (tmp_path / "a.json").write_text('{"a": 1}', encoding="utf-8")
        (tmp_path / "a.yaml").write_text("stage: jury\nfamilies: 3\n", encoding="utf-8")
        (tmp_path / "a.jsonl").write_text('{"a": 1}\n', encoding="utf-8")

        assert fu.read_txt(tmp_path / "a.txt") == "hello"
        assert fu.read_json(tmp_path / "a.json") == {"a": 1}
        assert fu.read_yaml(tmp_path / "a.yaml") == {"stage": "jury", "families": 3}
        assert fu.read_jsonlines(tmp_path / "a.jsonl") == [{"a": 1}]

    def test_read_yaml_does_not_construct_python_objects(
        self, tmp_path: pathlib.Path
    ) -> None:
        """safe_load: a parameter file must not be able to instantiate anything."""
        target = tmp_path / "params.yaml"
        target.write_text(
            "!!python/object/apply:os.system ['true']\n", encoding="utf-8"
        )
        assert fu.read_yaml(target) == {}
        with pytest.raises(yaml.YAMLError):
            yaml.safe_load(target.read_text(encoding="utf-8"))


class TestByteOrderMark:
    """A BOM'd file must not read as an empty one -- that is a silent config loss."""

    def test_read_json_strips_a_bom(self, tmp_path: pathlib.Path) -> None:
        target = tmp_path / "artifact.json"
        target.write_text('{"a": 1}', encoding="utf-8-sig")
        assert fu.read_json(target) == {"a": 1}

    def test_the_bom_would_otherwise_break_the_parse(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Non-vacuity: under plain utf-8 the same file is unparseable."""
        target = tmp_path / "artifact.json"
        target.write_text('{"a": 1}', encoding="utf-8-sig")
        with pytest.raises(json.JSONDecodeError):
            json.loads(target.read_text(encoding="utf-8"))

    def test_read_txt_strips_a_bom(self, tmp_path: pathlib.Path) -> None:
        target = tmp_path / "prompt.txt"
        target.write_text("Bạn là trợ lý", encoding="utf-8-sig")
        assert fu.read_txt(target) == "Bạn là trợ lý"

    def test_writers_never_emit_a_bom(self, tmp_path: pathlib.Path) -> None:
        target = tmp_path / "artifact.json"
        fu.write_json(target, {"a": 1})
        assert not target.read_bytes().startswith(b"\xef\xbb\xbf")


def test_adds_no_dependency_beyond_pyyaml() -> None:
    """T5's fourth criterion: `jsonlines` was dropped, nothing took its place."""
    source = pathlib.Path(fu.__file__).read_text(encoding="utf-8")
    roots: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])

    third_party = roots - sys.stdlib_module_names - {"agent_toolkit"}
    assert third_party == {"yaml"}, f"unexpected dependency: {third_party}"
