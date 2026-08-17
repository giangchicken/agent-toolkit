"""Config resolution: three resolvers, one precedence rule, no host directory.

Every test runs with the LLM_* environment cleared and the process-wide resolver
reset, so nothing here depends on the shell it was launched from and no test can
leak an installed resolver into the next one.
"""

import json
import pathlib
from collections.abc import Iterator

import pytest

from agent_toolkit.llm.config import (
    DictConfigResolver,
    EnvConfigResolver,
    JsonDirConfigResolver,
    LLMConfig,
    resolve_config,
    set_config_resolver,
)
from agent_toolkit.llm.exceptions import LLMConfigError

ENV_VARS = ["LLM_MODEL", "LLM_API_KEY", "LLM_BASE_URL"]


@pytest.fixture(autouse=True)
def isolated(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    set_config_resolver(None)
    yield
    set_config_resolver(None)


class TestPrecedence:
    def test_an_explicit_argument_beats_the_resolver_and_the_environment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """T6's fifth criterion: all three set, the argument wins."""
        monkeypatch.setenv("LLM_API_KEY", "from-env")
        monkeypatch.setenv("LLM_BASE_URL", "https://env.invalid")
        set_config_resolver(
            DictConfigResolver(
                {
                    "m": LLMConfig(
                        model="m",
                        api_key="from-resolver",
                        base_url="https://resolver.invalid",
                    )
                }
            )
        )

        config = resolve_config(
            model="m", api_key="from-argument", base_url="https://argument.invalid"
        )

        assert config.api_key == "from-argument"
        assert config.base_url == "https://argument.invalid"

    def test_the_resolver_beats_the_environment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LLM_API_KEY", "from-env")
        set_config_resolver(
            DictConfigResolver({"m": LLMConfig(model="m", api_key="from-resolver")})
        )
        assert resolve_config(model="m").api_key == "from-resolver"

    def test_an_installed_resolver_does_not_fall_back_to_the_environment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A shell variable must not seep into a run that installed a resolver."""
        monkeypatch.setenv("LLM_API_KEY", "from-env")
        set_config_resolver(DictConfigResolver({"m": LLMConfig(model="m")}))
        assert resolve_config(model="m").api_key == ""

    def test_the_environment_is_used_when_nothing_is_installed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LLM_MODEL", "env-model")
        monkeypatch.setenv("LLM_API_KEY", "from-env")
        monkeypatch.setenv("LLM_BASE_URL", "https://env.invalid")
        config = resolve_config()
        assert (config.model, config.api_key, config.base_url) == (
            "env-model",
            "from-env",
            "https://env.invalid",
        )

    def test_an_empty_api_key_argument_overrides_a_configured_one(self) -> None:
        """`is not None`, not truthiness: a local server may want no key at all."""
        set_config_resolver(
            DictConfigResolver({"m": LLMConfig(model="m", api_key="from-resolver")})
        )
        assert resolve_config(model="m", api_key="").api_key == ""

    def test_binding_falls_back_to_openai(self) -> None:
        set_config_resolver(DictConfigResolver({"m": LLMConfig(model="m", binding="")}))
        assert resolve_config(model="m").binding == "openai"

    def test_setting_the_resolver_to_none_restores_the_environment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LLM_MODEL", "env-model")
        set_config_resolver(DictConfigResolver({"m": LLMConfig(model="m")}))
        set_config_resolver(None)
        assert resolve_config().model == "env-model"


class TestExtraHeaders:
    def test_call_headers_merge_with_configured_ones(self) -> None:
        set_config_resolver(
            DictConfigResolver(
                {"m": LLMConfig(model="m", extra_headers={"X-Tenant": "a"})}
            )
        )
        config = resolve_config(model="m", extra_headers={"X-Trace": "b"})
        assert config.extra_headers == {"X-Tenant": "a", "X-Trace": "b"}

    def test_a_call_header_does_not_leak_into_the_next_call(self) -> None:
        """The harvested version updated the cached config's dict in place."""
        stored = LLMConfig(model="m", extra_headers={"X-Tenant": "a"})
        set_config_resolver(DictConfigResolver({"m": stored}))

        resolve_config(model="m", extra_headers={"X-Trace": "first"})
        second = resolve_config(model="m")

        assert second.extra_headers == {"X-Tenant": "a"}
        assert stored.extra_headers == {"X-Tenant": "a"}

    def test_no_headers_anywhere_yields_an_empty_mapping(self) -> None:
        set_config_resolver(DictConfigResolver({"m": LLMConfig(model="m")}))
        assert resolve_config(model="m").extra_headers == {}


class TestEnvConfigResolver:
    def test_a_missing_model_is_a_config_error(self) -> None:
        with pytest.raises(LLMConfigError, match="LLM_MODEL"):
            resolve_config()

    def test_the_argument_beats_llm_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LLM_MODEL", "env-model")
        assert resolve_config(model="argument-model").model == "argument-model"

    def test_the_environment_is_read_per_call_not_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Requirement 4: nothing is captured at import time."""
        monkeypatch.setenv("LLM_MODEL", "first")
        assert resolve_config().model == "first"
        monkeypatch.setenv("LLM_MODEL", "second")
        assert resolve_config().model == "second"

    def test_only_the_three_documented_variables_are_read(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LLM_MODEL", "m")
        monkeypatch.setenv("LLM_MAX_TOKENS", "1")
        monkeypatch.setenv("LLM_TEMPERATURE", "0.0")
        config = EnvConfigResolver().resolve(None)
        assert (config.max_tokens, config.temperature) == (4096, 0.7)


class TestDictConfigResolver:
    def test_an_unknown_model_names_the_known_ones(self) -> None:
        set_config_resolver(DictConfigResolver({"a": LLMConfig(model="a")}))
        with pytest.raises(LLMConfigError, match="known models: \\['a'\\]"):
            resolve_config(model="b")

    def test_a_missing_model_name_is_a_config_error(self) -> None:
        set_config_resolver(DictConfigResolver({"a": LLMConfig(model="a")}))
        with pytest.raises(LLMConfigError, match="model name is required"):
            resolve_config()

    def test_the_caller_s_mapping_is_copied(self) -> None:
        mapping = {"a": LLMConfig(model="a")}
        resolver = DictConfigResolver(mapping)
        mapping.clear()
        assert resolver.resolve("a").model == "a"


class TestJsonDirConfigResolver:
    """The harvested convention, preserved: <dir>/<lowercased model>.json."""

    @pytest.mark.parametrize(
        ("model", "filename"),
        [
            ("GLM-5.1", "glm-5.1.json"),
            ("glm-5.1", "glm-5.1.json"),
            ("My Model", "my_model.json"),
            ("  GLM-5.1  ", "glm-5.1.json"),
            ("gemma-4-31B-it", "gemma-4-31b-it.json"),
        ],
    )
    def test_the_model_name_becomes_the_filename(
        self, tmp_path: pathlib.Path, model: str, filename: str
    ) -> None:
        """T6's fourth criterion."""
        (tmp_path / filename).write_text(json.dumps({"api_key": "k"}), encoding="utf-8")
        assert JsonDirConfigResolver(tmp_path).resolve(model).api_key == "k"

    def test_every_field_is_read(self, tmp_path: pathlib.Path) -> None:
        (tmp_path / "m.json").write_text(
            json.dumps(
                {
                    "api_key": "k",
                    "base_url": "https://provider.invalid/v1",
                    "api_version": "2024-02-01",
                    "binding": "azure",
                    "extra_headers": {"X-Tenant": "a"},
                    "reasoning_effort": "high",
                    "max_tokens": 128,
                    "temperature": 0.1,
                    "max_concurrency": 4,
                    "requests_per_minute": 60,
                }
            ),
            encoding="utf-8",
        )
        config = JsonDirConfigResolver(tmp_path).resolve("m")
        assert config == LLMConfig(
            model="m",
            api_key="k",
            base_url="https://provider.invalid/v1",
            api_version="2024-02-01",
            binding="azure",
            extra_headers={"X-Tenant": "a"},
            reasoning_effort="high",
            max_tokens=128,
            temperature=0.1,
            max_concurrency=4,
            requests_per_minute=60,
        )

    def test_the_model_recorded_in_the_file_wins_over_the_lookup_key(
        self, tmp_path: pathlib.Path
    ) -> None:
        """So a file named for a local alias can still send the provider's name."""
        (tmp_path / "glm-5.1.json").write_text(
            json.dumps({"model": "glm-5.1-0710", "api_key": "k"}), encoding="utf-8"
        )
        assert (
            JsonDirConfigResolver(tmp_path).resolve("GLM-5.1").model == "glm-5.1-0710"
        )

    def test_the_lookup_key_is_used_when_the_file_names_no_model(
        self, tmp_path: pathlib.Path
    ) -> None:
        (tmp_path / "glm-5.1.json").write_text(
            json.dumps({"api_key": "k"}), encoding="utf-8"
        )
        assert JsonDirConfigResolver(tmp_path).resolve("GLM-5.1").model == "GLM-5.1"

    def test_unknown_keys_are_ignored(self, tmp_path: pathlib.Path) -> None:
        (tmp_path / "m.json").write_text(
            json.dumps({"api_key": "k", "traffic_controller": {}, "retired": 1}),
            encoding="utf-8",
        )
        assert JsonDirConfigResolver(tmp_path).resolve("m").api_key == "k"

    def test_a_missing_file_is_a_config_error(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(LLMConfigError, match="no usable LLM config"):
            JsonDirConfigResolver(tmp_path).resolve("m")

    def test_a_malformed_file_is_a_config_error(self, tmp_path: pathlib.Path) -> None:
        (tmp_path / "m.json").write_text('{"api_key": ', encoding="utf-8")
        with pytest.raises(LLMConfigError, match="no usable LLM config"):
            JsonDirConfigResolver(tmp_path).resolve("m")

    def test_a_json_array_is_a_config_error(self, tmp_path: pathlib.Path) -> None:
        (tmp_path / "m.json").write_text("[]", encoding="utf-8")
        with pytest.raises(LLMConfigError, match="no usable LLM config"):
            JsonDirConfigResolver(tmp_path).resolve("m")

    def test_the_error_names_the_file_it_looked_for(
        self, tmp_path: pathlib.Path
    ) -> None:
        with pytest.raises(LLMConfigError, match="glm-5.1.json"):
            JsonDirConfigResolver(tmp_path).resolve("GLM-5.1")

    def test_a_missing_model_name_is_a_config_error(
        self, tmp_path: pathlib.Path
    ) -> None:
        with pytest.raises(LLMConfigError, match="model name is required"):
            JsonDirConfigResolver(tmp_path).resolve(None)

    def test_it_works_through_the_installed_resolver(
        self, tmp_path: pathlib.Path
    ) -> None:
        """How `agent-evaluation` keeps working: install one at startup."""
        (tmp_path / "glm-5.1.json").write_text(
            json.dumps({"api_key": "k", "base_url": "https://provider.invalid/v1"}),
            encoding="utf-8",
        )
        set_config_resolver(JsonDirConfigResolver(tmp_path))
        config = resolve_config(model="GLM-5.1")
        assert (config.api_key, config.base_url) == (
            "k",
            "https://provider.invalid/v1",
        )


def test_the_resolver_protocol_accepts_a_plain_object() -> None:
    """Structural typing, so a host need not import or subclass anything."""

    class HostResolver:
        def resolve(self, model: str | None) -> LLMConfig:
            return LLMConfig(model=model or "host-default", api_key="from-host")

    set_config_resolver(HostResolver())
    assert resolve_config().api_key == "from-host"
