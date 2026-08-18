"""Config resolution: three resolvers, one precedence rule, no host directory.

Every test runs with the LLM_* environment cleared and the process-wide resolver
reset, so nothing here depends on the shell it was launched from and no test can
leak an installed resolver into the next one.
"""

import json
import pathlib
import textwrap
from collections.abc import Iterator

import pytest

from agent_toolkit.llm.config import (
    DictConfigResolver,
    EnvConfigResolver,
    JsonDirConfigResolver,
    LLMConfig,
    YamlConfigResolver,
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


class TestYamlConfigResolver:
    """One file, a ``defaults`` block, per-model overrides, and no credentials."""

    def _write(self, tmp_path: pathlib.Path, text: str) -> pathlib.Path:
        target = tmp_path / "models.yaml"
        target.write_text(textwrap.dedent(text), encoding="utf-8")
        return target

    def test_the_three_configs_from_the_host_pipeline(
        self, tmp_path: pathlib.Path
    ) -> None:
        """The file this feature exists for, read back field by field."""
        path = self._write(
            tmp_path,
            """
            defaults:
              temperature: 0.3
              top_p: 1.0
              max_concurrency: 10
              requests_per_minute: 600
              timeout: 120.0
              max_tokens: 4096

            models:
              gemma-4-31B-it: {}
              DeepSeek-V4-Flash:
                enable_thinking: false
                max_tokens: 8012
              Qwen3.6-27B:
                temperature: 0
                enable_thinking: false
            """,
        )
        resolver = YamlConfigResolver(path)

        gemma = resolver.resolve("gemma-4-31B-it")
        assert gemma == LLMConfig(
            model="gemma-4-31B-it",
            temperature=0.3,
            top_p=1.0,
            timeout=120.0,
            max_tokens=4096,
            max_concurrency=10,
            requests_per_minute=600,
        )

        deepseek = resolver.resolve("DeepSeek-V4-Flash")
        assert (deepseek.enable_thinking, deepseek.max_tokens) == (False, 8012)
        assert deepseek.temperature == 0.3  # still the default

        qwen = resolver.resolve("Qwen3.6-27B")
        assert (qwen.temperature, qwen.enable_thinking) == (0, False)
        assert qwen.max_tokens == 4096  # still the default

    def test_neither_credential_nor_endpoint_is_read_from_the_file(
        self, tmp_path: pathlib.Path
    ) -> None:
        path = self._write(tmp_path, "models:\n  m: {}\n")
        config = YamlConfigResolver(path).resolve("m")
        assert (config.api_key, config.base_url) == ("", None)

    @pytest.mark.parametrize(
        ("key", "value"), [("api_key", "sk-committed"), ("base_url", "https://p/v1")]
    )
    def test_a_caller_supplied_setting_in_a_model_block_is_refused(
        self, tmp_path: pathlib.Path, key: str, value: str
    ) -> None:
        """Silently using a committed secret or endpoint, or silently ignoring one,
        are both worse than saying so at load time."""
        path = self._write(tmp_path, f"models:\n  m:\n    {key}: {value}\n")
        with pytest.raises(LLMConfigError, match=f"sets '{key}'"):
            YamlConfigResolver(path)

    @pytest.mark.parametrize(
        ("key", "value"), [("api_key", "sk-committed"), ("base_url", "https://p/v1")]
    )
    def test_a_caller_supplied_setting_in_the_defaults_block_is_refused(
        self, tmp_path: pathlib.Path, key: str, value: str
    ) -> None:
        path = self._write(
            tmp_path, f"defaults:\n  {key}: {value}\nmodels:\n  m: {{}}\n"
        )
        with pytest.raises(LLMConfigError, match=f"sets '{key}'"):
            YamlConfigResolver(path)

    def test_the_call_site_supplies_the_key_and_the_endpoint(
        self, tmp_path: pathlib.Path
    ) -> None:
        """How a YAML-configured run reaches a provider: the arguments, as always."""
        path = self._write(
            tmp_path, "defaults:\n  temperature: 0.3\nmodels:\n  m: {}\n"
        )
        set_config_resolver(YamlConfigResolver(path))
        config = resolve_config(
            model="m", api_key="sk-from-the-host", base_url="https://p.invalid/v1"
        )
        assert (config.api_key, config.base_url, config.temperature) == (
            "sk-from-the-host",
            "https://p.invalid/v1",
            0.3,
        )

    def test_a_misspelled_setting_raises_instead_of_being_ignored(
        self, tmp_path: pathlib.Path
    ) -> None:
        """The whole point: a typo must not silently leave every call on the default."""
        path = self._write(tmp_path, "models:\n  m:\n    temperatur: 0\n")
        with pytest.raises(LLMConfigError, match="unknown setting"):
            YamlConfigResolver(path)

    def test_a_misspelled_default_names_the_defaults_block(
        self, tmp_path: pathlib.Path
    ) -> None:
        path = self._write(tmp_path, "defaults:\n  top-p: 1.0\nmodels:\n  m: {}\n")
        with pytest.raises(LLMConfigError, match="defaults has unknown setting"):
            YamlConfigResolver(path)

    def test_the_error_names_the_settable_keys(self, tmp_path: pathlib.Path) -> None:
        path = self._write(tmp_path, "models:\n  m:\n    nonsense: 1\n")
        with pytest.raises(LLMConfigError, match="top_p"):
            YamlConfigResolver(path)

    def test_a_typo_in_a_later_model_is_caught_at_load_not_at_the_call(
        self, tmp_path: pathlib.Path
    ) -> None:
        path = self._write(
            tmp_path, "models:\n  first: {}\n  second:\n    max_token: 8\n"
        )
        with pytest.raises(LLMConfigError, match="model 'second'"):
            YamlConfigResolver(path)

    def test_the_model_recorded_in_the_block_wins_over_the_key(
        self, tmp_path: pathlib.Path
    ) -> None:
        """So a short local alias can still send the provider's real name."""
        path = self._write(tmp_path, "models:\n  fast:\n    model: DeepSeek-V4-Flash\n")
        assert YamlConfigResolver(path).resolve("fast").model == "DeepSeek-V4-Flash"

    def test_an_unknown_model_names_the_known_ones(
        self, tmp_path: pathlib.Path
    ) -> None:
        path = self._write(tmp_path, "models:\n  a: {}\n")
        with pytest.raises(LLMConfigError, match="known models: \\['a'\\]"):
            YamlConfigResolver(path).resolve("b")

    def test_a_missing_model_name_is_a_config_error(
        self, tmp_path: pathlib.Path
    ) -> None:
        path = self._write(tmp_path, "models:\n  a: {}\n")
        with pytest.raises(LLMConfigError, match="model name is required"):
            YamlConfigResolver(path).resolve(None)

    def test_a_missing_file_is_a_config_error(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(LLMConfigError, match="no usable LLM config"):
            YamlConfigResolver(tmp_path / "absent.yaml")

    def test_a_malformed_file_is_a_config_error(self, tmp_path: pathlib.Path) -> None:
        path = self._write(tmp_path, "models: [unclosed\n")
        with pytest.raises(LLMConfigError, match="no usable LLM config"):
            YamlConfigResolver(path)

    def test_a_file_without_a_models_block_is_a_config_error(
        self, tmp_path: pathlib.Path
    ) -> None:
        path = self._write(tmp_path, "defaults:\n  temperature: 0.3\n")
        with pytest.raises(LLMConfigError, match="no 'models' block"):
            YamlConfigResolver(path)

    def test_a_model_that_is_not_a_mapping_is_a_config_error(
        self, tmp_path: pathlib.Path
    ) -> None:
        path = self._write(tmp_path, "models:\n  m: 0.3\n")
        with pytest.raises(LLMConfigError, match="must be a mapping"):
            YamlConfigResolver(path)

    def test_it_works_through_the_installed_resolver(
        self, tmp_path: pathlib.Path
    ) -> None:
        path = self._write(
            tmp_path,
            "defaults:\n  temperature: 0.3\n  max_tokens: 512\nmodels:\n  m: {}\n",
        )
        set_config_resolver(YamlConfigResolver(path))
        config = resolve_config(model="m")
        assert (config.temperature, config.max_tokens) == (0.3, 512)
