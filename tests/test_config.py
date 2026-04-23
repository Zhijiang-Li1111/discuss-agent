"""Tests for discuss_agent.config — ConfigLoader and config dataclasses."""

import yaml
import pytest

from discuss_agent.config import (
    AgentConfig,
    ConfigLoader,
    DiscussionConfig,
    HostConfig,
    ModelConfig,
    ToolConfig,
    resolve_env,
    _resolve_template_vars,
)


class TestConfigLoaderFullParse:
    """Test that a full YAML file is parsed into the correct dataclass tree."""

    def test_load_returns_discussion_config(self, sample_config_yaml):
        cfg = ConfigLoader.load(sample_config_yaml)
        assert isinstance(cfg, DiscussionConfig)

    def test_discussion_fields(self, sample_config_yaml):
        cfg = ConfigLoader.load(sample_config_yaml)
        assert cfg.min_rounds == 2
        assert cfg.max_rounds == 5
        assert cfg.model_config.model == "claude-sonnet-4-20250514"

    def test_agents(self, sample_config_yaml):
        cfg = ConfigLoader.load(sample_config_yaml)
        assert len(cfg.agents) == 2
        assert isinstance(cfg.agents[0], AgentConfig)
        assert cfg.agents[0].name == "Agent A"
        assert cfg.agents[0].system_prompt == "You are agent A."
        assert cfg.agents[1].name == "Agent B"

    def test_host(self, sample_config_yaml):
        cfg = ConfigLoader.load(sample_config_yaml)
        assert isinstance(cfg.host, HostConfig)
        assert cfg.host.convergence_prompt == "Judge convergence."
        assert cfg.host.summary_prompt == "Summarize the discussion."

    def test_tools_are_tool_configs(self, sample_config_yaml):
        cfg = ConfigLoader.load(sample_config_yaml)
        assert len(cfg.tools) == 2
        assert all(isinstance(t, ToolConfig) for t in cfg.tools)
        assert cfg.tools[0].path == "tests.helpers.FakeToolA"
        assert cfg.tools[1].path == "tests.helpers.FakeToolB"

    def test_context(self, sample_config_yaml):
        cfg = ConfigLoader.load(sample_config_yaml)
        assert isinstance(cfg.context, dict)
        assert cfg.context["research_dir"] == "~/research-data/"
        assert cfg.context["published_file"] == "PUBLISHED.md"
        assert cfg.context["research_days"] == 2


class TestConfigLoaderDefaults:
    """min_rounds and max_rounds should fall back to defaults when omitted."""

    def test_defaults_applied(self, tmp_path, sample_config_dict):
        # Remove min/max from discussion block
        d = dict(sample_config_dict)
        d["discussion"] = {"model": "claude-sonnet-4-20250514"}
        path = tmp_path / "no_rounds.yaml"
        path.write_text(yaml.dump(d, allow_unicode=True))

        cfg = ConfigLoader.load(str(path))
        assert cfg.min_rounds == 2
        assert cfg.max_rounds == 5


class TestConfigLoaderValidation:
    """Missing required keys should raise ValueError."""

    def test_missing_agents_raises(self, tmp_path, sample_config_dict):
        d = dict(sample_config_dict)
        del d["agents"]
        path = tmp_path / "bad.yaml"
        path.write_text(yaml.dump(d, allow_unicode=True))

        with pytest.raises(ValueError, match="agents"):
            ConfigLoader.load(str(path))

    def test_missing_host_raises(self, tmp_path, sample_config_dict):
        d = dict(sample_config_dict)
        del d["host"]
        path = tmp_path / "bad.yaml"
        path.write_text(yaml.dump(d, allow_unicode=True))

        with pytest.raises(ValueError, match="host"):
            ConfigLoader.load(str(path))

    def test_missing_tools_raises(self, tmp_path, sample_config_dict):
        d = dict(sample_config_dict)
        del d["tools"]
        path = tmp_path / "bad.yaml"
        path.write_text(yaml.dump(d, allow_unicode=True))

        with pytest.raises(ValueError, match="tools"):
            ConfigLoader.load(str(path))

    def test_missing_context_defaults_empty(self, tmp_path, sample_config_dict):
        d = dict(sample_config_dict)
        del d["context"]
        path = tmp_path / "no_ctx.yaml"
        path.write_text(yaml.dump(d, allow_unicode=True))

        cfg = ConfigLoader.load(str(path))
        assert cfg.context == {}

    def test_missing_model_raises(self, tmp_path, sample_config_dict):
        d = dict(sample_config_dict)
        d["discussion"] = {"min_rounds": 2}  # no model key
        path = tmp_path / "bad.yaml"
        path.write_text(yaml.dump(d, allow_unicode=True))

        with pytest.raises(ValueError, match="model"):
            ConfigLoader.load(str(path))

    def test_context_arbitrary_keys(self, tmp_path, sample_config_dict):
        d = dict(sample_config_dict)
        d["context"] = {"foo": "bar", "num": 42}
        path = tmp_path / "custom_ctx.yaml"
        path.write_text(yaml.dump(d, allow_unicode=True))

        cfg = ConfigLoader.load(str(path))
        assert cfg.context == {"foo": "bar", "num": 42}

    def test_tool_entry_missing_path_raises(self, tmp_path, sample_config_dict):
        d = dict(sample_config_dict)
        d["tools"] = [{"name": "bad"}]
        path = tmp_path / "bad_tool.yaml"
        path.write_text(yaml.dump(d, allow_unicode=True))

        with pytest.raises(ValueError, match="path"):
            ConfigLoader.load(str(path))

    def test_tool_entry_string_raises(self, tmp_path, sample_config_dict):
        d = dict(sample_config_dict)
        d["tools"] = ["just_a_string"]
        path = tmp_path / "bad_tool.yaml"
        path.write_text(yaml.dump(d, allow_unicode=True))

        with pytest.raises(ValueError, match="path"):
            ConfigLoader.load(str(path))

    def test_empty_tools_list_ok(self, tmp_path, sample_config_dict):
        d = dict(sample_config_dict)
        d["tools"] = []
        path = tmp_path / "empty_tools.yaml"
        path.write_text(yaml.dump(d, allow_unicode=True))

        cfg = ConfigLoader.load(str(path))
        assert cfg.tools == []


class TestAgentConfigPerAgentTools:
    """Test per-agent extra_tools and disable_tools parsing."""

    def test_extra_tools_parsed(self, tmp_path, sample_config_dict):
        d = dict(sample_config_dict)
        d["agents"][0]["extra_tools"] = [{"path": "pkg.ExtraTool"}]
        path = tmp_path / "extra.yaml"
        path.write_text(yaml.dump(d, allow_unicode=True))

        cfg = ConfigLoader.load(str(path))
        assert len(cfg.agents[0].extra_tools) == 1
        assert cfg.agents[0].extra_tools[0].path == "pkg.ExtraTool"

    def test_disable_tools_parsed(self, tmp_path, sample_config_dict):
        d = dict(sample_config_dict)
        d["agents"][0]["disable_tools"] = ["tests.helpers.FakeToolA"]
        path = tmp_path / "disable.yaml"
        path.write_text(yaml.dump(d, allow_unicode=True))

        cfg = ConfigLoader.load(str(path))
        assert cfg.agents[0].disable_tools == ["tests.helpers.FakeToolA"]

    def test_defaults_empty(self, sample_config_yaml):
        cfg = ConfigLoader.load(sample_config_yaml)
        assert cfg.agents[0].extra_tools == []
        assert cfg.agents[0].disable_tools == []


class TestContextBuilder:
    """Test context_builder top-level field parsing."""

    def test_context_builder_parsed(self, tmp_path, sample_config_dict):
        d = dict(sample_config_dict)
        d["context_builder"] = "my_pkg.context.build_ctx"
        path = tmp_path / "with_builder.yaml"
        path.write_text(yaml.dump(d, allow_unicode=True))

        cfg = ConfigLoader.load(str(path))
        assert cfg.context_builder == "my_pkg.context.build_ctx"

    def test_context_builder_defaults_none(self, sample_config_yaml):
        cfg = ConfigLoader.load(sample_config_yaml)
        assert cfg.context_builder is None


class TestLimitation:
    """Test limitation top-level field parsing."""

    def test_limitation_parsed(self, tmp_path, sample_config_dict):
        d = dict(sample_config_dict)
        d["limitation"] = "仅讨论技术可行性"
        path = tmp_path / "with_limitation.yaml"
        path.write_text(yaml.dump(d, allow_unicode=True))

        cfg = ConfigLoader.load(str(path))
        assert cfg.limitation == "仅讨论技术可行性"

    def test_limitation_defaults_none(self, sample_config_yaml):
        cfg = ConfigLoader.load(sample_config_yaml)
        assert cfg.limitation is None


# ---------------------------------------------------------------------------
# resolve_env (AC-02..04, AC-18)
# ---------------------------------------------------------------------------


class TestResolveEnv:
    """Tests for the env: prefix resolver."""

    def test_env_prefix_resolves(self, monkeypatch):
        monkeypatch.setenv("TEST_API_KEY", "sk-secret-123")
        assert resolve_env("env:TEST_API_KEY") == "sk-secret-123"

    def test_env_prefix_missing_raises(self, monkeypatch):
        monkeypatch.delenv("MISSING_VAR", raising=False)
        with pytest.raises(ValueError, match="MISSING_VAR"):
            resolve_env("env:MISSING_VAR")

    def test_literal_returned_as_is(self):
        assert resolve_env("sk-literal-key") == "sk-literal-key"

    def test_none_returns_none(self):
        assert resolve_env(None) is None


# ---------------------------------------------------------------------------
# ModelConfig (AC-01, AC-19, AC-23)
# ---------------------------------------------------------------------------


class TestModelConfig:
    """Tests for ModelConfig dataclass."""

    def test_all_defaults(self):
        mc = ModelConfig(model="claude-sonnet-4-20250514")
        assert mc.model == "claude-sonnet-4-20250514"
        assert mc.api_key is None
        assert mc.base_url is None
        assert mc.temperature is None
        assert mc.max_tokens is None

    def test_all_explicit(self):
        mc = ModelConfig(
            model="claude-opus-4-20250514",
            api_key="sk-test",
            base_url="http://localhost:8080",
            temperature=0.7,
            max_tokens=4096,
        )
        assert mc.model == "claude-opus-4-20250514"
        assert mc.api_key == "sk-test"
        assert mc.base_url == "http://localhost:8080"
        assert mc.temperature == 0.7
        assert mc.max_tokens == 4096

    def test_to_safe_dict_masks_api_key(self):
        mc = ModelConfig(model="m", api_key="sk-secret")
        safe = mc.to_safe_dict()
        assert safe["api_key"] == "***"
        assert safe["model"] == "m"

    def test_to_safe_dict_none_api_key_preserved(self):
        mc = ModelConfig(model="m")
        safe = mc.to_safe_dict()
        assert safe["api_key"] is None


# ---------------------------------------------------------------------------
# Host model override (AC-10..12, AC-20)
# ---------------------------------------------------------------------------


class TestHostModelOverride:
    """Tests for HostConfig.resolve_model merge logic."""

    def test_no_override_returns_discussion_copy(self):
        discussion = ModelConfig(
            model="claude-sonnet-4-20250514",
            api_key="sk-key",
            base_url="http://base",
            temperature=0.5,
            max_tokens=2048,
        )
        host = HostConfig(convergence_prompt="cp", summary_prompt="sp")
        resolved = host.resolve_model(discussion)
        assert resolved.model == "claude-sonnet-4-20250514"
        assert resolved.api_key == "sk-key"
        assert resolved.base_url == "http://base"
        assert resolved.temperature == 0.5
        assert resolved.max_tokens == 2048

    def test_partial_override(self):
        discussion = ModelConfig(
            model="claude-sonnet-4-20250514",
            api_key="sk-key",
            temperature=0.5,
        )
        host = HostConfig(
            convergence_prompt="cp",
            summary_prompt="sp",
            model="claude-haiku-4-5-20251001",
            temperature=0.3,
        )
        resolved = host.resolve_model(discussion)
        assert resolved.model == "claude-haiku-4-5-20251001"
        assert resolved.temperature == 0.3
        assert resolved.api_key == "sk-key"  # inherited

    def test_full_override(self):
        discussion = ModelConfig(model="a", api_key="k1", base_url="u1", temperature=0.5, max_tokens=100)
        host = HostConfig(
            convergence_prompt="cp",
            summary_prompt="sp",
            model="b",
            api_key="k2",
            base_url="u2",
            temperature=0.1,
            max_tokens=200,
        )
        resolved = host.resolve_model(discussion)
        assert resolved.model == "b"
        assert resolved.api_key == "k2"
        assert resolved.base_url == "u2"
        assert resolved.temperature == 0.1
        assert resolved.max_tokens == 200


# ---------------------------------------------------------------------------
# ConfigLoader model fields (AC-05..09)
# ---------------------------------------------------------------------------


class TestConfigLoaderModelFields:
    """Tests for model config parsing from YAML."""

    def test_model_only_backward_compatible(self, tmp_path, sample_config_dict):
        """Existing configs with only discussion.model should work."""
        path = tmp_path / "compat.yaml"
        path.write_text(yaml.dump(sample_config_dict, allow_unicode=True))

        cfg = ConfigLoader.load(str(path))
        assert cfg.model_config.model == "claude-sonnet-4-20250514"
        assert cfg.model_config.api_key is None
        assert cfg.model_config.base_url is None
        assert cfg.model_config.temperature is None
        assert cfg.model_config.max_tokens is None

    def test_full_model_config_parsed(self, tmp_path, sample_config_dict, monkeypatch):
        monkeypatch.setenv("MY_KEY", "sk-resolved")
        d = dict(sample_config_dict)
        d["discussion"] = {
            "model": "claude-opus-4-20250514",
            "api_key": "env:MY_KEY",
            "base_url": "http://proxy:8080",
            "temperature": 0.7,
            "max_tokens": 4096,
        }
        path = tmp_path / "full.yaml"
        path.write_text(yaml.dump(d, allow_unicode=True))

        cfg = ConfigLoader.load(str(path))
        assert cfg.model_config.model == "claude-opus-4-20250514"
        assert cfg.model_config.api_key == "sk-resolved"
        assert cfg.model_config.base_url == "http://proxy:8080"
        assert cfg.model_config.temperature == 0.7
        assert cfg.model_config.max_tokens == 4096

    def test_api_key_env_missing_raises(self, tmp_path, sample_config_dict, monkeypatch):
        monkeypatch.delenv("NONEXISTENT_KEY", raising=False)
        d = dict(sample_config_dict)
        d["discussion"] = {"model": "m", "api_key": "env:NONEXISTENT_KEY"}
        path = tmp_path / "bad_key.yaml"
        path.write_text(yaml.dump(d, allow_unicode=True))

        with pytest.raises(ValueError, match="NONEXISTENT_KEY"):
            ConfigLoader.load(str(path))

    def test_host_model_override_parsed(self, tmp_path, sample_config_dict):
        d = dict(sample_config_dict)
        d["host"]["model"] = "claude-haiku-4-5-20251001"
        d["host"]["temperature"] = 0.2
        path = tmp_path / "host_override.yaml"
        path.write_text(yaml.dump(d, allow_unicode=True))

        cfg = ConfigLoader.load(str(path))
        assert cfg.host.model == "claude-haiku-4-5-20251001"
        assert cfg.host.temperature == 0.2

    def test_host_api_key_env_resolved(self, tmp_path, sample_config_dict, monkeypatch):
        monkeypatch.setenv("HOST_KEY", "sk-host-key")
        d = dict(sample_config_dict)
        d["host"]["api_key"] = "env:HOST_KEY"
        path = tmp_path / "host_key.yaml"
        path.write_text(yaml.dump(d, allow_unicode=True))

        cfg = ConfigLoader.load(str(path))
        assert cfg.host.api_key == "sk-host-key"


class TestResolveTemplateVars:
    """Tests for {{var}} template variable resolution."""

    def test_top_level_string_replaces_in_agent_prompt(self, tmp_path, sample_config_dict):
        d = dict(sample_config_dict)
        d["reader_persona"] = "程序员，26-35岁"
        d["agents"] = [
            {"name": "A1", "system_prompt": "读者是{{reader_persona}}。"},
        ]
        path = tmp_path / "tpl.yaml"
        path.write_text(yaml.dump(d, allow_unicode=True))
        cfg = ConfigLoader.load(str(path))
        assert cfg.agents[0].system_prompt == "读者是程序员，26-35岁。"

    def test_top_level_string_replaces_in_host_prompts(self, tmp_path, sample_config_dict):
        d = dict(sample_config_dict)
        d["scope"] = "科技与AI领域"
        d["host"] = {
            "convergence_prompt": "范围：{{scope}}",
            "summary_prompt": "总结{{scope}}讨论",
        }
        path = tmp_path / "tpl2.yaml"
        path.write_text(yaml.dump(d, allow_unicode=True))
        cfg = ConfigLoader.load(str(path))
        assert cfg.host.convergence_prompt == "范围：科技与AI领域"
        assert cfg.host.summary_prompt == "总结科技与AI领域讨论"

    def test_multiple_vars_replaced(self, tmp_path, sample_config_dict):
        d = dict(sample_config_dict)
        d["var_a"] = "AAA"
        d["var_b"] = "BBB"
        d["agents"] = [
            {"name": "A1", "system_prompt": "{{var_a}} and {{var_b}}"},
        ]
        path = tmp_path / "multi.yaml"
        path.write_text(yaml.dump(d, allow_unicode=True))
        cfg = ConfigLoader.load(str(path))
        assert cfg.agents[0].system_prompt == "AAA and BBB"

    def test_unknown_placeholder_left_as_is(self, tmp_path, sample_config_dict):
        d = dict(sample_config_dict)
        d["agents"] = [
            {"name": "A1", "system_prompt": "Keep {{unknown_var}} intact."},
        ]
        path = tmp_path / "unknown.yaml"
        path.write_text(yaml.dump(d, allow_unicode=True))
        cfg = ConfigLoader.load(str(path))
        assert "{{unknown_var}}" in cfg.agents[0].system_prompt

    def test_reserved_keys_not_used_as_vars(self):
        raw = {
            "discussion": {"model": "m"},
            "agents": [{"name": "A", "system_prompt": "Use {{discussion}}"}],
            "host": {"convergence_prompt": "cp", "summary_prompt": "sp"},
            "tools": [],
        }
        _resolve_template_vars(raw)
        assert raw["agents"][0]["system_prompt"] == "Use {{discussion}}"

    def test_non_string_top_level_ignored(self):
        raw = {
            "discussion": {"model": "m"},
            "agents": [{"name": "A", "system_prompt": "{{num}}"}],
            "host": {"convergence_prompt": "cp", "summary_prompt": "sp"},
            "tools": [],
            "num": 42,
        }
        _resolve_template_vars(raw)
        assert raw["agents"][0]["system_prompt"] == "{{num}}"

    def test_no_vars_is_noop(self, sample_config_yaml):
        cfg = ConfigLoader.load(sample_config_yaml)
        assert cfg.agents[0].system_prompt == "You are agent A."
