"""Tests for discuss_agent.config — ConfigLoader and config dataclasses."""

import yaml
import pytest

from discuss_agent.config import (
    AgentConfig,
    ConfigLoader,
    ContextConfig,
    DiscussionConfig,
    HostConfig,
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
        assert cfg.model == "claude-sonnet-4-20250514"

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

    def test_tools(self, sample_config_yaml):
        cfg = ConfigLoader.load(sample_config_yaml)
        assert cfg.tools == ["research_list", "research_content", "trending", "published"]

    def test_context(self, sample_config_yaml):
        cfg = ConfigLoader.load(sample_config_yaml)
        assert isinstance(cfg.context, ContextConfig)
        assert cfg.context.research_dir == "~/ima-downloads/"
        assert cfg.context.published_file == "PUBLISHED.md"
        assert cfg.context.research_days == 2


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

    def test_missing_context_raises(self, tmp_path, sample_config_dict):
        d = dict(sample_config_dict)
        del d["context"]
        path = tmp_path / "bad.yaml"
        path.write_text(yaml.dump(d, allow_unicode=True))

        with pytest.raises(ValueError, match="context"):
            ConfigLoader.load(str(path))

    def test_missing_model_raises(self, tmp_path, sample_config_dict):
        d = dict(sample_config_dict)
        d["discussion"] = {"min_rounds": 2}  # no model key
        path = tmp_path / "bad.yaml"
        path.write_text(yaml.dump(d, allow_unicode=True))

        with pytest.raises(ValueError, match="model"):
            ConfigLoader.load(str(path))
