"""Tests for skill support in discuss-agent config and engine."""

import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import yaml
import pytest

from discuss_agent.config import (
    AgentConfig,
    ConfigLoader,
    DiscussionConfig,
    HostConfig,
    ModelConfig,
    SkillConfig,
    ToolConfig,
)

FIXTURES_DIR = str(Path(__file__).parent / "fixtures")
TEST_SKILL_PATH = str(Path(__file__).parent / "fixtures" / "test_skill")


# ---------------------------------------------------------------------------
# Config parsing: SkillConfig
# ---------------------------------------------------------------------------


class TestSkillConfigDataclass:
    def test_skill_config_has_path(self):
        sc = SkillConfig(path="/some/path")
        assert sc.path == "/some/path"


# ---------------------------------------------------------------------------
# Config parsing: global skills
# ---------------------------------------------------------------------------


class TestConfigLoaderGlobalSkills:
    def test_no_skills_defaults_none(self, sample_config_yaml):
        cfg = ConfigLoader.load(sample_config_yaml)
        assert cfg.skills is None

    def test_global_skills_parsed(self, tmp_path, sample_config_dict):
        d = dict(sample_config_dict)
        d["skills"] = [{"path": "/path/to/skill"}]
        path = tmp_path / "cfg.yaml"
        path.write_text(yaml.dump(d, allow_unicode=True))
        cfg = ConfigLoader.load(str(path))
        assert cfg.skills is not None
        assert len(cfg.skills) == 1
        assert cfg.skills[0].path == "/path/to/skill"

    def test_global_skills_missing_path_raises(self, tmp_path, sample_config_dict):
        d = dict(sample_config_dict)
        d["skills"] = [{"name": "bad"}]
        path = tmp_path / "cfg.yaml"
        path.write_text(yaml.dump(d, allow_unicode=True))
        with pytest.raises(ValueError, match="path"):
            ConfigLoader.load(str(path))


# ---------------------------------------------------------------------------
# Config parsing: per-agent skills
# ---------------------------------------------------------------------------


class TestConfigLoaderAgentSkills:
    def test_agent_skills_default_empty(self, sample_config_yaml):
        cfg = ConfigLoader.load(sample_config_yaml)
        assert cfg.agents[0].skills == []

    def test_agent_skills_parsed(self, tmp_path, sample_config_dict):
        d = dict(sample_config_dict)
        d["agents"][0]["skills"] = [{"path": "/skill/agent-a"}]
        path = tmp_path / "cfg.yaml"
        path.write_text(yaml.dump(d, allow_unicode=True))
        cfg = ConfigLoader.load(str(path))
        assert len(cfg.agents[0].skills) == 1
        assert cfg.agents[0].skills[0].path == "/skill/agent-a"
        assert cfg.agents[1].skills == []  # B has none


# ---------------------------------------------------------------------------
# Engine: skill loading
# ---------------------------------------------------------------------------


class TestEngineSkillLoading:
    """Test that skills are loaded and passed to Agent during engine init."""

    @patch("discuss_agent.engine.import_from_path")
    @patch("discuss_agent.engine.build_claude")
    @patch("discuss_agent.engine.Agent")
    @patch("discuss_agent.engine.ContextManager")
    def test_agent_receives_per_agent_skills(
        self, MockCtxMgr, MockAgent, mock_build_claude, mock_import
    ):
        from discuss_agent.engine import DiscussionEngine

        mock_build_claude.return_value = MagicMock()

        config = DiscussionConfig(
            min_rounds=2,
            max_rounds=5,
            model_config=MagicMock(),
            agents=[
                AgentConfig(
                    name="A1",
                    system_prompt="You are A1.",
                    skills=[SkillConfig(path=TEST_SKILL_PATH)],
                ),
                AgentConfig(name="A2", system_prompt="You are A2."),
            ],
            host=HostConfig(
                convergence_prompt="Judge.", summary_prompt="Summarize."
            ),
            tools=[],
            context={},
            skills=None,
        )

        engine = DiscussionEngine(config)

        # Discussion agents are created first, then host
        calls = MockAgent.call_args_list
        assert len(calls) >= 3  # 2 discussion + 1 host
        # Agent A1 should have skills
        assert calls[0].kwargs.get("skills") is not None
        # Agent A2 should not have skills
        assert calls[1].kwargs.get("skills") is None

    @patch("discuss_agent.engine.import_from_path")
    @patch("discuss_agent.engine.build_claude")
    @patch("discuss_agent.engine.Agent")
    @patch("discuss_agent.engine.ContextManager")
    def test_global_skills_shared(
        self, MockCtxMgr, MockAgent, mock_build_claude, mock_import
    ):
        from discuss_agent.engine import DiscussionEngine

        mock_build_claude.return_value = MagicMock()

        config = DiscussionConfig(
            min_rounds=2,
            max_rounds=5,
            model_config=MagicMock(),
            agents=[
                AgentConfig(name="A1", system_prompt="You are A1."),
                AgentConfig(name="A2", system_prompt="You are A2."),
            ],
            host=HostConfig(
                convergence_prompt="Judge.", summary_prompt="Summarize."
            ),
            tools=[],
            context={},
            skills=[SkillConfig(path=TEST_SKILL_PATH)],
        )

        engine = DiscussionEngine(config)

        calls = MockAgent.call_args_list
        # Both discussion agents should have global skills
        assert calls[0].kwargs.get("skills") is not None
        assert calls[1].kwargs.get("skills") is not None

    @patch("discuss_agent.engine.import_from_path")
    @patch("discuss_agent.engine.build_claude")
    @patch("discuss_agent.engine.Agent")
    @patch("discuss_agent.engine.ContextManager")
    def test_no_skills_passes_none(
        self, MockCtxMgr, MockAgent, mock_build_claude, mock_import
    ):
        from discuss_agent.engine import DiscussionEngine

        mock_build_claude.return_value = MagicMock()

        config = DiscussionConfig(
            min_rounds=2,
            max_rounds=5,
            model_config=MagicMock(),
            agents=[
                AgentConfig(name="A1", system_prompt="You are A1."),
            ],
            host=HostConfig(
                convergence_prompt="Judge.", summary_prompt="Summarize."
            ),
            tools=[],
            context={},
            skills=None,
        )

        engine = DiscussionEngine(config)

        # Discussion agent should have skills=None
        discussion_call = MockAgent.call_args_list[0]
        assert discussion_call.kwargs.get("skills") is None
