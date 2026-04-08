"""Tests for the tool registry (__init__.py)."""

import pytest

from discuss_agent.tools import TOOL_REGISTRY, get_tools
from discuss_agent.tools.research_list import ResearchListTools
from discuss_agent.tools.research_content import ResearchContentTools
from discuss_agent.tools.trending import TrendingTools
from discuss_agent.tools.published import PublishedTools
from discuss_agent.config import DiscussionConfig, AgentConfig, HostConfig, ContextConfig


def _make_config(research_dir: str = "~/ima-downloads/") -> DiscussionConfig:
    """Create a minimal DiscussionConfig for testing."""
    return DiscussionConfig(
        min_rounds=2,
        max_rounds=5,
        model="claude-sonnet-4-20250514",
        agents=[AgentConfig(name="A", system_prompt="prompt")],
        host=HostConfig(
            convergence_prompt="converge",
            summary_prompt="summarize",
        ),
        tools=["research_list", "research_content", "trending", "published"],
        context=ContextConfig(
            research_dir=research_dir,
            published_file="PUBLISHED.md",
            research_days=2,
        ),
    )


class TestGetToolsReturnsInstances:
    """test_get_tools_returns_instances — verify correct Toolkit instances."""

    def test_get_tools_returns_instances(self):
        config = _make_config()
        tools = get_tools(
            ["research_list", "research_content", "trending", "published"],
            config,
        )

        assert len(tools) == 4
        assert isinstance(tools[0], ResearchListTools)
        assert isinstance(tools[1], ResearchContentTools)
        assert isinstance(tools[2], TrendingTools)
        assert isinstance(tools[3], PublishedTools)

    def test_get_tools_single(self):
        config = _make_config()
        tools = get_tools(["trending"], config)

        assert len(tools) == 1
        assert isinstance(tools[0], TrendingTools)

    def test_research_list_receives_config_dir(self, tmp_path):
        config = _make_config(research_dir=str(tmp_path))
        tools = get_tools(["research_list"], config)

        assert isinstance(tools[0], ResearchListTools)
        assert tools[0].research_dir == str(tmp_path)


class TestUnknownToolRaises:
    """test_unknown_tool_raises — verify ValueError for unknown tool name."""

    def test_unknown_tool_raises(self):
        config = _make_config()

        with pytest.raises(ValueError, match="Unknown tool"):
            get_tools(["nonexistent_tool"], config)

    def test_unknown_tool_among_valid_raises(self):
        config = _make_config()

        with pytest.raises(ValueError, match="Unknown tool"):
            get_tools(["trending", "bad_tool_name"], config)


class TestToolRegistry:
    """Verify TOOL_REGISTRY contains all expected tools."""

    def test_registry_has_all_tools(self):
        expected = {"research_list", "research_content", "trending", "published"}
        assert set(TOOL_REGISTRY.keys()) == expected

    def test_registry_values_are_classes(self):
        for name, cls in TOOL_REGISTRY.items():
            assert isinstance(cls, type), f"{name} should map to a class"
