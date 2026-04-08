"""Tool registry for the multi-agent discussion framework."""

from __future__ import annotations

from typing import TYPE_CHECKING

from discuss_agent.tools.research_list import ResearchListTools
from discuss_agent.tools.research_content import ResearchContentTools
from discuss_agent.tools.trending import TrendingTools
from discuss_agent.tools.published import PublishedTools

if TYPE_CHECKING:
    from discuss_agent.config import DiscussionConfig

TOOL_REGISTRY: dict[str, type] = {
    "research_list": ResearchListTools,
    "research_content": ResearchContentTools,
    "trending": TrendingTools,
    "published": PublishedTools,
}


def get_tools(names: list[str], config: DiscussionConfig) -> list:
    """Instantiate toolkit classes by name.

    Parameters:
        names: List of tool names to instantiate (must be keys in TOOL_REGISTRY).
        config: A DiscussionConfig providing context such as research_dir.

    Returns:
        A list of instantiated Toolkit objects.

    Raises:
        ValueError: If a tool name is not found in the registry.
    """
    tools = []
    for name in names:
        if name not in TOOL_REGISTRY:
            raise ValueError(f"Unknown tool: {name}")
        cls = TOOL_REGISTRY[name]
        if name == "research_list":
            tools.append(cls(research_dir=config.context.research_dir))
        elif name == "research_content":
            tools.append(cls(allowed_dir=config.context.research_dir))
        else:
            tools.append(cls())
    return tools
