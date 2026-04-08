"""Plugin registry for the multi-agent discussion framework.

Discovers and loads plugins via Python entry_points, allowing external
packages to register custom tools and context builders.
"""

from __future__ import annotations

import importlib.metadata
import logging
from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)


class PluginRegistry:
    """Plugin registration interface for the discussion framework."""

    def __init__(self) -> None:
        self._tools: dict[str, type] = {}
        self._context_builder: Callable[[dict], Awaitable[str]] | None = None

    def register_tool(self, name: str, tool_class: type) -> None:
        """Register a tool class by name.

        The tool_class must be an Agno Toolkit subclass.
        When the YAML config's tools list references this name,
        the framework instantiates this class with context=config.context.
        """
        self._tools[name] = tool_class

    def register_context_builder(
        self, builder: Callable[[dict], Awaitable[str]]
    ) -> None:
        """Register a function that builds initial shared context.

        The builder receives the YAML config's 'context' dict
        and returns an assembled context string.
        Only one context builder is active; last registered wins.
        """
        self._context_builder = builder

    def get_tool_class(self, name: str) -> type:
        """Look up a registered tool class by name.

        Raises ValueError if the tool name is not registered.
        """
        if name not in self._tools:
            raise ValueError(
                f"Unknown tool: '{name}'. "
                f"Available: {list(self._tools.keys())}"
            )
        return self._tools[name]

    def get_context_builder(self) -> Callable[[dict], Awaitable[str]] | None:
        """Return the registered context builder, or None."""
        return self._context_builder


def load_plugins() -> PluginRegistry:
    """Discover and load all discuss_agent plugins via entry_points.

    Looks for entry points in the ``discuss_agent.plugins`` group.
    Each entry point must resolve to a callable that accepts a
    PluginRegistry instance.

    Returns a fully populated PluginRegistry.
    """
    registry = PluginRegistry()
    eps = importlib.metadata.entry_points(group="discuss_agent.plugins")
    if not eps:
        logger.warning(
            "No discuss_agent.plugins entry points found. "
            "Did you install your plugin package with 'pip install -e .'?"
        )
    for ep in eps:
        register_fn = ep.load()
        register_fn(registry)
    return registry
