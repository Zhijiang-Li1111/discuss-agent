"""Tests for discuss_agent.registry — PluginRegistry and load_plugins."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from discuss_agent.registry import PluginRegistry, load_plugins


# ---------------------------------------------------------------------------
# PluginRegistry
# ---------------------------------------------------------------------------


class TestRegisterTool:
    def test_register_and_get(self):
        registry = PluginRegistry()

        class FakeTool:
            pass

        registry.register_tool("my_tool", FakeTool)
        assert registry.get_tool_class("my_tool") is FakeTool

    def test_get_unknown_raises(self):
        registry = PluginRegistry()
        with pytest.raises(ValueError, match="Unknown tool: 'bad_name'"):
            registry.get_tool_class("bad_name")

    def test_error_message_lists_available(self):
        registry = PluginRegistry()
        registry.register_tool("alpha", type)
        registry.register_tool("beta", type)

        with pytest.raises(ValueError, match="Available:") as exc_info:
            registry.get_tool_class("gamma")

        msg = str(exc_info.value)
        assert "alpha" in msg
        assert "beta" in msg

    def test_overwrite_tool(self):
        registry = PluginRegistry()

        class ToolA:
            pass

        class ToolB:
            pass

        registry.register_tool("x", ToolA)
        registry.register_tool("x", ToolB)
        assert registry.get_tool_class("x") is ToolB


class TestRegisterContextBuilder:
    def test_register_and_get(self):
        registry = PluginRegistry()

        async def my_builder(ctx: dict) -> str:
            return "context"

        registry.register_context_builder(my_builder)
        assert registry.get_context_builder() is my_builder

    def test_default_is_none(self):
        registry = PluginRegistry()
        assert registry.get_context_builder() is None

    def test_last_registered_wins(self):
        registry = PluginRegistry()

        async def builder_a(ctx: dict) -> str:
            return "a"

        async def builder_b(ctx: dict) -> str:
            return "b"

        registry.register_context_builder(builder_a)
        registry.register_context_builder(builder_b)
        assert registry.get_context_builder() is builder_b


# ---------------------------------------------------------------------------
# load_plugins
# ---------------------------------------------------------------------------


class TestLoadPlugins:
    @patch("discuss_agent.registry.importlib.metadata.entry_points", return_value=[])
    def test_warns_when_no_plugins(self, mock_eps, caplog):
        with caplog.at_level(logging.WARNING, logger="discuss_agent.registry"):
            registry = load_plugins()

        assert "No discuss_agent.plugins entry points found" in caplog.text
        assert isinstance(registry, PluginRegistry)

    @patch("discuss_agent.registry.importlib.metadata.entry_points")
    def test_calls_register_function(self, mock_eps):
        called_with = []

        def fake_register(registry):
            called_with.append(registry)
            registry.register_tool("fake", type)

        ep = MagicMock()
        ep.load.return_value = fake_register
        mock_eps.return_value = [ep]

        registry = load_plugins()

        assert len(called_with) == 1
        assert called_with[0] is registry
        assert registry.get_tool_class("fake") is type

    @patch("discuss_agent.registry.importlib.metadata.entry_points")
    def test_multiple_plugins(self, mock_eps):
        def register_a(registry):
            registry.register_tool("tool_a", int)

        def register_b(registry):
            registry.register_tool("tool_b", str)

        ep_a = MagicMock()
        ep_a.load.return_value = register_a
        ep_b = MagicMock()
        ep_b.load.return_value = register_b
        mock_eps.return_value = [ep_a, ep_b]

        registry = load_plugins()

        assert registry.get_tool_class("tool_a") is int
        assert registry.get_tool_class("tool_b") is str

    @patch("discuss_agent.registry.importlib.metadata.entry_points")
    def test_propagates_register_exception(self, mock_eps):
        def bad_register(registry):
            raise RuntimeError("plugin broken")

        ep = MagicMock()
        ep.load.return_value = bad_register
        mock_eps.return_value = [ep]

        with pytest.raises(RuntimeError, match="plugin broken"):
            load_plugins()

    @patch("discuss_agent.registry.importlib.metadata.entry_points")
    def test_propagates_load_exception(self, mock_eps):
        ep = MagicMock()
        ep.load.side_effect = ImportError("module not found")
        mock_eps.return_value = [ep]

        with pytest.raises(ImportError, match="module not found"):
            load_plugins()
