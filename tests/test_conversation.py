"""Tests for AgentConversation tool execution loop."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from discuss_agent.conversation import AgentConversation


def _make_text_block(text: str):
    b = MagicMock()
    b.type = "text"
    b.text = text
    return b


def _make_tool_use_block(tool_id: str, name: str, input_args: dict):
    b = MagicMock()
    b.type = "tool_use"
    b.id = tool_id
    b.name = name
    b.input = input_args
    return b


def _make_response(*blocks, stop_reason="end_turn"):
    resp = MagicMock()
    resp.content = list(blocks)
    resp.stop_reason = stop_reason
    return resp


class TestConversationToolLoop:
    """Test that send() implements the tool-use loop correctly."""

    async def test_no_tools_returns_text(self):
        """When no tool_use blocks, text is returned directly."""
        conv = AgentConversation(
            agent_name="TestAgent",
            system_prompt="test",
            api_key="fake",
        )
        conv._client = MagicMock()
        conv._client.messages.create = AsyncMock(
            return_value=_make_response(
                _make_text_block("Hello world"),
            )
        )

        result = await conv.send("Hi")
        assert result == "Hello world"
        assert len(conv.messages) == 2  # user + assistant

    async def test_single_tool_call(self):
        """Claude calls one tool, gets result, then returns text."""
        def my_tool(query: str) -> str:
            return f"Result for: {query}"

        conv = AgentConversation(
            agent_name="TestAgent",
            system_prompt="test",
            api_key="fake",
            tools=[{"name": "search", "description": "Search", "input_schema": {}}],
            tool_callables={"search": my_tool},
        )
        conv._client = MagicMock()

        # First call: Claude wants to use a tool
        # Second call: Claude returns final text
        conv._client.messages.create = AsyncMock(side_effect=[
            _make_response(
                _make_tool_use_block("call_1", "search", {"query": "pig cycle"}),
            ),
            _make_response(
                _make_text_block("Based on my research: pig cycle is 4 years"),
            ),
        ])

        result = await conv.send("Research pig cycle")

        assert "pig cycle is 4 years" in result
        # Messages: user, assistant(tool_use), user(tool_result), assistant(text)
        assert len(conv.messages) == 4
        assert conv.messages[2]["role"] == "user"
        assert conv.messages[2]["content"][0]["type"] == "tool_result"
        assert "Result for: pig cycle" in conv.messages[2]["content"][0]["content"]

    async def test_multiple_tool_calls_in_one_turn(self):
        """Claude calls multiple tools in a single response."""
        call_log = []

        def tool_a(x: str) -> str:
            call_log.append(("a", x))
            return f"A:{x}"

        def tool_b(y: str) -> str:
            call_log.append(("b", y))
            return f"B:{y}"

        conv = AgentConversation(
            agent_name="TestAgent",
            system_prompt="test",
            api_key="fake",
            tools=[
                {"name": "tool_a", "description": "A", "input_schema": {}},
                {"name": "tool_b", "description": "B", "input_schema": {}},
            ],
            tool_callables={"tool_a": tool_a, "tool_b": tool_b},
        )
        conv._client = MagicMock()

        conv._client.messages.create = AsyncMock(side_effect=[
            _make_response(
                _make_tool_use_block("c1", "tool_a", {"x": "hello"}),
                _make_tool_use_block("c2", "tool_b", {"y": "world"}),
            ),
            _make_response(
                _make_text_block("Combined result"),
            ),
        ])

        result = await conv.send("Do both")
        assert result == "Combined result"
        assert len(call_log) == 2
        # Tool results are sent as a single user message with two entries
        tool_result_msg = conv.messages[2]
        assert len(tool_result_msg["content"]) == 2

    async def test_multi_iteration_tool_loop(self):
        """Claude calls tools across multiple iterations."""
        calls = {"n": 0}

        def search(q: str) -> str:
            calls["n"] += 1
            return f"result_{calls['n']}"

        conv = AgentConversation(
            agent_name="TestAgent",
            system_prompt="test",
            api_key="fake",
            tools=[{"name": "search", "description": "S", "input_schema": {}}],
            tool_callables={"search": search},
        )
        conv._client = MagicMock()

        # 3 iterations: tool → tool → text
        conv._client.messages.create = AsyncMock(side_effect=[
            _make_response(_make_tool_use_block("c1", "search", {"q": "a"})),
            _make_response(_make_tool_use_block("c2", "search", {"q": "b"})),
            _make_response(_make_text_block("Final answer")),
        ])

        result = await conv.send("Deep research")
        assert result == "Final answer"
        assert calls["n"] == 2

    async def test_async_tool(self):
        """Async tool functions are awaited correctly."""
        async def async_search(query: str) -> str:
            return f"async result for {query}"

        conv = AgentConversation(
            agent_name="TestAgent",
            system_prompt="test",
            api_key="fake",
            tools=[{"name": "search", "description": "S", "input_schema": {}}],
            tool_callables={"search": async_search},
        )
        conv._client = MagicMock()

        conv._client.messages.create = AsyncMock(side_effect=[
            _make_response(_make_tool_use_block("c1", "search", {"query": "test"})),
            _make_response(_make_text_block("Done")),
        ])

        result = await conv.send("Search")
        assert result == "Done"
        # Verify the tool result was sent back
        assert "async result for test" in conv.messages[2]["content"][0]["content"]

    async def test_unknown_tool_returns_error(self):
        """Unknown tool names produce an error result."""
        conv = AgentConversation(
            agent_name="TestAgent",
            system_prompt="test",
            api_key="fake",
            tools=[{"name": "known", "description": "K", "input_schema": {}}],
            tool_callables={},
        )
        conv._client = MagicMock()

        conv._client.messages.create = AsyncMock(side_effect=[
            _make_response(_make_tool_use_block("c1", "unknown_tool", {})),
            _make_response(_make_text_block("OK")),
        ])

        result = await conv.send("test")
        assert result == "OK"
        tool_result = conv.messages[2]["content"][0]["content"]
        assert "Unknown tool" in tool_result

    async def test_tool_exception_returns_error(self):
        """Tool that raises an exception sends error back to Claude."""
        def broken_tool() -> str:
            raise RuntimeError("DB connection failed")

        conv = AgentConversation(
            agent_name="TestAgent",
            system_prompt="test",
            api_key="fake",
            tools=[{"name": "broken", "description": "B", "input_schema": {}}],
            tool_callables={"broken": broken_tool},
        )
        conv._client = MagicMock()

        conv._client.messages.create = AsyncMock(side_effect=[
            _make_response(_make_tool_use_block("c1", "broken", {})),
            _make_response(_make_text_block("Tool failed, here is my answer anyway")),
        ])

        result = await conv.send("test")
        assert "my answer anyway" in result
        tool_result = conv.messages[2]["content"][0]["content"]
        assert "DB connection failed" in tool_result

    async def test_text_and_tool_use_in_same_response(self):
        """Claude returns both text and tool_use in the same response."""
        def search(q: str) -> str:
            return "data"

        conv = AgentConversation(
            agent_name="TestAgent",
            system_prompt="test",
            api_key="fake",
            tools=[{"name": "search", "description": "S", "input_schema": {}}],
            tool_callables={"search": search},
        )
        conv._client = MagicMock()

        # First response has both text and tool_use
        conv._client.messages.create = AsyncMock(side_effect=[
            _make_response(
                _make_text_block("Let me search for that"),
                _make_tool_use_block("c1", "search", {"q": "test"}),
            ),
            _make_response(
                _make_text_block("Based on the search: final answer"),
            ),
        ])

        result = await conv.send("find info")
        assert result == "Based on the search: final answer"

    async def test_message_history_preserved(self):
        """Multi-turn: second call preserves first call's history."""
        conv = AgentConversation(
            agent_name="TestAgent",
            system_prompt="test",
            api_key="fake",
        )
        conv._client = MagicMock()

        conv._client.messages.create = AsyncMock(
            return_value=_make_response(_make_text_block("reply"))
        )

        await conv.send("msg1")
        await conv.send("msg2")

        assert len(conv.messages) == 4  # user, assistant, user, assistant
        assert conv.messages[0]["content"] == "msg1"
        assert conv.messages[2]["content"] == "msg2"
