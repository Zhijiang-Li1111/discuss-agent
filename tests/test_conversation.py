"""Tests for AgentConversation tool execution loop."""

from __future__ import annotations

import base64
import json
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from agno.media import Image
from agno.tools.function import ToolResult
from PIL import Image as PILImage

import discuss_agent.conversation as conversation_module
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


def _make_openai_response(content=None, tool_calls=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls or [])
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _make_openai_tool_call(call_id: str, name: str, arguments: dict):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


class TestOpenAIConversation:
    """OpenAI-compatible messages and function-tool loop."""

    async def test_no_tools_returns_text_and_preserves_multiturn_history(self):
        conv = AgentConversation(
            agent_name="OpenAI-Agent",
            system_prompt="system",
            model="agent-maestro-openai/gpt-5.5",
            api_key="dummy",
            base_url="http://localhost:23333/api/anthropic",
        )
        conv._client = MagicMock()
        conv._client.chat.completions.create = AsyncMock(side_effect=[
            _make_openai_response("first reply"),
            _make_openai_response("second reply"),
        ])

        assert await conv.send("first") == "first reply"
        assert await conv.send("second") == "second reply"

        second_kwargs = conv._client.chat.completions.create.await_args_list[1].kwargs
        assert second_kwargs["messages"] == [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "first reply"},
            {"role": "user", "content": "second"},
        ]

    async def test_tool_loop_converts_schema_and_appends_tool_messages(self):
        def search(query: str) -> dict:
            return {"answer": f"found {query}"}

        conv = AgentConversation(
            agent_name="OpenAI-Agent",
            system_prompt="system",
            model="agent-maestro-openai/gpt-5.5",
            api_key="dummy",
            base_url="http://localhost:23333/api/anthropic",
            tools=[{
                "name": "search",
                "description": "Search reports",
                "input_schema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            }],
            tool_callables={"search": search},
        )
        conv._client = MagicMock()
        conv._client.chat.completions.create = AsyncMock(side_effect=[
            _make_openai_response(
                tool_calls=[_make_openai_tool_call("call_1", "search", {"query": "DRAM"})]
            ),
            _make_openai_response("final answer"),
        ])

        assert await conv.send("research") == "final answer"

        first_kwargs = conv._client.chat.completions.create.await_args_list[0].kwargs
        assert first_kwargs["tools"] == [{
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search reports",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        }]

        assert conv.messages[1]["role"] == "assistant"
        assert conv.messages[1]["tool_calls"][0]["id"] == "call_1"
        assert conv.messages[2] == {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": '{"answer": "found DRAM"}',
        }

        second_kwargs = conv._client.chat.completions.create.await_args_list[1].kwargs
        assert second_kwargs["messages"][0] == {"role": "system", "content": "system"}
        assert second_kwargs["messages"][3]["role"] == "tool"


class TestToolResultMedia:
    @staticmethod
    def _image_bytes(fmt="PNG"):
        output = BytesIO()
        PILImage.new("RGB", (4, 3), "white").save(output, format=fmt)
        return output.getvalue()

    async def test_anthropic_tool_result_contains_image(self):
        png = self._image_bytes()
        conv = AgentConversation(agent_name="media", system_prompt="system", api_key="dummy",
            tools=[{"name":"page","description":"page","input_schema":{}}],
            tool_callables={"page": lambda: ToolResult(content='{"page":68}', images=[Image(content=png)])})
        conv._client = MagicMock()
        conv._client.messages.create = AsyncMock(side_effect=[
            _make_response(_make_tool_use_block("c1","page",{})), _make_response(_make_text_block("done"))])
        assert await conv.send("read") == "done"
        payload = conv.messages[2]["content"][0]["content"]
        assert payload[1]["type"] == "image"
        assert base64.b64decode(payload[1]["source"]["data"]) == png

    async def test_openai_batches_tools_before_image(self):
        jpeg = self._image_bytes("JPEG")
        conv = AgentConversation(agent_name="media", system_prompt="system",
            model="agent-maestro-openai/gpt-5.5", api_key="dummy",
            tools=[{"name":"page","description":"page","input_schema":{}},{"name":"text","description":"text","input_schema":{}}],
            tool_callables={"page": lambda: ToolResult(content="page", images=[Image(content=jpeg)]), "text": lambda:"plain"})
        conv._client = MagicMock()
        conv._client.chat.completions.create = AsyncMock(side_effect=[
            _make_openai_response(tool_calls=[_make_openai_tool_call("c1","page",{}),_make_openai_tool_call("c2","text",{})]),
            _make_openai_response("done")])
        assert await conv.send("read") == "done"
        assert [m["role"] for m in conv.messages[1:5]] == ["assistant","tool","tool","user"]
        assert conv.messages[4]["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,")

    @pytest.mark.parametrize("format", ["PNG", "JPEG"])
    async def test_truncated_real_image_visible_error(self, format):
        payload = self._image_bytes(format)[:-12]
        conv = AgentConversation(agent_name="media", system_prompt="system", api_key="dummy",
            tools=[{"name":"page","description":"page","input_schema":{}}],
            tool_callables={"page": lambda: ToolResult(content="page", images=[Image(content=payload)])})
        conv._client = MagicMock()
        conv._client.messages.create = AsyncMock(side_effect=[
            _make_response(_make_tool_use_block("c1","page",{})), _make_response(_make_text_block("handled"))])
        assert await conv.send("read") == "handled"
        text = conv.messages[2]["content"][0]["content"]
        assert "complete valid PNG or JPEG" in text

    async def test_legacy_results_and_local_filepath(self, tmp_path):
        png = self._image_bytes()
        path = tmp_path / "page.png"
        path.write_bytes(png)
        string_conv = AgentConversation(agent_name="media", system_prompt="system", api_key="dummy",
            tool_callables={"tool": lambda: "plain"})
        dict_conv = AgentConversation(agent_name="media", system_prompt="system", api_key="dummy",
            tool_callables={"tool": lambda: {"answer": 42}})
        file_conv = AgentConversation(agent_name="media", system_prompt="system", api_key="dummy",
            tool_callables={"tool": lambda: ToolResult(content="page", images=[Image(filepath=path)])})

        assert (await string_conv._execute_tool("tool", {})).text == "plain"
        assert (await dict_conv._execute_tool("tool", {})).text == '{"answer": 42}'
        assert (await file_conv._execute_tool("tool", {})).images[0].data == png

    @pytest.mark.parametrize(
        ("images", "error"),
        [
            ([Image(url="https://example.com/image.png")], "remote image URLs"),
            ([Image(content=b"x" * (5 * 1024 * 1024 + 1))], "5 MiB"),
            ([Image(content=b"unused") for _ in range(4)], "3 image"),
        ],
    )
    async def test_rejected_media_is_visible_and_not_forwarded(self, images, error):
        conv = AgentConversation(agent_name="media", system_prompt="system", api_key="dummy",
            tool_callables={"tool": lambda: ToolResult(content="page", images=images)})

        result = await conv._execute_tool("tool", {})

        assert result.images == ()
        assert error in result.text

    async def test_corrupt_real_images_with_terminal_markers_are_rejected(self):
        for format in ("PNG", "JPEG"):
            original = self._image_bytes(format)
            trailer_size = 12 if format == "PNG" else 2
            payload = original[:20] + original[-trailer_size:]
            conv = AgentConversation(agent_name="media", system_prompt="system", api_key="dummy",
                tool_callables={"tool": lambda data=bytes(payload): ToolResult(
                    content="page", images=[Image(content=data)]
                )})

            result = await conv._execute_tool("tool", {})

            assert result.images == ()
            assert "complete valid PNG or JPEG" in result.text

    async def test_multiple_tool_results_cannot_exceed_turn_image_limit(self):
        png = self._image_bytes()
        conv = AgentConversation(agent_name="media", system_prompt="system", api_key="dummy",
            tools=[{"name":"a","description":"a","input_schema":{}},{"name":"b","description":"b","input_schema":{}}],
            tool_callables={
                "a": lambda: ToolResult(content="a", images=[Image(content=png), Image(content=png)]),
                "b": lambda: ToolResult(content="b", images=[Image(content=png), Image(content=png)]),
            })
        conv._client = MagicMock()
        conv._client.messages.create = AsyncMock(side_effect=[
            _make_response(_make_tool_use_block("c1", "a", {}), _make_tool_use_block("c2", "b", {})),
            _make_response(_make_text_block("handled")),
        ])

        assert await conv.send("read") == "handled"
        results = conv.messages[2]["content"]
        assert sum(
            block["type"] == "image"
            for result in results
            for block in result["content"]
            if isinstance(result["content"], list)
        ) <= 3
        assert any("3 image" in str(result["content"]) for result in results)

    async def test_anthropic_image_only_result_omits_empty_text_block(self):
        png = self._image_bytes()
        conv = AgentConversation(agent_name="media", system_prompt="system", api_key="dummy",
            tools=[{"name":"page","description":"page","input_schema":{}}],
            tool_callables={"page": lambda: ToolResult(content="", images=[Image(content=png)])})
        conv._client = MagicMock()
        conv._client.messages.create = AsyncMock(side_effect=[
            _make_response(_make_tool_use_block("c1", "page", {})),
            _make_response(_make_text_block("seen")),
        ])

        assert await conv.send("read") == "seen"
        content = conv.messages[2]["content"][0]["content"]
        assert [block["type"] for block in content] == ["image"]

    async def test_async_callable_object_is_awaited(self):
        class AsyncTool:
            async def __call__(self):
                return "awaited"

        conv = AgentConversation(agent_name="media", system_prompt="system", api_key="dummy",
            tool_callables={"tool": AsyncTool()})

        assert (await conv._execute_tool("tool", {})).text == "awaited"

    async def test_decompression_bomb_returns_visible_error(self, monkeypatch):
        png = self._image_bytes()
        monkeypatch.setattr(conversation_module, "_MAX_TOOL_IMAGE_PIXELS", 1)
        conv = AgentConversation(agent_name="media", system_prompt="system", api_key="dummy",
            tool_callables={"tool": lambda: ToolResult(content="page", images=[Image(content=png)])})

        result = await conv._execute_tool("tool", {})

        assert result.images == ()
        assert "safe pixel limit" in result.text

    async def test_openai_multiple_tools_enforce_turn_image_limit(self):
        png = self._image_bytes()
        conv = AgentConversation(agent_name="media", system_prompt="system",
            model="agent-maestro-openai/gpt-5.5", api_key="dummy",
            tools=[{"name":"a","description":"a","input_schema":{}},{"name":"b","description":"b","input_schema":{}}],
            tool_callables={
                "a": lambda: ToolResult(content="a", images=[Image(content=png), Image(content=png)]),
                "b": lambda: ToolResult(content="b", images=[Image(content=png), Image(content=png)]),
            })
        conv._client = MagicMock()
        conv._client.chat.completions.create = AsyncMock(side_effect=[
            _make_openai_response(tool_calls=[
                _make_openai_tool_call("c1", "a", {}), _make_openai_tool_call("c2", "b", {})
            ]),
            _make_openai_response("handled"),
        ])

        assert await conv.send("read") == "handled"
        assert [message["role"] for message in conv.messages[1:5]] == [
            "assistant", "tool", "tool", "user"
        ]
        assert "3 image" in conv.messages[3]["content"]
        assert len(conv.messages[4]["content"]) == 3

    async def test_oversized_filepath_rejected_before_read(self, tmp_path, monkeypatch):
        path = tmp_path / "large.png"
        path.write_bytes(b"x" * (5 * 1024 * 1024 + 1))
        read_bytes = Path.read_bytes

        def guarded_read(candidate):
            if candidate == path:
                raise AssertionError("oversized filepath was read")
            return read_bytes(candidate)

        monkeypatch.setattr(Path, "read_bytes", guarded_read)
        conv = AgentConversation(agent_name="media", system_prompt="system", api_key="dummy",
            tool_callables={"tool": lambda: ToolResult(content="page", images=[Image(filepath=path)])})

        result = await conv._execute_tool("tool", {})

        assert result.images == ()
        assert "5 MiB" in result.text

    async def test_openai_media_history_is_preserved_on_next_turn(self):
        png = self._image_bytes()
        conv = AgentConversation(agent_name="media", system_prompt="system",
            model="agent-maestro-openai/gpt-5.5", api_key="dummy",
            tools=[{"name":"page","description":"page","input_schema":{}}],
            tool_callables={"page": lambda: ToolResult(content="page", images=[Image(content=png)])})
        conv._client = MagicMock()
        conv._client.chat.completions.create = AsyncMock(side_effect=[
            _make_openai_response(tool_calls=[_make_openai_tool_call("c1", "page", {})]),
            _make_openai_response("seen"),
            _make_openai_response("remembered"),
        ])

        assert await conv.send("read") == "seen"
        assert await conv.send("recall") == "remembered"
        third_messages = conv._client.chat.completions.create.await_args_list[2].kwargs["messages"]
        assert third_messages[3:6] == conv.messages[2:5]
