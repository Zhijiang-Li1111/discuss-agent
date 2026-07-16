"""AgentConversation — provider-aware multi-turn conversation manager."""

from __future__ import annotations

import inspect
import json
import logging
from typing import Any

import anthropic
import openai

from discuss_agent.config import infer_provider, normalize_base_url

logger = logging.getLogger(__name__)

# Maximum tool-use iterations to prevent infinite loops
_MAX_TOOL_ITERATIONS = 20


class AgentConversation:
    """Manage a single agent's multi-turn conversation and tool-use loop.

    Claude models use Anthropic Messages while ``agent-maestro-openai/*``
    models use the OpenAI-compatible Chat Completions protocol.  ``messages``
    stores provider-native history, allowing either protocol to preserve
    context across discussion rounds.
    """

    def __init__(
        self,
        agent_name: str,
        system_prompt: str,
        model: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 4096,
        temperature: float | None = None,
        tools: list[dict] | None = None,
        tool_callables: dict[str, callable] | None = None,
    ):
        self.agent_name = agent_name
        self.system_prompt = system_prompt
        self.model = model
        self.provider = infer_provider(model)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.tools = tools or []
        self._tool_callables = tool_callables or {}
        self.messages: list[dict[str, Any]] = []

        client_kwargs: dict[str, Any] = {"timeout": 600.0}
        if api_key:
            client_kwargs["api_key"] = api_key
        normalized_url = normalize_base_url(base_url, self.provider)
        if normalized_url:
            client_kwargs["base_url"] = normalized_url
        if self.provider == "openai":
            self._client = openai.AsyncOpenAI(**client_kwargs)
        else:
            self._client = anthropic.AsyncAnthropic(**client_kwargs)

    def _build_anthropic_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": self.system_prompt,
            "messages": self.messages,
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.tools:
            kwargs["tools"] = self.tools
        return kwargs

    def _build_openai_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                *self.messages,
            ],
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.tools:
            kwargs["tools"] = [self._to_openai_tool(tool) for tool in self.tools]
        return kwargs

    # Kept for callers/tests that inspected the old Anthropic helper.
    def _build_api_kwargs(self) -> dict[str, Any]:
        if self.provider == "openai":
            return self._build_openai_kwargs()
        return self._build_anthropic_kwargs()

    @staticmethod
    def _to_openai_tool(tool: dict) -> dict:
        """Convert the framework's Anthropic tool schema to function tools."""
        return {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get(
                    "input_schema", {"type": "object", "properties": {}}
                ),
            },
        }

    async def _execute_tool(self, name: str, input_args: dict) -> str:
        """Execute a tool by name and return its result as a string."""
        fn = self._tool_callables.get(name)
        if fn is None:
            return json.dumps({"error": f"Unknown tool: {name}"})
        try:
            if inspect.iscoroutinefunction(fn):
                result = await fn(**input_args)
            else:
                result = fn(**input_args)
            if isinstance(result, str):
                return result
            return json.dumps(result, ensure_ascii=False, default=str)
        except Exception as exc:
            logger.warning(
                "Tool '%s' failed for agent '%s': %s",
                name, self.agent_name, exc,
            )
            return json.dumps({"error": str(exc)})

    async def send(self, user_message: str) -> str:
        """Send a user message and return the final text after any tool loop."""
        self.messages.append({"role": "user", "content": user_message})
        if self.provider == "openai":
            return await self._send_openai()
        return await self._send_anthropic()

    async def _send_anthropic(self) -> str:
        for iteration in range(_MAX_TOOL_ITERATIONS):
            response = await self._client.messages.create(
                **self._build_anthropic_kwargs()
            )
            self.messages.append({
                "role": "assistant",
                "content": [self._block_to_dict(b) for b in response.content],
            })
            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
            if not tool_use_blocks:
                final_text = "\n".join(
                    b.text for b in response.content if b.type == "text"
                )
                self._log_response(final_text, iteration)
                return final_text

            tool_results = []
            for block in tool_use_blocks:
                logger.info(
                    "Agent '%s' calling tool '%s' with args: %s",
                    self.agent_name, block.name, block.input,
                )
                result_str = await self._execute_tool(block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_str,
                })
            self.messages.append({"role": "user", "content": tool_results})

        return self._max_iterations_text()

    async def _send_openai(self) -> str:
        for iteration in range(_MAX_TOOL_ITERATIONS):
            response = await self._client.chat.completions.create(
                **self._build_openai_kwargs()
            )
            message = response.choices[0].message
            tool_calls = message.tool_calls or []
            assistant_message: dict[str, Any] = {
                "role": "assistant",
                "content": message.content,
            }
            if tool_calls:
                assistant_message["tool_calls"] = [
                    self._openai_tool_call_to_dict(call) for call in tool_calls
                ]
            self.messages.append(assistant_message)

            if not tool_calls:
                final_text = message.content or ""
                self._log_response(final_text, iteration)
                return final_text

            for call in tool_calls:
                name = call.function.name
                try:
                    input_args = json.loads(call.function.arguments or "{}")
                    if not isinstance(input_args, dict):
                        raise ValueError("tool arguments must be a JSON object")
                except (json.JSONDecodeError, ValueError) as exc:
                    result_str = json.dumps({"error": f"Invalid tool arguments: {exc}"})
                else:
                    logger.info(
                        "Agent '%s' calling tool '%s' with args: %s",
                        self.agent_name, name, input_args,
                    )
                    result_str = await self._execute_tool(name, input_args)
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": result_str,
                })

        return self._max_iterations_text()

    def _max_iterations_text(self) -> str:
        logger.warning(
            "Agent '%s' hit max tool iterations (%d)",
            self.agent_name, _MAX_TOOL_ITERATIONS,
        )
        if not self.messages or self.messages[-1]["role"] != "assistant":
            return ""
        content = self.messages[-1].get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(
                block["text"] for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
        return ""

    def _log_response(self, text: str, iteration: int) -> None:
        logger.info(
            "Agent '%s' responded (%d chars, %d messages, %d tool iterations)",
            self.agent_name, len(text), len(self.messages), iteration,
        )

    @staticmethod
    def _openai_tool_call_to_dict(call) -> dict:
        return {
            "id": call.id,
            "type": getattr(call, "type", "function"),
            "function": {
                "name": call.function.name,
                "arguments": call.function.arguments,
            },
        }

    @staticmethod
    def _block_to_dict(block) -> dict:
        """Convert an Anthropic content block to a plain dict."""
        if block.type == "text":
            return {"type": "text", "text": block.text}
        if block.type == "tool_use":
            return {
                "type": "tool_use",
                "id": block.id,
                "name": block.name,
                "input": block.input,
            }
        return {"type": block.type}

    def get_history_length(self) -> int:
        return len(self.messages)
