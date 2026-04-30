"""AgentConversation — multi-turn conversation manager for a single agent."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
from typing import Any

import anthropic

logger = logging.getLogger(__name__)

# Maximum tool-use iterations to prevent infinite loops
_MAX_TOOL_ITERATIONS = 20


class AgentConversation:
    """Manage a single agent's multi-turn conversation via the Anthropic messages API.

    Maintains the full message history so the agent has context from previous rounds.
    Handles the tool-use loop: when Claude requests a tool call, the tool is executed
    and the result is sent back until Claude produces a final text response.
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
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.tools = tools or []
        self._tool_callables = tool_callables or {}
        self.messages: list[dict[str, Any]] = []

        client_kwargs: dict[str, Any] = {"timeout": 600.0}
        if api_key:
            client_kwargs["api_key"] = api_key
        if base_url:
            client_kwargs["base_url"] = base_url
        self._client = anthropic.AsyncAnthropic(**client_kwargs)

    def _build_api_kwargs(self) -> dict[str, Any]:
        """Build the kwargs dict for messages.create()."""
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
            # Convert result to string
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
        """Send a user message and get the assistant's final text response.

        Implements the tool-use loop:
        1. Send message to Claude
        2. If Claude returns tool_use blocks → execute tools → send results back
        3. Repeat until Claude returns end_turn (no more tool_use blocks)
        """
        self.messages.append({"role": "user", "content": user_message})

        for iteration in range(_MAX_TOOL_ITERATIONS):
            kwargs = self._build_api_kwargs()
            response = await self._client.messages.create(**kwargs)

            # Store the raw content blocks as the assistant message
            # (Anthropic API requires the full content array, not just text)
            self.messages.append({
                "role": "assistant",
                "content": [self._block_to_dict(b) for b in response.content],
            })

            # Check if there are any tool_use blocks
            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]

            if not tool_use_blocks:
                # No tool calls — extract text and return
                text_parts = [b.text for b in response.content if b.type == "text"]
                final_text = "\n".join(text_parts)
                logger.info(
                    "Agent '%s' responded (%d chars, %d messages, %d tool iterations)",
                    self.agent_name, len(final_text), len(self.messages), iteration,
                )
                return final_text

            # Execute all tool calls and send results back
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

            # Append tool results as a user message (Anthropic API convention)
            self.messages.append({"role": "user", "content": tool_results})

        # Safety: max iterations reached
        logger.warning(
            "Agent '%s' hit max tool iterations (%d)",
            self.agent_name, _MAX_TOOL_ITERATIONS,
        )
        # Extract whatever text we have from the last response
        text_parts = []
        if self.messages and self.messages[-1]["role"] == "assistant":
            content = self.messages[-1]["content"]
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(block["text"])
        return "\n".join(text_parts)

    @staticmethod
    def _block_to_dict(block) -> dict:
        """Convert an API content block to a plain dict for message storage."""
        if block.type == "text":
            return {"type": "text", "text": block.text}
        if block.type == "tool_use":
            return {
                "type": "tool_use",
                "id": block.id,
                "name": block.name,
                "input": block.input,
            }
        # Fallback for unknown block types
        return {"type": block.type}

    def get_history_length(self) -> int:
        """Return the number of messages in the conversation history."""
        return len(self.messages)
