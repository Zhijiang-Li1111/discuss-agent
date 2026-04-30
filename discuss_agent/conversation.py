"""AgentConversation — multi-turn conversation manager for a single agent."""

from __future__ import annotations

import logging
from typing import Any

import anthropic

logger = logging.getLogger(__name__)


class AgentConversation:
    """Manage a single agent's multi-turn conversation via the Anthropic messages API.

    Maintains the full message history so the agent has context from previous rounds.
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
    ):
        self.agent_name = agent_name
        self.system_prompt = system_prompt
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.tools = tools or []
        self.messages: list[dict[str, Any]] = []

        client_kwargs: dict[str, Any] = {"timeout": 600.0}
        if api_key:
            client_kwargs["api_key"] = api_key
        if base_url:
            client_kwargs["base_url"] = base_url
        self._client = anthropic.AsyncAnthropic(**client_kwargs)

    async def send(self, user_message: str) -> str:
        """Send a user message and get the assistant's response.

        Appends both to the message history for multi-turn context.
        """
        self.messages.append({"role": "user", "content": user_message})

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

        response = await self._client.messages.create(**kwargs)

        # Extract text content from response
        text_parts = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)

        assistant_text = "\n".join(text_parts)
        self.messages.append({"role": "assistant", "content": assistant_text})

        logger.info(
            "Agent '%s' responded (%d chars, %d messages in history)",
            self.agent_name,
            len(assistant_text),
            len(self.messages),
        )
        return assistant_text

    def get_history_length(self) -> int:
        """Return the number of messages in the conversation history."""
        return len(self.messages)
