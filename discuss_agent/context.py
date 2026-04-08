"""ContextManager — builds initial shared context and compresses history."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from copy import deepcopy

from discuss_agent.config import DiscussionConfig
from discuss_agent.models import RoundRecord

logger = logging.getLogger(__name__)


class ContextManager:
    """Manages context assembly and compression for the discussion framework."""

    def __init__(
        self,
        config: DiscussionConfig,
        context_builder: Callable[[dict], Awaitable[str]] | None = None,
    ):
        self._config = config
        self._context_builder = context_builder

    async def build_initial_context(self) -> str:
        """Build Round 0 shared context by delegating to the registered builder."""
        if self._context_builder:
            return await self._context_builder(self._config.context)
        logger.warning(
            "No context builder registered. Starting discussion with empty context."
        )
        return ""

    async def compress(
        self, history: list[RoundRecord], current_round: int
    ) -> list[RoundRecord]:
        """Compress old rounds when context gets large.

        Strategy:
        - Only compress when estimated tokens (total_chars / 4) > 120,000.
        - Keep the last 2 rounds (relative to *current_round*) in full.
        - Earlier rounds are compressed via an LLM agent.
        - If <= 2 rounds total, never compress.
        - On LLM failure, keep the round uncompressed (safe fallback).
        """
        if current_round <= 2:
            return history

        # Estimate total tokens from non-summary rounds
        total_chars = sum(
            sum(len(u.content) for u in r.expressions)
            + sum(len(u.content) for u in r.challenges)
            for r in history
            if not r.is_summary
        )
        if total_chars / 4 < 120_000:  # below 120K token threshold
            return history

        # Compress rounds older than current_round - 2
        threshold = current_round - 2
        compressed = deepcopy(history)
        for record in compressed:
            if record.round_num > threshold or record.is_summary:
                continue
            # Compress this round
            text = _format_round_for_compression(record)
            try:
                summary = await self._compress_round(text)
                record.is_summary = True
                record.summary_text = summary
                record.expressions = []
                record.challenges = []
            except Exception:
                pass  # Safe fallback: keep uncompressed

        return compressed

    async def _compress_round(self, round_text: str) -> str:
        """Compress a single round's text using an LLM agent."""
        from agno.agent import Agent
        from discuss_agent.config import build_claude

        agent = Agent(
            name="Compressor",
            model=build_claude(self._config.model_config),
            system_message=(
                "你是一个讨论记录压缩助手。"
                "将以下讨论轮次的发言浓缩为一段简洁的摘要，"
                "保留关键论点、证据和分歧。不要添加新信息。"
            ),
        )
        result = await agent.arun(input=round_text, stream=False)
        return result.content


def _format_round_for_compression(record: RoundRecord) -> str:
    """Format a RoundRecord into readable text for compression."""
    parts: list[str] = [f"=== 第 {record.round_num} 轮 ==="]
    if record.expressions:
        parts.append("【发言】")
        for u in record.expressions:
            parts.append(f"{u.agent_name}: {u.content}")
    if record.challenges:
        parts.append("【质疑】")
        for u in record.challenges:
            parts.append(f"{u.agent_name}: {u.content}")
    return "\n".join(parts)
