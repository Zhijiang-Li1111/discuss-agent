"""ContextManager — builds initial shared context and compresses history."""

from __future__ import annotations

from copy import deepcopy

from discuss_agent.config import DiscussionConfig
from discuss_agent.models import RoundRecord
from discuss_agent.tools.research_list import ResearchListTools
from discuss_agent.tools.published import PublishedTools
from discuss_agent.tools.trending import TrendingTools


class ContextManager:
    """Manages context assembly and compression for the discussion framework."""

    def __init__(self, config: DiscussionConfig):
        self._config = config
        self._research_list = ResearchListTools(research_dir=config.context.research_dir)
        self._published = PublishedTools()
        self._trending = TrendingTools()

    async def build_initial_context(self) -> str:
        """Build Round 0 shared context: published history + research titles + trending."""
        published = self._published.get_published(self._config.context.published_file)
        research = self._research_list.list_research(days=self._config.context.research_days)
        trending = self._trending.get_trending()
        return (
            f"## 已发布历史\n{published}\n\n"
            f"## 近期研报标题\n{research}\n\n"
            f"## 当前热榜\n{trending}"
        )

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
        from agno.models.anthropic import Claude

        agent = Agent(
            name="Compressor",
            model=Claude(id=self._config.model),
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
