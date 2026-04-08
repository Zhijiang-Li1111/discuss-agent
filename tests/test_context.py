"""Tests for the ContextManager (Tasks 11–12)."""

from __future__ import annotations

import pytest
from unittest.mock import patch, AsyncMock

from discuss_agent.config import DiscussionConfig, AgentConfig, HostConfig, ContextConfig
from discuss_agent.models import RoundRecord, AgentUtterance
from discuss_agent.context import ContextManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config() -> DiscussionConfig:
    """Return a minimal DiscussionConfig for testing."""
    return DiscussionConfig(
        min_rounds=2,
        max_rounds=5,
        model="claude-sonnet-4-20250514",
        agents=[
            AgentConfig(name="A", system_prompt="prompt A"),
            AgentConfig(name="B", system_prompt="prompt B"),
        ],
        host=HostConfig(
            convergence_prompt="converge",
            summary_prompt="summarize",
        ),
        tools=["research_list", "published", "trending"],
        context=ContextConfig(
            research_dir="~/ima-downloads/",
            published_file="PUBLISHED.md",
            research_days=2,
        ),
    )


def _make_round(round_num: int, content_size: int = 100) -> RoundRecord:
    """Create a RoundRecord with expressions/challenges of a given content size."""
    text = "x" * content_size
    return RoundRecord(
        round_num=round_num,
        expressions=[AgentUtterance(agent_name="A", content=text)],
        challenges=[AgentUtterance(agent_name="B", content=text)],
    )


# ===========================================================================
# Task 11 — build_initial_context
# ===========================================================================


class TestBuildInitialContext:
    """Tests for ContextManager.build_initial_context."""

    @pytest.mark.asyncio
    async def test_build_initial_context_contains_sections(self):
        """Mock the three tool methods and verify the result contains all sections."""
        config = _make_config()
        mgr = ContextManager(config)

        with (
            patch.object(mgr._published, "get_published", return_value="pub-data-here"),
            patch.object(mgr._research_list, "list_research", return_value="research-titles-here"),
            patch.object(mgr._trending, "get_trending", return_value="trending-data-here"),
        ):
            result = await mgr.build_initial_context()

        # All three section headers must be present
        assert "已发布历史" in result
        assert "近期研报标题" in result
        assert "当前热榜" in result

        # The mocked content must appear in the output
        assert "pub-data-here" in result
        assert "research-titles-here" in result
        assert "trending-data-here" in result

    @pytest.mark.asyncio
    async def test_initial_context_metadata_only(self):
        """The returned context should contain title-level data, not PDF content."""
        config = _make_config()
        mgr = ContextManager(config)

        title_data = "20260407-GoldmanSachs-半导体行业深度"
        with (
            patch.object(mgr._published, "get_published", return_value="some published"),
            patch.object(mgr._research_list, "list_research", return_value=title_data),
            patch.object(mgr._trending, "get_trending", return_value="热榜条目"),
        ):
            result = await mgr.build_initial_context()

        # Should contain the title-level string we returned
        assert title_data in result
        # Should NOT contain any raw PDF content markers
        assert "%PDF" not in result
        assert "stream\n" not in result


# ===========================================================================
# Task 12 — compress
# ===========================================================================


class TestCompressNoOp:
    """Cases where compress should return history unchanged."""

    @pytest.mark.asyncio
    async def test_no_compression_when_few_rounds(self):
        """2 rounds with small content -> returned unchanged."""
        config = _make_config()
        mgr = ContextManager(config)

        history = [_make_round(1, 100), _make_round(2, 100)]
        result = await mgr.compress(history, current_round=2)

        assert result is history
        for r in result:
            assert r.is_summary is False
            assert r.summary_text is None

    @pytest.mark.asyncio
    async def test_no_compression_when_below_threshold(self):
        """4 rounds but total chars < 480K -> no compression."""
        config = _make_config()
        mgr = ContextManager(config)

        # Each round: 2 utterances * 1000 chars = 2000 chars per round
        # 4 rounds: 8000 chars total. Well below 480K.
        history = [_make_round(i, 1000) for i in range(1, 5)]
        result = await mgr.compress(history, current_round=4)

        assert result is history
        for r in result:
            assert r.is_summary is False


class TestCompressOverThreshold:
    """Cases where compress should actually compress old rounds."""

    @pytest.mark.asyncio
    async def test_compresses_old_rounds_when_over_threshold(self):
        """4 rounds with total chars > 480K. Old rounds get compressed."""
        config = _make_config()
        mgr = ContextManager(config)

        # 4 rounds, each with 150K chars in expressions + 150K in challenges
        # -> total non-summary chars = 4 * (150K + 150K) = 1.2M >> 480K
        history = [_make_round(i, 150_000) for i in range(1, 5)]

        mock_summary = "compressed summary of round"

        with patch.object(
            mgr, "_compress_round", new_callable=AsyncMock, return_value=mock_summary
        ):
            result = await mgr.compress(history, current_round=4)

        # Rounds 1 and 2 should be compressed (below threshold of current_round - 2 = 2)
        # Rounds 3 and 4 (the last 2) remain full.
        for r in result:
            if r.round_num <= 2:  # rounds 1 and 2
                assert r.is_summary is True
                assert r.summary_text == mock_summary
                assert r.expressions == []
                assert r.challenges == []
            else:  # rounds 3 and 4
                assert r.is_summary is False
                assert r.summary_text is None
                assert len(r.expressions) > 0

    @pytest.mark.asyncio
    async def test_compression_failure_keeps_round(self):
        """If _compress_round raises, the round stays uncompressed."""
        config = _make_config()
        mgr = ContextManager(config)

        history = [_make_round(i, 150_000) for i in range(1, 5)]

        with patch.object(
            mgr, "_compress_round", new_callable=AsyncMock, side_effect=RuntimeError("LLM down")
        ):
            result = await mgr.compress(history, current_round=4)

        # All rounds should remain uncompressed due to the error
        for r in result:
            assert r.is_summary is False
            assert len(r.expressions) > 0
