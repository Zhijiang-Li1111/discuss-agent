"""Tests for the ContextManager."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch

from discuss_agent.config import DiscussionConfig, AgentConfig, HostConfig, ModelConfig
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
        model_config=ModelConfig(model="claude-sonnet-4-20250514"),
        agents=[
            AgentConfig(name="A", system_prompt="prompt A"),
            AgentConfig(name="B", system_prompt="prompt B"),
        ],
        host=HostConfig(
            convergence_prompt="converge",
            summary_prompt="summarize",
        ),
        tools=[],
        context={"research_dir": "~/ima-downloads/", "published_file": "PUBLISHED.md", "research_days": 2},
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
# build_initial_context — pluggable context builder
# ===========================================================================


class TestBuildInitialContext:
    """Tests for ContextManager.build_initial_context with pluggable builder."""

    @pytest.mark.asyncio
    async def test_calls_context_builder(self):
        """When a context_builder is provided, it should be called with config.context."""
        config = _make_config()
        mock_builder = AsyncMock(return_value="built context")
        mgr = ContextManager(config, context_builder=mock_builder)

        result = await mgr.build_initial_context()

        mock_builder.assert_called_once_with(config.context)
        assert result == "built context"

    @pytest.mark.asyncio
    async def test_no_builder_returns_empty(self):
        """When no context_builder is provided, should return empty string."""
        config = _make_config()
        mgr = ContextManager(config, context_builder=None)

        result = await mgr.build_initial_context()

        assert result == ""

    @pytest.mark.asyncio
    async def test_no_builder_default(self):
        """context_builder defaults to None when not provided."""
        config = _make_config()
        mgr = ContextManager(config)

        result = await mgr.build_initial_context()

        assert result == ""


# ===========================================================================
# compress
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
