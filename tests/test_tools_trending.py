"""Tests for the trending tool (stub)."""

import pytest

from discuss_agent.tools.trending import TrendingTools


class TestTrendingStub:
    """test_returns_degradation_message — verify stub returns graceful fallback."""

    def test_returns_degradation_message(self):
        tool = TrendingTools()
        result = tool.get_trending()

        assert isinstance(result, str)
        assert "热榜服务暂不可用" in result

    def test_returns_nonempty_string(self):
        tool = TrendingTools()
        result = tool.get_trending()

        assert len(result) > 0
