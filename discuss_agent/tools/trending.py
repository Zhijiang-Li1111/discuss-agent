"""Trending tool — fetches trending topics via MCP, with graceful degradation."""

from __future__ import annotations

import logging

from agno.tools import Toolkit

logger = logging.getLogger(__name__)

_DEGRADATION_MSG = "热榜服务暂不可用，请基于研报数据和自身判断进行讨论。"


class TrendingTools(Toolkit):
    """Toolkit for retrieving trending market topics and news headlines."""

    def __init__(self, mcp_url: str | None = None):
        super().__init__(name="trending")
        self._mcp_url = mcp_url

    def get_trending(self) -> str:
        """Fetch current trending financial topics and market headlines.

        Attempts to retrieve real-time trending topics from the NewsNow MCP
        Server. If the MCP server is unavailable or not configured, returns a
        graceful degradation message so that agents can continue discussing
        based on research report data.

        Returns:
            A string with trending topics, or a fallback message if the service
            is unavailable.
        """
        if not self._mcp_url:
            return _DEGRADATION_MSG

        try:
            # MCP integration placeholder — when NewsNow MCP server is deployed,
            # replace this with actual MCPTools call:
            #
            #   from agno.tools.mcp import MCPTools
            #   async with MCPTools(url=self._mcp_url) as mcp:
            #       return await mcp.call("get_trending")
            #
            return _DEGRADATION_MSG
        except Exception:
            logger.warning("Failed to connect to trending MCP server", exc_info=True)
            return _DEGRADATION_MSG
