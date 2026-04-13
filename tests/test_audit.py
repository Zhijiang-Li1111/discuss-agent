"""Tests for per-agent audit log (discuss_agent.audit + engine integration)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from discuss_agent.audit import AuditLogger, _now_iso, _truncate, _summarize_args


# -----------------------------------------------------------------------
# AuditLogger unit tests
# -----------------------------------------------------------------------


class TestAuditLoggerUnit:
    def test_creates_audit_dir(self, tmp_path):
        audit = AuditLogger(str(tmp_path))
        assert (tmp_path / "audit").is_dir()
        audit.close()

    def test_log_call_start(self, tmp_path):
        audit = AuditLogger(str(tmp_path))
        audit.log_call_start("Bull", "Market is going up")
        audit.close()

        log_file = tmp_path / "audit" / "Bull.jsonl"
        assert log_file.exists()
        event = json.loads(log_file.read_text().strip())
        assert event["event"] == "call_start"
        assert event["agent"] == "Bull"

    def test_log_call_start_with_extras(self, tmp_path):
        audit = AuditLogger(str(tmp_path))
        audit.log_call_start(
            "Bull", "Market is going up",
            system_prompt_size_chars=400,
            system_prompt_size_tokens_est=100,
            skill_tools_loaded=["web_search", "calculator"],
        )
        audit.close()

        event = json.loads((tmp_path / "audit" / "Bull.jsonl").read_text().strip())
        assert event["system_prompt_size_chars"] == 400
        assert event["system_prompt_size_tokens_est"] == 100
        assert event["skill_tools_loaded"] == ["web_search", "calculator"]

    def test_log_tool_call(self, tmp_path):
        audit = AuditLogger(str(tmp_path))
        audit.log_tool_call(
            "Bear",
            tool_name="web_search",
            args={"query": "GDP data"},
            response_size=5000,
            duration_ms=2100.0,
        )
        audit.close()

        event = json.loads((tmp_path / "audit" / "Bear.jsonl").read_text().strip())
        assert event["event"] == "tool_call"
        assert event["tool"] == "web_search"
        assert event["response_size"] == 5000
        assert event["duration_ms"] == 2100.0

    def test_log_api_request(self, tmp_path):
        audit = AuditLogger(str(tmp_path))
        audit.log_api_request(
            "Host",
            model="claude-opus-4.6-1m",
            input_tokens=20000,
            output_tokens=1500,
            duration_ms=15000.0,
        )
        audit.close()

        event = json.loads((tmp_path / "audit" / "Host.jsonl").read_text().strip())
        assert event["event"] == "api_request"
        assert event["model"] == "claude-opus-4.6-1m"

    def test_log_error(self, tmp_path):
        audit = AuditLogger(str(tmp_path))
        audit.log_error("Bull", "TimeoutError", 60000.0)
        audit.close()

        event = json.loads((tmp_path / "audit" / "Bull.jsonl").read_text().strip())
        assert event["event"] == "error"
        assert "Timeout" in event["error"]

    def test_multiple_agents_separate_files(self, tmp_path):
        audit = AuditLogger(str(tmp_path))
        audit.log_call_start("Bull", "prompt1")
        audit.log_call_start("Bear", "prompt2")
        audit.log_call_start("Host", "prompt3")
        audit.close()

        assert (tmp_path / "audit" / "Bull.jsonl").exists()
        assert (tmp_path / "audit" / "Bear.jsonl").exists()
        assert (tmp_path / "audit" / "Host.jsonl").exists()

    def test_log_from_run_output_with_tools(self, tmp_path):
        audit = AuditLogger(str(tmp_path))

        mock_tool = MagicMock()
        mock_tool.tool_name = "web_search"
        mock_tool.tool_args = {"query": "test"}
        mock_tool.result = "search results..."
        mock_tool.tool_call_error = False
        mock_tool.metrics = MagicMock(duration=0.5)

        mock_output = MagicMock()
        mock_output.tools = [mock_tool]
        mock_output.messages = None
        mock_output.metrics = None
        mock_output.model = None

        audit.log_from_run_output("Bull", mock_output)
        audit.close()

        lines = (tmp_path / "audit" / "Bull.jsonl").read_text().strip().split("\n")
        events = [json.loads(line) for line in lines]
        tool_events = [e for e in events if e["event"] == "tool_call"]
        assert len(tool_events) == 1
        assert tool_events[0]["tool"] == "web_search"
        assert tool_events[0]["duration_ms"] == 500.0
        # Should also have messages_summary
        summary_events = [e for e in events if e["event"] == "messages_summary"]
        assert len(summary_events) == 1
        assert summary_events[0]["messages_count"] == 0

    def test_log_from_run_output_with_metrics(self, tmp_path):
        audit = AuditLogger(str(tmp_path))

        mock_output = MagicMock()
        mock_output.tools = []
        mock_output.messages = None
        mock_output.model = "claude-opus-4.6-1m"
        mock_output.metrics = MagicMock(
            input_tokens=30000, output_tokens=2000, duration=20.0,
            reasoning_tokens=0, cache_read_tokens=0, cache_write_tokens=0,
            total_tokens=32000, cost=None,
        )

        audit.log_from_run_output("Bear", mock_output)
        audit.close()

        lines = (tmp_path / "audit" / "Bear.jsonl").read_text().strip().split("\n")
        events = [json.loads(line) for line in lines]
        api_events = [e for e in events if e["event"] == "api_request"]
        assert len(api_events) == 1
        assert api_events[0]["input_tokens"] == 30000
        # Should also have run_metrics event
        rm_events = [e for e in events if e["event"] == "run_metrics"]
        assert len(rm_events) == 1
        assert rm_events[0]["total_tokens"] == 32000

    def test_log_from_run_output_none(self, tmp_path):
        audit = AuditLogger(str(tmp_path))
        audit.log_from_run_output("X", None)
        audit.close()
        assert not (tmp_path / "audit" / "X.jsonl").exists()

    def test_log_call_end_with_extras(self, tmp_path):
        audit = AuditLogger(str(tmp_path))
        audit.log_call_end(
            "Bull", 5000.0, "output", "end_turn",
            messages_count=5,
            total_tool_response_chars=2000,
        )
        audit.close()

        event = json.loads((tmp_path / "audit" / "Bull.jsonl").read_text().strip())
        assert event["messages_count"] == 5
        assert event["total_tool_response_chars"] == 2000

    def test_log_from_run_output_roundtrip_tokens(self, tmp_path):
        """Per-roundtrip token events from assistant messages."""
        audit = AuditLogger(str(tmp_path))

        msg1 = MagicMock()
        msg1.role = "assistant"
        msg1.content = "Hello"
        msg1.metrics = MagicMock(
            input_tokens=800, output_tokens=150,
            reasoning_tokens=0, cache_read_tokens=0, cache_write_tokens=0,
            cost=None, duration=None, time_to_first_token=None,
        )

        msg2 = MagicMock()
        msg2.role = "tool"
        msg2.content = "tool result"
        msg2.metrics = MagicMock(input_tokens=0, output_tokens=0)

        msg3 = MagicMock()
        msg3.role = "assistant"
        msg3.content = "World"
        msg3.metrics = MagicMock(
            input_tokens=1200, output_tokens=250,
            reasoning_tokens=0, cache_read_tokens=0, cache_write_tokens=0,
            cost=None, duration=None, time_to_first_token=None,
        )

        mock_output = MagicMock()
        mock_output.tools = []
        mock_output.messages = [msg1, msg2, msg3]
        mock_output.metrics = None

        audit.log_from_run_output("Bull", mock_output)
        audit.close()

        lines = (tmp_path / "audit" / "Bull.jsonl").read_text().strip().split("\n")
        events = [json.loads(line) for line in lines]
        rt_events = [e for e in events if e["event"] == "roundtrip_tokens"]
        assert len(rt_events) == 2
        assert rt_events[0]["input_tokens"] == 800
        assert rt_events[0]["roundtrip_idx"] == 0
        assert rt_events[1]["input_tokens"] == 1200
        assert rt_events[1]["roundtrip_idx"] == 1
        # Check messages_summary
        summary = [e for e in events if e["event"] == "messages_summary"]
        assert len(summary) == 1
        assert summary[0]["messages_count"] == 3

    def test_extract_call_start_extras(self):
        """extract_call_start_extras returns system_prompt_size and tool names."""
        mock_agent = MagicMock()
        mock_agent.system_message = "You are a helpful agent." * 10

        tool1 = MagicMock()
        tool1.name = "web_search"
        tool1.__name__ = "web_search"
        mock_agent.tools = [tool1]

        extras = AuditLogger.extract_call_start_extras(mock_agent)
        assert extras["system_prompt_size_chars"] == len(str(mock_agent.system_message))
        assert extras["system_prompt_size_tokens_est"] == extras["system_prompt_size_chars"] // 4
        assert "web_search" in extras["skill_tools_loaded"]

    def test_extract_call_end_extras(self):
        """extract_call_end_extras returns messages_count and total_tool_response_chars."""
        tool1 = MagicMock()
        tool1.result = "x" * 100

        mock_output = MagicMock()
        mock_output.messages = [MagicMock()] * 3
        mock_output.tools = [tool1]

        extras = AuditLogger.extract_call_end_extras(mock_output)
        assert extras["messages_count"] == 3
        assert extras["total_tool_response_chars"] == 100

    def test_extract_call_end_extras_none(self):
        assert AuditLogger.extract_call_end_extras(None) == {}


# -----------------------------------------------------------------------
# Engine integration tests
# -----------------------------------------------------------------------


class MockRunOutput:
    def __init__(self, content=None, tools=None, messages=None, metrics=None, model=None):
        self.content = content
        self.tools = tools or []
        self.messages = messages or []
        self.metrics = metrics
        self.model = model


class TestEngineAuditIntegration:
    """Test that DiscussionEngine creates audit log files during runs."""

    @patch("discuss_agent.engine.import_from_path")
    @patch("discuss_agent.engine.ContextManager")
    @patch("discuss_agent.engine.Agent")
    @pytest.mark.asyncio
    async def test_audit_files_created(self, MockAgent, MockCtxMgr, mock_import, tmp_path):
        """A normal run should produce audit/*.jsonl files."""
        from discuss_agent.config import (
            AgentConfig,
            DiscussionConfig,
            HostConfig,
            ModelConfig,
        )
        from discuss_agent.engine import DiscussionEngine

        config = DiscussionConfig(
            min_rounds=1,
            max_rounds=2,
            model_config=ModelConfig(model="claude-opus-4.6-1m"),
            agents=[
                AgentConfig(name="Bull", system_prompt="bullish"),
                AgentConfig(name="Bear", system_prompt="bearish"),
            ],
            host=HostConfig(
                convergence_prompt="judge convergence",
                summary_prompt="summarize",
            ),
            tools=[],
            context={},
        )

        engine = DiscussionEngine(config)
        engine._archiver._base_dir = str(tmp_path)

        # Mock agents
        mock_bull = AsyncMock()
        mock_bull.name = "Bull"
        mock_bull.arun = AsyncMock(return_value=MockRunOutput(content="Bullish view"))

        mock_bear = AsyncMock()
        mock_bear.name = "Bear"
        mock_bear.arun = AsyncMock(return_value=MockRunOutput(content="Bearish view"))

        engine._agents = [mock_bull, mock_bear]

        # Mock host returns converged
        mock_host = AsyncMock()
        mock_host.name = "Host"
        mock_host.arun = AsyncMock(
            return_value=MockRunOutput(
                content='{"converged": true, "reason": "agreed", "remaining_disputes": []}'
            )
        )
        engine._host = mock_host

        # Mock context builder
        mock_ctx = MockCtxMgr.return_value
        mock_ctx.build_initial_context = AsyncMock(return_value="background context")
        mock_ctx.compress = AsyncMock(side_effect=lambda h, r: h)

        # Mock summary agent
        with patch("discuss_agent.engine.Agent") as MockInnerAgent:
            mock_summary_agent = AsyncMock()
            mock_summary_agent.arun = AsyncMock(
                return_value=MockRunOutput(content="Final summary")
            )
            MockInnerAgent.return_value = mock_summary_agent

            result = await engine.run()

        assert result.converged is True

        # Find session directory
        session_dirs = [d for d in tmp_path.iterdir() if d.is_dir()]
        assert len(session_dirs) == 1
        audit_dir = session_dirs[0] / "audit"
        assert audit_dir.is_dir(), "audit/ directory should exist"

        # Check audit files
        audit_files = list(audit_dir.glob("*.jsonl"))
        assert len(audit_files) >= 1

        # Collect all events
        all_events = []
        for f in audit_files:
            for line in f.read_text().strip().split("\n"):
                if line:
                    all_events.append(json.loads(line))

        event_types = [e["event"] for e in all_events]
        assert "call_start" in event_types
        assert "call_end" in event_types

    @patch("discuss_agent.engine.import_from_path")
    @patch("discuss_agent.engine.ContextManager")
    @patch("discuss_agent.engine.Agent")
    @pytest.mark.asyncio
    async def test_audit_logs_error_on_failure(self, MockAgent, MockCtxMgr, mock_import, tmp_path):
        """Error events should be logged when agents fail."""
        from discuss_agent.config import (
            AgentConfig,
            DiscussionConfig,
            HostConfig,
            ModelConfig,
        )
        from discuss_agent.engine import DiscussionEngine

        config = DiscussionConfig(
            min_rounds=1,
            max_rounds=2,
            model_config=ModelConfig(model="claude-opus-4.6-1m"),
            agents=[
                AgentConfig(name="Bull", system_prompt="bullish"),
            ],
            host=HostConfig(
                convergence_prompt="judge",
                summary_prompt="summarize",
            ),
            tools=[],
            context={},
        )

        engine = DiscussionEngine(config)
        engine._archiver._base_dir = str(tmp_path)

        # Mock agent fails
        mock_bull = AsyncMock()
        mock_bull.name = "Bull"
        mock_bull.arun = AsyncMock(side_effect=RuntimeError("API timeout"))
        engine._agents = [mock_bull]

        mock_ctx = MockCtxMgr.return_value
        mock_ctx.build_initial_context = AsyncMock(return_value="context")

        result = await engine.run()
        assert result.terminated_by_error is True

        # Check audit
        session_dirs = [d for d in tmp_path.iterdir() if d.is_dir()]
        if session_dirs:
            audit_dir = session_dirs[0] / "audit"
            if audit_dir.exists():
                all_events = []
                for f in audit_dir.glob("*.jsonl"):
                    for line in f.read_text().strip().split("\n"):
                        if line:
                            all_events.append(json.loads(line))
                event_types = [e["event"] for e in all_events]
                assert "call_start" in event_types
