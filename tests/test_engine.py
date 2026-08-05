"""Integration tests for DiscussionEngine."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from discuss_agent.claims import ClaimsManager
from discuss_agent.config import (
    AgentConfig,
    DiscussionConfig,
    HostConfig,
    ModelConfig,
    ToolConfig,
)
from discuss_agent.models import DiscussionResult


def _make_config(num_agents: int = 2, max_rounds: int = 3) -> DiscussionConfig:
    agents = [
        AgentConfig(
            name=f"Agent-{chr(65 + i)}",
            system_prompt=f"You are agent {chr(65 + i)}.",
        )
        for i in range(num_agents)
    ]
    return DiscussionConfig(
        min_rounds=1,
        max_rounds=max_rounds,
        model_config=ModelConfig(model="claude-sonnet-4-20250514", api_key="test-key"),
        agents=agents,
        host=HostConfig(
            convergence_prompt="Judge convergence.",
            summary_prompt="Summarize.",
            skip_summary=True,
        ),
        tools=[],
        context={},
    )


class TestDiscussionEngineIntegration:
    """Test 2 agents running 2 rounds with CLAIM state transitions."""

    @patch("discuss_agent.engine.generate_usage_summary")
    @patch("discuss_agent.engine.AuditLogger")
    @patch("discuss_agent.engine.Archiver")
    @patch("discuss_agent.engine.ContextManager")
    @patch("discuss_agent.engine.AgentConversation")
    @patch("discuss_agent.engine.anthropic")
    async def test_two_agents_two_rounds_convergence(
        self,
        mock_anthropic_mod,
        MockConversation,
        MockCtxMgr,
        MockArchiver,
        MockAuditLogger,
        mock_usage_summary,
    ):
        from discuss_agent.engine import DiscussionEngine

        config = _make_config(num_agents=2, max_rounds=3)

        # --- Mock Archiver ---
        archiver_inst = MagicMock()
        archiver_inst.start_session.return_value = "/tmp/test_session"
        archiver_inst.save_round = MagicMock()
        archiver_inst.save_context = MagicMock()
        archiver_inst.save_summary = MagicMock()
        MockArchiver.return_value = archiver_inst

        # --- Mock AuditLogger ---
        audit_inst = MagicMock()
        MockAuditLogger.return_value = audit_inst

        # --- Mock ContextManager ---
        ctx_inst = MagicMock()
        ctx_inst.build_initial_context = AsyncMock(return_value="猪周期分析议题")
        MockCtxMgr.return_value = ctx_inst

        # --- Mock AgentConversation ---
        # Round 1: both agents propose claims
        # Round 2: both agents accept each other's claims
        call_counts = {"Agent-A": 0, "Agent-B": 0}

        def make_conv(**kwargs):
            name = kwargs["agent_name"]
            conv = MagicMock()
            conv.agent_name = name
            conv.messages = []

            async def _send(prompt):
                call_counts[name] += 1
                count = call_counts[name]
                if count == 1:
                    if name == "Agent-A":
                        return "[NEW_CLAIM:能繁去化] 当前3904万头"
                    else:
                        return "[NEW_CLAIM:成本优势] 头均14.5元"
                else:
                    if name == "Agent-A":
                        return "[ACCEPT TO:成本优势] 招商证券确认"
                    else:
                        return "[ACCEPT TO:能繁去化] 农业部数据确认"

            conv.send = AsyncMock(side_effect=_send)
            return conv

        MockConversation.side_effect = make_conv

        # --- Mock host judge (Anthropic API) ---
        # After round 2: close both claims
        mock_host_response = MagicMock()
        mock_host_response.content = [
            MagicMock(type="text", text='[{"claim":"能繁去化","verdict":"CLOSED:共识","reason":"双方一致"},{"claim":"成本优势","verdict":"CLOSED:共识","reason":"双方一致"}]')
        ]

        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=mock_host_response)
        mock_anthropic_mod.AsyncAnthropic.return_value = mock_client

        # --- Run engine ---
        engine = DiscussionEngine(config)
        result = await engine.run()

        # --- Assertions ---
        assert result.converged is True
        assert result.rounds_completed >= 2
        assert result.remaining_disputes == []

        # Verify agents were called (round 1 + round 2)
        assert call_counts["Agent-A"] == 2
        assert call_counts["Agent-B"] == 2

    @patch("discuss_agent.engine.generate_usage_summary")
    @patch("discuss_agent.engine.AuditLogger")
    @patch("discuss_agent.engine.Archiver")
    @patch("discuss_agent.engine.ContextManager")
    @patch("discuss_agent.engine.AgentConversation")
    @patch("discuss_agent.engine.anthropic")
    async def test_max_rounds_without_convergence(
        self,
        mock_anthropic_mod,
        MockConversation,
        MockCtxMgr,
        MockArchiver,
        MockAuditLogger,
        mock_usage_summary,
    ):
        from discuss_agent.engine import DiscussionEngine

        config = _make_config(num_agents=2, max_rounds=2)
        config.host.skip_summary = False

        archiver_inst = MagicMock()
        archiver_inst.start_session.return_value = "/tmp/test_session2"
        archiver_inst.save_round = MagicMock()
        archiver_inst.save_context = MagicMock()
        MockArchiver.return_value = archiver_inst

        MockAuditLogger.return_value = MagicMock()

        ctx_inst = MagicMock()
        ctx_inst.build_initial_context = AsyncMock(return_value="议题")
        MockCtxMgr.return_value = ctx_inst

        call_counts = {"Agent-A": 0, "Agent-B": 0}

        def make_conv(**kwargs):
            name = kwargs["agent_name"]
            conv = MagicMock()
            conv.agent_name = name
            conv.messages = []

            async def _send(prompt):
                call_counts[name] += 1
                if call_counts[name] == 1:
                    return f"[NEW_CLAIM:claim_{name}] content from {name}"
                return f"[REBUTTAL TO:claim_Agent-{'B' if name == 'Agent-A' else 'A'}] disagree"

            conv.send = AsyncMock(side_effect=_send)
            return conv

        MockConversation.side_effect = make_conv

        # Host always says CONTINUE
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(type="text", text='[{"claim":"claim_Agent-A","verdict":"CONTINUE","reason":""},{"claim":"claim_Agent-B","verdict":"CONTINUE","reason":""}]')
        ]
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)
        mock_anthropic_mod.AsyncAnthropic.return_value = mock_client

        engine = DiscussionEngine(config)
        result = await engine.run()

        assert result.converged is False
        assert result.rounds_completed == 2
        assert len(result.remaining_disputes) > 0
        assert result.summary is None
        archiver_inst.save_summary.assert_not_called()


class TestOpenAIHostRouting:
    """Host judge and summary use the OpenAI-compatible protocol when configured."""

    @patch("discuss_agent.engine.openai.AsyncOpenAI")
    async def test_openai_host_judge_and_summary(self, MockAsyncOpenAI):
        from types import SimpleNamespace

        from discuss_agent.engine import DiscussionEngine

        config = _make_config(num_agents=1)
        config.model_config = ModelConfig(
            model="agent-maestro-openai/gpt-5.5",
            api_key="dummy",
            base_url="http://localhost:23333/api/anthropic",
            max_tokens=100,
        )
        config.host.skip_summary = False

        client = MagicMock()
        client.chat.completions.create = AsyncMock(side_effect=[
            SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
                content='[{"claim":"cost","verdict":"CLOSED:共识","reason":"ok"}]'
            ))]),
            SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
                content="host summary"
            ))]),
        ])
        MockAsyncOpenAI.return_value = client

        claim = MagicMock()
        claim.format.return_value = "## [OPEN] cost"
        claims_mgr = MagicMock()
        claims_mgr.get_open_claims.return_value = [claim]
        claims_mgr.format_file.return_value = "all claims"

        engine = DiscussionEngine(config)
        verdicts = await engine._host_judge(claims_mgr, round_num=2)
        summary = await engine._host_summarize(claims_mgr)

        assert verdicts[0]["claim"] == "cost"
        assert summary == "host summary"
        assert MockAsyncOpenAI.call_args.kwargs["base_url"] == (
            "http://localhost:23333/api/openai/v1"
        )
        assert client.chat.completions.create.await_count == 2
        judge_messages = client.chat.completions.create.await_args_list[0].kwargs["messages"]
        summary_messages = client.chat.completions.create.await_args_list[1].kwargs["messages"]
        assert judge_messages[0] == {"role": "system", "content": "Judge convergence."}
        assert summary_messages[0] == {"role": "system", "content": "Summarize."}
