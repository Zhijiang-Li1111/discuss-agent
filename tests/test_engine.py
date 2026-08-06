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
        round_one_response = MagicMock()
        round_one_response.content = [
            MagicMock(
                type="text",
                text=(
                    '[{"claim":"能繁去化","verdict":"CONTINUE","reason":"需交叉核查",'
                    '"missing":"成本方审阅","needs_agents":["Agent-B"]},'
                    '{"claim":"成本优势","verdict":"CONTINUE","reason":"需交叉核查",'
                    '"missing":"供给方审阅","needs_agents":["Agent-A"]}]'
                ),
            )
        ]
        mock_client.messages.create = AsyncMock(
            side_effect=[round_one_response, mock_host_response],
        )
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
        claims_mgr.get_host_candidates.return_value = [claim]
        claims_mgr.format_host_candidate_batches.return_value = [
            "##CLAIM:cost [OPEN]##",
        ]
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

    async def test_host_judge_reviews_and_combines_every_bounded_batch(self):
        from discuss_agent.engine import DiscussionEngine

        engine = DiscussionEngine(_make_config(num_agents=2))
        claims_mgr = MagicMock()
        claims_mgr.topic = "topic"
        claims_mgr.get_host_candidates.return_value = [MagicMock(), MagicMock()]
        claims_mgr.format_host_candidate_batches.return_value = [
            "##CLAIM:first [OPEN]##",
            "##CLAIM:second [OPEN]##",
        ]
        claims_mgr.format_host_global_context.return_value = (
            "GLOBAL: first depends on second"
        )
        engine._call_host = AsyncMock(side_effect=[
            '[{"claim":"first","verdict":"CLOSED:共识","reason":"enough"}]',
            '[{"claim":"second","verdict":"CONTINUE","reason":"gap",'
            '"missing":"source","needs_agents":["Agent-B"]}]',
        ])

        verdicts = await engine._host_judge(claims_mgr, round_num=2)

        assert [item["claim"] for item in verdicts] == ["first", "second"]
        assert engine._call_host.await_count == 2
        prompts = [call.args[1] for call in engine._call_host.await_args_list]
        assert "候选批次：1/2" in prompts[0]
        assert "候选批次：2/2" in prompts[1]
        assert all("GLOBAL: first depends on second" in prompt for prompt in prompts)

    async def test_host_judge_retries_valid_but_incomplete_batch(self):
        from discuss_agent.engine import DiscussionEngine

        engine = DiscussionEngine(_make_config(num_agents=2))
        claims_mgr = MagicMock()
        claims_mgr.topic = "topic"
        claims_mgr.get_host_candidates.return_value = [MagicMock(), MagicMock()]
        claims_mgr.format_host_candidate_batches.return_value = [
            "##CLAIM:first [OPEN]##\n##CLAIM:second [OPEN]##",
        ]
        engine._call_host = AsyncMock(side_effect=[
            '[{"claim":"first","verdict":"CLOSED:共识","reason":"enough"}]',
            (
                '[{"claim":"first","verdict":"CLOSED:共识","reason":"enough"},'
                '{"claim":"second","verdict":"CONTINUE","reason":"gap",'
                '"missing":"source","needs_agents":["Agent-B"]}]'
            ),
        ])

        verdicts = await engine._host_judge(claims_mgr, round_num=2)

        assert [item["claim"] for item in verdicts] == ["first", "second"]
        assert engine._call_host.await_count == 2

    async def test_host_judge_retries_duplicate_conflicting_verdicts(self):
        from discuss_agent.engine import DiscussionEngine

        engine = DiscussionEngine(_make_config(num_agents=1))
        claims_mgr = MagicMock()
        claims_mgr.topic = "topic"
        claims_mgr.get_host_candidates.return_value = [MagicMock()]
        claims_mgr.format_host_candidate_batches.return_value = [
            "##CLAIM:first [OPEN]##",
        ]
        engine._call_host = AsyncMock(side_effect=[
            (
                '[{"claim":"first","verdict":"CLOSED:共识","reason":"yes"},'
                '{"claim":"first","verdict":"CONTINUE","reason":"no"}]'
            ),
            '[{"claim":"first","verdict":"CLOSED:共识","reason":"enough"}]',
        ])

        verdicts = await engine._host_judge(claims_mgr, round_num=1)

        assert verdicts == [{
            "claim": "first",
            "verdict": "CLOSED:共识",
            "reason": "enough",
        }]
        assert engine._call_host.await_count == 2
        assert [
            item["reason"] for item in engine._host_protocol_rejections
        ] == ["duplicate verdict"]

    async def test_host_judge_audits_extra_and_missing_ids_before_retry(self):
        from discuss_agent.engine import DiscussionEngine

        engine = DiscussionEngine(_make_config(num_agents=1))
        claims_mgr = MagicMock()
        claims_mgr.topic = "topic"
        claims_mgr.get_host_candidates.return_value = [MagicMock(), MagicMock()]
        claims_mgr.format_host_candidate_batches.return_value = [
            "##CLAIM:first [OPEN]##\n##CLAIM:second [OPEN]##",
        ]
        claims_mgr.format_host_global_context.return_value = ""
        engine._call_host = AsyncMock(side_effect=[
            (
                '[{"claim":"first","verdict":"CLOSED:共识","reason":"enough"},'
                '{"claim":"extra","verdict":"CLOSED:共识","reason":"other"}]'
            ),
            (
                '[{"claim":"first","verdict":"CLOSED:共识","reason":"enough"},'
                '{"claim":"second","verdict":"CLOSED:共识","reason":"enough"}]'
            ),
        ])

        await engine._host_judge(claims_mgr, round_num=1)

        assert {
            (item["claim"], item["reason"])
            for item in engine._host_protocol_rejections
        } == {
            ("extra", "claim was not offered to the host"),
            ("second", "missing verdict"),
        }

    async def test_host_judge_preserves_bracketed_claim_keyword(self):
        from discuss_agent.engine import DiscussionEngine

        engine = DiscussionEngine(_make_config(num_agents=1))
        claims_mgr = MagicMock()
        claims_mgr.topic = "topic"
        claims_mgr.get_host_candidates.return_value = [MagicMock()]
        claims_mgr.format_host_candidate_batches.return_value = [
            "##CLAIM:EPS [FY26] [OPEN]##",
        ]
        claims_mgr.format_host_global_context.return_value = ""
        engine._call_host = AsyncMock(return_value=(
            '[{"claim":"EPS [FY26]","verdict":"CLOSED:共识",'
            '"reason":"enough"}]'
        ))

        verdicts = await engine._host_judge(claims_mgr, round_num=1)

        assert verdicts[0]["claim"] == "EPS [FY26]"
        assert engine._call_host.await_count == 1
        assert engine._host_protocol_rejections == []

    async def test_host_summary_prompt_is_bounded(self):
        from discuss_agent.engine import DiscussionEngine

        engine = DiscussionEngine(_make_config(num_agents=1))
        claims_mgr = MagicMock()
        claims_mgr.format_file.return_value = "history" * 100_000
        engine._call_host = AsyncMock(return_value="summary")

        await engine._host_summarize(claims_mgr, round_num=2)

        prompt = engine._call_host.await_args.args[1]
        assert len(prompt) < 110_000
        assert "截断" in prompt

    async def test_round_one_host_prompt_fail_closes_unsupported_claims(self):
        from discuss_agent.engine import DiscussionEngine

        engine = DiscussionEngine(_make_config(num_agents=2))
        claims_mgr = MagicMock()
        claims_mgr.topic = "topic"
        claims_mgr.get_host_candidates.return_value = [MagicMock()]
        claims_mgr.format_host_candidate_batches.return_value = [
            "##CLAIM:first [OPEN]##",
        ]
        engine._call_host = AsyncMock(return_value=(
            '[{"claim":"first","verdict":"CONTINUE","reason":"gap",'
            '"missing":"source","needs_agents":["Agent-B"]}]'
        ))

        await engine._host_judge(claims_mgr, round_num=1)

        prompt = engine._call_host.await_args.args[1]
        assert "第1轮也必须遵守" in prompt
        assert "证据不足" in prompt
        assert "关键反驳被忽略" in prompt
        assert "UNKNOWN" in prompt
        assert "实质缺口时，必须 CONTINUE" in prompt
        assert "上下文截断" in prompt
        assert "必须 CONTINUE" in prompt
        assert "定向给相关 Agent" in prompt

    async def test_host_judge_accepts_bracketed_claim_keyword(self):
        from discuss_agent.claims import Claim, ClaimsManager
        from discuss_agent.engine import DiscussionEngine

        engine = DiscussionEngine(_make_config(num_agents=1))
        claims_mgr = ClaimsManager()
        claims_mgr.claims["EPS [FY26]"] = Claim("EPS [FY26]", "OPEN")
        engine._call_host = AsyncMock(return_value=(
            '[{"claim":"EPS [FY26]","verdict":"CLOSED:共识",'
            '"reason":"supported"}]'
        ))

        verdicts = await engine._host_judge(claims_mgr, round_num=1)

        assert verdicts == [{
            "claim": "EPS [FY26]",
            "verdict": "CLOSED:共识",
            "reason": "supported",
        }]

    async def test_final_round_prompt_requests_semantic_terminal_judgment(self):
        from discuss_agent.engine import DiscussionEngine

        engine = DiscussionEngine(_make_config(num_agents=2, max_rounds=3))
        claims_mgr = MagicMock()
        claims_mgr.topic = "topic"
        claims_mgr.get_host_candidates.return_value = [MagicMock()]
        claims_mgr.format_host_candidate_batches.return_value = [
            "##CLAIM:first [OPEN]##",
        ]
        claims_mgr.format_host_global_context.return_value = "global context"
        engine._call_host = AsyncMock(return_value=(
            '[{"claim":"first","verdict":"CONTINUE","reason":"source unavailable",'
            '"missing":"decisive source","needs_agents":["Agent-B"]}]'
        ))

        verdicts = await engine._host_judge(claims_mgr, round_num=3)

        prompt = engine._call_host.await_args.args[1]
        assert verdicts[0]["verdict"] == "CONTINUE"
        assert "安全上限" in prompt
        assert "最终语义裁决" in prompt
        assert "保持 OPEN" in prompt
        assert "不得因回应数量" in prompt
        assert "UNKNOWN是否会改变结论" in prompt
