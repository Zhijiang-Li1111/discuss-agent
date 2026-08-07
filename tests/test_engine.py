"""Integration tests for DiscussionEngine."""

from __future__ import annotations

import json
from pathlib import Path
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


def _closed_judgment(claim: str, reason: str = "enough") -> dict:
    return {
        "claim": claim,
        "verdict": "CLOSED:共识",
        "reason": reason,
    }


def _continue_judgment(
    claim: str,
    reason: str = "gap",
    missing: str = "source",
    needs_agents: list[str] | None = None,
) -> dict:
    return {
        "claim": claim,
        "verdict": "CONTINUE",
        "reason": reason,
        "missing": missing,
        "needs_agents": ["Agent-B"] if needs_agents is None else needs_agents,
        "allow_unknown_progress": False,
    }


def _room_judgment(status: str, reason: str = "room semantics resolved") -> dict:
    return {
        "status": status,
        "reason": reason,
    }


def _conditional_seal_cases() -> list[dict]:
    path = Path(__file__).parent / "fixtures" / "host_conditional_seal_cases.json"
    return json.loads(path.read_text())


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
                        return (
                            "[NEW_CLAIM:能繁去化] "
                            "source://population 当前3904万头"
                        )
                    else:
                        return (
                            "[NEW_CLAIM:成本优势] "
                            "source://cost 头均14.5元"
                        )
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
            MagicMock(type="text", text=json.dumps([
                {
                    "claim": "能繁去化",
                    "verdict": "CLOSED:共识",
                    "reason": "双方一致",
                },
                {
                    "claim": "成本优势",
                    "verdict": "CLOSED:共识",
                    "reason": "双方一致",
                },
            ], ensure_ascii=False))
        ]

        mock_client = MagicMock()
        round_one_response = MagicMock()
        round_one_response.content = [
            MagicMock(
                type="text",
                text=(
                    '[{"claim":"能繁去化","verdict":"CONTINUE","reason":"需交叉核查",'
                    '"missing":"成本方审阅","needs_agents":["Agent-B"],'
                    '"allow_unknown_progress":false},'
                    '{"claim":"成本优势","verdict":"CONTINUE","reason":"需交叉核查",'
                    '"missing":"供给方审阅","needs_agents":["Agent-A"],'
                    '"allow_unknown_progress":false}]'
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

    @patch("discuss_agent.engine.generate_usage_summary")
    @patch("discuss_agent.engine.AuditLogger")
    @patch("discuss_agent.engine.Archiver")
    @patch("discuss_agent.engine.ContextManager")
    @patch("discuss_agent.engine.AgentConversation")
    async def test_open_claim_can_remain_when_host_declares_room_complete(
        self,
        MockConversation,
        MockCtxMgr,
        MockArchiver,
        MockAuditLogger,
        mock_usage_summary,
    ):
        from discuss_agent.engine import DiscussionEngine

        config = _make_config(num_agents=2, max_rounds=1)
        archiver = MagicMock()
        archiver.start_session.return_value = "/tmp/test_semantic_completion"
        MockArchiver.return_value = archiver
        MockAuditLogger.return_value = MagicMock()
        MockCtxMgr.return_value.build_initial_context = AsyncMock(
            return_value="generic topic",
        )

        def make_conv(**kwargs):
            conv = MagicMock(messages=[])
            if kwargs["agent_name"] == "Agent-A":
                conv.send = AsyncMock(
                    return_value="[NEW_CLAIM:future] future value is UNKNOWN",
                )
            else:
                conv.send = AsyncMock(return_value="")
            return conv

        MockConversation.side_effect = make_conv
        engine = DiscussionEngine(config)

        async def judge(_claims_mgr, _round_num):
            engine._host_room_adjudication = _room_judgment("CONVERGED")
            return [
                _continue_judgment(
                    "future",
                    reason="encapsulated by an update trigger",
                    missing="future observation",
                    needs_agents=[],
                )
                | {"allow_unknown_progress": True}
            ]

        engine._host_judge = AsyncMock(side_effect=judge)

        result = await engine.run()

        assert result.converged is True
        assert result.rounds_completed == 1
        assert result.remaining_disputes == []
        host_record = next(
            call.args[2]
            for call in archiver.save_round.call_args_list
            if call.args[1] == "host"
        )
        assert host_record["room_adjudication"] == _room_judgment("CONVERGED")
        assert host_record["accepted_verdicts"][0]["verdict"] == "CONTINUE"

    @patch("discuss_agent.engine.generate_usage_summary")
    @patch("discuss_agent.engine.AuditLogger")
    @patch("discuss_agent.engine.Archiver")
    @patch("discuss_agent.engine.ContextManager")
    @patch("discuss_agent.engine.AgentConversation")
    async def test_material_open_claim_continues_when_host_declares_not_converged(
        self,
        MockConversation,
        MockCtxMgr,
        MockArchiver,
        MockAuditLogger,
        mock_usage_summary,
    ):
        from discuss_agent.engine import DiscussionEngine

        config = _make_config(num_agents=2, max_rounds=2)
        archiver = MagicMock()
        archiver.start_session.return_value = "/tmp/test_material_open"
        MockArchiver.return_value = archiver
        MockAuditLogger.return_value = MagicMock()
        MockCtxMgr.return_value.build_initial_context = AsyncMock(
            return_value="generic topic",
        )
        calls = {"Agent-A": 0, "Agent-B": 0}

        def make_conv(**kwargs):
            name = kwargs["agent_name"]
            conv = MagicMock(messages=[])

            async def send(_prompt):
                calls[name] += 1
                if calls[name] == 1 and name == "Agent-A":
                    return "[NEW_CLAIM:material] unsupported material assertion"
                if calls[name] > 1:
                    return "[REBUTTAL TO:material] still lacks a decisive boundary"
                return "[ACCEPT TO:material] initial review"

            conv.send = AsyncMock(side_effect=send)
            return conv

        MockConversation.side_effect = make_conv
        engine = DiscussionEngine(config)

        async def judge(_claims_mgr, _round_num):
            engine._host_room_adjudication = _room_judgment(
                "NOT_CONVERGED",
                "material claim remains unencapsulated",
            )
            return [_continue_judgment("material", needs_agents=["Agent-B"])]

        engine._host_judge = AsyncMock(side_effect=judge)

        result = await engine.run()

        assert result.converged is False
        assert result.rounds_completed == 2
        assert result.remaining_disputes == ["material"]
        assert calls == {"Agent-A": 2, "Agent-B": 2}

    @patch("discuss_agent.engine.generate_usage_summary")
    @patch("discuss_agent.engine.AuditLogger")
    @patch("discuss_agent.engine.Archiver")
    @patch("discuss_agent.engine.ContextManager")
    @patch("discuss_agent.engine.AgentConversation")
    async def test_legacy_host_output_still_requires_all_claims_closed(
        self,
        MockConversation,
        MockCtxMgr,
        MockArchiver,
        MockAuditLogger,
        mock_usage_summary,
    ):
        from discuss_agent.engine import DiscussionEngine

        config = _make_config(num_agents=1, max_rounds=1)
        archiver = MagicMock()
        archiver.start_session.return_value = "/tmp/test_legacy_host"
        MockArchiver.return_value = archiver
        MockAuditLogger.return_value = MagicMock()
        MockCtxMgr.return_value.build_initial_context = AsyncMock(return_value="topic")
        conv = MagicMock(messages=[])
        conv.send = AsyncMock(return_value="[NEW_CLAIM:legacy] unresolved")
        MockConversation.return_value = conv
        engine = DiscussionEngine(config)
        engine._host_judge = AsyncMock(return_value=[
            _continue_judgment("legacy", needs_agents=[]),
        ])

        result = await engine.run()

        assert result.converged is False
        assert result.remaining_disputes == ["legacy"]


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
                content=json.dumps([_closed_judgment("cost", "ok")])
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
            json.dumps([_closed_judgment("first")]),
            json.dumps([_continue_judgment("second")]),
        ])

        verdicts = await engine._host_judge(claims_mgr, round_num=2)

        assert [item["claim"] for item in verdicts] == ["first", "second"]
        assert engine._call_host.await_count == 2
        prompts = [call.args[1] for call in engine._call_host.await_args_list]
        assert "候选批次：1/2" in prompts[0]
        assert "候选批次：2/2" in prompts[1]
        assert all("GLOBAL: first depends on second" in prompt for prompt in prompts)

    async def test_host_judge_accepts_explicit_room_adjudication(self):
        from discuss_agent.engine import DiscussionEngine

        engine = DiscussionEngine(_make_config(num_agents=2))
        claims_mgr = MagicMock()
        claims_mgr.topic = "topic"
        claims_mgr.get_host_candidates.return_value = [MagicMock(keyword="future")]
        claims_mgr.format_host_candidate_batches.return_value = [
            "##CLAIM:future [OPEN]##",
        ]
        claims_mgr.format_host_global_context.return_value = "future is bounded"
        verdict = _continue_judgment(
            "future",
            reason="condition and update trigger are explicit",
            missing="future observation",
            needs_agents=[],
        ) | {"allow_unknown_progress": True}
        engine._call_host = AsyncMock(return_value=json.dumps({
            "room_adjudication": _room_judgment("CONVERGED"),
            "verdicts": [verdict],
        }))

        verdicts = await engine._host_judge(claims_mgr, round_num=2)

        assert verdicts == [verdict]
        assert engine._host_room_adjudication == _room_judgment("CONVERGED")

    async def test_explicit_completion_cannot_bypass_truncation_gate(self):
        from discuss_agent.claims import Claim, ClaimEntry, ClaimsManager
        from discuss_agent.engine import DiscussionEngine

        engine = DiscussionEngine(_make_config(num_agents=2))
        claims_mgr = ClaimsManager()
        claims_mgr.topic = "topic"
        claims_mgr.claims["large"] = Claim("large", "OPEN", [
            ClaimEntry("FROM", "Agent-A", 1, "x" * 10_000),
        ])
        engine._call_host = AsyncMock(return_value=json.dumps({
            "room_adjudication": _room_judgment("CONVERGED"),
            "verdicts": [_closed_judgment("large")],
        }))

        verdicts = await engine._host_judge(claims_mgr, round_num=2)
        offered = {"large"}
        accepted, rejected = engine._apply_host_verdicts(
            claims_mgr,
            verdicts,
            offered,
            round_num=2,
        )

        assert verdicts[0]["verdict"] == "CONTINUE"
        assert engine._room_converged(
            claims_mgr,
            accepted=accepted,
            rejected=rejected,
            offered_keywords=offered,
        ) is False

    async def test_invalid_explicit_wrapper_cannot_bypass_schema_and_identity(self):
        from discuss_agent.engine import DiscussionEngine

        engine = DiscussionEngine(_make_config(num_agents=1))
        claims_mgr = MagicMock()
        claims_mgr.topic = "topic"
        claims_mgr.get_host_candidates.return_value = [MagicMock(keyword="X")]
        claims_mgr.format_host_candidate_batches.return_value = [
            "##CLAIM:X [OPEN]##",
        ]
        claims_mgr.format_host_global_context.return_value = "global"
        engine._call_host = AsyncMock(side_effect=[
            json.dumps({
                "room_adjudication": _room_judgment("CONVERGED"),
                "verdicts": [_closed_judgment("wrong")],
            }),
            json.dumps({
                "room_adjudication": {
                    "status": "CONVERGED",
                    "reason": "",
                },
                "verdicts": [_closed_judgment("X")],
            }),
        ])

        verdicts = await engine._host_judge(claims_mgr, round_num=2)

        assert verdicts == []
        assert engine._host_room_adjudication is None
        assert engine._call_host.await_count == 2

    async def test_host_judge_does_not_force_unrelated_claim_on_global_truncation(self):
        from discuss_agent.engine import DiscussionEngine

        engine = DiscussionEngine(_make_config(num_agents=2))
        claims_mgr = MagicMock()
        claims_mgr.topic = "topic"
        claims_mgr.get_host_candidates.return_value = [MagicMock(keyword="first")]
        claims_mgr.format_host_candidate_batches.return_value = [
            "##CLAIM:first [OPEN]##",
        ]
        claims_mgr.format_host_global_context.return_value = "[MUST_CONT"
        claims_mgr.host_global_context_truncated = True
        engine._call_host = AsyncMock(return_value=json.dumps([{
            "claim": "first",
            "verdict": "CLOSED:共识",
            "reason": "complete without omitted global context",
        }]))

        verdicts = await engine._host_judge(claims_mgr, round_num=1)

        assert verdicts[0]["verdict"] == "CLOSED:共识"
        prompt = engine._call_host.await_args.args[1]
        assert "仅当当前 claim 依赖缺失的全局上下文" in prompt

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
            json.dumps([_closed_judgment("first")]),
            json.dumps([
                _closed_judgment("first"),
                _continue_judgment("second"),
            ]),
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
            json.dumps([
                _closed_judgment("first", "yes"),
                _continue_judgment("first", "no"),
            ]),
            json.dumps([_closed_judgment("first")]),
        ])

        verdicts = await engine._host_judge(claims_mgr, round_num=1)

        assert verdicts == [_closed_judgment("first")]
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
            json.dumps([
                _closed_judgment("first"),
                _closed_judgment("extra", "other"),
            ]),
            json.dumps([
                _closed_judgment("first"),
                _closed_judgment("second"),
            ]),
        ])

        await engine._host_judge(claims_mgr, round_num=1)

        assert {
            (item["claim"], item["reason"])
            for item in engine._host_protocol_rejections
        } == {
            ("extra", "claim was not offered to the host"),
            ("second", "missing verdict"),
        }

    async def test_host_protocol_rejections_identify_retry_attempts(self):
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
            '[{"claim":"first","verdict":"CONTINUE"}]',
            '[{"claim":"first","verdict":"CONTINUE"}]',
        ])

        await engine._host_judge(claims_mgr, round_num=1)

        missing_second = [
            item for item in engine._host_protocol_rejections
            if item["claim"] == "second" and item["reason"] == "missing verdict"
        ]
        assert [item["attempt"] for item in missing_second] == [1, 2]

    async def test_host_protocol_rejections_audit_malformed_retry_attempts(self):
        from discuss_agent.engine import DiscussionEngine

        engine = DiscussionEngine(_make_config(num_agents=1))
        claims_mgr = MagicMock()
        claims_mgr.topic = "topic"
        claims_mgr.get_host_candidates.return_value = [MagicMock()]
        claims_mgr.format_host_candidate_batches.return_value = [
            "##CLAIM:first [OPEN]##",
        ]
        claims_mgr.format_host_global_context.return_value = ""
        engine._call_host = AsyncMock(side_effect=["not JSON", "{}"])

        verdicts = await engine._host_judge(claims_mgr, round_num=1)

        assert verdicts == []
        assert [
            (item["attempt"], item["reason"])
            for item in engine._host_protocol_rejections
        ] == [
            (1, "invalid JSON array"),
            (2, "JSON payload is not an array"),
        ]

    @pytest.mark.parametrize(
        ("response", "reason"),
        [
            ('prefix [{"claim":"first"}]', "invalid JSON array"),
            ('[{"claim":"first"}] suffix', "invalid JSON array"),
            ('[{"claim":"first"}', "invalid JSON array"),
            ('{"claim":"first"}', "JSON payload is not an array"),
        ],
    )
    async def test_host_judge_requires_whole_response_json_array(
        self, response, reason,
    ):
        from discuss_agent.engine import DiscussionEngine

        engine = DiscussionEngine(_make_config(num_agents=1))
        claims_mgr = MagicMock()
        claims_mgr.topic = "topic"
        claims_mgr.get_host_candidates.return_value = [MagicMock()]
        claims_mgr.format_host_candidate_batches.return_value = [
            "##CLAIM:first [OPEN]##",
        ]
        claims_mgr.format_host_global_context.return_value = ""
        engine._call_host = AsyncMock(return_value=response)

        assert await engine._host_judge(claims_mgr, round_num=1) == []
        assert [item["reason"] for item in engine._host_protocol_rejections] == [
            reason,
            reason,
        ]

    async def test_host_judge_accepts_whitespace_around_json_array(self):
        from discuss_agent.engine import DiscussionEngine

        engine = DiscussionEngine(_make_config(num_agents=1))
        claims_mgr = MagicMock()
        claims_mgr.topic = "topic"
        claims_mgr.get_host_candidates.return_value = [MagicMock()]
        claims_mgr.format_host_candidate_batches.return_value = [
            "##CLAIM:first [OPEN]##",
        ]
        claims_mgr.format_host_global_context.return_value = ""
        engine._call_host = AsyncMock(
            return_value="\n  " + json.dumps([_closed_judgment("first")]) + " \t",
        )

        assert await engine._host_judge(claims_mgr, round_num=1) == [
            _closed_judgment("first"),
        ]

    async def test_host_retries_and_audits_schema_invalid_verdict(self):
        from discuss_agent.engine import DiscussionEngine

        engine = DiscussionEngine(_make_config(num_agents=1))
        claims_mgr = MagicMock()
        claims_mgr.topic = "topic"
        claims_mgr.get_host_candidates.return_value = [MagicMock()]
        claims_mgr.format_host_candidate_batches.return_value = [
            "##CLAIM:first [OPEN]##",
        ]
        claims_mgr.format_host_global_context.return_value = ""
        engine._call_host = AsyncMock(side_effect=[
            (
                '[{"claim":"first","verdict":"CLOSED:共识","reason":"enough",'
                '"unknown_field":true}]'
            ),
            (
                '[{"claim":"first","verdict":"CONTINUE","reason":"gap",'
                '"missing":"source","needs_agents":["Agent-A"],'
                '"allow_unknown_progress":false}]'
            ),
        ])

        verdicts = await engine._host_judge(claims_mgr, round_num=1)

        assert verdicts[0]["verdict"] == "CONTINUE"
        assert engine._host_protocol_rejections == [{
            "claim": "first",
            "verdict": "CLOSED:共识",
            "unknown_field": True,
            "host_reason": "enough",
            "reason": "invalid closed verdict schema",
            "attempt": 1,
        }]

    async def test_host_retries_continue_with_unknown_agent(self):
        from discuss_agent.engine import DiscussionEngine

        engine = DiscussionEngine(_make_config(num_agents=1))
        claims_mgr = MagicMock()
        claims_mgr.topic = "topic"
        claims_mgr.get_host_candidates.return_value = [MagicMock()]
        claims_mgr.format_host_candidate_batches.return_value = [
            "##CLAIM:first [OPEN]##",
        ]
        claims_mgr.format_host_global_context.return_value = ""
        engine._call_host = AsyncMock(side_effect=[
            json.dumps([
                _continue_judgment(
                    "first",
                    needs_agents=["Unknown-Agent"],
                ),
            ]),
            json.dumps([
                _continue_judgment(
                    "first",
                    needs_agents=["Agent-A"],
                ),
            ]),
        ])

        verdicts = await engine._host_judge(claims_mgr, round_num=1)

        assert verdicts == [
            _continue_judgment("first", needs_agents=["Agent-A"]),
        ]
        assert engine._call_host.await_count == 2
        assert engine._host_protocol_rejections[0]["reason"] == (
            "invalid continue verdict routing"
        )

    async def test_host_retries_non_string_verdict_as_schema_rejection(self):
        from discuss_agent.engine import DiscussionEngine

        engine = DiscussionEngine(_make_config(num_agents=1))
        claims_mgr = MagicMock()
        claims_mgr.topic = "topic"
        claims_mgr.get_host_candidates.return_value = [MagicMock()]
        claims_mgr.format_host_candidate_batches.return_value = [
            "##CLAIM:first [OPEN]##",
        ]
        claims_mgr.format_host_global_context.return_value = ""
        engine._call_host = AsyncMock(side_effect=[
            '[{"claim":"first","verdict":[],"reason":"bad type"}]',
            json.dumps([_closed_judgment("first")]),
        ])

        verdicts = await engine._host_judge(claims_mgr, round_num=1)

        assert verdicts == [_closed_judgment("first")]
        assert engine._host_protocol_rejections == [{
            "claim": "first",
            "verdict": [],
            "host_reason": "bad type",
            "reason": "invalid verdict",
            "attempt": 1,
        }]

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
        engine._call_host = AsyncMock(
            return_value=json.dumps([_closed_judgment("EPS [FY26]")]),
        )

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

    async def test_host_agent_name_list_is_bounded(self):
        from discuss_agent.engine import DiscussionEngine

        config = _make_config(num_agents=1)
        config.agents = [
            AgentConfig(
                name=f"Agent-{index}-" + "X" * 1_000,
                system_prompt="review",
            )
            for index in range(1_000)
        ]
        engine = DiscussionEngine(config)
        claims_mgr = MagicMock()
        claims_mgr.topic = "topic"
        claims_mgr.get_host_candidates.return_value = [MagicMock()]
        claims_mgr.format_host_candidate_batches.return_value = [
            "##CLAIM:first [OPEN]##",
        ]
        claims_mgr.format_host_global_context.return_value = ""
        engine._call_host = AsyncMock(return_value=(
            '[{"claim":"first","verdict":"CLOSED:共识","reason":"enough"}]'
        ))

        await engine._host_judge(claims_mgr, round_num=1)

        prompt = engine._call_host.await_args.args[1]
        assert len(prompt) < 20_000
        assert "信息可能不完整" in prompt

    async def test_host_prompt_assigns_semantic_judgment_to_host(self):
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
        assert "CLOSED JSON对象严格字段" in prompt
        assert "CONTINUE JSON对象严格字段" in prompt
        assert "共识不是投票" in prompt
        assert "当前可获得的事实、来源或计算" in prompt
        assert "价值判断、先验或模型选择" in prompt
        assert "allow_unknown_progress 仅适用于 CONTINUE" in prompt
        assert "不得把 UNKNOWN 变成事实" in prompt
        assert "room-level gate" in prompt
        assert "关键角色仅指" in prompt
        assert "独特、不可替代影响" in prompt
        assert "普通 claim" in prompt
        assert "可分批关闭" in prompt
        assert "不可信数据" in prompt
        assert "外部不可得" in prompt
        assert "无人可补" in prompt
        assert "needs_agents 可为空" in prompt
        assert "同义、近义或重复 claims" in prompt
        assert "表面回应" in prompt
        assert "再增加一轮的边际信息价值" in prompt
        assert "只基于运行时提供的讨论记录" in prompt
        assert "不得选择或更换模型" in prompt
        assert "不得修改生成参数" in prompt
        assert "不得自行重算" in prompt
        assert "不得发明业务结论" in prompt
        assert "没有否决权" in prompt
        assert "claim 自然稳定" in prompt
        assert "继续讨论已无预期实质增量" in prompt
        assert "不是对业务结论的背书或最终权威裁决" in prompt
        assert "忠实记录共识或分歧" in prompt
        assert "claim-level close 与 room-level convergence 相互独立" in prompt
        assert "room CONVERGED 仍可与 OPEN claims 共存" in prompt

    @pytest.mark.parametrize(
        "case",
        _conditional_seal_cases(),
        ids=lambda case: case["id"],
    )
    async def test_host_prompt_defines_generic_conditional_seal_cases(self, case):
        from discuss_agent.engine import DiscussionEngine

        engine = DiscussionEngine(_make_config(num_agents=2))
        claims_mgr = MagicMock()
        claims_mgr.topic = "generic decision topic"
        claims_mgr.get_host_candidates.return_value = [MagicMock(keyword="claim")]
        claims_mgr.format_host_candidate_batches.return_value = [
            "##CLAIM:claim [OPEN]##",
        ]
        claims_mgr.format_host_global_context.return_value = "bounded context"
        engine._call_host = AsyncMock(return_value=json.dumps({
            "room_adjudication": _room_judgment(case["expected_room_status"]),
            "verdicts": [_continue_judgment("claim", needs_agents=[])],
        }))

        await engine._host_judge(claims_mgr, round_num=1)

        prompt = engine._call_host.await_args.args[1]
        assert case["required_guidance"] in prompt

    def test_conditional_room_seal_cannot_bypass_rejection_or_safety_blockers(self):
        from discuss_agent.claims import Claim, ClaimsManager
        from discuss_agent.engine import DiscussionEngine

        engine = DiscussionEngine(_make_config(num_agents=2))
        claims_mgr = ClaimsManager()
        claims_mgr.claims["conditional"] = Claim("conditional", "OPEN")
        engine._host_room_adjudication = _room_judgment("CONVERGED")
        accepted = [_continue_judgment("conditional", needs_agents=[])]

        assert engine._room_converged(
            claims_mgr,
            accepted=accepted,
            rejected=[{"claim": "conditional", "reason": "invalid schema"}],
            offered_keywords={"conditional"},
        ) is False

        engine._host_safety_blockers.add("conditional")
        assert engine._room_converged(
            claims_mgr,
            accepted=accepted,
            rejected=[],
            offered_keywords={"conditional"},
        ) is False

    async def test_host_judge_forces_truncated_claim_to_continue(self):
        from discuss_agent.claims import Claim, ClaimEntry, ClaimsManager
        from discuss_agent.engine import DiscussionEngine

        engine = DiscussionEngine(_make_config(num_agents=2))
        claims_mgr = ClaimsManager()
        claims_mgr.claims["X"] = Claim("X", "OPEN", [
            ClaimEntry("FROM", "Agent-A", 1, "original"),
            *[
                ClaimEntry(
                    "REBUTTAL",
                    f"reviewer-{index:05d}",
                    1,
                    "counterexample",
                )
                for index in range(5_000)
            ],
        ])
        engine._call_host = AsyncMock(return_value=(
            '[{"claim":"X","verdict":"CLOSED:共识","reason":"looks complete"}]'
        ))

        verdicts = await engine._host_judge(claims_mgr, round_num=1)

        assert verdicts == [{
            "claim": "X",
            "verdict": "CONTINUE",
            "reason": "上下文截断，无法安全关闭 claim",
            "missing": "完整的 claim 证据、反驳和身份上下文",
            "needs_agents": ["Agent-A", "Agent-B"],
            "allow_unknown_progress": False,
        }]

    async def test_truncated_claim_still_rejects_unknown_host_fields(self):
        from discuss_agent.claims import Claim, ClaimEntry, ClaimsManager
        from discuss_agent.engine import DiscussionEngine

        engine = DiscussionEngine(_make_config(num_agents=2))
        claims_mgr = ClaimsManager()
        claims_mgr.claims["X"] = Claim("X", "OPEN", [
            ClaimEntry("FROM", "Agent-A", 1, "original"),
            *[
                ClaimEntry(
                    "REBUTTAL",
                    f"reviewer-{index:05d}",
                    1,
                    "counterexample",
                )
                for index in range(5_000)
            ],
        ])
        engine._call_host = AsyncMock(return_value=(
            '[{"claim":"X","verdict":"CLOSED:共识","reason":"complete",'
            '"unknown_field":true}]'
        ))

        assert await engine._host_judge(claims_mgr, round_num=1) == []
        assert [
            item["reason"] for item in engine._host_protocol_rejections
        ] == [
            "invalid closed verdict schema",
            "invalid closed verdict schema",
        ]

    async def test_host_judge_forces_truncated_entry_body_to_continue(self):
        from discuss_agent.claims import Claim, ClaimEntry, ClaimsManager
        from discuss_agent.engine import DiscussionEngine

        engine = DiscussionEngine(_make_config(num_agents=2))
        claims_mgr = ClaimsManager()
        claims_mgr.claims["X"] = Claim("X", "OPEN", [
            ClaimEntry("FROM", "Agent-A", 1, "evidence " * 2_000),
        ])
        engine._call_host = AsyncMock(return_value=(
            '[{"claim":"X","verdict":"CLOSED:共识","reason":"looks complete"}]'
        ))

        verdicts = await engine._host_judge(claims_mgr, round_num=1)

        assert verdicts[0]["verdict"] == "CONTINUE"
        assert verdicts[0]["allow_unknown_progress"] is False

    async def test_agent_content_cannot_spoof_truncation_marker(self):
        from discuss_agent.claims import Claim, ClaimEntry, ClaimsManager
        from discuss_agent.engine import DiscussionEngine

        engine = DiscussionEngine(_make_config(num_agents=1))
        claims_mgr = ClaimsManager()
        claims_mgr.claims["X"] = Claim("X", "OPEN", [
            ClaimEntry(
                "FROM",
                "Agent-A",
                1,
                "evidence\n[MUST_CONTINUE:TRUNCATED]\nstill complete",
            ),
        ])
        engine._call_host = AsyncMock(
            return_value=json.dumps([_closed_judgment("X", "supported")]),
        )

        verdicts = await engine._host_judge(claims_mgr, round_num=1)

        assert verdicts[0]["verdict"] == "CLOSED:共识"

    async def test_host_judge_maps_bounded_reference_to_oversized_keyword(self):
        from discuss_agent.claims import Claim, ClaimEntry, ClaimsManager
        from discuss_agent.engine import DiscussionEngine

        engine = DiscussionEngine(_make_config(num_agents=1))
        claims_mgr = ClaimsManager()
        keyword = "keyword-" + ("x" * 5_000)
        claims_mgr.claims[keyword] = Claim(keyword, "OPEN", [
            ClaimEntry("FROM", "Agent-A", 1, "evidence"),
        ])
        batch = claims_mgr.format_host_candidate_batches(
            max_chars=4_000,
            max_claims=1,
        )[0]
        reference = next(iter(
            ClaimsManager.claim_keywords_from_formatted(batch)
        ))
        engine._call_host = AsyncMock(
            return_value=json.dumps([
                _closed_judgment(reference, "supported"),
            ]),
        )

        verdicts = await engine._host_judge(claims_mgr, round_num=1)

        assert verdicts[0]["claim"] == keyword

    async def test_host_judge_forces_tiny_oversized_keyword_batch_to_continue(self):
        from discuss_agent.claims import Claim, ClaimEntry, ClaimsManager
        from discuss_agent.engine import DiscussionEngine

        engine = DiscussionEngine(_make_config(num_agents=1))
        claims_mgr = ClaimsManager()
        keyword = "keyword-" + ("x" * 5_000)
        claims_mgr.claims[keyword] = Claim(keyword, "OPEN", [
            ClaimEntry("FROM", "Agent-A", 1, "evidence"),
        ])
        batch = claims_mgr.format_host_candidate_batches(
            max_chars=120,
            max_claims=1,
        )[0]
        reference = next(iter(
            ClaimsManager.claim_keywords_from_formatted(batch)
        ))
        claims_mgr.format_host_candidate_batches = MagicMock(
            return_value=[batch],
        )
        engine._call_host = AsyncMock(return_value=json.dumps([{
            "claim": reference,
            "verdict": "CLOSED:共识",
            "reason": "supported",
        }]))

        verdicts = await engine._host_judge(claims_mgr, round_num=1)

        assert verdicts[0]["claim"] == keyword
        assert verdicts[0]["verdict"] == "CONTINUE"

    async def test_host_references_cannot_collide_with_real_keyword(self):
        from discuss_agent.claims import Claim, ClaimEntry, ClaimsManager
        from discuss_agent.engine import DiscussionEngine

        engine = DiscussionEngine(_make_config(num_agents=1))
        claims_mgr = ClaimsManager()
        long_keyword = "keyword-" + ("x" * 5_000)
        colliding_keyword = ClaimsManager.host_reference(long_keyword)
        claims_mgr.claims = {
            long_keyword: Claim(long_keyword, "OPEN", [
                ClaimEntry("FROM", "Agent-A", 1, "long evidence"),
            ]),
            colliding_keyword: Claim(colliding_keyword, "OPEN", [
                ClaimEntry("FROM", "Agent-A", 1, "short evidence"),
            ]),
        }
        references = ClaimsManager.build_host_references(
            claims_mgr.get_host_candidates(),
        )
        batches = claims_mgr.format_host_candidate_batches()
        offered = set().union(*(
            ClaimsManager.claim_keywords_from_formatted(batch)
            for batch in batches
        ))
        engine._call_host = AsyncMock(side_effect=[
            json.dumps([
                _closed_judgment(reference, "supported")
                for reference in ClaimsManager.claim_keywords_from_formatted(
                    batch,
                )
            ])
            for batch in batches
        ])

        verdicts = await engine._host_judge(claims_mgr, round_num=1)

        assert len(set(references.values())) == 2
        assert offered == set(references.values())
        assert {item["claim"] for item in verdicts} == {
            long_keyword,
            colliding_keyword,
        }

    async def test_host_judge_accepts_bracketed_claim_keyword(self):
        from discuss_agent.claims import Claim, ClaimsManager
        from discuss_agent.engine import DiscussionEngine

        engine = DiscussionEngine(_make_config(num_agents=1))
        claims_mgr = ClaimsManager()
        claims_mgr.claims["EPS [FY26]"] = Claim("EPS [FY26]", "OPEN")
        engine._call_host = AsyncMock(
            return_value=json.dumps([
                _closed_judgment("EPS [FY26]", "supported"),
            ]),
        )

        verdicts = await engine._host_judge(claims_mgr, round_num=1)

        assert verdicts == [_closed_judgment("EPS [FY26]", "supported")]

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
        engine._call_host = AsyncMock(return_value=json.dumps([
            _continue_judgment(
                "first",
                "source unavailable",
                "decisive source",
            ),
        ]))

        verdicts = await engine._host_judge(claims_mgr, round_num=3)

        prompt = engine._call_host.await_args.args[1]
        assert verdicts[0]["verdict"] == "CONTINUE"
        assert "安全上限" in prompt
        assert "最终语义裁决" in prompt
        assert "保持 OPEN" in prompt
        assert "不得因回应数量" in prompt
        assert "UNKNOWN是否会改变结论" in prompt
