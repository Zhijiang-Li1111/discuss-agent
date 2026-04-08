"""Tests for DiscussionEngine in discuss_agent.engine."""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

from discuss_agent.config import (
    AgentConfig,
    DiscussionConfig,
    HostConfig,
    ModelConfig,
    ToolConfig,
)
from discuss_agent.models import AgentUtterance, RoundRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockRunOutput:
    """Stand-in for agno RunOutput."""

    def __init__(self, content: str | None):
        self.content = content


def _make_config(
    min_rounds: int = 2,
    max_rounds: int = 5,
    num_agents: int = 2,
    tools: list[ToolConfig] | None = None,
    context_builder: str | None = None,
    agent_overrides: dict | None = None,
) -> DiscussionConfig:
    agents = []
    for i in range(num_agents):
        name = f"Agent-{chr(65 + i)}"
        overrides = (agent_overrides or {}).get(name, {})
        agents.append(
            AgentConfig(
                name=name,
                system_prompt=f"You are agent {chr(65 + i)}.",
                extra_tools=overrides.get("extra_tools", []),
                disable_tools=overrides.get("disable_tools", []),
            )
        )
    return DiscussionConfig(
        min_rounds=min_rounds,
        max_rounds=max_rounds,
        model_config=ModelConfig(model="claude-sonnet-4-20250514"),
        agents=agents,
        host=HostConfig(
            convergence_prompt="Judge convergence.",
            summary_prompt="Summarize the discussion.",
        ),
        tools=tools or [],
        context={},
        context_builder=context_builder,
    )


# ---------------------------------------------------------------------------
# Agent Creation
# ---------------------------------------------------------------------------


class TestAgentCreation:
    @patch("discuss_agent.engine.import_from_path")
    @patch("discuss_agent.engine.ContextManager")
    @patch("discuss_agent.engine.Agent")
    def test_creates_agents_from_config(self, MockAgent, MockCtxMgr, mock_import):
        from discuss_agent.engine import DiscussionEngine

        config = _make_config(num_agents=3)
        engine = DiscussionEngine(config)

        assert MockAgent.call_count == 4
        agent_calls = MockAgent.call_args_list
        discussion_names = [c.kwargs["name"] for c in agent_calls[:3]]
        assert discussion_names == ["Agent-A", "Agent-B", "Agent-C"]
        assert agent_calls[3].kwargs["name"] == "Host"


# ---------------------------------------------------------------------------
# Per-agent Tools
# ---------------------------------------------------------------------------


class TestPerAgentTools:

    @patch("discuss_agent.engine.ContextManager")
    @patch("discuss_agent.engine.Agent")
    def test_global_tools_inherited(self, MockAgent, MockCtxMgr):
        from discuss_agent.engine import DiscussionEngine

        class FakeTool:
            def __init__(self, context=None):
                pass

        with patch("discuss_agent.engine.import_from_path", return_value=FakeTool):
            config = _make_config(
                num_agents=2,
                tools=[ToolConfig(path="pkg.FakeTool")],
            )
            engine = DiscussionEngine(config)

        for call_obj in MockAgent.call_args_list[:2]:
            tools_arg = call_obj.kwargs.get("tools")
            assert tools_arg is not None
            assert len(tools_arg) == 1

    @patch("discuss_agent.engine.ContextManager")
    @patch("discuss_agent.engine.Agent")
    def test_extra_tools_added(self, MockAgent, MockCtxMgr):
        from discuss_agent.engine import DiscussionEngine

        class FakeGlobal:
            def __init__(self, context=None):
                pass

        class FakeExtra:
            def __init__(self, context=None):
                pass

        def mock_import(path):
            if path == "pkg.FakeGlobal":
                return FakeGlobal
            if path == "pkg.FakeExtra":
                return FakeExtra
            raise ImportError(path)

        with patch("discuss_agent.engine.import_from_path", side_effect=mock_import):
            config = _make_config(
                num_agents=2,
                tools=[ToolConfig(path="pkg.FakeGlobal")],
                agent_overrides={
                    "Agent-A": {"extra_tools": [ToolConfig(path="pkg.FakeExtra")]},
                },
            )
            engine = DiscussionEngine(config)

        agent_a_tools = MockAgent.call_args_list[0].kwargs.get("tools")
        assert len(agent_a_tools) == 2
        agent_b_tools = MockAgent.call_args_list[1].kwargs.get("tools")
        assert len(agent_b_tools) == 1

    @patch("discuss_agent.engine.ContextManager")
    @patch("discuss_agent.engine.Agent")
    def test_disable_tools_removed(self, MockAgent, MockCtxMgr):
        from discuss_agent.engine import DiscussionEngine

        class FakeToolA:
            def __init__(self, context=None):
                pass

        class FakeToolB:
            def __init__(self, context=None):
                pass

        def mock_import(path):
            if path == "pkg.FakeToolA":
                return FakeToolA
            if path == "pkg.FakeToolB":
                return FakeToolB
            raise ImportError(path)

        with patch("discuss_agent.engine.import_from_path", side_effect=mock_import):
            config = _make_config(
                num_agents=2,
                tools=[
                    ToolConfig(path="pkg.FakeToolA"),
                    ToolConfig(path="pkg.FakeToolB"),
                ],
                agent_overrides={
                    "Agent-A": {"disable_tools": ["pkg.FakeToolA"]},
                },
            )
            engine = DiscussionEngine(config)

        agent_a_tools = MockAgent.call_args_list[0].kwargs.get("tools")
        assert len(agent_a_tools) == 1
        agent_b_tools = MockAgent.call_args_list[1].kwargs.get("tools")
        assert len(agent_b_tools) == 2

    @patch("discuss_agent.engine.ContextManager")
    @patch("discuss_agent.engine.Agent")
    def test_disable_nonexistent_warns(self, MockAgent, MockCtxMgr, caplog):
        from discuss_agent.engine import DiscussionEngine

        with patch("discuss_agent.engine.import_from_path"):
            with caplog.at_level(logging.WARNING, logger="discuss_agent.engine"):
                config = _make_config(
                    num_agents=1,
                    agent_overrides={
                        "Agent-A": {"disable_tools": ["pkg.NonExistent"]},
                    },
                )
                engine = DiscussionEngine(config)

        assert "pkg.NonExistent" in caplog.text
        assert "does not match" in caplog.text

    @patch("discuss_agent.engine.ContextManager")
    @patch("discuss_agent.engine.Agent")
    def test_duplicate_in_extra_deduped(self, MockAgent, MockCtxMgr):
        from discuss_agent.engine import DiscussionEngine

        class FakeTool:
            def __init__(self, context=None):
                pass

        with patch("discuss_agent.engine.import_from_path", return_value=FakeTool):
            config = _make_config(
                num_agents=1,
                tools=[ToolConfig(path="pkg.FakeTool")],
                agent_overrides={
                    "Agent-A": {"extra_tools": [ToolConfig(path="pkg.FakeTool")]},
                },
            )
            engine = DiscussionEngine(config)

        agent_tools = MockAgent.call_args_list[0].kwargs.get("tools")
        assert len(agent_tools) == 1


# ---------------------------------------------------------------------------
# Express (Parallel)
# ---------------------------------------------------------------------------


class TestExpress:

    @patch("discuss_agent.engine.import_from_path")
    @patch("discuss_agent.engine.ContextManager")
    @pytest.mark.asyncio
    async def test_express_returns_utterances(self, MockCtxMgr, mock_import):
        from discuss_agent.engine import DiscussionEngine

        mock_agents = [MagicMock(name="Agent-A"), MagicMock(name="Agent-B")]
        mock_agents[0].name = "Agent-A"
        mock_agents[1].name = "Agent-B"
        mock_host = MagicMock(name="Host")
        mock_host.name = "Host"

        with patch("discuss_agent.engine.Agent", side_effect=mock_agents + [mock_host]):
            config = _make_config(num_agents=2)
            engine = DiscussionEngine(config)

        async def mock_safe_call(agent, prompt):
            return f"Opinion from {agent.name}"

        with patch.object(engine, "_safe_agent_call", side_effect=mock_safe_call):
            result = await engine._express(1, "context", [])

        assert len(result) == 2
        assert all(isinstance(u, AgentUtterance) for u in result)
        names = {u.agent_name for u in result}
        assert names == {"Agent-A", "Agent-B"}

    @patch("discuss_agent.engine.import_from_path")
    @patch("discuss_agent.engine.ContextManager")
    @patch("discuss_agent.engine.Agent")
    @pytest.mark.asyncio
    async def test_express_prompt_includes_history(self, MockAgent, MockCtxMgr, mock_import):
        from discuss_agent.engine import DiscussionEngine

        config = _make_config(num_agents=1)
        engine = DiscussionEngine(config)
        engine._agents[0].name = "Agent-A"

        history = [
            RoundRecord(
                round_num=1,
                expressions=[AgentUtterance("Agent-A", "First opinion")],
                challenges=[AgentUtterance("Agent-A", "First challenge")],
            )
        ]

        captured_prompts = []

        async def mock_safe_call(agent, prompt):
            captured_prompts.append(prompt)
            return "Response"

        with patch.object(engine, "_safe_agent_call", side_effect=mock_safe_call):
            await engine._express(2, "context-text", history)

        assert len(captured_prompts) == 1
        prompt = captured_prompts[0]
        assert "First opinion" in prompt
        assert "First challenge" in prompt
        assert "context-text" in prompt


# ---------------------------------------------------------------------------
# Challenge (Parallel)
# ---------------------------------------------------------------------------


class TestChallenge:

    @patch("discuss_agent.engine.import_from_path")
    @patch("discuss_agent.engine.ContextManager")
    @pytest.mark.asyncio
    async def test_challenge_excludes_own_expression(self, MockCtxMgr, mock_import):
        from discuss_agent.engine import DiscussionEngine

        mock_agents = [MagicMock(), MagicMock()]
        mock_agents[0].name = "Agent-A"
        mock_agents[1].name = "Agent-B"
        mock_host = MagicMock()
        mock_host.name = "Host"

        with patch("discuss_agent.engine.Agent", side_effect=mock_agents + [mock_host]):
            config = _make_config(num_agents=2)
            engine = DiscussionEngine(config)

        expressions = [
            AgentUtterance("Agent-A", "Opinion from A"),
            AgentUtterance("Agent-B", "Opinion from B"),
        ]

        captured = {}

        async def mock_safe_call(agent, prompt):
            captured[agent.name] = prompt
            return f"Challenge from {agent.name}"

        with patch.object(engine, "_safe_agent_call", side_effect=mock_safe_call):
            result = await engine._challenge(1, expressions)

        assert "[Agent-A]" not in captured["Agent-A"]
        assert "Opinion from A" not in captured["Agent-A"]
        assert "Opinion from B" in captured["Agent-A"]
        assert "[Agent-B]" not in captured["Agent-B"]
        assert "Opinion from B" not in captured["Agent-B"]
        assert "Opinion from A" in captured["Agent-B"]


# ---------------------------------------------------------------------------
# Host Judgment
# ---------------------------------------------------------------------------


class TestHostJudgment:

    @patch("discuss_agent.engine.import_from_path")
    @patch("discuss_agent.engine.ContextManager")
    @patch("discuss_agent.engine.Agent")
    @pytest.mark.asyncio
    async def test_host_judge_parses_json(self, MockAgent, MockCtxMgr, mock_import):
        from discuss_agent.engine import DiscussionEngine

        config = _make_config()
        engine = DiscussionEngine(config)

        json_response = '{"converged": true, "reason": "All agree", "remaining_disputes": []}'
        engine._host.arun = AsyncMock(return_value=MockRunOutput(json_response))

        history = [
            RoundRecord(
                round_num=1,
                expressions=[AgentUtterance("Agent-A", "Opinion")],
                challenges=[AgentUtterance("Agent-A", "Challenge")],
            )
        ]

        result = await engine._host_judge(history)
        assert result["converged"] is True
        assert result["reason"] == "All agree"

    @patch("discuss_agent.engine.import_from_path")
    @patch("discuss_agent.engine.ContextManager")
    @patch("discuss_agent.engine.Agent")
    @pytest.mark.asyncio
    async def test_host_judge_malformed_defaults_not_converged(
        self, MockAgent, MockCtxMgr, mock_import
    ):
        from discuss_agent.engine import DiscussionEngine

        config = _make_config()
        engine = DiscussionEngine(config)
        engine._host.arun = AsyncMock(return_value=MockRunOutput("This is not JSON at all!!!"))

        history = [
            RoundRecord(
                round_num=1,
                expressions=[AgentUtterance("Agent-A", "Opinion")],
                challenges=[AgentUtterance("Agent-A", "Challenge")],
            )
        ]

        result = await engine._host_judge(history)
        assert result["converged"] is False


# ---------------------------------------------------------------------------
# Host Summary
# ---------------------------------------------------------------------------


class TestHostSummary:

    @patch("discuss_agent.engine.import_from_path")
    @patch("discuss_agent.engine.ContextManager")
    @patch("discuss_agent.engine.Agent")
    @pytest.mark.asyncio
    async def test_host_summarize_returns_content(self, MockAgent, MockCtxMgr, mock_import):
        from discuss_agent.engine import DiscussionEngine

        config = _make_config()
        engine = DiscussionEngine(config)

        history = [
            RoundRecord(
                round_num=1,
                expressions=[AgentUtterance("Agent-A", "Opinion")],
                challenges=[AgentUtterance("Agent-A", "Challenge")],
            )
        ]

        summary_text = "Final summary: everyone agrees on X."

        with patch("discuss_agent.engine.Agent") as PatchedAgent:
            mock_summary_agent = MagicMock()
            mock_summary_agent.arun = AsyncMock(return_value=MockRunOutput(summary_text))
            PatchedAgent.return_value = mock_summary_agent

            result = await engine._host_summarize(history)

        assert result == summary_text


# ---------------------------------------------------------------------------
# Error Handling
# ---------------------------------------------------------------------------


class TestErrorHandling:

    @patch("discuss_agent.engine.import_from_path")
    @patch("discuss_agent.engine.ContextManager")
    @pytest.mark.asyncio
    async def test_single_agent_failure_retries_and_skips(self, MockCtxMgr, mock_import):
        from discuss_agent.engine import DiscussionEngine

        mock_agents = [MagicMock(), MagicMock()]
        mock_agents[0].name = "Agent-A"
        mock_agents[1].name = "Agent-B"
        mock_host = MagicMock()
        mock_host.name = "Host"

        with patch("discuss_agent.engine.Agent", side_effect=mock_agents + [mock_host]):
            config = _make_config(num_agents=2)
            engine = DiscussionEngine(config)

        async def mock_safe_call(agent, prompt):
            if agent.name == "Agent-A":
                return None
            return "Agent-B opinion"

        with patch.object(engine, "_safe_agent_call", side_effect=mock_safe_call):
            result = await engine._express(1, "context", [])

        assert len(result) == 1
        assert result[0].agent_name == "Agent-B"

    @patch("discuss_agent.engine.import_from_path")
    @patch("discuss_agent.engine.ContextManager")
    @pytest.mark.asyncio
    async def test_all_agents_fail_raises(self, MockCtxMgr, mock_import):
        from discuss_agent.engine import DiscussionEngine, AllAgentsFailedError

        mock_agents = [MagicMock(), MagicMock()]
        mock_agents[0].name = "Agent-A"
        mock_agents[1].name = "Agent-B"
        mock_host = MagicMock()
        mock_host.name = "Host"

        with patch("discuss_agent.engine.Agent", side_effect=mock_agents + [mock_host]):
            config = _make_config(num_agents=2)
            engine = DiscussionEngine(config)

        async def mock_safe_call(agent, prompt):
            return None

        with patch.object(engine, "_safe_agent_call", side_effect=mock_safe_call):
            with pytest.raises(AllAgentsFailedError):
                await engine._express(1, "context", [])


# ---------------------------------------------------------------------------
# Main Loop
# ---------------------------------------------------------------------------


class TestMainLoop:

    def _patch_engine(self, engine, judgments):
        round_counter = {"n": 0}

        async def mock_express(round_num, context, history):
            return [
                AgentUtterance("Agent-A", f"Expr-A-R{round_num}"),
                AgentUtterance("Agent-B", f"Expr-B-R{round_num}"),
            ]

        async def mock_challenge(round_num, expressions):
            return [
                AgentUtterance("Agent-A", f"Chal-A-R{round_num}"),
                AgentUtterance("Agent-B", f"Chal-B-R{round_num}"),
            ]

        async def mock_host_judge(history):
            idx = round_counter["n"]
            round_counter["n"] += 1
            if idx < len(judgments):
                return judgments[idx]
            return {"converged": False, "reason": "", "remaining_disputes": []}

        async def mock_host_summarize(history):
            return "Final summary content"

        engine._express = AsyncMock(side_effect=mock_express)
        engine._challenge = AsyncMock(side_effect=mock_challenge)
        engine._host_judge = AsyncMock(side_effect=mock_host_judge)
        engine._host_summarize = AsyncMock(side_effect=mock_host_summarize)

    @patch("discuss_agent.engine.import_from_path")
    @patch("discuss_agent.engine.ContextManager")
    @patch("discuss_agent.engine.Agent")
    @pytest.mark.asyncio
    async def test_full_loop_converges(self, MockAgent, MockCtxMgr, mock_import):
        from discuss_agent.engine import DiscussionEngine

        config = _make_config(min_rounds=2, max_rounds=5)
        engine = DiscussionEngine(config)

        engine._archiver = MagicMock()
        engine._archiver.start_session.return_value = "/tmp/session"
        engine._context_mgr.build_initial_context = AsyncMock(return_value="Initial context")
        engine._context_mgr.compress = AsyncMock(side_effect=lambda h, r: h)

        judgments = [
            {"converged": False, "reason": "Still debating", "remaining_disputes": ["topic X"]},
            {"converged": True, "reason": "Agreement reached", "remaining_disputes": []},
        ]
        self._patch_engine(engine, judgments)

        result = await engine.run()

        assert result.converged is True
        assert result.rounds_completed == 2
        assert result.archive_path == "/tmp/session"
        assert result.summary == "Final summary content"
        assert result.remaining_disputes == []

    @patch("discuss_agent.engine.import_from_path")
    @patch("discuss_agent.engine.ContextManager")
    @patch("discuss_agent.engine.Agent")
    @pytest.mark.asyncio
    async def test_min_rounds_enforced(self, MockAgent, MockCtxMgr, mock_import):
        from discuss_agent.engine import DiscussionEngine

        config = _make_config(min_rounds=2, max_rounds=5)
        engine = DiscussionEngine(config)

        engine._archiver = MagicMock()
        engine._archiver.start_session.return_value = "/tmp/session"
        engine._context_mgr.build_initial_context = AsyncMock(return_value="ctx")
        engine._context_mgr.compress = AsyncMock(side_effect=lambda h, r: h)

        judgments = [
            {"converged": True, "reason": "Early convergence", "remaining_disputes": []},
            {"converged": True, "reason": "Still converged", "remaining_disputes": []},
        ]
        self._patch_engine(engine, judgments)

        result = await engine.run()

        assert result.converged is True
        assert result.rounds_completed == 2

    @patch("discuss_agent.engine.import_from_path")
    @patch("discuss_agent.engine.ContextManager")
    @patch("discuss_agent.engine.Agent")
    @pytest.mark.asyncio
    async def test_max_rounds_enforced(self, MockAgent, MockCtxMgr, mock_import):
        from discuss_agent.engine import DiscussionEngine

        config = _make_config(min_rounds=1, max_rounds=3)
        engine = DiscussionEngine(config)

        engine._archiver = MagicMock()
        engine._archiver.start_session.return_value = "/tmp/session"
        engine._context_mgr.build_initial_context = AsyncMock(return_value="ctx")
        engine._context_mgr.compress = AsyncMock(side_effect=lambda h, r: h)

        judgments = [
            {"converged": False, "reason": "No", "remaining_disputes": ["X"]},
            {"converged": False, "reason": "No", "remaining_disputes": ["X"]},
            {"converged": False, "reason": "No", "remaining_disputes": ["X", "Y"]},
        ]
        self._patch_engine(engine, judgments)

        result = await engine.run()

        assert result.converged is False
        assert result.rounds_completed == 3
        assert result.summary is None
        assert "X" in result.remaining_disputes

    @patch("discuss_agent.engine.import_from_path")
    @patch("discuss_agent.engine.ContextManager")
    @patch("discuss_agent.engine.Agent")
    @pytest.mark.asyncio
    async def test_error_termination(self, MockAgent, MockCtxMgr, mock_import):
        from discuss_agent.engine import DiscussionEngine, AllAgentsFailedError

        config = _make_config(min_rounds=1, max_rounds=5)
        engine = DiscussionEngine(config)

        engine._archiver = MagicMock()
        engine._archiver.start_session.return_value = "/tmp/session"
        engine._context_mgr.build_initial_context = AsyncMock(return_value="ctx")
        engine._context_mgr.compress = AsyncMock(side_effect=lambda h, r: h)

        call_count = {"n": 0}

        async def mock_express(round_num, context, history):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return [AgentUtterance("Agent-A", "R1 expression")]
            raise AllAgentsFailedError("All agents failed")

        async def mock_challenge(round_num, expressions):
            return [AgentUtterance("Agent-A", "R1 challenge")]

        async def mock_host_judge(history):
            return {"converged": False, "reason": "No", "remaining_disputes": []}

        engine._express = AsyncMock(side_effect=mock_express)
        engine._challenge = AsyncMock(side_effect=mock_challenge)
        engine._host_judge = AsyncMock(side_effect=mock_host_judge)

        result = await engine.run()

        assert result.terminated_by_error is True
        assert result.rounds_completed == 1
        assert result.converged is False

        save_round_calls = engine._archiver.save_round.call_args_list
        phases_r1 = [c.args[1] for c in save_round_calls if c.args[0] == 1]
        assert "express" in phases_r1
        assert "challenge" in phases_r1
        assert "host" in phases_r1

        engine._archiver.save_error_log.assert_called_once()

    @patch("discuss_agent.engine.import_from_path")
    @patch("discuss_agent.engine.ContextManager")
    @patch("discuss_agent.engine.Agent")
    @pytest.mark.asyncio
    async def test_three_save_round_calls_per_round(self, MockAgent, MockCtxMgr, mock_import):
        from discuss_agent.engine import DiscussionEngine

        config = _make_config(min_rounds=1, max_rounds=2)
        engine = DiscussionEngine(config)

        engine._archiver = MagicMock()
        engine._archiver.start_session.return_value = "/tmp/session"
        engine._context_mgr.build_initial_context = AsyncMock(return_value="ctx")
        engine._context_mgr.compress = AsyncMock(side_effect=lambda h, r: h)

        judgments = [
            {"converged": False, "reason": "No", "remaining_disputes": []},
            {"converged": True, "reason": "Yes", "remaining_disputes": []},
        ]
        self._patch_engine(engine, judgments)

        result = await engine.run()

        save_round_calls = engine._archiver.save_round.call_args_list
        assert len(save_round_calls) == 6

    @patch("discuss_agent.engine.import_from_path")
    @patch("discuss_agent.engine.ContextManager")
    @patch("discuss_agent.engine.Agent")
    @pytest.mark.asyncio
    async def test_no_tools_empty_list_runs(self, MockAgent, MockCtxMgr, mock_import):
        from discuss_agent.engine import DiscussionEngine

        config = _make_config(min_rounds=1, max_rounds=1)
        engine = DiscussionEngine(config)

        engine._archiver = MagicMock()
        engine._archiver.start_session.return_value = "/tmp/session"
        engine._context_mgr.build_initial_context = AsyncMock(return_value="")
        engine._context_mgr.compress = AsyncMock(side_effect=lambda h, r: h)

        judgments = [
            {"converged": True, "reason": "Quick agreement", "remaining_disputes": []},
        ]
        self._patch_engine(engine, judgments)

        result = await engine.run()

        assert result.converged is True
        assert result.rounds_completed == 1
