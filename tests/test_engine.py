"""Tests for DiscussionEngine in discuss_agent.engine."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

from discuss_agent.config import (
    AgentConfig,
    ContextConfig,
    DiscussionConfig,
    HostConfig,
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
) -> DiscussionConfig:
    agents = [
        AgentConfig(name=f"Agent-{chr(65 + i)}", system_prompt=f"You are agent {chr(65 + i)}.")
        for i in range(num_agents)
    ]
    return DiscussionConfig(
        min_rounds=min_rounds,
        max_rounds=max_rounds,
        model="claude-sonnet-4-20250514",
        agents=agents,
        host=HostConfig(
            convergence_prompt="Judge convergence.",
            summary_prompt="Summarize the discussion.",
        ),
        tools=[],
        context=ContextConfig(
            research_dir="~/research",
            published_file="PUBLISHED.md",
            research_days=2,
        ),
    )


# ---------------------------------------------------------------------------
# Task 13 — Agent Creation
# ---------------------------------------------------------------------------


class TestAgentCreation:
    """Verify that DiscussionEngine creates the correct agents."""

    @patch("discuss_agent.engine.get_tools", return_value=[])
    @patch("discuss_agent.engine.ContextManager")
    @patch("discuss_agent.engine.Agent")
    def test_creates_agents_from_config(self, MockAgent, MockCtxMgr, mock_get_tools):
        """Engine should create len(config.agents) discussion agents plus 1 host."""
        from discuss_agent.engine import DiscussionEngine

        config = _make_config(num_agents=3)
        engine = DiscussionEngine(config)

        # 3 discussion agents + 1 host = 4 Agent() calls
        assert MockAgent.call_count == 4, (
            f"Expected 4 Agent() calls (3 agents + 1 host), got {MockAgent.call_count}"
        )

        # Verify discussion agent names
        agent_calls = MockAgent.call_args_list
        discussion_names = [c.kwargs["name"] for c in agent_calls[:3]]
        assert discussion_names == ["Agent-A", "Agent-B", "Agent-C"]

        # Verify host name
        host_name = agent_calls[3].kwargs["name"]
        assert host_name == "Host"


# ---------------------------------------------------------------------------
# Task 14 — Express (Parallel)
# ---------------------------------------------------------------------------


class TestExpress:
    """Verify the _express step."""

    @patch("discuss_agent.engine.get_tools", return_value=[])
    @patch("discuss_agent.engine.ContextManager")
    @pytest.mark.asyncio
    async def test_express_returns_utterances(self, MockCtxMgr, mock_get_tools):
        """_express should return AgentUtterance for each agent that succeeds."""
        from discuss_agent.engine import DiscussionEngine

        # Create distinct mock agents so each has its own .name
        mock_agents = [MagicMock(name="Agent-A"), MagicMock(name="Agent-B")]
        mock_agents[0].name = "Agent-A"
        mock_agents[1].name = "Agent-B"
        mock_host = MagicMock(name="Host")
        mock_host.name = "Host"

        with patch("discuss_agent.engine.Agent", side_effect=mock_agents + [mock_host]):
            config = _make_config(num_agents=2)
            engine = DiscussionEngine(config)

        # Mock _safe_agent_call to return content keyed by agent name
        async def mock_safe_call(agent, prompt):
            return f"Opinion from {agent.name}"

        with patch.object(engine, "_safe_agent_call", side_effect=mock_safe_call):
            result = await engine._express(1, "context", [])

        assert len(result) == 2
        assert all(isinstance(u, AgentUtterance) for u in result)
        names = {u.agent_name for u in result}
        assert names == {"Agent-A", "Agent-B"}

    @patch("discuss_agent.engine.get_tools", return_value=[])
    @patch("discuss_agent.engine.ContextManager")
    @patch("discuss_agent.engine.Agent")
    @pytest.mark.asyncio
    async def test_express_prompt_includes_history(self, MockAgent, MockCtxMgr, mock_get_tools):
        """The prompt passed to agents should include the formatted history."""
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
# Task 15 — Challenge (Parallel)
# ---------------------------------------------------------------------------


class TestChallenge:
    """Verify the _challenge step."""

    @patch("discuss_agent.engine.get_tools", return_value=[])
    @patch("discuss_agent.engine.ContextManager")
    @pytest.mark.asyncio
    async def test_challenge_excludes_own_expression(self, MockCtxMgr, mock_get_tools):
        """Each agent's challenge prompt must exclude its own expression."""
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

        # Agent-A's prompt should NOT contain Agent-A's expression
        assert "[Agent-A]" not in captured["Agent-A"]
        assert "Opinion from A" not in captured["Agent-A"]
        # But SHOULD contain Agent-B's expression
        assert "Opinion from B" in captured["Agent-A"]

        # Agent-B's prompt should NOT contain Agent-B's expression
        assert "[Agent-B]" not in captured["Agent-B"]
        assert "Opinion from B" not in captured["Agent-B"]
        # But SHOULD contain Agent-A's expression
        assert "Opinion from A" in captured["Agent-B"]


# ---------------------------------------------------------------------------
# Task 16 — Host Judgment
# ---------------------------------------------------------------------------


class TestHostJudgment:
    """Verify host convergence judgment parsing."""

    @patch("discuss_agent.engine.get_tools", return_value=[])
    @patch("discuss_agent.engine.ContextManager")
    @patch("discuss_agent.engine.Agent")
    @pytest.mark.asyncio
    async def test_host_judge_parses_json(self, MockAgent, MockCtxMgr, mock_get_tools):
        """Host judge should parse valid JSON with converged=true."""
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
        assert result["remaining_disputes"] == []

    @patch("discuss_agent.engine.get_tools", return_value=[])
    @patch("discuss_agent.engine.ContextManager")
    @patch("discuss_agent.engine.Agent")
    @pytest.mark.asyncio
    async def test_host_judge_malformed_defaults_not_converged(
        self, MockAgent, MockCtxMgr, mock_get_tools
    ):
        """Malformed host response should default to converged=False."""
        from discuss_agent.engine import DiscussionEngine

        config = _make_config()
        engine = DiscussionEngine(config)

        # Return garbage for both attempts
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
# Task 17 — Host Summary
# ---------------------------------------------------------------------------


class TestHostSummary:
    """Verify host summary generation."""

    @patch("discuss_agent.engine.get_tools", return_value=[])
    @patch("discuss_agent.engine.ContextManager")
    @patch("discuss_agent.engine.Agent")
    @pytest.mark.asyncio
    async def test_host_summarize_returns_content(self, MockAgent, MockCtxMgr, mock_get_tools):
        """_host_summarize should return the content from the summary agent."""
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

        # Patch Agent constructor so _host_summarize's internal Agent returns our content
        with patch("discuss_agent.engine.Agent") as PatchedAgent:
            mock_summary_agent = MagicMock()
            mock_summary_agent.arun = AsyncMock(return_value=MockRunOutput(summary_text))
            PatchedAgent.return_value = mock_summary_agent

            result = await engine._host_summarize(history)

        assert result == summary_text
        assert len(result) > 0


# ---------------------------------------------------------------------------
# Task 18 — Error Handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Verify retry and error handling in agent calls."""

    @patch("discuss_agent.engine.get_tools", return_value=[])
    @patch("discuss_agent.engine.ContextManager")
    @pytest.mark.asyncio
    async def test_single_agent_failure_retries_and_skips(
        self, MockCtxMgr, mock_get_tools
    ):
        """If one agent fails, _express should still return utterances from the other."""
        from discuss_agent.engine import DiscussionEngine

        mock_agents = [MagicMock(), MagicMock()]
        mock_agents[0].name = "Agent-A"
        mock_agents[1].name = "Agent-B"
        mock_host = MagicMock()
        mock_host.name = "Host"

        with patch("discuss_agent.engine.Agent", side_effect=mock_agents + [mock_host]):
            config = _make_config(num_agents=2)
            engine = DiscussionEngine(config)

        # Agent-A always fails, Agent-B succeeds
        async def mock_safe_call(agent, prompt):
            if agent.name == "Agent-A":
                return None  # simulates failure after retries
            return "Agent-B opinion"

        with patch.object(engine, "_safe_agent_call", side_effect=mock_safe_call):
            result = await engine._express(1, "context", [])

        assert len(result) == 1
        assert result[0].agent_name == "Agent-B"

    @patch("discuss_agent.engine.get_tools", return_value=[])
    @patch("discuss_agent.engine.ContextManager")
    @pytest.mark.asyncio
    async def test_all_agents_fail_raises(self, MockCtxMgr, mock_get_tools):
        """If ALL agents fail, _express should raise AllAgentsFailedError."""
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
# Task 19 — Main Loop
# ---------------------------------------------------------------------------


class TestMainLoop:
    """Verify the full discussion loop."""

    def _patch_engine(self, engine, judgments):
        """Set up standard mocks for the main loop.

        Parameters
        ----------
        engine : DiscussionEngine
            The engine instance to patch.
        judgments : list[dict]
            One judgment per round. Must have len >= max_rounds or until convergence.
        """
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

    @patch("discuss_agent.engine.get_tools", return_value=[])
    @patch("discuss_agent.engine.ContextManager")
    @patch("discuss_agent.engine.Agent")
    @pytest.mark.asyncio
    async def test_full_loop_converges(self, MockAgent, MockCtxMgr, mock_get_tools):
        """When host converges at round 2 (>= min_rounds), result should show converged."""
        from discuss_agent.engine import DiscussionEngine

        config = _make_config(min_rounds=2, max_rounds=5)
        engine = DiscussionEngine(config)

        # Mock archiver
        engine._archiver = MagicMock()
        engine._archiver.start_session.return_value = "/tmp/session"

        # Mock context manager
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

    @patch("discuss_agent.engine.get_tools", return_value=[])
    @patch("discuss_agent.engine.ContextManager")
    @patch("discuss_agent.engine.Agent")
    @pytest.mark.asyncio
    async def test_min_rounds_enforced(self, MockAgent, MockCtxMgr, mock_get_tools):
        """Host converges at round 1 but min_rounds=2; should continue to round 2."""
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

        # Should NOT converge at round 1 because min_rounds=2
        assert result.converged is True
        assert result.rounds_completed == 2  # not 1

    @patch("discuss_agent.engine.get_tools", return_value=[])
    @patch("discuss_agent.engine.ContextManager")
    @patch("discuss_agent.engine.Agent")
    @pytest.mark.asyncio
    async def test_max_rounds_enforced(self, MockAgent, MockCtxMgr, mock_get_tools):
        """Host never converges; should stop at max_rounds with converged=False."""
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

    @patch("discuss_agent.engine.get_tools", return_value=[])
    @patch("discuss_agent.engine.ContextManager")
    @patch("discuss_agent.engine.Agent")
    @pytest.mark.asyncio
    async def test_error_termination(self, MockAgent, MockCtxMgr, mock_get_tools):
        """AllAgentsFailedError at round 2 => terminated_by_error, round 1 archived."""
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
        assert result.rounds_completed == 1  # round 1 completed, round 2 failed
        assert result.converged is False

        # Verify round 1 was archived (save_round called for express, challenge, host)
        save_round_calls = engine._archiver.save_round.call_args_list
        phases_r1 = [c.args[1] for c in save_round_calls if c.args[0] == 1]
        assert "express" in phases_r1
        assert "challenge" in phases_r1
        assert "host" in phases_r1

        # Verify error log was saved (AC-8.3)
        engine._archiver.save_error_log.assert_called_once()
        error_msg = engine._archiver.save_error_log.call_args.args[0]
        assert "All agents failed" in error_msg

    @patch("discuss_agent.engine.get_tools", return_value=[])
    @patch("discuss_agent.engine.ContextManager")
    @patch("discuss_agent.engine.Agent")
    @pytest.mark.asyncio
    async def test_three_save_round_calls_per_round(self, MockAgent, MockCtxMgr, mock_get_tools):
        """Each round should produce exactly 3 save_round calls: express, challenge, host."""
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
        # 2 rounds x 3 phases = 6 calls
        assert len(save_round_calls) == 6

        # Check per-round phases
        for round_num in [1, 2]:
            phases = [c.args[1] for c in save_round_calls if c.args[0] == round_num]
            assert phases == ["express", "challenge", "host"], (
                f"Round {round_num} should have phases [express, challenge, host], got {phases}"
            )
