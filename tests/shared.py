"""Shared test helpers used by test_engine.py and test_resume.py."""

from __future__ import annotations

from unittest.mock import AsyncMock

from discuss_agent.config import (
    AgentConfig,
    DiscussionConfig,
    HostConfig,
    ModelConfig,
    ToolConfig,
)
from discuss_agent.models import AgentUtterance


class MockRunOutput:
    """Stand-in for agno RunOutput."""

    def __init__(self, content: str | None):
        self.content = content


def make_config(
    min_rounds: int = 2,
    max_rounds: int = 5,
    num_agents: int = 2,
    tools: list[ToolConfig] | None = None,
    context_builder: str | None = None,
    agent_overrides: dict | None = None,
    limitation: str | None = None,
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
        limitation=limitation,
    )


def patch_engine(engine, judgments):
    """Replace internal engine methods with deterministic mocks."""
    round_counter = {"n": 0}

    async def mock_express(round_num, context, history, **kwargs):
        return [
            AgentUtterance("Agent-A", f"Expr-A-R{round_num}"),
            AgentUtterance("Agent-B", f"Expr-B-R{round_num}"),
        ]

    async def mock_challenge(round_num, expressions, **kwargs):
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
