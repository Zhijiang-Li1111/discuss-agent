"""Data structures for the multi-agent discussion framework."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AgentUtterance:
    agent_name: str
    content: str


@dataclass
class RoundRecord:
    round_num: int
    expressions: list[AgentUtterance]
    challenges: list[AgentUtterance]
    host_judgment: dict | None = None
    is_summary: bool = False
    summary_text: str | None = None


@dataclass
class DiscussionResult:
    converged: bool
    rounds_completed: int
    archive_path: str
    summary: str | None
    remaining_disputes: list[str]
    terminated_by_error: bool = False
