"""Tests for discuss_agent.models data structures."""

import dataclasses

from discuss_agent.models import AgentUtterance, DiscussionResult, RoundRecord


class TestAgentUtterance:
    def test_instantiation(self):
        u = AgentUtterance(agent_name="Alice", content="I think X.")
        assert u.agent_name == "Alice"
        assert u.content == "I think X."

    def test_asdict(self):
        u = AgentUtterance(agent_name="Bob", content="Evidence Y.")
        d = dataclasses.asdict(u)
        assert d == {"agent_name": "Bob", "content": "Evidence Y."}


class TestRoundRecord:
    def test_instantiation_defaults(self):
        r = RoundRecord(
            round_num=1,
            expressions=[AgentUtterance("A", "expr")],
            challenges=[AgentUtterance("B", "chal")],
        )
        assert r.round_num == 1
        assert len(r.expressions) == 1
        assert len(r.challenges) == 1
        assert r.host_judgment is None
        assert r.is_summary is False
        assert r.summary_text is None

    def test_instantiation_with_all_fields(self):
        r = RoundRecord(
            round_num=3,
            expressions=[],
            challenges=[],
            host_judgment={"converged": True, "reason": "agreed"},
            is_summary=True,
            summary_text="Final summary.",
        )
        assert r.host_judgment == {"converged": True, "reason": "agreed"}
        assert r.is_summary is True
        assert r.summary_text == "Final summary."

    def test_asdict(self):
        r = RoundRecord(
            round_num=2,
            expressions=[AgentUtterance("A", "hello")],
            challenges=[],
            host_judgment={"converged": False},
        )
        d = dataclasses.asdict(r)
        assert d["round_num"] == 2
        assert d["expressions"] == [{"agent_name": "A", "content": "hello"}]
        assert d["challenges"] == []
        assert d["host_judgment"] == {"converged": False}
        assert d["is_summary"] is False
        assert d["summary_text"] is None


class TestDiscussionResult:
    def test_instantiation_defaults(self):
        result = DiscussionResult(
            converged=True,
            rounds_completed=3,
            archive_path="/tmp/archive",
            summary="All agreed.",
            remaining_disputes=[],
        )
        assert result.converged is True
        assert result.rounds_completed == 3
        assert result.archive_path == "/tmp/archive"
        assert result.summary == "All agreed."
        assert result.remaining_disputes == []
        assert result.terminated_by_error is False

    def test_instantiation_with_error(self):
        result = DiscussionResult(
            converged=False,
            rounds_completed=1,
            archive_path="/tmp/err",
            summary=None,
            remaining_disputes=["topic A"],
            terminated_by_error=True,
        )
        assert result.terminated_by_error is True
        assert result.summary is None
        assert result.remaining_disputes == ["topic A"]

    def test_asdict(self):
        result = DiscussionResult(
            converged=False,
            rounds_completed=5,
            archive_path="/tmp/out",
            summary=None,
            remaining_disputes=["dispute1", "dispute2"],
            terminated_by_error=False,
        )
        d = dataclasses.asdict(result)
        assert d == {
            "converged": False,
            "rounds_completed": 5,
            "archive_path": "/tmp/out",
            "summary": None,
            "remaining_disputes": ["dispute1", "dispute2"],
            "terminated_by_error": False,
        }
