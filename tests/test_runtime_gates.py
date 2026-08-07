from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from discuss_agent.audit import AuditLogger
from discuss_agent.config import AgentConfig, DiscussionConfig, HostConfig, ModelConfig, ToolConfig
from discuss_agent.conversation import AgentConversation


def _config(*, min_rounds=1, max_rounds=3, strict=False, tools=None, extra_tools=None):
    return DiscussionConfig(
        min_rounds=min_rounds,
        max_rounds=max_rounds,
        model_config=ModelConfig(model="claude-sonnet-4-20250514", api_key="test"),
        agents=[
            AgentConfig(name="A", system_prompt="A", extra_tools=extra_tools or []),
            AgentConfig(name="B", system_prompt="B"),
        ],
        host=HostConfig(convergence_prompt="judge", summary_prompt="summary", skip_summary=False),
        tools=tools or [],
        context={},
        strict_tool_loading=strict,
    )


def _reviewed_claim(
    keyword="X",
    *,
    proposer="A",
    reviewer="B",
    source="source://primary",
):
    from discuss_agent.claims import Claim, ClaimEntry

    return Claim(keyword, "OPEN", [
        ClaimEntry("FROM", proposer, 1, f"supported by {source}"),
        ClaimEntry("ACCEPT", reviewer, 1, "independently reviewed"),
    ])


def _closed_verdict(
    claim="X",
    *,
    reason="source verified and independently reviewed",
    **overrides,
):
    verdict = {
        "claim": claim,
        "verdict": "CLOSED:共识",
        "reason": reason,
    }
    verdict.update(overrides)
    return verdict


def test_config_strict_tool_loading_defaults_false_and_parses_true(tmp_path):
    from discuss_agent.config import ConfigLoader

    raw = {
        "discussion": {"model": "model"},
        "agents": [{"name": "A", "system_prompt": "A"}],
        "host": {"convergence_prompt": "judge", "summary_prompt": "summary"},
        "tools": [],
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    assert ConfigLoader.load(str(path)).strict_tool_loading is False
    raw["discussion"]["strict_tool_loading"] = True
    path.write_text(yaml.safe_dump(raw))
    assert ConfigLoader.load(str(path)).strict_tool_loading is True


def test_readme_matches_current_cli_and_archive_layout():
    readme = (Path(__file__).parents[1] / "README.md").read_text()

    assert "--resume" not in readme
    assert "--rounds" not in readme
    assert "round_1_express.json" not in readme
    assert "round_1_challenge.json" not in readme
    assert "round_1_agents.json" in readme
    assert "round_1_host.json" in readme
    assert "claims.md" in readme
    assert "audit/" in readme
    assert "usage_summary.md" in readme
    assert "summary.md    (if generated)" in readme
    assert "summary when generated" in readme
    assert "skip_summary" in readme
    assert "round_1_host.json (when reviewed)" in readme
    assert "usage_summary.md (with audit data)" in readme
    assert "In the measured Muyuan replay" not in readme
    assert "does not currently support resuming" in readme


@patch("discuss_agent.engine.generate_usage_summary")
@patch("discuss_agent.engine.AuditLogger")
@patch("discuss_agent.engine.Archiver")
@patch("discuss_agent.engine.ContextManager")
@patch("discuss_agent.engine.AgentConversation")
async def test_natural_convergence_ignores_legacy_min_rounds(
    MockConversation, MockContext, MockArchiver, MockAudit, _summary,
):
    """Legacy min_rounds must not delay a closable second round."""
    from discuss_agent.engine import DiscussionEngine

    config = _config(min_rounds=99, max_rounds=4)
    config.host.skip_summary = True
    archiver = MagicMock()
    archiver.start_session.return_value = "/tmp/runtime-natural-convergence"
    MockArchiver.return_value = archiver
    MockAudit.return_value = MagicMock()
    MockContext.return_value.build_initial_context = AsyncMock(return_value="generic topic")
    counts = {"A": 0, "B": 0}

    def conversation(**kwargs):
        name = kwargs["agent_name"]
        conv = MagicMock(messages=[])

        async def send(_prompt):
            counts[name] += 1
            if counts[name] == 1:
                return f"[NEW_CLAIM:item-{name}] source://item-{name}"
            other = "B" if name == "A" else "A"
            return f"[ACCEPT TO:item-{other}] accepted"

        conv.send = AsyncMock(side_effect=send)
        return conv

    MockConversation.side_effect = conversation
    engine = DiscussionEngine(config)
    engine._host_judge = AsyncMock(side_effect=[
        [
            {
                "claim": "item-A",
                "verdict": "CONTINUE",
                "reason": "peer review needed",
                "missing": "B review",
                "needs_agents": ["B"],
                "allow_unknown_progress": False,
            },
            {
                "claim": "item-B",
                "verdict": "CONTINUE",
                "reason": "peer review needed",
                "missing": "A review",
                "needs_agents": ["A"],
                "allow_unknown_progress": False,
            },
        ],
        [
            _closed_verdict("item-A"),
            _closed_verdict("item-B"),
        ],
    ])

    result = await engine.run()

    assert result.converged is True
    assert result.rounds_completed == 2
    assert counts == {"A": 2, "B": 2}
    assert engine._host_judge.await_count == 2
    assert [call.args[1] for call in engine._host_judge.await_args_list] == [1, 2]


@patch("discuss_agent.engine.generate_usage_summary")
@patch("discuss_agent.engine.AuditLogger")
@patch("discuss_agent.engine.Archiver")
@patch("discuss_agent.engine.ContextManager")
@patch("discuss_agent.engine.AgentConversation")
async def test_host_can_semantically_close_claim_in_round_one(
    MockConversation, MockContext, MockArchiver, MockAudit, _summary,
):
    from discuss_agent.engine import DiscussionEngine

    config = _config(max_rounds=6)
    config.host.skip_summary = True
    archiver = MagicMock()
    archiver.start_session.return_value = "/tmp/runtime-round-one"
    MockArchiver.return_value = archiver
    MockAudit.return_value = MagicMock()
    MockContext.return_value.build_initial_context = AsyncMock(return_value="topic")

    def conversation(**kwargs):
        conv = MagicMock(messages=[])
        text = (
            "[NEW_CLAIM:sufficient] source://round-one"
            if kwargs["agent_name"] == "A"
            else "[ACCEPT TO:sufficient] independently reviewed"
        )
        conv.send = AsyncMock(return_value=text)
        return conv

    MockConversation.side_effect = conversation
    engine = DiscussionEngine(config)
    engine._host_judge = AsyncMock(return_value=[
        _closed_verdict("sufficient"),
    ])

    result = await engine.run()

    assert result.converged is True
    assert result.rounds_completed == 1
    engine._host_judge.assert_awaited_once()
    assert engine._host_judge.await_args.args[1] == 1


@patch("discuss_agent.engine.generate_usage_summary")
@patch("discuss_agent.engine.AuditLogger")
@patch("discuss_agent.engine.Archiver")
@patch("discuss_agent.engine.ContextManager")
@patch("discuss_agent.engine.AgentConversation")
async def test_silent_agents_do_not_block_host_review_of_existing_claims(
    MockConversation, MockContext, MockArchiver, MockAudit, _summary,
):
    from discuss_agent.engine import DiscussionEngine

    config = _config(max_rounds=2)
    config.host.skip_summary = True
    archiver = MagicMock()
    archiver.start_session.return_value = "/tmp/runtime-silent-agents"
    MockArchiver.return_value = archiver
    MockAudit.return_value = MagicMock()
    MockContext.return_value.build_initial_context = AsyncMock(return_value="topic")

    def conversation(**kwargs):
        conv = MagicMock(messages=[])
        if kwargs["agent_name"] == "A":
            conv.send = AsyncMock(side_effect=["[NEW_CLAIM:X] evidence", "", ""])
        else:
            conv.send = AsyncMock(return_value="")
        return conv

    MockConversation.side_effect = conversation
    engine = DiscussionEngine(config)
    engine._host_judge = AsyncMock(side_effect=[
        [{
            "claim": "X",
            "verdict": "CONTINUE",
            "reason": "Host will reassess",
            "missing": "independent reassessment",
            "needs_agents": ["A"],
            "allow_unknown_progress": False,
        }],
        [_closed_verdict()],
    ])

    result = await engine.run()

    assert result.converged is True
    assert result.rounds_completed == 2
    assert result.remaining_disputes == []
    assert [call.args[1] for call in engine._host_judge.await_args_list] == [1, 2]


def test_host_precondition_allows_any_open_claim_without_agent_response_gate():
    from discuss_agent.claims import Claim, ClaimEntry, ClaimsManager
    from discuss_agent.engine import DiscussionEngine

    config = _config(max_rounds=4)
    config.agents.append(AgentConfig(name="C", system_prompt="C"))
    mgr = ClaimsManager()
    mgr.claims["unreviewed"] = Claim("unreviewed", "OPEN", [
        ClaimEntry("FROM", "A", 1, "claim"),
    ])

    assert DiscussionEngine(config)._check_convergence_precondition(mgr, round_num=1) is True


def test_host_verdicts_accept_any_offered_open_claim_and_apply_targeted_continue():
    from discuss_agent.claims import Claim, ClaimEntry, ClaimsManager
    from discuss_agent.engine import DiscussionEngine

    mgr = ClaimsManager()
    mgr.claims["unreviewed"] = Claim("unreviewed", "OPEN", [
        ClaimEntry("FROM", "A", 1, "claim"),
    ])
    verdicts = [
        {
            "claim": "unreviewed",
            "verdict": "CONTINUE",
            "reason": "more work",
            "needs_agents": ["B"],
            "missing": "counterevidence",
            "allow_unknown_progress": False,
        },
        {"claim": "unreviewed", "verdict": "CLOSED:共识", "reason": "duplicate"},
        {"claim": "missing", "verdict": "CLOSED:分歧", "reason": "unknown"},
    ]

    accepted, rejected = DiscussionEngine(_config())._apply_host_verdicts(
        mgr, verdicts, {"unreviewed"}, round_num=2,
    )

    assert accepted == [verdicts[0]]
    assert [item["reason"] for item in rejected] == [
        "duplicate verdict",
        "claim was not offered to the host",
    ]
    assert mgr.claims["unreviewed"].status == "OPEN"
    assert mgr._host_request(mgr.claims["unreviewed"]) == (
        {"B"}, "counterevidence",
    )


def test_host_verdicts_record_offered_claim_omitted_by_host():
    from discuss_agent.claims import Claim, ClaimsManager
    from discuss_agent.engine import DiscussionEngine

    mgr = ClaimsManager()
    mgr.claims["X"] = Claim("X", "OPEN")

    accepted, rejected = DiscussionEngine(_config())._apply_host_verdicts(
        mgr, [], {"X"}, round_num=2,
    )

    assert accepted == []
    assert rejected == [{
        "claim": "X",
        "verdict": None,
        "host_reason": "",
        "reason": "missing verdict",
    }]


def test_semantically_bad_but_structurally_valid_closed_is_host_judgment():
    from discuss_agent.claims import AgentOutput, ClaimsManager
    from discuss_agent.engine import DiscussionEngine

    mgr = ClaimsManager()
    mgr.merge_round([
        AgentOutput("A", 1, "[NEW_CLAIM:X] unsupported assertion; no evidence"),
    ])

    accepted, rejected = DiscussionEngine(_config())._apply_host_verdicts(
        mgr,
        [{
            "claim": "X",
            "verdict": "CLOSED:共识",
            "reason": "no one objected",
        }],
        {"X"},
        round_num=1,
    )

    assert accepted == [{
        "claim": "X",
        "verdict": "CLOSED:共识",
        "reason": "no one objected",
    }]
    assert rejected == []
    assert mgr.claims["X"].status == "CLOSED:共识"


@pytest.mark.parametrize("verdict", ["CLOSED:共识", "CLOSED:分歧"])
def test_host_applies_semantic_claim_close(verdict):
    from discuss_agent.claims import ClaimsManager
    from discuss_agent.engine import DiscussionEngine

    mgr = ClaimsManager()
    mgr.claims["X"] = _reviewed_claim()

    accepted, rejected = DiscussionEngine(_config())._apply_host_verdicts(
        mgr,
        [{
            "claim": "X",
            "verdict": verdict,
            "reason": "bounded claim has no expected material increment",
        }],
        {"X"},
        round_num=2,
    )

    assert rejected == []
    assert accepted[0]["verdict"] == verdict
    assert mgr.claims["X"].status == verdict
    assert mgr.claims["X"].entries[-1].entry_type == "HOST"
    assert "no expected material increment" in (
        mgr.claims["X"].entries[-1].content
    )


@pytest.mark.parametrize(
    "mutation",
    [
        {"unresolved_evidence": "missing source"},
        {"allow_unknown_progress": False},
        {"missing": None},
        {"needs_agents": ()},
    ],
)
def test_closed_verdict_uses_strict_schema(mutation):
    from discuss_agent.claims import ClaimsManager
    from discuss_agent.engine import DiscussionEngine

    mgr = ClaimsManager()
    mgr.claims["X"] = _reviewed_claim()

    accepted, rejected = DiscussionEngine(_config())._apply_host_verdicts(
        mgr,
        [_closed_verdict(**mutation)],
        {"X"},
        round_num=1,
    )

    assert accepted == []
    assert rejected[0]["reason"] == "invalid closed verdict schema"
    assert mgr.claims["X"].status == "OPEN"


def test_closed_reason_content_is_not_a_runtime_semantic_gate():
    from discuss_agent.claims import Claim, ClaimsManager
    from discuss_agent.engine import DiscussionEngine

    mgr = ClaimsManager()
    mgr.claims["X"] = Claim("X", "OPEN")
    verdict = _closed_verdict(reason="unsupported because nobody objected")

    accepted, rejected = DiscussionEngine(_config())._apply_host_verdicts(
        mgr,
        [verdict],
        {"X"},
        round_num=1,
    )

    assert accepted == [verdict]
    assert rejected == []
    assert mgr.claims["X"].status == "CLOSED:共识"


def test_material_continue_false_cannot_coexist_with_room_converged():
    from discuss_agent.claims import Claim, ClaimsManager
    from discuss_agent.engine import DiscussionEngine

    mgr = ClaimsManager()
    mgr.claims["X"] = Claim("X", "OPEN")
    engine = DiscussionEngine(_config())
    engine._host_room_adjudication = {
        "status": "CONVERGED",
        "reason": "incorrectly claims completion",
    }
    verdict = {
        "claim": "X",
        "verdict": "CONTINUE",
        "reason": "material current conflict remains",
        "missing": "same-parameter arithmetic reconciliation",
        "needs_agents": ["A"],
        "allow_unknown_progress": False,
    }

    accepted, rejected = engine._apply_host_verdicts(
        mgr, [verdict], {"X"}, round_num=2,
    )

    assert engine._room_converged(
        mgr,
        accepted=accepted,
        rejected=rejected,
        offered_keywords={"X"},
    ) is False


def test_same_round_indented_protocol_marker_warns_without_semantic_gate():
    from discuss_agent.claims import AgentOutput, ClaimsManager
    from discuss_agent.engine import DiscussionEngine

    mgr = ClaimsManager()
    mgr.merge_round([
        AgentOutput("A", 1, "[NEW_CLAIM:X] source://primary"),
        AgentOutput(
            "B",
            1,
            "[ACCEPT TO:X] independently reviewed\n"
            "  [REBUTTAL TO:X] malformed counterevidence",
        ),
    ])

    accepted, rejected = DiscussionEngine(_config())._apply_host_verdicts(
        mgr,
        [_closed_verdict()],
        {"X"},
        round_num=1,
    )

    assert accepted == [_closed_verdict()]
    assert rejected == []
    assert mgr.claims["X"].status == "CLOSED:共识"
    assert mgr.unmatched_responses[-1]["response_type"] == "PROTOCOL_VIOLATION"


@pytest.mark.parametrize(
    "mutation",
    [
        {"unexpected": "field"},
        {"allow_unknown_progress": "false"},
        {"needs_agents": ("A",)},
        {"missing": ["source"]},
    ],
)
def test_continue_verdict_uses_strict_schema(mutation):
    from discuss_agent.claims import ClaimsManager
    from discuss_agent.engine import DiscussionEngine

    mgr = ClaimsManager()
    mgr.claims["X"] = _reviewed_claim()
    verdict = {
        "claim": "X",
        "verdict": "CONTINUE",
        "reason": "evidence gap",
        "missing": "source",
        "needs_agents": ["A"],
        "allow_unknown_progress": False,
        **mutation,
    }

    accepted, rejected = DiscussionEngine(_config())._apply_host_verdicts(
        mgr,
        [verdict],
        {"X"},
        round_num=1,
    )

    assert accepted == []
    assert rejected[0]["reason"] == "invalid continue verdict schema"
    assert mgr.claims["X"].status == "OPEN"


def test_applying_multiple_host_verdicts_persists_once():
    from discuss_agent.claims import Claim, ClaimsManager
    from discuss_agent.engine import DiscussionEngine

    mgr = ClaimsManager()
    mgr.claims = {
        "X": _reviewed_claim(),
        "Y": _reviewed_claim("Y"),
    }
    mgr.save = MagicMock()

    accepted, rejected = DiscussionEngine(_config())._apply_host_verdicts(
        mgr,
        [
            _closed_verdict(),
            _closed_verdict("Y", verdict="CLOSED:分歧"),
        ],
        {"X", "Y"},
        round_num=2,
    )

    assert len(accepted) == 2
    assert rejected == []
    mgr.save.assert_called_once_with()


def test_host_verdicts_reject_non_string_fields_without_crashing():
    from discuss_agent.claims import Claim, ClaimsManager
    from discuss_agent.engine import DiscussionEngine

    mgr = ClaimsManager()
    mgr.claims["X"] = Claim("X", "OPEN")
    verdicts = [
        {"claim": [], "verdict": "CLOSED:共识", "reason": "bad claim"},
        {"claim": "X", "verdict": ["CLOSED:共识"], "reason": "bad verdict"},
    ]

    accepted, rejected = DiscussionEngine(_config())._apply_host_verdicts(
        mgr, verdicts, {"X"}, round_num=2,
    )

    assert accepted == []
    assert [item["reason"] for item in rejected] == [
        "invalid claim field",
        "invalid verdict field",
        "missing verdict",
    ]


def test_continue_accepts_external_or_unassigned_blockers():
    from discuss_agent.claims import Claim, ClaimsManager
    from discuss_agent.engine import DiscussionEngine

    mgr = ClaimsManager()
    mgr.claims = {
        "X": Claim("X", "OPEN"),
        "Y": Claim("Y", "OPEN"),
    }
    verdicts = [
        {
            "claim": "X",
            "verdict": "CONTINUE",
            "reason": "evidence gap",
            "missing": "",
            "needs_agents": [],
            "allow_unknown_progress": False,
        },
        {
            "claim": "Y",
            "verdict": "CONTINUE",
            "reason": "counterexample review",
            "missing": "external archive unavailable; nobody can retrieve it",
            "needs_agents": [],
            "allow_unknown_progress": True,
        },
    ]

    accepted, rejected = DiscussionEngine(_config())._apply_host_verdicts(
        mgr, verdicts, {"X", "Y"}, round_num=1,
    )

    assert accepted == verdicts
    assert rejected == []
    assert all(claim.status == "OPEN" for claim in mgr.claims.values())
    assert mgr._host_request(mgr.claims["X"]) == (set(), "")
    assert mgr._host_request(mgr.claims["Y"]) == (
        set(), "external archive unavailable; nobody can retrieve it",
    )
    assert mgr._host_routing(mgr.claims["X"])["allow_unknown_progress"] is False
    assert mgr._host_routing(mgr.claims["Y"])["allow_unknown_progress"] is True


def test_empty_continue_routing_round_trips(tmp_path):
    from discuss_agent.claims import Claim, ClaimsManager
    from discuss_agent.engine import DiscussionEngine

    path = tmp_path / "claims.md"
    mgr = ClaimsManager(str(path))
    mgr.claims["X"] = Claim("X", "OPEN")
    verdict = {
        "claim": "X",
        "verdict": "CONTINUE",
        "reason": "external blocker",
        "missing": "",
        "needs_agents": [],
        "allow_unknown_progress": True,
    }

    accepted, rejected = DiscussionEngine(_config())._apply_host_verdicts(
        mgr, [verdict], {"X"}, round_num=2,
    )
    reloaded = ClaimsManager(str(path))
    reloaded.parse_claims_file()

    assert accepted == [verdict]
    assert rejected == []
    assert reloaded._host_routing(reloaded.claims["X"]) == {
        "needs_agents": [],
        "missing": "",
        "allow_unknown_progress": True,
    }


@pytest.mark.parametrize("allow_unknown", [None, "false", 0])
def test_continue_verdict_requires_explicit_boolean_unknown_policy(
    allow_unknown,
):
    from discuss_agent.claims import Claim, ClaimsManager
    from discuss_agent.engine import DiscussionEngine

    mgr = ClaimsManager()
    mgr.claims["X"] = Claim("X", "OPEN")
    verdict = {
        "claim": "X",
        "verdict": "CONTINUE",
        "reason": "evidence gap",
        "missing": "source",
        "needs_agents": ["A"],
    }
    if allow_unknown is not None:
        verdict["allow_unknown_progress"] = allow_unknown

    accepted, rejected = DiscussionEngine(_config())._apply_host_verdicts(
        mgr, [verdict], {"X"}, round_num=1,
    )

    assert accepted == []
    assert rejected[0]["reason"] == "invalid continue verdict schema"
    assert mgr.claims["X"].status == "OPEN"


@pytest.mark.parametrize("reason", ["", "   ", ["not", "text"]])
def test_closed_verdict_requires_nonempty_reason_and_falls_back_open(reason):
    from discuss_agent.claims import Claim, ClaimsManager
    from discuss_agent.engine import DiscussionEngine

    mgr = ClaimsManager()
    mgr.claims["X"] = Claim("X", "OPEN")

    accepted, rejected = DiscussionEngine(_config())._apply_host_verdicts(
        mgr,
        [{"claim": "X", "verdict": "CLOSED:共识", "reason": reason}],
        {"X"},
        round_num=1,
    )

    assert accepted == []
    assert rejected[0]["reason"] == "reason must be a non-empty string"
    assert mgr.claims["X"].status == "OPEN"
    assert mgr._host_request(mgr.claims["X"])[0] == {"A", "B"}


@pytest.mark.parametrize(
    "metadata",
    [
        {"missing": "decisive source"},
        {"needs_agents": ["A"]},
        {"missing": 0},
        {"needs_agents": {}},
        {"needs_agents": ""},
    ],
)
def test_closed_verdict_rejects_unknown_metadata(metadata):
    from discuss_agent.claims import Claim, ClaimsManager
    from discuss_agent.engine import DiscussionEngine

    mgr = ClaimsManager()
    mgr.claims["X"] = _reviewed_claim()

    accepted, rejected = DiscussionEngine(_config())._apply_host_verdicts(
        mgr,
        [_closed_verdict(**metadata)],
        {"X"},
        round_num=1,
    )

    assert accepted == []
    assert rejected[0]["reason"] == "invalid closed verdict schema"
    assert mgr.claims["X"].status == "OPEN"


@patch("discuss_agent.engine.generate_usage_summary")
@patch("discuss_agent.engine.AuditLogger")
@patch("discuss_agent.engine.Archiver")
@patch("discuss_agent.engine.ContextManager")
@patch("discuss_agent.engine.AgentConversation")
async def test_host_semantically_continues_new_claim_while_closing_sufficient_claims(
    MockConversation, MockContext, MockArchiver, MockAudit, _summary,
):
    from discuss_agent.engine import DiscussionEngine

    config = _config(max_rounds=2)
    config.host.skip_summary = True
    archiver = MagicMock()
    archiver.start_session.return_value = "/tmp/runtime-partial-closure"
    MockArchiver.return_value = archiver
    MockAudit.return_value = MagicMock()
    MockContext.return_value.build_initial_context = AsyncMock(return_value="topic")
    counts = {"A": 0, "B": 0}

    def conversation(**kwargs):
        name = kwargs["agent_name"]
        conv = MagicMock(messages=[])

        async def send(_prompt):
            counts[name] += 1
            if counts[name] == 1:
                return (
                    "[NEW_CLAIM:old] source://old"
                    if name == "A"
                    else "[NEW_CLAIM:peer] source://peer"
                )
            if name == "A":
                return (
                    "[ACCEPT TO:peer] ok\n"
                    "[NEW_CLAIM:new] source://new genuinely new"
                )
            return "[ACCEPT TO:old] ok"

        conv.send = AsyncMock(side_effect=send)
        return conv

    MockConversation.side_effect = conversation
    engine = DiscussionEngine(config)
    judged = []

    async def judge(mgr, round_num):
        candidates = [claim.keyword for claim in mgr.get_host_candidates()]
        judged.extend(candidates)
        if round_num == 1:
            return [
                {
                    "claim": keyword,
                    "verdict": "CONTINUE",
                    "reason": "cross-review",
                    "missing": "peer review",
                    "needs_agents": ["B" if keyword == "old" else "A"],
                    "allow_unknown_progress": False,
                }
                for keyword in candidates
            ]
        return [
            (
                {
                    "claim": keyword,
                    "verdict": "CONTINUE",
                    "reason": "new evidence needs review",
                    "missing": "B review",
                    "needs_agents": ["B"],
                    "allow_unknown_progress": False,
                }
                if keyword == "new"
                else _closed_verdict(keyword)
            )
            for keyword in candidates
        ]

    engine._host_judge = AsyncMock(side_effect=judge)
    result = await engine.run()

    assert set(judged) == {"old", "peer", "new"}
    assert result.converged is False
    assert result.remaining_disputes == ["new"]


@patch("discuss_agent.engine.generate_usage_summary")
@patch("discuss_agent.engine.AuditLogger")
@patch("discuss_agent.engine.Archiver")
@patch("discuss_agent.engine.ContextManager")
@patch("discuss_agent.engine.AgentConversation")
async def test_host_can_close_same_round_claim_without_all_agents_responding(
    MockConversation, MockContext, MockArchiver, MockAudit, _summary,
):
    from discuss_agent.engine import DiscussionEngine

    config = _config(max_rounds=2)
    config.agents.append(AgentConfig(name="C", system_prompt="C"))
    config.host.skip_summary = True
    archiver = MagicMock()
    archiver.start_session.return_value = "/tmp/runtime-three-agent-partial"
    MockArchiver.return_value = archiver
    MockAudit.return_value = MagicMock()
    MockContext.return_value.build_initial_context = AsyncMock(return_value="topic")
    counts = {"A": 0, "B": 0, "C": 0}

    def conversation(**kwargs):
        name = kwargs["agent_name"]
        conv = MagicMock(messages=[])

        async def send(_prompt):
            counts[name] += 1
            if counts[name] == 1:
                return "[NEW_CLAIM:old] source://old" if name == "A" else ""
            if name == "A":
                return "[NEW_CLAIM:new] source://new same-round new"
            if name == "B":
                return "[ACCEPT TO:old] ok\n[ACCEPT TO:new] independently reviewed"
            return "[ACCEPT TO:old] ok"

        conv.send = AsyncMock(side_effect=send)
        return conv

    MockConversation.side_effect = conversation
    engine = DiscussionEngine(config)
    engine._host_judge = AsyncMock(side_effect=[
        [{
            "claim": "old",
            "verdict": "CONTINUE",
            "reason": "review needed",
            "missing": "peer review",
            "needs_agents": ["B"],
            "allow_unknown_progress": False,
        }],
        [
            _closed_verdict("old"),
            _closed_verdict("new"),
        ],
    ])

    result = await engine.run()

    assert result.converged is True
    assert result.remaining_disputes == []
    host_record = [
        call.args[2]
        for call in archiver.save_round.call_args_list
        if call.args[1] == "host"
    ][-1]
    assert {
        item["claim"] for item in host_record["accepted_verdicts"]
    } == {"old", "new"}
    assert host_record["rejected_verdicts"] == []


@pytest.mark.parametrize("strict", [True, False])
def test_strict_tool_loading_controls_fail_closed(strict):
    from discuss_agent.engine import DiscussionEngine

    engine = DiscussionEngine(_config(strict=strict, tools=[ToolConfig("missing.module.Toolkit")]))
    if strict:
        with pytest.raises(RuntimeError, match=r"global.*missing\.module\.Toolkit"):
            engine._create_conversations()
    else:
        engine._create_conversations()
        assert set(engine._conversations) == {"A", "B"}


def test_strict_extra_toolkit_requires_callable_function():
    from discuss_agent.engine import DiscussionEngine

    empty = SimpleNamespace(functions={}, async_functions={})
    config = _config(strict=True, extra_tools=[ToolConfig("pkg.Empty")])
    engine = DiscussionEngine(config)
    with patch("discuss_agent.engine.import_from_path", return_value=lambda: empty):
        with pytest.raises(RuntimeError, match=r"extra tool.*pkg\.Empty.*callable"):
            engine._create_conversations()


@pytest.mark.parametrize(
    ("loaded", "message"),
    [
        (lambda: (_ for _ in ()).throw(RuntimeError("constructor failed")), "constructor failed"),
        (
            lambda: SimpleNamespace(
                functions={"bad": SimpleNamespace(entrypoint=None)},
                async_functions={},
            ),
            "no callable entrypoint",
        ),
    ],
)
def test_strict_toolkit_initialization_and_entrypoint_failures(loaded, message):
    from discuss_agent.engine import DiscussionEngine

    config = _config(strict=True, tools=[ToolConfig("pkg.Toolkit")])
    engine = DiscussionEngine(config)
    with patch("discuss_agent.engine.import_from_path", return_value=loaded):
        with pytest.raises(RuntimeError, match=message):
            engine._create_conversations()


@pytest.mark.parametrize("scope", ["global", "extra"])
def test_toolkit_context_is_passed_for_global_and_extra_tools(scope):
    from discuss_agent.engine import DiscussionEngine

    research_dir = (
        "/home/zhijiang/.openclaw/workspace-research/"
        "shadow_runs/muyuan_v20_20260807/materials"
    )
    context = {"research_dirs": [research_dir], "topic": "readiness"}
    received = []

    class ContextToolkit:
        def __init__(self, context):
            received.append(context)
            self.functions = {}
            self.async_functions = {}

    tool = ToolConfig("pkg.ContextToolkit")
    config = _config(**({"tools": [tool]} if scope == "global" else {"extra_tools": [tool]}))
    config.context = context
    engine = DiscussionEngine(config)

    with patch("discuss_agent.engine.import_from_path", return_value=ContextToolkit):
        engine._create_conversations()

    assert received == [context]
    assert received[0] is config.context


def test_zero_arg_toolkit_still_instantiates():
    from discuss_agent.engine import DiscussionEngine

    instantiated = []

    class ZeroArgToolkit:
        def __init__(self):
            instantiated.append(True)
            self.functions = {}
            self.async_functions = {}

    engine = DiscussionEngine(_config(tools=[ToolConfig("pkg.ZeroArgToolkit")]))
    with patch("discuss_agent.engine.import_from_path", return_value=ZeroArgToolkit):
        engine._create_conversations()

    assert instantiated == [True]


def test_conversations_share_one_run_scoped_tool_result_cache():
    from discuss_agent.engine import DiscussionEngine

    engine = DiscussionEngine(_config())
    with patch(
        "discuss_agent.engine.AgentConversation"
    ) as conversation:
        engine._create_conversations()

    caches = [
        call.kwargs["tool_result_cache"]
        for call in conversation.call_args_list
    ]
    assert len(caches) == 2
    assert caches[0] is caches[1]


def test_variadic_constructor_is_not_treated_as_context_aware():
    from discuss_agent.engine import DiscussionEngine

    received = []

    class VariadicToolkit:
        def __init__(self, **context):
            received.append(context)
            self.functions = {}
            self.async_functions = {}

    engine = DiscussionEngine(_config(tools=[ToolConfig("pkg.VariadicToolkit")]))
    with patch("discuss_agent.engine.import_from_path", return_value=VariadicToolkit):
        engine._create_conversations()

    assert received == [{}]


def test_positional_only_context_preserves_preceding_defaults():
    from discuss_agent.engine import DiscussionEngine

    received = []

    class PositionalContextToolkit:
        def __init__(self, label="default", context=None, /):
            received.append((label, context))
            self.functions = {}
            self.async_functions = {}

    config = _config(tools=[ToolConfig("pkg.PositionalContextToolkit")])
    config.context = {"research_dirs": ["/tmp/research"]}
    engine = DiscussionEngine(config)

    with patch("discuss_agent.engine.import_from_path", return_value=PositionalContextToolkit):
        engine._create_conversations()

    assert received == [("default", config.context)]


def test_internal_constructor_type_error_fails_strict_loading():
    from discuss_agent.engine import DiscussionEngine

    class BrokenContextToolkit:
        def __init__(self, context):
            raise TypeError("internal constructor type error")

    config = _config(strict=True, tools=[ToolConfig("pkg.BrokenContextToolkit")])
    config.context = {"research_dirs": ["/tmp/research"]}
    engine = DiscussionEngine(config)

    with patch("discuss_agent.engine.import_from_path", return_value=BrokenContextToolkit):
        with pytest.raises(RuntimeError, match="internal constructor type error"):
            engine._create_conversations()


@patch("discuss_agent.engine.generate_usage_summary")
@patch("discuss_agent.engine.AuditLogger")
@patch("discuss_agent.engine.Archiver")
@patch("discuss_agent.engine.ContextManager")
async def test_strict_loading_run_fails_before_context_or_agent_calls(
    MockContext, MockArchiver, MockAudit, _summary,
):
    from discuss_agent.engine import DiscussionEngine

    archiver = MagicMock()
    archiver.start_session.return_value = "/tmp/runtime-gate-strict"
    MockArchiver.return_value = archiver
    context = MagicMock()
    context.build_initial_context = AsyncMock(return_value="must not run")
    MockContext.return_value = context
    result = await DiscussionEngine(
        _config(strict=True, tools=[ToolConfig("missing.module.Toolkit")])
    ).run()
    assert result.terminated_by_error is True
    context.build_initial_context.assert_not_awaited()
    error = archiver.save_error_log.call_args.args[0]
    assert "global tool" in error and "missing.module.Toolkit" in error


async def test_anthropic_sync_and_async_tools_are_audited(tmp_path):
    audit = AuditLogger(str(tmp_path))

    def sync_tool(value):
        return value * 2

    async def async_tool(value):
        return value + 1

    conv = AgentConversation(
        agent_name="agent",
        system_prompt="system",
        api_key="test",
        tools=[{"name": "sync", "input_schema": {}}, {"name": "async", "input_schema": {}}],
        tool_callables={"sync": sync_tool, "async": async_tool},
        audit_logger=audit,
    )
    conv.set_round(2)
    conv._client = MagicMock()
    blocks = lambda *xs: SimpleNamespace(content=list(xs), stop_reason="end_turn")
    use = lambda ident, name, args: SimpleNamespace(type="tool_use", id=ident, name=name, input=args)
    text = lambda value: SimpleNamespace(type="text", text=value)
    conv._client.messages.create = AsyncMock(side_effect=[
        blocks(use("1", "sync", {"value": 2}), use("2", "async", {"value": 2})),
        blocks(text("complete")),
    ])

    assert await conv.send("go") == "complete"
    audit.close()
    events = [json.loads(line) for line in (tmp_path / "audit" / "agent.jsonl").read_text().splitlines()]
    assert events[0]["event"] == "call_start" and events[0]["round"] == 2
    tools = [event for event in events if event["event"] == "tool_call"]
    assert [event["tool"] for event in tools] == ["sync", "async"]
    assert all(event["round"] == 2 and event["response_size"] > 0 for event in tools)
    assert events[-1]["event"] == "call_end" and events[-1]["round"] == 2


async def test_openai_invalid_and_failing_tools_are_audited(tmp_path):
    audit = AuditLogger(str(tmp_path))

    def failing():
        raise RuntimeError("failed safely")

    conv = AgentConversation(
        agent_name="openai-agent",
        system_prompt="system",
        model="agent-maestro-openai/model",
        api_key="test",
        tools=[{"name": "failing", "input_schema": {}}],
        tool_callables={"failing": failing},
        audit_logger=audit,
    )
    conv.set_round(4)
    conv._client = MagicMock()

    def call(ident, name, args):
        return SimpleNamespace(id=ident, type="function", function=SimpleNamespace(name=name, arguments=args))

    conv._client.chat.completions.create = AsyncMock(side_effect=[
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=[
            call("1", "failing", "{}"), call("2", "failing", "not-json")
        ]))]),
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="complete", tool_calls=[]))]),
    ])

    assert await conv.send("go") == "complete"
    audit.close()
    events = [json.loads(line) for line in (tmp_path / "audit" / "openai-agent.jsonl").read_text().splitlines()]
    tools = [event for event in events if event["event"] == "tool_call"]
    assert len(tools) == 2
    assert "failed safely" in tools[0]["error"]
    assert "Invalid tool arguments" in tools[1]["error"]
    assert all(event["round"] == 4 and "duration_ms" in event for event in tools)


async def test_failed_model_call_still_has_audited_start_error_and_end(tmp_path):
    audit = AuditLogger(str(tmp_path))
    conv = AgentConversation(
        agent_name="failed-agent", system_prompt="system", api_key="test",
        audit_logger=audit,
    )
    conv.set_round(3)
    conv._client = MagicMock()
    conv._client.messages.create = AsyncMock(side_effect=RuntimeError("model unavailable"))

    with pytest.raises(RuntimeError, match="model unavailable"):
        await conv.send("go")
    audit.close()
    events = [
        json.loads(line)
        for line in (tmp_path / "audit" / "failed-agent.jsonl").read_text().splitlines()
    ]
    assert [event["event"] for event in events] == ["call_start", "error", "call_end"]
    assert all(event["round"] == 3 for event in events)
    assert events[-1]["stop_reason"] == "error"
