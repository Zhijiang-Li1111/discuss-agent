"""Tests for ClaimsManager in discuss_agent.claims."""

from __future__ import annotations

import pytest

from discuss_agent.claims import (
    AgentOutput,
    Claim,
    ClaimEntry,
    ClaimsManager,
    ParsedResponse,
    parse_agent_output,
)


# ---------------------------------------------------------------------------
# parse_agent_output
# ---------------------------------------------------------------------------


class TestParseAgentOutput:
    def test_parse_new_claim(self):
        text = "[NEW_CLAIM:能繁去化进度] 当前能繁母猪3904万头，目标3650万"
        results = parse_agent_output(text)
        assert len(results) == 1
        assert results[0].response_type == "NEW_CLAIM"
        assert results[0].target == "能繁去化进度"
        assert "3904万头" in results[0].content

    def test_parse_rebuttal(self):
        text = "[REBUTTAL TO:能繁去化进度] 数据需交叉验证，官方口径可能有偏差"
        results = parse_agent_output(text)
        assert len(results) == 1
        assert results[0].response_type == "REBUTTAL"
        assert results[0].target == "能繁去化进度"

    def test_parse_accept(self):
        text = "[ACCEPT TO:牧原成本优势] 招商证券研报确认头均14.5元"
        results = parse_agent_output(text)
        assert len(results) == 1
        assert results[0].response_type == "ACCEPT"
        assert results[0].target == "牧原成本优势"

    def test_parse_multiple_responses(self):
        text = (
            "[NEW_CLAIM:饲料产量下降] Q1猪饲料产量同比-8%\n"
            "[REBUTTAL TO:能繁去化进度] 数据来源存疑\n"
            "[ACCEPT TO:牧原成本优势] 同意"
        )
        results = parse_agent_output(text)
        assert len(results) == 3
        assert results[0].response_type == "NEW_CLAIM"
        assert results[1].response_type == "REBUTTAL"
        assert results[2].response_type == "ACCEPT"

    def test_parse_empty_text(self):
        assert parse_agent_output("") == []
        assert parse_agent_output("some random text without markers") == []


# ---------------------------------------------------------------------------
# ClaimsManager — merge
# ---------------------------------------------------------------------------


class TestClaimsManagerMerge:
    def test_merge_new_claims(self):
        mgr = ClaimsManager()
        outputs = [
            AgentOutput(
                agent_name="分析师A",
                round_num=1,
                raw_text="[NEW_CLAIM:能繁去化] 当前3904万头\n[NEW_CLAIM:成本优势] 头均14.5元",
            ),
            AgentOutput(
                agent_name="分析师B",
                round_num=1,
                raw_text="[NEW_CLAIM:饲料下降] Q1同比-8%",
            ),
        ]
        mgr.merge_round(outputs)

        assert len(mgr.claims) == 3
        assert "能繁去化" in mgr.claims
        assert "成本优势" in mgr.claims
        assert "饲料下降" in mgr.claims

        # Check FROM tags
        claim = mgr.claims["能繁去化"]
        assert claim.status == "OPEN"
        assert len(claim.entries) == 1
        assert claim.entries[0].entry_type == "FROM"
        assert claim.entries[0].agent_name == "分析师A"
        assert claim.entries[0].round_num == 1

    def test_merge_rebuttal_and_accept(self):
        mgr = ClaimsManager()

        # Round 1: create claims
        mgr.merge_round([
            AgentOutput("A", 1, "[NEW_CLAIM:X] claim X content"),
            AgentOutput("B", 1, "[NEW_CLAIM:Y] claim Y content"),
        ])

        # Round 2: rebuttals and accepts
        mgr.merge_round([
            AgentOutput("B", 2, "[REBUTTAL TO:X] 我反驳X"),
            AgentOutput("A", 2, "[ACCEPT TO:Y] 我同意Y"),
        ])

        assert len(mgr.claims["X"].entries) == 2  # FROM + REBUTTAL
        assert mgr.claims["X"].entries[1].entry_type == "REBUTTAL"
        assert mgr.claims["X"].entries[1].agent_name == "B"
        assert mgr.claims["X"].entries[1].round_num == 2

        assert len(mgr.claims["Y"].entries) == 2  # FROM + ACCEPT
        assert mgr.claims["Y"].entries[1].entry_type == "ACCEPT"
        assert mgr.claims["Y"].entries[1].agent_name == "A"

    def test_merge_ignores_rebuttal_to_nonexistent_claim(self):
        mgr = ClaimsManager()
        mgr.merge_round([
            AgentOutput("A", 1, "[REBUTTAL TO:不存在] 反驳"),
        ])
        assert len(mgr.claims) == 0
        assert mgr.unmatched_responses == [{
            "agent_name": "A",
            "round_num": 1,
            "response_type": "REBUTTAL",
            "target": "不存在",
        }]

    def test_merge_batch_accept_targets_without_silent_loss(self):
        mgr = ClaimsManager()
        mgr.merge_round([
            AgentOutput("A", 1, "[NEW_CLAIM:X] x"),
            AgentOutput("B", 1, "[NEW_CLAIM:Y] y"),
        ])
        mgr.merge_round([AgentOutput("C", 2, "[ACCEPT TO:X、Y、Z] batch")])

        assert mgr.claims["X"].entries[-1].entry_type == "ACCEPT"
        assert mgr.claims["Y"].entries[-1].entry_type == "ACCEPT"
        assert mgr.unmatched_responses[-1]["target"] == "Z"

    def test_current_round_updated(self):
        mgr = ClaimsManager()
        mgr.merge_round([AgentOutput("A", 1, "[NEW_CLAIM:X] content")])
        assert mgr.current_round == 1
        mgr.merge_round([AgentOutput("B", 3, "[ACCEPT TO:X] ok")])
        assert mgr.current_round == 3


# ---------------------------------------------------------------------------
# ClaimsManager — get_open_claims
# ---------------------------------------------------------------------------


class TestGetOpenClaims:
    def test_returns_only_open(self):
        mgr = ClaimsManager()
        mgr.claims["X"] = Claim("X", "OPEN")
        mgr.claims["Y"] = Claim("Y", "CLOSED:共识")
        mgr.claims["Z"] = Claim("Z", "OPEN")

        open_claims = mgr.get_open_claims()
        keywords = {c.keyword for c in open_claims}
        assert keywords == {"X", "Z"}

    def test_maturity_uses_cumulative_responses_per_claim(self):
        mgr = ClaimsManager()
        mgr.claims["old"] = Claim("old", "OPEN", [
            ClaimEntry("FROM", "A", 1, "claim"),
            ClaimEntry("ACCEPT", "B", 2, "ok"),
            ClaimEntry("ACCEPT", "C", 3, "ok"),
        ])
        mgr.claims["new"] = Claim("new", "OPEN", [
            ClaimEntry("FROM", "B", 3, "claim"),
        ])

        assert [claim.keyword for claim in mgr.get_mature_claims({"A", "B", "C"})] == ["old"]


# ---------------------------------------------------------------------------
# ClaimsManager — close_claim
# ---------------------------------------------------------------------------


class TestCloseClaim:
    def test_close_with_verdict(self):
        mgr = ClaimsManager()
        mgr.claims["X"] = Claim("X", "OPEN", [
            ClaimEntry("FROM", "A", 1, "content"),
        ])
        mgr.close_claim("X", "共识", "各方一致认可", round_num=2)

        assert mgr.claims["X"].status == "CLOSED:共识"
        assert len(mgr.claims["X"].entries) == 2
        host_entry = mgr.claims["X"].entries[-1]
        assert host_entry.entry_type == "HOST"
        assert host_entry.round_num == 2
        assert "裁决" in host_entry.content

    def test_close_nonexistent_is_noop(self):
        mgr = ClaimsManager()
        mgr.close_claim("不存在", "共识", "reason", round_num=1)
        assert len(mgr.claims) == 0


# ---------------------------------------------------------------------------
# ClaimsManager — parse claims file
# ---------------------------------------------------------------------------


class TestParseClaimsFile:
    def test_parse_formatted_output(self):
        mgr = ClaimsManager()

        # Build some claims, format, then re-parse
        mgr.claims["能繁去化"] = Claim("能繁去化", "OPEN", [
            ClaimEntry("FROM", "分析师A", 1, "当前3904万头"),
            ClaimEntry("REBUTTAL", "分析师B", 1, "数据需交叉验证"),
            ClaimEntry("RESPONSE", "分析师A", 2, "农业农村部月报确认"),
        ])
        mgr.claims["成本优势"] = Claim("成本优势", "CLOSED:共识", [
            ClaimEntry("FROM", "分析师B", 1, "头均14.5元"),
            ClaimEntry("ACCEPT", "分析师A", 1, "招商证券确认"),
            ClaimEntry("HOST", "HOST", 1, "裁决：各方一致认可"),
        ])

        text = mgr.format_file()

        # Re-parse
        mgr2 = ClaimsManager()
        mgr2._parse_text(text)

        assert len(mgr2.claims) == 2
        assert mgr2.claims["能繁去化"].status == "OPEN"
        assert len(mgr2.claims["能繁去化"].entries) == 3
        assert mgr2.claims["成本优势"].status == "CLOSED:共识"
        assert len(mgr2.claims["成本优势"].entries) == 3

        # Verify FROM tags preserved
        first = mgr2.claims["能繁去化"].entries[0]
        assert first.entry_type == "FROM"
        assert first.agent_name == "分析师A"
        assert first.round_num == 1

        host = mgr2.claims["成本优势"].entries[2]
        assert host.entry_type == "HOST"
        assert host.round_num == 1


# ---------------------------------------------------------------------------
# ClaimsManager — save/load round-trip
# ---------------------------------------------------------------------------


class TestSaveLoad:
    def test_round_trip(self, tmp_path):
        claims_file = str(tmp_path / "claims.md")
        mgr = ClaimsManager(claims_file)
        mgr.topic = "猪周期分析"

        mgr.merge_round([
            AgentOutput("A", 1, "[NEW_CLAIM:X] content X"),
            AgentOutput("B", 1, "[NEW_CLAIM:Y] content Y"),
        ])
        mgr.merge_round([
            AgentOutput("B", 2, "[REBUTTAL TO:X] rebuttal"),
            AgentOutput("A", 2, "[ACCEPT TO:Y] accept"),
        ])
        mgr.close_claim("Y", "共识", "一致同意", round_num=2)

        # Reload
        mgr2 = ClaimsManager(claims_file)
        mgr2.parse_claims_file()

        assert len(mgr2.claims) == 2
        assert mgr2.claims["X"].status == "OPEN"
        assert mgr2.claims["Y"].status == "CLOSED:共识"
        assert len(mgr2.claims["X"].entries) == 2  # FROM + REBUTTAL
        assert len(mgr2.claims["Y"].entries) == 3  # FROM + ACCEPT + HOST


# ---------------------------------------------------------------------------
# ClaimsManager — generate_initial_prompt
# ---------------------------------------------------------------------------


class TestGenerateInitialPrompt:
    def test_includes_full_context(self):
        mgr = ClaimsManager()
        prompt = mgr.generate_initial_prompt(
            "OCR 策略",
            limitation="全部文件必须入库",
            context="成本事实：12,823 页，119.28 美元",
        )

        assert "议题：OCR 策略" in prompt
        assert "全部文件必须入库" in prompt
        assert "## 已确认背景材料" in prompt
        assert "12,823 页" in prompt

    def test_context_is_optional(self):
        mgr = ClaimsManager()
        prompt = mgr.generate_initial_prompt("OCR 策略")
        assert "议题：OCR 策略" in prompt
        assert "已确认背景材料" not in prompt


# ---------------------------------------------------------------------------
# ClaimsManager — generate_update_prompt
# ---------------------------------------------------------------------------


class TestGenerateUpdatePrompt:
    def test_update_includes_open_full_text(self):
        mgr = ClaimsManager()
        mgr.current_round = 2
        mgr.claims["X"] = Claim("X", "OPEN", [
            ClaimEntry("FROM", "A", 1, "content X"),
            ClaimEntry("REBUTTAL", "B", 1, "rebuttal X"),
        ])
        mgr.claims["Y"] = Claim("Y", "CLOSED:共识", [
            ClaimEntry("FROM", "B", 1, "content Y"),
            ClaimEntry("HOST", "HOST", 2, "裁决：共识"),
        ])

        prompt = mgr.generate_update_prompt(prev_round=1)

        # OPEN claim full text should be present
        assert "CLAIM:X" in prompt
        assert "content X" in prompt
        # CLOSED claim should show status change but not full text
        assert "[已关闭] CLAIM:Y" in prompt
        # Task instructions should be present
        assert "REBUTTAL TO" in prompt

    def test_update_with_no_changes(self):
        mgr = ClaimsManager()
        mgr.current_round = 1
        prompt = mgr.generate_update_prompt(prev_round=0)
        assert "你的任务" in prompt

    def test_per_agent_prompt_separates_needs_response_from_awaiting_host(self):
        mgr = ClaimsManager()
        mgr.current_round = 3
        mgr.claims["mature"] = Claim("mature", "OPEN", [
            ClaimEntry("FROM", "A", 1, "mature body"),
            ClaimEntry("ACCEPT", "B", 2, "ok"),
            ClaimEntry("ACCEPT", "C", 2, "ok"),
        ])
        mgr.claims["needs-c"] = Claim("needs-c", "OPEN", [
            ClaimEntry("FROM", "A", 1, "needs body"),
            ClaimEntry("ACCEPT", "B", 2, "ok"),
        ])

        prompt = mgr.generate_update_prompt(
            prev_round=2,
            agent_name="C",
            all_agent_names={"A", "B", "C"},
        )

        assert "## NEEDS_YOUR_RESPONSE" in prompt
        assert "CLAIM:needs-c" in prompt
        assert "## AWAITING_HOST" in prompt
        assert "CLAIM:mature" in prompt
        assert "mature body" not in prompt
