"""Tests for usage summary generation (discuss_agent.audit.generate_usage_summary)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from discuss_agent.audit import (
    _format_duration_ms,
    _format_number,
    _parse_audit_files,
    generate_usage_summary,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, events: list[dict]) -> None:
    """Write a list of dicts as JSONL to the given path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")


def _make_roundtrip(
    agent: str,
    *,
    input_tokens: int = 1000,
    output_tokens: int = 200,
    cache_read: int = 0,
    cache_write: int = 0,
    reasoning: int = 0,
    cost: float | None = None,
    roundtrip_idx: int = 0,
) -> dict:
    ev = {
        "ts": "2026-04-12T23:00:00Z",
        "agent": agent,
        "event": "roundtrip_tokens",
        "roundtrip_idx": roundtrip_idx,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }
    if cache_read:
        ev["cache_read_tokens"] = cache_read
    if cache_write:
        ev["cache_write_tokens"] = cache_write
    if reasoning:
        ev["reasoning_tokens"] = reasoning
    if cost is not None:
        ev["cost"] = cost
    return ev


def _make_call_start(agent: str) -> dict:
    return {
        "ts": "2026-04-12T23:00:00Z",
        "agent": agent,
        "event": "call_start",
        "prompt_preview": "test prompt...",
    }


def _make_call_end(agent: str, *, duration_ms: float = 60000.0) -> dict:
    return {
        "ts": "2026-04-12T23:01:00Z",
        "agent": agent,
        "event": "call_end",
        "duration_ms": duration_ms,
    }


# ---------------------------------------------------------------------------
# _format_number tests
# ---------------------------------------------------------------------------


class TestFormatNumber:
    def test_zero(self):
        assert _format_number(0) == "0"

    def test_small(self):
        assert _format_number(42) == "42"

    def test_thousands(self):
        assert _format_number(1234567) == "1,234,567"

    def test_float(self):
        assert _format_number(1234.56) == "1,234.56"

    def test_negative(self):
        assert _format_number(-1000) == "-1,000"


# ---------------------------------------------------------------------------
# _format_duration_ms tests
# ---------------------------------------------------------------------------


class TestFormatDurationMs:
    def test_zero(self):
        assert _format_duration_ms(0) == "0s"

    def test_negative(self):
        assert _format_duration_ms(-1000) == "0s"

    def test_seconds_only(self):
        assert _format_duration_ms(45000) == "45s"

    def test_minutes_and_seconds(self):
        assert _format_duration_ms(1112000) == "18m 32s"

    def test_hours_minutes_seconds(self):
        assert _format_duration_ms(3661000) == "1h 1m 1s"

    def test_exact_minute(self):
        assert _format_duration_ms(60000) == "1m"

    def test_exact_hour(self):
        assert _format_duration_ms(3600000) == "1h"


# ---------------------------------------------------------------------------
# _parse_audit_files tests
# ---------------------------------------------------------------------------


class TestParseAuditFiles:
    def test_empty_dir(self, tmp_path):
        audit_dir = tmp_path / "audit"
        audit_dir.mkdir()
        result = _parse_audit_files(audit_dir)
        assert result == {}

    def test_nonexistent_dir(self, tmp_path):
        audit_dir = tmp_path / "audit"
        result = _parse_audit_files(audit_dir)
        assert result == {}

    def test_single_agent(self, tmp_path):
        audit_dir = tmp_path / "audit"
        events = [
            _make_call_start("Author"),
            _make_roundtrip("Author", input_tokens=500, output_tokens=100),
            _make_call_end("Author", duration_ms=30000),
        ]
        _write_jsonl(audit_dir / "Author.jsonl", events)
        result = _parse_audit_files(audit_dir)
        assert "Author" in result
        assert len(result["Author"]) == 3

    def test_multiple_agents(self, tmp_path):
        audit_dir = tmp_path / "audit"
        _write_jsonl(audit_dir / "Author.jsonl", [_make_call_start("Author")])
        _write_jsonl(audit_dir / "Reviewer.jsonl", [_make_call_start("Reviewer")])
        result = _parse_audit_files(audit_dir)
        assert len(result) == 2

    def test_skips_invalid_json_lines(self, tmp_path):
        audit_dir = tmp_path / "audit"
        audit_dir.mkdir(parents=True, exist_ok=True)
        path = audit_dir / "Agent.jsonl"
        with open(path, "w") as f:
            f.write('{"event": "call_start"}\n')
            f.write("this is not json\n")
            f.write('{"event": "call_end"}\n')
        result = _parse_audit_files(audit_dir)
        assert len(result["Agent"]) == 2

    def test_skips_empty_lines(self, tmp_path):
        audit_dir = tmp_path / "audit"
        audit_dir.mkdir(parents=True, exist_ok=True)
        path = audit_dir / "Agent.jsonl"
        with open(path, "w") as f:
            f.write('{"event": "call_start"}\n')
            f.write("\n")
            f.write('{"event": "call_end"}\n')
        result = _parse_audit_files(audit_dir)
        assert len(result["Agent"]) == 2


# ---------------------------------------------------------------------------
# generate_usage_summary tests
# ---------------------------------------------------------------------------


class TestGenerateUsageSummary:
    def _setup_basic_session(self, tmp_path, *, num_rounds=2, num_agents=2):
        """Create a session dir with audit data for testing."""
        session_dir = tmp_path / "2026-04-12_2332"
        audit_dir = session_dir / "audit"

        agents = [f"Agent{i}" for i in range(num_agents)]
        for agent in agents:
            events = []
            for r in range(num_rounds):
                events.append(_make_call_start(agent))
                events.append(_make_roundtrip(
                    agent,
                    input_tokens=1000 * (r + 1),
                    output_tokens=200 * (r + 1),
                    cache_read=500 * (r + 1),
                    cache_write=300 * (r + 1),
                    cost=0.05 * (r + 1),
                    roundtrip_idx=0,
                ))
                events.append(_make_call_end(agent, duration_ms=60000.0 * (r + 1)))
            _write_jsonl(audit_dir / f"{agent}.jsonl", events)

        return str(session_dir)

    def test_returns_none_when_no_audit_dir(self, tmp_path):
        session_dir = tmp_path / "empty_session"
        session_dir.mkdir()
        result = generate_usage_summary(str(session_dir))
        assert result is None

    def test_returns_none_when_empty_audit_dir(self, tmp_path):
        session_dir = tmp_path / "empty_audit"
        (session_dir / "audit").mkdir(parents=True)
        result = generate_usage_summary(str(session_dir))
        assert result is None

    def test_generates_file(self, tmp_path):
        session_dir = self._setup_basic_session(tmp_path)
        result = generate_usage_summary(session_dir)
        assert result is not None
        assert Path(result).exists()
        assert Path(result).name == "usage_summary.md"

    def test_content_has_header(self, tmp_path):
        session_dir = self._setup_basic_session(tmp_path)
        generate_usage_summary(session_dir, model_name="claude-opus-4.6-1m")
        content = (Path(session_dir) / "usage_summary.md").read_text()
        assert "# Token Usage Summary" in content
        assert "Run: 2026-04-12_2332" in content
        assert "Model: claude-opus-4.6-1m" in content

    def test_per_agent_table(self, tmp_path):
        session_dir = self._setup_basic_session(tmp_path, num_rounds=2, num_agents=2)
        generate_usage_summary(session_dir)
        content = (Path(session_dir) / "usage_summary.md").read_text()
        assert "## Per Agent" in content
        assert "Agent0" in content
        assert "Agent1" in content
        assert "**Total**" in content

    def test_per_round_table(self, tmp_path):
        session_dir = self._setup_basic_session(tmp_path, num_rounds=3)
        generate_usage_summary(session_dir)
        content = (Path(session_dir) / "usage_summary.md").read_text()
        assert "## Per Round" in content
        # Should have 3 round rows
        assert "| 1 " in content
        assert "| 2 " in content
        assert "| 3 " in content

    def test_cost_section_when_cost_present(self, tmp_path):
        session_dir = self._setup_basic_session(tmp_path)
        generate_usage_summary(session_dir)
        content = (Path(session_dir) / "usage_summary.md").read_text()
        assert "## Cost" in content
        assert "$" in content

    def test_no_cost_section_when_zero_cost(self, tmp_path):
        session_dir = tmp_path / "no_cost_session"
        audit_dir = session_dir / "audit"
        events = [
            _make_call_start("Author"),
            _make_roundtrip("Author", input_tokens=1000, output_tokens=200),
            _make_call_end("Author"),
        ]
        _write_jsonl(audit_dir / "Author.jsonl", events)
        generate_usage_summary(str(session_dir))
        content = (session_dir / "usage_summary.md").read_text()
        assert "## Cost" not in content

    def test_thousands_separator_in_tokens(self, tmp_path):
        session_dir = tmp_path / "big_tokens"
        audit_dir = session_dir / "audit"
        events = [
            _make_call_start("Author"),
            _make_roundtrip("Author", input_tokens=2104320, output_tokens=35420),
            _make_call_end("Author"),
        ]
        _write_jsonl(audit_dir / "Author.jsonl", events)
        generate_usage_summary(str(session_dir))
        content = (session_dir / "usage_summary.md").read_text()
        assert "2,104,320" in content
        assert "35,420" in content

    def test_human_readable_duration(self, tmp_path):
        session_dir = tmp_path / "duration_test"
        audit_dir = session_dir / "audit"
        events = [
            _make_call_start("Author"),
            _make_roundtrip("Author"),
            _make_call_end("Author", duration_ms=1112000),  # 18m 32s
        ]
        _write_jsonl(audit_dir / "Author.jsonl", events)
        generate_usage_summary(str(session_dir))
        content = (session_dir / "usage_summary.md").read_text()
        assert "18m 32s" in content

    def test_total_rounds_override(self, tmp_path):
        session_dir = self._setup_basic_session(tmp_path, num_rounds=2)
        generate_usage_summary(session_dir, total_rounds=6)
        content = (Path(session_dir) / "usage_summary.md").read_text()
        assert "Rounds: 6" in content

    def test_run_name_from_dir(self, tmp_path):
        session_dir = self._setup_basic_session(tmp_path)
        generate_usage_summary(session_dir)
        content = (Path(session_dir) / "usage_summary.md").read_text()
        assert "Run: 2026-04-12_2332" in content

    def test_custom_run_name(self, tmp_path):
        session_dir = self._setup_basic_session(tmp_path)
        generate_usage_summary(session_dir, run_name="custom-run-42")
        content = (Path(session_dir) / "usage_summary.md").read_text()
        assert "Run: custom-run-42" in content

    def test_chinese_agent_names(self, tmp_path):
        """Agent names can be Chinese characters."""
        session_dir = tmp_path / "chinese_session"
        audit_dir = session_dir / "audit"
        events = [
            _make_call_start("正文写手"),
            _make_roundtrip("正文写手", input_tokens=5000, output_tokens=1000),
            _make_call_end("正文写手", duration_ms=120000),
        ]
        _write_jsonl(audit_dir / "正文写手.jsonl", events)
        generate_usage_summary(str(session_dir))
        content = (session_dir / "usage_summary.md").read_text()
        assert "正文写手" in content

    def test_graceful_with_incomplete_data(self, tmp_path):
        """Should handle files with only some event types."""
        session_dir = tmp_path / "incomplete"
        audit_dir = session_dir / "audit"
        # Only roundtrip_tokens, no call_start/call_end
        events = [
            _make_roundtrip("Agent", input_tokens=500, output_tokens=100),
        ]
        _write_jsonl(audit_dir / "Agent.jsonl", events)
        result = generate_usage_summary(str(session_dir))
        assert result is not None
        content = (session_dir / "usage_summary.md").read_text()
        assert "Agent" in content
        assert "500" in content

    def test_cache_and_reasoning_tokens(self, tmp_path):
        """Cache read/write and reasoning tokens should be aggregated."""
        session_dir = tmp_path / "cache_test"
        audit_dir = session_dir / "audit"
        events = [
            _make_call_start("Agent"),
            _make_roundtrip(
                "Agent",
                input_tokens=10000,
                output_tokens=2000,
                cache_read=5000,
                cache_write=3000,
                reasoning=1000,
            ),
            _make_roundtrip(
                "Agent",
                input_tokens=8000,
                output_tokens=1500,
                cache_read=4000,
                cache_write=2000,
                reasoning=500,
                roundtrip_idx=1,
            ),
            _make_call_end("Agent"),
        ]
        _write_jsonl(audit_dir / "Agent.jsonl", events)
        generate_usage_summary(str(session_dir))
        content = (session_dir / "usage_summary.md").read_text()
        # 5000 + 4000 = 9000 cache read
        assert "9,000" in content
        # 3000 + 2000 = 5000 cache write
        assert "5,000" in content
        # 1000 + 500 = 1500 reasoning
        assert "1,500" in content

    def test_multiple_rounds_per_agent(self, tmp_path):
        """Per-round stats should separate each call correctly."""
        session_dir = tmp_path / "multi_round"
        audit_dir = session_dir / "audit"
        events = [
            # Round 1
            _make_call_start("Agent"),
            _make_roundtrip("Agent", input_tokens=1000, output_tokens=200),
            _make_call_end("Agent", duration_ms=30000),
            # Round 2
            _make_call_start("Agent"),
            _make_roundtrip("Agent", input_tokens=2000, output_tokens=400),
            _make_call_end("Agent", duration_ms=60000),
        ]
        _write_jsonl(audit_dir / "Agent.jsonl", events)
        generate_usage_summary(str(session_dir))
        content = (session_dir / "usage_summary.md").read_text()
        # Per Round table should have 2 rows
        assert "| 1 " in content
        assert "| 2 " in content
        # Totals: 2 calls
        assert "**2**" in content
