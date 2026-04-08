"""Tests for the Archiver class in discuss_agent.persistence."""

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import yaml
import pytest


@dataclass
class _MockConfig:
    """Minimal dataclass to stand in for DiscussionConfig in tests."""

    min_rounds: int = 2
    max_rounds: int = 5
    model: str = "claude-sonnet-4-20250514"


# ---------------------------------------------------------------------------
# Task 4 — Session Setup
# ---------------------------------------------------------------------------


class TestStartSession:
    """Tests for Archiver.start_session."""

    def test_start_session_creates_dirs(self, tmp_path: Path):
        """start_session must create a timestamped dir and rounds/ subdir."""
        from discuss_agent.persistence import Archiver

        archiver = Archiver(base_dir=str(tmp_path))
        session_dir = archiver.start_session(_MockConfig())

        session_path = Path(session_dir)
        assert session_path.exists(), "session directory should exist"
        assert session_path.is_dir(), "session directory should be a directory"
        assert (session_path / "rounds").exists(), "rounds/ subdir should exist"
        assert (session_path / "rounds").is_dir(), "rounds/ should be a directory"
        # The session dir should be under the base dir
        assert str(session_path).startswith(str(tmp_path))

    def test_start_session_copies_config(self, tmp_path: Path):
        """start_session must write config.yaml with correct data."""
        from discuss_agent.persistence import Archiver

        cfg = _MockConfig(min_rounds=3, max_rounds=10, model="test-model")
        archiver = Archiver(base_dir=str(tmp_path))
        session_dir = archiver.start_session(cfg)

        config_path = Path(session_dir) / "config.yaml"
        assert config_path.exists(), "config.yaml should exist in session dir"

        loaded = yaml.safe_load(config_path.read_text())
        expected = asdict(cfg)
        assert loaded == expected, (
            f"config.yaml content should match dataclasses.asdict(config): "
            f"got {loaded}, expected {expected}"
        )


# ---------------------------------------------------------------------------
# Task 5 — Round / Summary Persistence
# ---------------------------------------------------------------------------


class TestSaveRound:
    """Tests for Archiver.save_round."""

    def _make_archiver(self, tmp_path: Path):
        from discuss_agent.persistence import Archiver

        archiver = Archiver(base_dir=str(tmp_path))
        archiver.start_session(_MockConfig())
        return archiver

    def test_save_round_express(self, tmp_path: Path):
        """save_round(1, 'express', data) creates rounds/round_1_express.json."""
        archiver = self._make_archiver(tmp_path)
        data = {"agent": "Agent A", "content": "I think X is important."}
        archiver.save_round(1, "express", data)

        round_file = Path(archiver._session_dir) / "rounds" / "round_1_express.json"
        assert round_file.exists(), "round_1_express.json should exist"

        loaded = json.loads(round_file.read_text())
        assert loaded == data, (
            f"JSON content should match: got {loaded}, expected {data}"
        )

    def test_save_round_host(self, tmp_path: Path):
        """save_round(1, 'host', data) creates rounds/round_1_host.json."""
        archiver = self._make_archiver(tmp_path)
        data = {"converged": False, "reason": "Still debating."}
        archiver.save_round(1, "host", data)

        round_file = Path(archiver._session_dir) / "rounds" / "round_1_host.json"
        assert round_file.exists(), "round_1_host.json should exist"

        loaded = json.loads(round_file.read_text())
        assert loaded == data


class TestSaveContextAndSummary:
    """Tests for Archiver.save_context and save_summary."""

    def _make_archiver(self, tmp_path: Path):
        from discuss_agent.persistence import Archiver

        archiver = Archiver(base_dir=str(tmp_path))
        archiver.start_session(_MockConfig())
        return archiver

    def test_save_context(self, tmp_path: Path):
        """save_context('text') creates context.md with that content."""
        archiver = self._make_archiver(tmp_path)
        text = "Here is the initial context for the discussion."
        archiver.save_context(text)

        ctx_path = Path(archiver._session_dir) / "context.md"
        assert ctx_path.exists(), "context.md should exist"
        assert ctx_path.read_text() == text

    def test_save_summary(self, tmp_path: Path):
        """save_summary('text') creates summary.md with that content."""
        archiver = self._make_archiver(tmp_path)
        text = "Final summary of the discussion."
        archiver.save_summary(text)

        summary_path = Path(archiver._session_dir) / "summary.md"
        assert summary_path.exists(), "summary.md should exist"
        assert summary_path.read_text() == text

    def test_save_error_log(self, tmp_path: Path):
        """save_error_log writes error.log with the error details."""
        archiver = self._make_archiver(tmp_path)
        archiver.save_error_log("All agents failed during express step")

        error_path = Path(archiver._session_dir) / "error.log"
        assert error_path.exists(), "error.log should exist"
        content = error_path.read_text()
        assert "All agents failed" in content
