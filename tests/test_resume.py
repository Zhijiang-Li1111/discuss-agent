"""Tests for the --resume / --rounds feature across persistence, engine, and CLI."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from discuss_agent.models import AgentUtterance, DiscussionResult

from tests.shared import make_config, patch_engine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_archive(
    base: Path, num_rounds: int = 3, with_host: bool = True
) -> Path:
    """Create a fake archive directory with *num_rounds* of data."""
    archive = base / "2026-04-11_2159"
    rounds_dir = archive / "rounds"
    rounds_dir.mkdir(parents=True)

    # Write context.md
    (archive / "context.md").write_text("Loaded context from archive")

    for rn in range(1, num_rounds + 1):
        express = {
            "utterances": [
                {"agent_name": "Agent-A", "content": f"Expr-A-R{rn}"},
                {"agent_name": "Agent-B", "content": f"Expr-B-R{rn}"},
            ]
        }
        challenge = {
            "utterances": [
                {"agent_name": "Agent-A", "content": f"Chal-A-R{rn}"},
                {"agent_name": "Agent-B", "content": f"Chal-B-R{rn}"},
            ]
        }
        (rounds_dir / f"round_{rn}_express.json").write_text(
            json.dumps(express, ensure_ascii=False)
        )
        (rounds_dir / f"round_{rn}_challenge.json").write_text(
            json.dumps(challenge, ensure_ascii=False)
        )
        if with_host:
            host = {"converged": False, "reason": "Ongoing", "remaining_disputes": []}
            (rounds_dir / f"round_{rn}_host.json").write_text(
                json.dumps(host, ensure_ascii=False)
            )

    return archive


# ---------------------------------------------------------------------------
# Persistence: resume_session
# ---------------------------------------------------------------------------


class TestArchiverResume:

    def test_resume_session_sets_session_dir(self, tmp_path: Path):
        from discuss_agent.persistence import Archiver

        archive = _create_archive(tmp_path)
        archiver = Archiver(base_dir=str(tmp_path))
        result = archiver.resume_session(str(archive))
        assert result == str(archive.resolve())
        assert archiver._session_dir == str(archive.resolve())

    def test_resume_session_nonexistent_raises(self, tmp_path: Path):
        from discuss_agent.persistence import Archiver

        archiver = Archiver(base_dir=str(tmp_path))
        with pytest.raises(FileNotFoundError, match="Archive directory not found"):
            archiver.resume_session(str(tmp_path / "nonexistent"))

    def test_resume_session_no_rounds_dir_raises(self, tmp_path: Path):
        from discuss_agent.persistence import Archiver

        no_rounds = tmp_path / "no_rounds_archive"
        no_rounds.mkdir()
        archiver = Archiver(base_dir=str(tmp_path))
        with pytest.raises(FileNotFoundError, match="rounds/ subdirectory not found"):
            archiver.resume_session(str(no_rounds))

    def test_resume_allows_saving_new_rounds(self, tmp_path: Path):
        """After resume_session, save_round should write into the same archive."""
        from discuss_agent.persistence import Archiver

        archive = _create_archive(tmp_path, num_rounds=2)
        archiver = Archiver(base_dir=str(tmp_path))
        archiver.resume_session(str(archive))

        archiver.save_round(3, "express", {"utterances": [{"agent_name": "Agent-A", "content": "new"}]})
        new_file = archive / "rounds" / "round_3_express.json"
        assert new_file.exists()
        data = json.loads(new_file.read_text())
        assert data["utterances"][0]["content"] == "new"


# ---------------------------------------------------------------------------
# Persistence: load_history
# ---------------------------------------------------------------------------


class TestLoadHistory:

    def test_load_history_returns_correct_count(self, tmp_path: Path):
        from discuss_agent.persistence import Archiver

        archive = _create_archive(tmp_path, num_rounds=3)
        archiver = Archiver()
        archiver.resume_session(str(archive))
        history = archiver.load_history()
        assert len(history) == 3

    def test_load_history_round_nums_correct(self, tmp_path: Path):
        from discuss_agent.persistence import Archiver

        archive = _create_archive(tmp_path, num_rounds=3)
        archiver = Archiver()
        archiver.resume_session(str(archive))
        history = archiver.load_history()
        assert [r.round_num for r in history] == [1, 2, 3]

    def test_load_history_expressions_populated(self, tmp_path: Path):
        from discuss_agent.persistence import Archiver

        archive = _create_archive(tmp_path, num_rounds=2)
        archiver = Archiver()
        archiver.resume_session(str(archive))
        history = archiver.load_history()
        r1 = history[0]
        assert len(r1.expressions) == 2
        assert r1.expressions[0].agent_name == "Agent-A"
        assert r1.expressions[0].content == "Expr-A-R1"

    def test_load_history_challenges_populated(self, tmp_path: Path):
        from discuss_agent.persistence import Archiver

        archive = _create_archive(tmp_path, num_rounds=2)
        archiver = Archiver()
        archiver.resume_session(str(archive))
        history = archiver.load_history()
        r1 = history[0]
        assert len(r1.challenges) == 2
        assert r1.challenges[1].agent_name == "Agent-B"
        assert r1.challenges[1].content == "Chal-B-R1"

    def test_load_history_host_judgment_loaded(self, tmp_path: Path):
        from discuss_agent.persistence import Archiver

        archive = _create_archive(tmp_path, num_rounds=1, with_host=True)
        archiver = Archiver()
        archiver.resume_session(str(archive))
        history = archiver.load_history()
        assert history[0].host_judgment is not None
        assert history[0].host_judgment["converged"] is False

    def test_load_history_no_host_is_none(self, tmp_path: Path):
        from discuss_agent.persistence import Archiver

        archive = _create_archive(tmp_path, num_rounds=1, with_host=False)
        archiver = Archiver()
        archiver.resume_session(str(archive))
        history = archiver.load_history()
        assert history[0].host_judgment is None


# ---------------------------------------------------------------------------
# Persistence: load_context
# ---------------------------------------------------------------------------


class TestLoadContext:

    def test_load_context_reads_file(self, tmp_path: Path):
        from discuss_agent.persistence import Archiver

        archive = _create_archive(tmp_path)
        archiver = Archiver()
        archiver.resume_session(str(archive))
        ctx = archiver.load_context()
        assert ctx == "Loaded context from archive"

    def test_load_context_missing_raises(self, tmp_path: Path):
        from discuss_agent.persistence import Archiver

        no_ctx = tmp_path / "no_ctx"
        no_ctx.mkdir()
        (no_ctx / "rounds").mkdir()
        archiver = Archiver()
        archiver.resume_session(str(no_ctx))
        with pytest.raises(FileNotFoundError, match="context.md not found"):
            archiver.load_context()


# ---------------------------------------------------------------------------
# Engine: resume run
# ---------------------------------------------------------------------------


class TestEngineResume:

    @patch("discuss_agent.engine.import_from_path")
    @patch("discuss_agent.engine.ContextManager")
    @patch("discuss_agent.engine.Agent")
    @pytest.mark.asyncio
    async def test_resume_loads_history_and_continues(
        self, MockAgent, MockCtxMgr, mock_import, tmp_path
    ):
        """Resume from 3 rounds, run 1 more -> round_num starts at 4."""
        from discuss_agent.engine import DiscussionEngine

        archive = _create_archive(tmp_path, num_rounds=3)
        config = make_config(min_rounds=1, max_rounds=10)
        engine = DiscussionEngine(config)

        # Don't mock archiver — use real one so load_history works
        from discuss_agent.persistence import Archiver

        real_archiver = Archiver(base_dir=str(tmp_path))
        engine._archiver = real_archiver

        engine._context_mgr.compress = AsyncMock(side_effect=lambda h, r: h)

        judgments = [
            {"converged": True, "reason": "Agreement", "remaining_disputes": []},
        ]
        patch_engine(engine, judgments)

        result = await engine.run(resume_path=str(archive), extra_rounds=1)

        assert result.converged is True
        # Round 4 was the only new round, and it converged
        assert result.rounds_completed == 4
        # Express was called once (for round 4) with round_num=4
        engine._express.assert_called_once()
        call_args = engine._express.call_args
        assert call_args[0][0] == 4  # round_num

    @patch("discuss_agent.engine.import_from_path")
    @patch("discuss_agent.engine.ContextManager")
    @patch("discuss_agent.engine.Agent")
    @pytest.mark.asyncio
    async def test_resume_agents_see_prior_history(
        self, MockAgent, MockCtxMgr, mock_import, tmp_path
    ):
        """Agents should receive loaded history in their prompts."""
        from discuss_agent.engine import DiscussionEngine

        archive = _create_archive(tmp_path, num_rounds=2)
        config = make_config(min_rounds=1, max_rounds=10)
        engine = DiscussionEngine(config)

        from discuss_agent.persistence import Archiver

        real_archiver = Archiver(base_dir=str(tmp_path))
        engine._archiver = real_archiver
        engine._context_mgr.compress = AsyncMock(side_effect=lambda h, r: h)

        # Capture the history passed to _express
        captured_history = []

        async def mock_express(round_num, context, history, **kwargs):
            captured_history.append(list(history))
            return [AgentUtterance("Agent-A", f"Expr-A-R{round_num}")]

        async def mock_challenge(round_num, expressions, **kwargs):
            return [AgentUtterance("Agent-A", f"Chal-A-R{round_num}")]

        async def mock_host_judge(history):
            return {"converged": True, "reason": "Done", "remaining_disputes": []}

        async def mock_host_summarize(history):
            return "Summary"

        engine._express = AsyncMock(side_effect=mock_express)
        engine._challenge = AsyncMock(side_effect=mock_challenge)
        engine._host_judge = AsyncMock(side_effect=mock_host_judge)
        engine._host_summarize = AsyncMock(side_effect=mock_host_summarize)

        await engine.run(resume_path=str(archive), extra_rounds=1)

        # The history passed to express in round 3 should contain 2 loaded rounds
        assert len(captured_history) == 1
        assert len(captured_history[0]) == 2
        assert captured_history[0][0].round_num == 1
        assert captured_history[0][1].round_num == 2

    @patch("discuss_agent.engine.import_from_path")
    @patch("discuss_agent.engine.ContextManager")
    @patch("discuss_agent.engine.Agent")
    @pytest.mark.asyncio
    async def test_resume_appends_to_same_directory(
        self, MockAgent, MockCtxMgr, mock_import, tmp_path
    ):
        """New rounds should be written to the same archive directory."""
        from discuss_agent.engine import DiscussionEngine

        archive = _create_archive(tmp_path, num_rounds=2)
        config = make_config(min_rounds=1, max_rounds=10)
        engine = DiscussionEngine(config)

        from discuss_agent.persistence import Archiver

        real_archiver = Archiver(base_dir=str(tmp_path))
        engine._archiver = real_archiver
        engine._context_mgr.compress = AsyncMock(side_effect=lambda h, r: h)

        judgments = [
            {"converged": False, "reason": "No", "remaining_disputes": ["X"]},
        ]
        patch_engine(engine, judgments)

        result = await engine.run(resume_path=str(archive), extra_rounds=1)

        # Result points at same directory
        assert str(archive.resolve()) in result.archive_path

        # New round files exist in same rounds/ dir
        rounds_dir = archive / "rounds"
        assert (rounds_dir / "round_3_express.json").exists()
        assert (rounds_dir / "round_3_challenge.json").exists()
        assert (rounds_dir / "round_3_host.json").exists()

    @patch("discuss_agent.engine.import_from_path")
    @patch("discuss_agent.engine.ContextManager")
    @patch("discuss_agent.engine.Agent")
    @pytest.mark.asyncio
    async def test_resume_context_from_archive(
        self, MockAgent, MockCtxMgr, mock_import, tmp_path
    ):
        """Context should come from archive, not regenerated."""
        from discuss_agent.engine import DiscussionEngine

        archive = _create_archive(tmp_path, num_rounds=1)
        config = make_config(min_rounds=1, max_rounds=10)
        engine = DiscussionEngine(config)

        from discuss_agent.persistence import Archiver

        real_archiver = Archiver(base_dir=str(tmp_path))
        engine._archiver = real_archiver
        engine._context_mgr.compress = AsyncMock(side_effect=lambda h, r: h)

        captured_context = []

        async def mock_express(round_num, context, history, **kwargs):
            captured_context.append(context)
            return [AgentUtterance("Agent-A", "opinion")]

        engine._express = AsyncMock(side_effect=mock_express)
        engine._challenge = AsyncMock(
            return_value=[AgentUtterance("Agent-A", "challenge")]
        )
        engine._host_judge = AsyncMock(
            return_value={"converged": True, "reason": "Done", "remaining_disputes": []}
        )
        engine._host_summarize = AsyncMock(return_value="Summary")

        await engine.run(resume_path=str(archive), extra_rounds=1)

        # Context should be the one loaded from archive
        assert captured_context[0] == "Loaded context from archive"
        # build_initial_context should NOT be called
        engine._context_mgr.build_initial_context.assert_not_called()

    @patch("discuss_agent.engine.import_from_path")
    @patch("discuss_agent.engine.ContextManager")
    @patch("discuss_agent.engine.Agent")
    @pytest.mark.asyncio
    async def test_resume_nonexistent_path_raises(
        self, MockAgent, MockCtxMgr, mock_import, tmp_path
    ):
        from discuss_agent.engine import DiscussionEngine

        config = make_config()
        engine = DiscussionEngine(config)

        with pytest.raises(FileNotFoundError):
            await engine.run(
                resume_path=str(tmp_path / "nonexistent"), extra_rounds=1
            )

    @patch("discuss_agent.engine.import_from_path")
    @patch("discuss_agent.engine.ContextManager")
    @patch("discuss_agent.engine.Agent")
    @pytest.mark.asyncio
    async def test_resume_extra_rounds_none_raises(
        self, MockAgent, MockCtxMgr, mock_import, tmp_path
    ):
        """Engine should raise ValueError when extra_rounds is None on resume."""
        from discuss_agent.engine import DiscussionEngine

        archive = _create_archive(tmp_path, num_rounds=1)
        config = make_config()
        engine = DiscussionEngine(config)

        with pytest.raises(ValueError, match="extra_rounds must be a positive integer"):
            await engine.run(resume_path=str(archive), extra_rounds=None)

    @patch("discuss_agent.engine.import_from_path")
    @patch("discuss_agent.engine.ContextManager")
    @patch("discuss_agent.engine.Agent")
    @pytest.mark.asyncio
    async def test_resume_extra_rounds_zero_raises(
        self, MockAgent, MockCtxMgr, mock_import, tmp_path
    ):
        """Engine should raise ValueError when extra_rounds is 0 on resume."""
        from discuss_agent.engine import DiscussionEngine

        archive = _create_archive(tmp_path, num_rounds=1)
        config = make_config()
        engine = DiscussionEngine(config)

        with pytest.raises(ValueError, match="extra_rounds must be a positive integer"):
            await engine.run(resume_path=str(archive), extra_rounds=0)

    @patch("discuss_agent.engine.import_from_path")
    @patch("discuss_agent.engine.ContextManager")
    @patch("discuss_agent.engine.Agent")
    @pytest.mark.asyncio
    async def test_normal_run_unaffected(
        self, MockAgent, MockCtxMgr, mock_import
    ):
        """run() without resume params behaves exactly as before."""
        from discuss_agent.engine import DiscussionEngine

        config = make_config(min_rounds=1, max_rounds=2)
        engine = DiscussionEngine(config)

        engine._archiver = MagicMock()
        engine._archiver.start_session.return_value = "/tmp/session"
        engine._context_mgr.build_initial_context = AsyncMock(return_value="ctx")
        engine._context_mgr.compress = AsyncMock(side_effect=lambda h, r: h)

        judgments = [
            {"converged": True, "reason": "Quick", "remaining_disputes": []},
        ]
        patch_engine(engine, judgments)

        result = await engine.run()

        assert result.converged is True
        assert result.rounds_completed == 1
        engine._archiver.start_session.assert_called_once()
        engine._context_mgr.build_initial_context.assert_called_once()

    @patch("discuss_agent.engine.import_from_path")
    @patch("discuss_agent.engine.ContextManager")
    @patch("discuss_agent.engine.Agent")
    @pytest.mark.asyncio
    async def test_resume_max_rounds_from_extra(
        self, MockAgent, MockCtxMgr, mock_import, tmp_path
    ):
        """With 3 loaded rounds + extra_rounds=2, loop should run rounds 4 and 5."""
        from discuss_agent.engine import DiscussionEngine

        archive = _create_archive(tmp_path, num_rounds=3)
        config = make_config(min_rounds=1, max_rounds=10)
        engine = DiscussionEngine(config)

        from discuss_agent.persistence import Archiver

        real_archiver = Archiver(base_dir=str(tmp_path))
        engine._archiver = real_archiver
        engine._context_mgr.compress = AsyncMock(side_effect=lambda h, r: h)

        judgments = [
            {"converged": False, "reason": "No", "remaining_disputes": ["X"]},
            {"converged": False, "reason": "No", "remaining_disputes": ["X"]},
        ]
        patch_engine(engine, judgments)

        result = await engine.run(resume_path=str(archive), extra_rounds=2)

        assert result.converged is False
        assert result.rounds_completed == 5
        assert engine._express.call_count == 2


# ---------------------------------------------------------------------------
# CLI: --resume and --rounds
# ---------------------------------------------------------------------------


class TestCLIResume:

    def test_resume_without_rounds_exits(self, sample_config_yaml):
        """--resume without --rounds should exit with code 1."""
        with patch("sys.argv", ["discuss_agent", sample_config_yaml, "--resume", "/some/path"]):
            from discuss_agent.main import main

            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    def test_resume_with_rounds_passes_to_engine(self, sample_config_yaml):
        """--resume and --rounds should be forwarded to engine.run()."""
        mock_result = DiscussionResult(
            converged=True,
            rounds_completed=4,
            archive_path="/tmp/discussions/2026-04-11_2159",
            summary="Test summary",
            remaining_disputes=[],
        )

        with (
            patch(
                "sys.argv",
                [
                    "discuss_agent",
                    sample_config_yaml,
                    "--resume",
                    "/tmp/discussions/2026-04-11_2159",
                    "--rounds",
                    "2",
                ],
            ),
            patch("discuss_agent.main.DiscussionEngine") as mock_engine_cls,
            patch("discuss_agent.main.ConfigLoader") as mock_loader,
        ):
            mock_loader.load.return_value = MagicMock()
            mock_engine = MagicMock()
            mock_engine.run = AsyncMock(return_value=mock_result)
            mock_engine_cls.return_value = mock_engine

            from discuss_agent.main import main

            main()

            mock_engine.run.assert_called_once_with(
                resume_path="/tmp/discussions/2026-04-11_2159",
                extra_rounds=2,
                guidance=None,
            )

    def test_normal_run_passes_none(self, sample_config_yaml):
        """Without --resume, engine.run() gets None for both params."""
        mock_result = DiscussionResult(
            converged=True,
            rounds_completed=2,
            archive_path="/tmp/sess",
            summary="Summary",
            remaining_disputes=[],
        )

        with (
            patch("sys.argv", ["discuss_agent", sample_config_yaml]),
            patch("discuss_agent.main.DiscussionEngine") as mock_engine_cls,
            patch("discuss_agent.main.ConfigLoader") as mock_loader,
        ):
            mock_loader.load.return_value = MagicMock()
            mock_engine = MagicMock()
            mock_engine.run = AsyncMock(return_value=mock_result)
            mock_engine_cls.return_value = mock_engine

            from discuss_agent.main import main

            main()

            mock_engine.run.assert_called_once_with(
                resume_path=None,
                extra_rounds=None,
                guidance=None,
            )

    def test_rounds_without_resume_exits(self, sample_config_yaml):
        """--rounds without --resume should exit with code 1."""
        with patch("sys.argv", ["discuss_agent", sample_config_yaml, "--rounds", "2"]):
            from discuss_agent.main import main

            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    def test_rounds_zero_exits(self, sample_config_yaml):
        """--rounds 0 should exit with code 1."""
        with patch(
            "sys.argv",
            ["discuss_agent", sample_config_yaml, "--resume", "/some/path", "--rounds", "0"],
        ):
            from discuss_agent.main import main

            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    def test_rounds_negative_exits(self, sample_config_yaml):
        """--rounds -1 should exit with code 1."""
        with patch(
            "sys.argv",
            ["discuss_agent", sample_config_yaml, "--resume", "/some/path", "--rounds", "-1"],
        ):
            from discuss_agent.main import main

            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1
