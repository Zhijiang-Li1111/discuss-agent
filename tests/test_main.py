"""Tests for the CLI entry point."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from discuss_agent.models import DiscussionResult


class TestCLI:
    def test_accepts_yaml_path(self, sample_config_yaml):
        """AC-7.1: CLI accepts YAML config file path as argument."""
        mock_result = DiscussionResult(
            converged=True,
            rounds_completed=2,
            archive_path="/tmp/discussions/2026-04-08_2100",
            summary="Test summary",
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

            mock_loader.load.assert_called_once_with(sample_config_yaml)
            mock_engine_cls.assert_called_once()
            mock_engine.run.assert_called_once()

    def test_prints_archive_path(self, sample_config_yaml, capsys):
        """AC-7.2: Prints archive directory path after completion."""
        mock_result = DiscussionResult(
            converged=True,
            rounds_completed=2,
            archive_path="/tmp/discussions/2026-04-08_2100",
            summary="Test summary",
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

            captured = capsys.readouterr()
            assert "/tmp/discussions/2026-04-08_2100" in captured.out

    def test_nonexistent_config_exits(self):
        """AC-7.3: Nonexistent config file exits with code 1."""
        with patch("sys.argv", ["discuss_agent", "/nonexistent/config.yaml"]):
            from discuss_agent.main import main

            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1
