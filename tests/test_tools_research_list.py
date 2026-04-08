"""Tests for the research_list tool."""

import os
from datetime import datetime, timedelta

import pytest

from discuss_agent.tools.research_list import ResearchListTools


class TestListRecentPdfs:
    """test_list_recent_pdfs — only recent PDFs within `days` are returned."""

    def test_list_recent_pdfs(self, tmp_path):
        # Set up directory structure mimicking ~/ima-downloads/
        research_subdir = tmp_path / "七、中金、中信、华泰等内资研报"
        research_subdir.mkdir()

        today = datetime.now()
        yesterday = today - timedelta(days=1)
        old_date = today - timedelta(days=10)

        today_str = today.strftime("%Y%m%d")
        yesterday_str = yesterday.strftime("%Y%m%d")
        old_str = old_date.strftime("%Y%m%d")

        # Create PDFs: recent ones and an old one
        (research_subdir / f"{today_str}-GoldmanSachs-Market Outlook.pdf").touch()
        (research_subdir / f"{yesterday_str}-CICC-Strategy Report.pdf").touch()
        (research_subdir / f"{old_str}-Citi-Old Analysis.pdf").touch()

        tool = ResearchListTools(research_dir=str(tmp_path))
        result = tool.list_research(days=2)

        assert today_str in result
        assert yesterday_str in result
        assert old_str not in result
        assert "GoldmanSachs" in result
        assert "CICC" in result
        assert "Old Analysis" not in result


class TestFormatOutput:
    """test_format_output — output format is 'YYYYMMDD-Institution-Title' per line."""

    def test_format_output(self, tmp_path):
        research_subdir = tmp_path / "二、高盛、摩根等外资研报"
        research_subdir.mkdir()

        today_str = datetime.now().strftime("%Y%m%d")
        (research_subdir / f"{today_str}-GoldmanSachs-Q2 Macro View.pdf").touch()

        tool = ResearchListTools(research_dir=str(tmp_path))
        result = tool.list_research(days=2)

        lines = [line for line in result.strip().splitlines() if line.strip()]
        assert len(lines) >= 1
        # Each line should follow the pattern YYYYMMDD-Institution-Title
        for line in lines:
            parts = line.split("-", 2)
            assert len(parts) == 3, f"Expected 3 parts, got: {line}"
            assert len(parts[0]) == 8, f"Date part should be 8 chars: {parts[0]}"
            assert parts[0].isdigit(), f"Date part should be digits: {parts[0]}"


class TestFiltersCategory:
    """test_filters_category — only 研报 dirs are scanned, others ignored."""

    def test_filters_category(self, tmp_path):
        # 研报 directory
        yanbao_dir = tmp_path / "七、中金、中信、华泰等内资研报"
        yanbao_dir.mkdir()

        # Non-研报 directory
        books_dir = tmp_path / "📚投资书籍"
        books_dir.mkdir()

        today_str = datetime.now().strftime("%Y%m%d")
        (yanbao_dir / f"{today_str}-CICC-Good Report.pdf").touch()
        (books_dir / f"{today_str}-Author-Some Book.pdf").touch()

        tool = ResearchListTools(research_dir=str(tmp_path))
        result = tool.list_research(days=2)

        assert "Good Report" in result
        assert "Some Book" not in result


class TestDirectoryNotFound:
    """Error handling when research directory does not exist."""

    def test_dir_not_found_returns_error_string(self):
        tool = ResearchListTools(research_dir="/tmp/nonexistent_research_dir_xyz")
        result = tool.list_research(days=2)
        assert isinstance(result, str)
        assert "不存在" in result or "not found" in result.lower() or "error" in result.lower()
