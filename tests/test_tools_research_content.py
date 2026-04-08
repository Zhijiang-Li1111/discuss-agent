"""Tests for the research_content tool."""

import subprocess
from unittest.mock import patch

import pytest

from discuss_agent.tools.research_content import ResearchContentTools


class TestReadPdfContent:
    """test_read_pdf_content — mock subprocess to simulate pdftotext extraction."""

    def test_read_pdf_content(self, tmp_path):
        # Create a dummy file inside the allowed directory
        pdf_path = tmp_path / "test_report.pdf"
        pdf_path.touch()

        mock_result = subprocess.CompletedProcess(
            args=["pdftotext", str(pdf_path), "-"],
            returncode=0,
            stdout="This is the extracted text content from a PDF report.",
            stderr="",
        )

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            tool = ResearchContentTools(allowed_dir=str(tmp_path))
            result = tool.read_content(file_path=str(pdf_path))

            mock_run.assert_called_once()
            assert "extracted text content" in result


class TestFileNotFound:
    """test_file_not_found — nonexistent path returns error message string."""

    def test_file_not_found(self, tmp_path):
        tool = ResearchContentTools(allowed_dir=str(tmp_path))
        result = tool.read_content(file_path=str(tmp_path / "nonexistent.pdf"))

        assert isinstance(result, str)
        assert "不存在" in result

    def test_path_traversal_blocked(self, tmp_path):
        """Paths outside the allowed directory are rejected."""
        tool = ResearchContentTools(allowed_dir=str(tmp_path))
        result = tool.read_content(file_path="/etc/passwd")

        assert isinstance(result, str)
        assert "不在允许的研报目录内" in result


class TestSubprocessFailure:
    """Verify graceful handling when pdftotext fails."""

    def test_subprocess_failure(self, tmp_path):
        pdf_path = tmp_path / "corrupt.pdf"
        pdf_path.touch()

        mock_result = subprocess.CompletedProcess(
            args=["pdftotext", str(pdf_path), "-"],
            returncode=1,
            stdout="",
            stderr="Error reading PDF",
        )

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            tool = ResearchContentTools(allowed_dir=str(tmp_path))
            result = tool.read_content(file_path=str(pdf_path))

            assert isinstance(result, str)
            assert "失败" in result
