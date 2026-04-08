"""Research content tool — reads PDF file content using pdftotext."""

from __future__ import annotations

import os
import subprocess

from agno.tools import Toolkit


class ResearchContentTools(Toolkit):
    """Toolkit for extracting text content from PDF research reports."""

    def __init__(self, allowed_dir: str = "~/ima-downloads/"):
        super().__init__(name="research_content")
        self._allowed_dir = os.path.realpath(os.path.expanduser(allowed_dir))

    def read_content(self, file_path: str) -> str:
        """Extract text content from a PDF research report using pdftotext.

        Reads the specified PDF file and returns its text content. The file must
        exist on disk, be a valid PDF, and reside within the allowed research
        directory. Uses the system pdftotext utility for extraction.

        Parameters:
            file_path: Path to the PDF file to read. Must be within the
                configured research directory.

        Returns:
            The extracted text content of the PDF, or an error message string if
            the file does not exist, is outside the allowed directory, or
            extraction fails.
        """
        real_path = os.path.realpath(os.path.expanduser(file_path))
        if not real_path.startswith(self._allowed_dir):
            return f"路径不在允许的研报目录内: {file_path}"

        if not os.path.isfile(real_path):
            return f"文件不存在: {file_path}"

        try:
            result = subprocess.run(
                ["pdftotext", real_path, "-"],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except Exception as exc:
            return f"PDF读取失败: {exc}"

        if result.returncode != 0:
            return f"PDF读取失败 (exit {result.returncode}): {result.stderr.strip()}"

        return result.stdout
