"""Published tool — reads and parses the PUBLISHED.md file of past articles."""

from __future__ import annotations

import os

from agno.tools import Toolkit


class PublishedTools(Toolkit):
    """Toolkit for reading previously published discussion articles."""

    def __init__(self):
        super().__init__(name="published")

    def get_published(self, file_path: str = "PUBLISHED.md") -> str:
        """Read and parse the PUBLISHED.md file listing previously published articles.

        Parses a Markdown table with columns for date, topic, and core argument.
        This helps agents avoid repeating topics that have already been covered and
        build on previous discussions.

        Parameters:
            file_path: Path to the PUBLISHED.md file (default 'PUBLISHED.md' in cwd).

        Returns:
            A formatted list of previously published articles with date, topic, and
            core argument, or '暂无已发布文章记录。' if the file does not exist.
        """
        if not os.path.isfile(file_path):
            return "暂无已发布文章记录。"

        try:
            with open(file_path, "r", encoding="utf-8") as fh:
                content = fh.read()
        except Exception:
            return "暂无已发布文章记录。"

        lines = content.strip().splitlines()
        entries: list[str] = []

        for line in lines:
            line = line.strip()
            if not line.startswith("|"):
                continue
            # Skip header row and separator row
            cells = [c.strip() for c in line.split("|")]
            # split on | gives empty strings at start/end: ['', 'col1', 'col2', ...]
            cells = [c for c in cells if c]
            if len(cells) < 3:
                continue
            # Skip the separator line (|------|------|----------|)
            if all(set(c) <= {"-"} for c in cells):
                continue
            # Skip the header row
            if cells[0] == "日期":
                continue

            date_val = cells[0]
            topic = cells[1]
            argument = cells[2]
            entries.append(f"{date_val} | {topic} | {argument}")

        if not entries:
            return "暂无已发布文章记录。"

        return "\n".join(entries)
