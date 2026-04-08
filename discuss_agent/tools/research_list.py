"""Research list tool — scans research directories for recent PDF reports."""

from __future__ import annotations

import os
from datetime import datetime, timedelta

from agno.tools import Toolkit


_CATEGORY_KEYWORDS = ("行业", "宏观", "策略")


class ResearchListTools(Toolkit):
    """Toolkit for listing recently downloaded research reports."""

    def __init__(self, research_dir: str = "~/ima-downloads/"):
        super().__init__(name="research_list")
        self.research_dir = os.path.expanduser(research_dir)

    def list_research(self, days: int = 2) -> str:
        """List recent industry and macro research reports from the local directory.

        Scans sub-directories whose names contain '外资研报' or '内资研报' for PDF
        files with date-prefixed filenames (YYYYMMDD-Institution-Title.pdf).
        Within those, only sub-paths containing industry/macro category keywords
        (行业, 宏观, 策略) are included — individual stock reports are excluded.
        Only reports published within the specified number of days are returned.

        Parameters:
            days: How many days back to look (default 2). A value of 2 means
                  today and yesterday.

        Returns:
            A newline-separated list of 'YYYYMMDD-Institution-Title' entries, or
            an error message string if the research directory does not exist.
        """
        if not os.path.isdir(self.research_dir):
            return f"研报目录不存在: {self.research_dir}"

        cutoff = datetime.now() - timedelta(days=days)
        results: list[str] = []

        for entry in os.listdir(self.research_dir):
            subdir_path = os.path.join(self.research_dir, entry)
            if not os.path.isdir(subdir_path):
                continue
            # Only scan directories that contain 研报 category markers
            if "外资研报" not in entry and "内资研报" not in entry:
                continue

            # Walk recursively — PDFs may be in nested subdirectories
            for dirpath, _dirnames, filenames in os.walk(subdir_path):
                # Only include paths with industry/macro category keywords
                rel_path = os.path.relpath(dirpath, subdir_path)
                if rel_path != "." and not any(
                    kw in rel_path for kw in _CATEGORY_KEYWORDS
                ):
                    continue
                for fname in filenames:
                    if not fname.lower().endswith(".pdf"):
                        continue
                    # Parse date prefix
                    stem = fname[:-4]  # strip .pdf
                    parts = stem.split("-", 2)
                    if len(parts) < 3:
                        continue
                    date_str = parts[0]
                    if len(date_str) != 8 or not date_str.isdigit():
                        continue
                    try:
                        file_date = datetime.strptime(date_str, "%Y%m%d")
                    except ValueError:
                        continue
                    if file_date < cutoff:
                        continue
                    results.append(f"{parts[0]}-{parts[1]}-{parts[2]}")

        if not results:
            return "未找到近期研报。"

        results.sort(reverse=True)
        return "\n".join(results)
