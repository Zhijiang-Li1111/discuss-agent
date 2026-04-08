"""Tests for the published tool."""

import pytest

from discuss_agent.tools.published import PublishedTools


SAMPLE_PUBLISHED_MD = """\
# 已发布文章

| 日期 | 话题 | 核心论点 |
|------|------|----------|
| 2026-04-01 | 半导体行业展望 | AI芯片需求持续增长，国产替代加速 |
| 2026-03-28 | 新能源汽车 | 电动车渗透率将突破50% |
"""


class TestParsePublishedMd:
    """test_parse_published_md — verify parsing returns structured info."""

    def test_parse_published_md(self, tmp_path):
        pub_file = tmp_path / "PUBLISHED.md"
        pub_file.write_text(SAMPLE_PUBLISHED_MD, encoding="utf-8")

        tool = PublishedTools()
        result = tool.get_published(file_path=str(pub_file))

        assert isinstance(result, str)
        assert "2026-04-01" in result
        assert "半导体行业展望" in result
        assert "AI芯片需求持续增长" in result
        assert "2026-03-28" in result
        assert "新能源汽车" in result

    def test_multiple_rows_all_present(self, tmp_path):
        pub_file = tmp_path / "PUBLISHED.md"
        pub_file.write_text(SAMPLE_PUBLISHED_MD, encoding="utf-8")

        tool = PublishedTools()
        result = tool.get_published(file_path=str(pub_file))

        # Both rows should appear in the result
        assert "半导体" in result
        assert "新能源" in result


class TestFileNotFound:
    """test_file_not_found_returns_message — nonexistent path returns specific message."""

    def test_file_not_found_returns_message(self):
        tool = PublishedTools()
        result = tool.get_published(file_path="/tmp/nonexistent_published_xyz.md")

        assert result == "暂无已发布文章记录。"

    def test_default_path_not_found(self, tmp_path, monkeypatch):
        """When default PUBLISHED.md doesn't exist, returns the fallback message."""
        monkeypatch.chdir(tmp_path)
        tool = PublishedTools()
        result = tool.get_published()

        assert result == "暂无已发布文章记录。"
