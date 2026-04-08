"""Tests for discuss_agent.registry — import_from_path."""

from __future__ import annotations

import pytest

from discuss_agent.registry import import_from_path


class TestImportFromPath:
    def test_imports_builtin_class(self):
        result = import_from_path("collections.OrderedDict")
        from collections import OrderedDict
        assert result is OrderedDict

    def test_imports_nested_module(self):
        result = import_from_path("os.path.join")
        import os.path
        assert result is os.path.join

    def test_no_dot_raises_value_error(self):
        with pytest.raises(ValueError, match="Invalid dotted path"):
            import_from_path("nodot")

    def test_bad_module_raises_import_error(self):
        with pytest.raises(ImportError, match="nonexistent_module_xyz"):
            import_from_path("nonexistent_module_xyz.SomeClass")

    def test_bad_attr_raises_import_error(self):
        with pytest.raises(ImportError, match="NoSuchAttr"):
            import_from_path("os.path.NoSuchAttr")

    def test_error_includes_dotted_path(self):
        with pytest.raises(ImportError, match="bad_pkg.BadClass"):
            import_from_path("bad_pkg.BadClass")
