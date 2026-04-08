"""Runtime class/function loader for the multi-agent discussion framework.

Replaces the entry_points-based PluginRegistry with a simple importlib loader.
"""

from __future__ import annotations

import importlib


def import_from_path(dotted_path: str):
    """Import a class or function from a Python dotted path.

    Example: ``import_from_path('my_package.tools.MyTool')`` imports and
    returns the ``MyTool`` attribute from ``my_package.tools``.

    Raises ``ImportError`` if the module cannot be found, or
    ``AttributeError`` if the attribute does not exist on the module.
    """
    if "." not in dotted_path:
        raise ValueError(
            f"Invalid dotted path: '{dotted_path}'. "
            "Must contain at least one '.' (e.g., 'package.module.ClassName')."
        )
    module_path, attr_name = dotted_path.rsplit(".", 1)
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as exc:
        raise ImportError(
            f"Cannot import '{dotted_path}': module '{module_path}' not found."
        ) from exc
    try:
        return getattr(module, attr_name)
    except AttributeError as exc:
        raise ImportError(
            f"Cannot import '{dotted_path}': module '{module_path}' "
            f"has no attribute '{attr_name}'."
        ) from exc
