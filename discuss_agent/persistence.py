"""Archiver — discussion session persistence.

Saves configuration, per-round data, initial context, and the final summary
to a timestamped directory on the local filesystem.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import yaml


class Archiver:
    """Persist discussion artefacts to disk.

    Directory layout created by a single session::

        {base_dir}/{YYYY-MM-DD_HHMM}/
            config.yaml
            context.md
            rounds/
                round_1_express.json
                round_1_challenge.json
                round_1_host.json
                ...
            summary.md
    """

    def __init__(self, base_dir: str = "discussions") -> None:
        self._base_dir = base_dir
        self._session_dir: str | None = None

    # ------------------------------------------------------------------
    # Session setup
    # ------------------------------------------------------------------

    def start_session(self, config) -> str:
        """Create session directory, ``rounds/`` subdir, and write *config.yaml*.

        Parameters
        ----------
        config:
            Any dataclass-compatible object.  Serialised via
            ``dataclasses.asdict(config)`` then ``yaml.dump()``.

        Returns
        -------
        str
            Absolute path of the newly created session directory.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
        session_path = Path(self._base_dir) / timestamp
        rounds_path = session_path / "rounds"

        os.makedirs(rounds_path, exist_ok=True)

        # Persist config
        config_dict = asdict(config)
        config_file = session_path / "config.yaml"
        config_file.write_text(yaml.dump(config_dict, allow_unicode=True))

        self._session_dir = str(session_path.resolve())
        return self._session_dir

    # ------------------------------------------------------------------
    # Per-round persistence
    # ------------------------------------------------------------------

    def save_round(self, round_num: int, phase: str, data: dict) -> None:
        """Save *data* as JSON to ``rounds/round_{num}_{phase}.json``."""
        rounds_dir = Path(self._session_dir) / "rounds"
        filename = f"round_{round_num}_{phase}.json"
        (rounds_dir / filename).write_text(
            json.dumps(data, ensure_ascii=False, indent=2)
        )

    # ------------------------------------------------------------------
    # Context & summary persistence
    # ------------------------------------------------------------------

    def save_context(self, context: str) -> None:
        """Save initial context as ``context.md``."""
        path = Path(self._session_dir) / "context.md"
        path.write_text(context)

    def save_summary(self, summary: str) -> None:
        """Save final summary as ``summary.md``."""
        path = Path(self._session_dir) / "summary.md"
        path.write_text(summary)

    def save_error_log(self, error: str) -> None:
        """Save error details when discussion terminates abnormally."""
        path = Path(self._session_dir) / "error.log"
        path.write_text(f"Discussion terminated by error:\n{error}\n")
