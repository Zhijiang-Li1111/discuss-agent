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


def _mask_api_keys(d: dict) -> None:
    """Recursively mask any ``api_key`` values in a nested dict."""
    for key, value in d.items():
        if key == "api_key" and value is not None:
            d[key] = "***"
        elif isinstance(value, dict):
            _mask_api_keys(value)


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

        # Persist config (with secrets masked)
        config_dict = asdict(config)
        _mask_api_keys(config_dict)
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

    def resume_session(self, archive_path: str) -> str:
        """Point at an existing archive directory for appending new rounds.

        Parameters
        ----------
        archive_path:
            Path to an existing session directory (must contain ``rounds/``).

        Returns
        -------
        str
            Absolute path of the session directory.

        Raises
        ------
        FileNotFoundError
            If *archive_path* or its ``rounds/`` sub-directory does not exist.
        """
        session_path = Path(archive_path).resolve()
        if not session_path.is_dir():
            raise FileNotFoundError(f"Archive directory not found: {archive_path}")
        rounds_path = session_path / "rounds"
        if not rounds_path.is_dir():
            raise FileNotFoundError(f"rounds/ subdirectory not found in: {archive_path}")
        self._session_dir = str(session_path)
        return self._session_dir

    def load_history(self, archive_path: str):
        """Load round history from an archive's ``rounds/`` directory.

        Returns
        -------
        list[RoundRecord]
            Ordered list of reconstructed round records.
        """
        from discuss_agent.models import AgentUtterance, RoundRecord

        rounds_dir = Path(archive_path) / "rounds"
        # Discover which round numbers exist
        round_nums: set[int] = set()
        for f in rounds_dir.iterdir():
            if f.name.startswith("round_") and f.name.endswith("_express.json"):
                num = int(f.name.split("_")[1])
                round_nums.add(num)

        history: list[RoundRecord] = []
        for rn in sorted(round_nums):
            express_file = rounds_dir / f"round_{rn}_express.json"
            challenge_file = rounds_dir / f"round_{rn}_challenge.json"
            host_file = rounds_dir / f"round_{rn}_host.json"

            expressions: list[AgentUtterance] = []
            if express_file.exists():
                data = json.loads(express_file.read_text())
                expressions = [
                    AgentUtterance(agent_name=u["agent_name"], content=u["content"])
                    for u in data.get("utterances", [])
                ]

            challenges: list[AgentUtterance] = []
            if challenge_file.exists():
                data = json.loads(challenge_file.read_text())
                challenges = [
                    AgentUtterance(agent_name=u["agent_name"], content=u["content"])
                    for u in data.get("utterances", [])
                ]

            host_judgment = None
            if host_file.exists():
                host_judgment = json.loads(host_file.read_text())

            record = RoundRecord(
                round_num=rn,
                expressions=expressions,
                challenges=challenges,
                host_judgment=host_judgment,
            )
            history.append(record)

        return history

    def load_context(self, archive_path: str) -> str:
        """Read ``context.md`` from an archive directory."""
        ctx_path = Path(archive_path) / "context.md"
        if not ctx_path.exists():
            raise FileNotFoundError(f"context.md not found in: {archive_path}")
        return ctx_path.read_text()

    def save_error_log(self, error: str) -> None:
        """Save error details when discussion terminates abnormally."""
        path = Path(self._session_dir) / "error.log"
        path.write_text(f"Discussion terminated by error:\n{error}\n")
