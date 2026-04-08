"""Configuration loading for the multi-agent discussion framework."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import yaml


@dataclass
class AgentConfig:
    name: str
    system_prompt: str


@dataclass
class HostConfig:
    convergence_prompt: str
    summary_prompt: str


@dataclass
class DiscussionConfig:
    min_rounds: int
    max_rounds: int
    model: str
    agents: list[AgentConfig]
    host: HostConfig
    tools: list[str]
    context: dict


_REQUIRED_TOP_KEYS = ("agents", "host", "tools")


class ConfigLoader:
    """Load and validate a discussion YAML configuration file."""

    @staticmethod
    def load(path: str) -> DiscussionConfig:
        """Read *path*, validate required keys, and return a DiscussionConfig."""
        with open(path, "r", encoding="utf-8") as fh:
            raw: dict[str, Any] = yaml.safe_load(fh)

        # --- validate required top-level keys ---
        for key in _REQUIRED_TOP_KEYS:
            if key not in raw:
                raise ValueError(
                    f"Missing required configuration key: '{key}'"
                )

        # --- discussion block (with defaults) ---
        disc = raw.get("discussion", {}) or {}
        if "model" not in disc:
            raise ValueError(
                "Missing required configuration key: 'discussion.model'"
            )

        min_rounds: int = disc.get("min_rounds", 2)
        max_rounds: int = disc.get("max_rounds", 5)
        model: str = disc["model"]

        if min_rounds > max_rounds:
            raise ValueError(
                f"min_rounds ({min_rounds}) must not exceed max_rounds ({max_rounds})"
            )

        # --- agents ---
        agents = [
            AgentConfig(name=a["name"], system_prompt=a["system_prompt"])
            for a in raw["agents"]
        ]
        if not agents:
            raise ValueError("'agents' list must not be empty")

        # --- host ---
        host_raw = raw["host"]
        host = HostConfig(
            convergence_prompt=host_raw["convergence_prompt"],
            summary_prompt=host_raw["summary_prompt"],
        )

        # --- context (optional, opaque dict passed to context builder) ---
        context: dict = raw.get("context", {}) or {}

        # --- tools ---
        tools: list[str] = raw["tools"]

        return DiscussionConfig(
            min_rounds=min_rounds,
            max_rounds=max_rounds,
            model=model,
            agents=agents,
            host=host,
            tools=tools,
            context=context,
        )
