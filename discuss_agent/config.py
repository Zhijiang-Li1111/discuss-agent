"""Configuration loading for the multi-agent discussion framework."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, asdict
from typing import Any

import yaml
from agno.models.anthropic import Claude
from agno.models.openai import OpenAIChat


# ---------------------------------------------------------------------------
# Template variable resolution
# ---------------------------------------------------------------------------

_TEMPLATE_RE = re.compile(r"\{\{(\w+)\}\}")

# Keys consumed by the framework — never treated as template variables.
_RESERVED_KEYS = frozenset({
    "discussion", "agents", "host", "tools", "context",
    "context_builder", "limitation", "skills",
})


def _resolve_template_vars(raw: dict[str, Any]) -> None:
    """Replace ``{{key}}`` placeholders in-place using top-level string values.

    Only top-level keys whose values are plain strings (and that are not
    reserved framework keys) are available as template variables.
    """
    # 1. Collect template variables from top-level string fields.
    template_vars: dict[str, str] = {}
    for key, value in raw.items():
        if key not in _RESERVED_KEYS and isinstance(value, str):
            template_vars[key] = value

    if not template_vars:
        return

    def _replace(s: str) -> str:
        def _sub(m: re.Match) -> str:
            name = m.group(1)
            return template_vars.get(name, m.group(0))
        return _TEMPLATE_RE.sub(_sub, s)

    def _walk(obj: Any) -> Any:
        if isinstance(obj, str):
            return _replace(obj)
        if isinstance(obj, dict):
            return {k: _walk(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_walk(item) for item in obj]
        return obj

    # 2. Walk and replace in all framework-consumed keys (not the var defs).
    for key in list(raw.keys()):
        if key in _RESERVED_KEYS:
            raw[key] = _walk(raw[key])


# ---------------------------------------------------------------------------
# env: resolver
# ---------------------------------------------------------------------------


def resolve_env(value: str | None) -> str | None:
    """Resolve a value that may use the ``env:VAR_NAME`` prefix.

    - ``None`` → ``None``
    - ``"env:FOO"`` → ``os.environ["FOO"]`` (raises ``ValueError`` if unset)
    - Any other string → returned as-is
    """
    if value is None:
        return None
    if value.startswith("env:"):
        var_name = value[4:]
        env_val = os.environ.get(var_name)
        if env_val is None:
            raise ValueError(
                f"Environment variable '{var_name}' is not set "
                f"(referenced by api_key: {value})"
            )
        return env_val
    return value


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    """LLM connection and generation parameters."""

    model: str
    api_key: str | None = None
    base_url: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None

    def to_safe_dict(self) -> dict:
        """Return a dict with ``api_key`` masked for safe persistence."""
        d = asdict(self)
        if d.get("api_key") is not None:
            d["api_key"] = "***"
        return d


def infer_provider(model: str) -> str:
    """Infer the wire protocol provider from a configured model name.

    Maestro exposes explicit provider-prefixed model IDs.  All existing model
    names remain Anthropic by default so legacy Claude configurations preserve
    their current behaviour.
    """
    if model.lower().startswith("agent-maestro-openai/"):
        return "openai"
    return "anthropic"


def normalize_base_url(base_url: str | None, provider: str) -> str | None:
    """Normalize legacy Maestro URLs for the selected wire protocol."""
    if base_url is None:
        return None
    normalized = base_url.rstrip("/")
    if provider == "openai" and normalized.endswith("/api/anthropic"):
        return normalized[: -len("/api/anthropic")] + "/api/openai/v1"
    return normalized if base_url.endswith("/") else base_url


def build_model(model_config: ModelConfig) -> Claude | OpenAIChat:
    """Create an Agno model using the protocol implied by ``model``."""
    provider = infer_provider(model_config.model)
    base_url = normalize_base_url(model_config.base_url, provider)
    common: dict[str, Any] = {"id": model_config.model}
    if model_config.api_key is not None:
        common["api_key"] = model_config.api_key
    if model_config.temperature is not None:
        common["temperature"] = model_config.temperature
    if model_config.max_tokens is not None:
        common["max_tokens"] = model_config.max_tokens

    if provider == "openai":
        if base_url is not None:
            common["base_url"] = base_url
        common["timeout"] = 600.0
        return OpenAIChat(**common)

    client_params: dict[str, Any] = {"timeout": 600.0}
    if base_url is not None:
        client_params["base_url"] = base_url
    common["client_params"] = client_params
    return Claude(**common)


def build_claude(model_config: ModelConfig) -> Claude | OpenAIChat:
    """Compatibility alias for the now provider-aware Agno model builder."""
    return build_model(model_config)


@dataclass
class ToolConfig:
    """Reference to a tool class via Python dotted path."""

    path: str


@dataclass
class SkillConfig:
    """Reference to a skill directory path."""
    path: str


@dataclass
class AgentConfig:
    name: str
    system_prompt: str
    extra_tools: list[ToolConfig] = field(default_factory=list)
    disable_tools: list[str] = field(default_factory=list)
    skills: list[SkillConfig] = field(default_factory=list)


@dataclass
class HostConfig:
    convergence_prompt: str
    summary_prompt: str
    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    skip_summary: bool = False

    def resolve_model(self, discussion_model: ModelConfig) -> ModelConfig:
        """Merge host overrides onto *discussion_model*, returning a new ModelConfig."""
        return ModelConfig(
            model=self.model if self.model is not None else discussion_model.model,
            api_key=self.api_key if self.api_key is not None else discussion_model.api_key,
            base_url=self.base_url if self.base_url is not None else discussion_model.base_url,
            temperature=self.temperature if self.temperature is not None else discussion_model.temperature,
            max_tokens=self.max_tokens if self.max_tokens is not None else discussion_model.max_tokens,
        )


@dataclass
class DiscussionConfig:
    min_rounds: int
    max_rounds: int
    model_config: ModelConfig
    agents: list[AgentConfig]
    host: HostConfig
    tools: list[ToolConfig]
    context: dict
    context_builder: str | None = None
    limitation: str | None = None
    skills: list[SkillConfig] | None = None
    strict_tool_loading: bool = False


_REQUIRED_TOP_KEYS = ("agents", "host", "tools")


def _validated_bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean")
    return value


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

        # --- resolve {{var}} template variables from top-level strings ---
        _resolve_template_vars(raw)

        # --- discussion block (with defaults) ---
        disc = raw.get("discussion", {}) or {}
        if "model" not in disc:
            raise ValueError(
                "Missing required configuration key: 'discussion.model'"
            )

        min_rounds: int = disc.get("min_rounds", 2)
        max_rounds: int = disc.get("max_rounds", 6)
        strict_tool_loading = _validated_bool(
            disc.get("strict_tool_loading", False),
            "discussion.strict_tool_loading",
        )
        if (
            not isinstance(max_rounds, int)
            or isinstance(max_rounds, bool)
            or max_rounds < 1
        ):
            raise ValueError("discussion.max_rounds must be a positive integer")

        model_config = ModelConfig(
            model=disc["model"],
            api_key=resolve_env(disc.get("api_key")),
            base_url=resolve_env(disc.get("base_url")),
            temperature=disc.get("temperature"),
            max_tokens=disc.get("max_tokens"),
        )

        # --- agents ---
        agents = []
        for a in raw["agents"]:
            extra_tools = [
                ToolConfig(path=t["path"]) for t in (a.get("extra_tools") or [])
            ]
            disable_tools = a.get("disable_tools") or []
            agent_skills = [
                SkillConfig(path=s["path"]) for s in (a.get("skills") or [])
            ]
            agents.append(
                AgentConfig(
                    name=a["name"],
                    system_prompt=a["system_prompt"],
                    extra_tools=extra_tools,
                    disable_tools=disable_tools,
                    skills=agent_skills,
                )
            )
        if not agents:
            raise ValueError("'agents' list must not be empty")

        # --- host ---
        host_raw = raw["host"]
        host = HostConfig(
            convergence_prompt=host_raw["convergence_prompt"],
            summary_prompt=host_raw["summary_prompt"],
            model=host_raw.get("model"),
            api_key=resolve_env(host_raw.get("api_key")),
            base_url=resolve_env(host_raw.get("base_url")),
            temperature=host_raw.get("temperature"),
            max_tokens=host_raw.get("max_tokens"),
            skip_summary=_validated_bool(
                host_raw.get("skip_summary", False),
                "host.skip_summary",
            ),
        )

        # --- context (optional, opaque dict passed to context builder) ---
        context: dict = raw.get("context", {}) or {}

        # --- context_builder (optional top-level dotted path) ---
        context_builder: str | None = raw.get("context_builder")

        # --- limitation (optional top-level string) ---
        limitation: str | None = raw.get("limitation")

        # --- tools (list of {path: "..."} dicts) ---
        raw_tools = raw["tools"]
        tools: list[ToolConfig] = []
        for i, entry in enumerate(raw_tools):
            if not isinstance(entry, dict) or "path" not in entry:
                raise ValueError(
                    f"tools[{i}]: each tool must be a dict with a 'path' key, "
                    f"got {entry!r}"
                )
            tools.append(ToolConfig(path=entry["path"]))

        # --- skills (optional, global skills shared by all agents) ---
        raw_skills = raw.get("skills") or []
        skills: list[SkillConfig] = []
        for i, entry in enumerate(raw_skills):
            if not isinstance(entry, dict) or "path" not in entry:
                raise ValueError(
                    f"skills[{i}]: each skill must be a dict with a 'path' key, "
                    f"got {entry!r}"
                )
            skills.append(SkillConfig(path=entry["path"]))

        return DiscussionConfig(
            min_rounds=min_rounds,
            max_rounds=max_rounds,
            model_config=model_config,
            agents=agents,
            host=host,
            tools=tools,
            context=context,
            context_builder=context_builder,
            limitation=limitation,
            skills=skills if skills else None,
            strict_tool_loading=strict_tool_loading,
        )
