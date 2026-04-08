"""discuss-agent — generic multi-agent adversarial discussion framework."""

from discuss_agent.config import (
    AgentConfig,
    ConfigLoader,
    DiscussionConfig,
    HostConfig,
    ModelConfig,
    ToolConfig,
    build_claude,
    resolve_env,
)
from discuss_agent.engine import DiscussionEngine
from discuss_agent.models import AgentUtterance, DiscussionResult, RoundRecord
from discuss_agent.registry import import_from_path

__all__ = [
    "AgentConfig",
    "AgentUtterance",
    "ConfigLoader",
    "DiscussionConfig",
    "DiscussionEngine",
    "DiscussionResult",
    "HostConfig",
    "ModelConfig",
    "RoundRecord",
    "ToolConfig",
    "build_claude",
    "import_from_path",
    "resolve_env",
]
