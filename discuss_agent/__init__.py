"""discuss-agent — generic multi-agent adversarial discussion framework."""

from discuss_agent.config import AgentConfig, ConfigLoader, DiscussionConfig, HostConfig
from discuss_agent.engine import DiscussionEngine
from discuss_agent.models import AgentUtterance, DiscussionResult, RoundRecord
from discuss_agent.registry import PluginRegistry, load_plugins

__all__ = [
    "AgentConfig",
    "AgentUtterance",
    "ConfigLoader",
    "DiscussionConfig",
    "DiscussionEngine",
    "DiscussionResult",
    "HostConfig",
    "PluginRegistry",
    "RoundRecord",
    "load_plugins",
]
