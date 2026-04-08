# discuss-agent

Generic multi-agent adversarial discussion framework.

- **N agents** express opinions in parallel
- **Cross-challenge**: each agent critiques others' views
- **Host convergence judgment** with structured JSON output
- **YAML-driven** configuration (agents, host, tools, context)
- **Automatic session archiving** (config, rounds, summary)
- **Plugin system** via Python entry_points for custom tools and context

## Install

```bash
pip install git+https://github.com/Zhijiang-Li1111/discuss-agent.git
```

## Quick Start

Create a YAML config file:

```yaml
# config.yaml
discussion:
  model: "claude-sonnet-4-20250514"
  api_key: env:ANTHROPIC_API_KEY       # reads from environment variable
  # base_url: http://localhost:8080/v1 # optional — custom endpoint / proxy
  # temperature: 0.7                   # optional — SDK default if omitted
  # max_tokens: 8192                   # optional — SDK default if omitted

agents:
  - name: "Optimist"
    system_prompt: |
      You are an optimist. You see opportunity in every situation.
      Support your views with reasoning. Challenge pessimistic views constructively.
  - name: "Skeptic"
    system_prompt: |
      You are a skeptic. You question assumptions and look for risks.
      Support your views with reasoning. Challenge optimistic views constructively.

host:
  # model: "claude-haiku-4-5-20251001" # optional — use a cheaper model for the host
  # temperature: 0.3                    # optional — override per-field
  convergence_prompt: |
    You are the discussion moderator. Judge whether the discussion has converged.
    Return JSON: {"converged": bool, "reason": "...", "remaining_disputes": [...]}
  summary_prompt: |
    Summarize the discussion. Include key agreements, remaining disagreements,
    and a balanced conclusion.

tools: []
```

Run:

```bash
export ANTHROPIC_API_KEY="your-key"
python -m discuss_agent config.yaml
```

Output is archived to `discussions/{timestamp}/` with config, rounds, and summary.

## Configuration

| Block | Field | Required | Default | Description |
|-------|-------|----------|---------|-------------|
| `discussion` | `model` | Yes | — | Claude model ID |
| `discussion` | `api_key` | No | — | API key; supports `env:VAR_NAME` to read from env |
| `discussion` | `base_url` | No | — | Custom API endpoint / proxy URL |
| `discussion` | `temperature` | No | SDK default | Sampling temperature |
| `discussion` | `max_tokens` | No | SDK default | Maximum output tokens |
| `discussion` | `min_rounds` | No | 2 | Minimum rounds before convergence allowed |
| `discussion` | `max_rounds` | No | 5 | Maximum rounds before forced exit |
| `agents` | `name` | Yes | — | Agent display name |
| `agents` | `system_prompt` | Yes | — | Agent system prompt |
| `host` | `convergence_prompt` | Yes | — | Prompt for convergence judgment |
| `host` | `summary_prompt` | Yes | — | Prompt for final summary generation |
| `host` | `model` | No | inherits | Override model for host agent |
| `host` | `api_key` | No | inherits | Override API key for host; supports `env:` prefix |
| `host` | `base_url` | No | inherits | Override base URL for host |
| `host` | `temperature` | No | inherits | Override temperature for host |
| `host` | `max_tokens` | No | inherits | Override max_tokens for host |
| `tools` | — | Yes | — | List of tool names (resolved via plugin registry) |
| `context` | — | No | `{}` | Arbitrary dict passed to the context builder |

## Plugin Development

### 1. Create tools

Subclass `agno.tools.Toolkit`. The constructor must accept `context: dict | None = None`:

```python
from agno.tools import Toolkit

class MyTool(Toolkit):
    def __init__(self, context: dict | None = None):
        super().__init__(name="my_tool")
        ctx = context or {}
        self._api_url = ctx.get("api_url", "https://default.example.com")

    def fetch_data(self) -> str:
        """Fetch data from the API. Called by agents during discussion."""
        ...
```

### 2. Create a context builder

An async function that receives the YAML `context` dict and returns a string:

```python
async def build_context(context: dict) -> str:
    tool = MyTool(context)
    data = tool.fetch_data()
    return f"## Background Data\n{data}"
```

### 3. Write the register function

```python
from discuss_agent.registry import PluginRegistry

def register(registry: PluginRegistry) -> None:
    registry.register_tool("my_tool", MyTool)
    registry.register_context_builder(build_context)
```

### 4. Declare the entry point

In your package's `pyproject.toml`:

```toml
[project.entry-points."discuss_agent.plugins"]
my_plugin = "my_package:register"
```

### 5. Install

```bash
pip install -e .
python -m discuss_agent config.yaml
```

The framework discovers your plugin automatically via `entry_points`.

## API Reference

### PluginRegistry

```python
class PluginRegistry:
    def register_tool(self, name: str, tool_class: type) -> None: ...
    def register_context_builder(self, builder: Callable[[dict], Awaitable[str]]) -> None: ...
    def get_tool_class(self, name: str) -> type: ...          # raises ValueError if not found
    def get_context_builder(self) -> Callable | None: ...
```

### DiscussionEngine

```python
from discuss_agent import ConfigLoader, DiscussionEngine

config = ConfigLoader.load("config.yaml")
engine = DiscussionEngine(config)
result = await engine.run()
```

### DiscussionResult

| Field | Type | Description |
|-------|------|-------------|
| `converged` | `bool` | Whether the discussion reached convergence |
| `rounds_completed` | `int` | Number of rounds completed |
| `archive_path` | `str` | Path to the archived session directory |
| `summary` | `str \| None` | Final summary (only if converged) |
| `remaining_disputes` | `list[str]` | Unresolved disagreements |
| `terminated_by_error` | `bool` | Whether the discussion was terminated by an error |

### ConfigLoader

```python
from discuss_agent import ConfigLoader

config = ConfigLoader.load("path/to/config.yaml")
# Returns DiscussionConfig dataclass
```
