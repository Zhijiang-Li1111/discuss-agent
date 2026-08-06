# discuss-agent

Generic multi-agent adversarial discussion framework.

- **N agents** express opinions in parallel, then cross-challenge each other
- **Host convergence judgment** with structured JSON output
- **YAML-driven** configuration — agents, host, tools, context, model settings
- **Full LLM API config** — `api_key` (with `env:` prefix), `base_url`, `temperature`, `max_tokens`; host can override model settings independently
- **Per-agent tools** — global tools inherited by all agents, with per-agent `extra_tools` / `disable_tools`
- **Discussion scope control** — optional `limitation` field restricts agents to a defined topic
- **Runtime tool loading** via Python dotted paths — no `pip install` or entry_points needed
- **Automatic session archiving** with secret redaction (config, rounds, summary)

## Architecture

```
                          ┌─────────────────────────────────────┐
                          │           YAML Config               │
                          │                                     │
                          │  discussion:                        │
                          │    model, api_key, base_url,        │
                          │    temperature, max_tokens           │
                          │  agents:    [{name, system_prompt}] │
                          │  host:      {prompts, model override}│
                          │  tools:     [{path: "pkg.Tool"}]    │
                          │  context:   {arbitrary data}        │
                          │  limitation: "scope restriction"    │
                          └──────────────┬──────────────────────┘
                                         │
                                         ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                          DiscussionEngine                                  │
│                                                                            │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  Round 0: Build shared context (via context_builder)                 │  │
│  └─────────────────────────────────┬────────────────────────────────────┘  │
│                                    ▼                                       │
│  ┌ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ Round Loop ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┐   │
│                                                                         │  │
│  │  ┌────────────────────────────────────────────────────────────┐   │  │
│     │  1. EXPRESS — agents independently decide relevant input    │      │
│  │  │                                                            │   │  │
│     │   ┌───────────┐  ┌───────────┐       ┌───────────┐        │      │
│  │  │   │  Agent A  │  │  Agent B  │  ...  │  Agent N  │        │   │  │
│     │   │ + tools   │  │ + tools   │       │ + tools   │        │      │
│  │  │   └───────────┘  └───────────┘       └───────────┘        │   │  │
│     └────────────────────────────┬───────────────────────────────┘      │
│  │                               ▼                                  │  │
│     ┌────────────────────────────────────────────────────────────┐      │
│  │  │  2. CHALLENGE — relevant agents add evidence/rebuttals     │  │  │
│     │                                                            │      │
│  │  │   Agent A reviews B,C  │  Agent B reviews A,C  │  ...     │  │  │
│     └────────────────────────────┬───────────────────────────────┘      │
│  │                               ▼                                  │  │
│     ┌────────────────────────────────────────────────────────────┐      │
│  │  │  3. HOST JUDGE — convergence check                         │  │  │
│     │                                                            │      │
│  │  │   ┌────────────┐    [{"claim": "...",                      │  │  │
│     │   │    Host     │      "verdict": "CLOSED:*|CONTINUE",     │      │
│  │  │   │   Agent    │      "reason": "...", ...}]               │  │  │
│     │   └────────────┘                                           │      │
│  │  └──────────┬──────────────────────────┬──────────────────────┘  │  │
│                │                          │                            │
│  │     Host closes all           OPEN claims &&                       │  │
│        OPEN claims               round < max_rounds safety cap         │
│  │             │                          │                         │  │
│                ▼                          └──────── loop back ──┐       │
│  └ ─ ─ ─ ─ ─ ─│─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─│─ ┘  │
│                │                                                │      │
│                ▼                                                       │
│  ┌──────────────────────────┐                                          │
│  │  Host Summary (if conv.) │                                          │
│  └──────────────────────────┘                                          │
│                                                                        │
└────────────────────────┬───────────────────────────────────────────────┘
                         │
                         ▼
          ┌──────────────────────────────┐
          │     discussions/{timestamp}/ │
          │                              │
          │  config.yaml   (secrets masked)
          │  context.md                  │
          │  rounds/                     │
          │    round_1_express.json      │
          │    round_1_challenge.json    │
          │    round_1_host.json         │
          │    ...                       │
          │  summary.md    (if converged)│
          └──────────────────────────────┘
```

Bounded prompts prevent large legacy sessions from being mistaken for convergence.
In the measured Muyuan replay, all 71 claims remained OPEN and all 71 remained
Host candidates; the Host batch was 37,053 characters and each agent prompt was
37,007 characters. This is bounded review state, not natural convergence.

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

tools:
  - path: my_package.tools.MyTool

context_builder: my_package.context.build_context

# limitation: "仅讨论技术方案的可行性，不涉及预算"  # optional — restrict discussion scope

agents:
  - name: "Optimist"
    system_prompt: |
      You are an optimist. You see opportunity in every situation.
      Support your views with reasoning. Challenge pessimistic views constructively.
  - name: "Skeptic"
    system_prompt: |
      You are a skeptic. You question assumptions and look for risks.
      Support your views with reasoning. Challenge optimistic views constructively.
    extra_tools:
      - path: my_package.tools.ExtraTool
    disable_tools:
      - my_package.tools.MyTool

host:
  # model: "claude-haiku-4-5-20251001" # optional — use a cheaper model for the host
  # temperature: 0.3                    # optional — override per-field
  convergence_prompt: |
    You are the discussion moderator. Semantically judge each OPEN claim from
    its evidence, rebuttals, and UNKNOWNs. Do not require fixed response counts.
    Return one verdict per claim: CLOSED:共识, CLOSED:分歧, or CONTINUE.
  summary_prompt: |
    Summarize the discussion. Include key agreements, remaining disagreements,
    and a balanced conclusion.

context:
  api_url: "https://example.com"
```

Run:

```bash
export ANTHROPIC_API_KEY="your-key"
python -m discuss_agent config.yaml
```

Output is archived to `discussions/{timestamp}/` with config, rounds, and summary.

### Resume a Discussion

Resume from a previous discussion archive and run additional rounds:

```bash
python3 -m discuss_agent config.yaml --resume discussions/2026-04-11_2159 --rounds 2
```

- Loads conversation history from the archive directory
- Context is loaded from the archive (not regenerated)
- New rounds are appended to the same archive directory
- Round numbering continues from where it left off

## Configuration

| Block | Field | Required | Default | Description |
|-------|-------|----------|---------|-------------|
| `discussion` | `model` | Yes | — | Claude model ID |
| `discussion` | `api_key` | No | — | API key; supports `env:VAR_NAME` to read from env |
| `discussion` | `base_url` | No | — | Custom API endpoint / proxy URL |
| `discussion` | `temperature` | No | SDK default | Sampling temperature |
| `discussion` | `max_tokens` | No | SDK default | Maximum output tokens |
| `discussion` | `min_rounds` | No | 2 | Legacy compatibility field; does not gate convergence |
| `discussion` | `max_rounds` | No | 6 | Safety cap on rounds, not a convergence requirement |
| `tools` | `path` | Yes | — | Python dotted path to a Toolkit subclass |
| `context_builder` | — | No | — | Python dotted path to an async context builder function |
| `limitation` | — | No | — | Restrict discussion scope; injected as a warning prefix in every round |
| `agents` | `name` | Yes | — | Agent display name |
| `agents` | `system_prompt` | Yes | — | Agent system prompt |
| `agents` | `extra_tools` | No | `[]` | Additional tools for this agent (`[{path: "..."}]`) |
| `agents` | `disable_tools` | No | `[]` | Dotted paths of global tools to disable for this agent |
| `host` | `convergence_prompt` | Yes | — | Prompt for convergence judgment |
| `host` | `summary_prompt` | Yes | — | Prompt for final summary generation |
| `host` | `model` | No | inherits | Override model for host agent |
| `host` | `api_key` | No | inherits | Override API key for host; supports `env:` prefix |
| `host` | `base_url` | No | inherits | Override base URL for host |
| `host` | `temperature` | No | inherits | Override temperature for host |
| `host` | `max_tokens` | No | inherits | Override max_tokens for host |
| `context` | — | No | `{}` | Arbitrary dict passed to the context builder |

## Discussion Scope Control (`limitation`)

The optional top-level `limitation` field restricts agents to a specific topic. When set, the engine prepends a warning to every express and challenge prompt:

```
⚠️ 本次讨论范围仅限于：{limitation}
```

This keeps agents focused and prevents off-topic drift. When omitted, no prefix is injected and existing behavior is unchanged.

```yaml
# Example: restrict a product strategy discussion to technical feasibility only
limitation: "仅讨论技术方案的可行性，不涉及预算和人力资源"
```

## Tool Development

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

### 3. Reference in YAML

Point to your tool classes and context builder via Python dotted paths. No `pip install` or entry_points needed — just ensure the module is importable:

```yaml
tools:
  - path: my_package.tools.MyTool

context_builder: my_package.context.build_context
```

## API Reference

### import_from_path

```python
from discuss_agent import import_from_path

cls = import_from_path("my_package.tools.MyTool")  # returns the class
```

### ToolConfig

```python
from discuss_agent import ToolConfig

tc = ToolConfig(path="my_package.tools.MyTool")
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
