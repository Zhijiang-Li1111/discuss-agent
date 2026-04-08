# Spec 004 — Model & API Configuration

**Status:** Draft
**Depends on:** spec-002 (framework separation)

## Problem

Currently `DiscussionConfig` only carries a `model: str` field. All agents (discussion agents, host, compressor) are created with `Claude(id=config.model)` and no other LLM parameters. This means:

- No way to set `api_key` without relying on the `ANTHROPIC_API_KEY` environment variable that the Anthropic SDK reads by default.
- No way to use a custom endpoint / proxy (`base_url`).
- No way to tune `temperature` or `max_tokens` per discussion.
- The host agent is forced to use the same model as discussion agents, even though a cheaper/faster model would often suffice.

## Solution

Introduce a `ModelConfig` dataclass that captures LLM connection and generation parameters. Wire it through config loading, agent instantiation, and context compression.

### YAML schema change

```yaml
discussion:
  model: claude-opus-4-20250514          # required — model id
  api_key: env:ANTHROPIC_API_KEY         # optional — literal key or env: prefix
  base_url: http://localhost:23333/v1    # optional — custom endpoint
  temperature: 0.7                       # optional (default: None → SDK default)
  max_tokens: 8192                       # optional (default: None → SDK default)
  min_rounds: 2
  max_rounds: 5

host:
  model: claude-sonnet-4-20250514        # optional — override for host agent
  temperature: 0.3                       # optional — override per-field
  convergence_prompt: "..."
  summary_prompt: "..."
```

### `env:` prefix resolution

When an `api_key` value starts with `env:`, the remainder is treated as an environment variable name. `ConfigLoader` reads `os.environ[VAR_NAME]` and raises `ValueError` if the variable is not set. A literal string (without the prefix) is used as-is — though users should strongly prefer the `env:` form so that secrets never appear in committed YAML files.

This resolution applies to **all** `api_key` fields — both `discussion.api_key` and `host.api_key`. Both are resolved at config load time.

### `ModelConfig` dataclass

```
ModelConfig
  model: str              # required
  api_key: str | None     # resolved (env: already expanded), default None
  base_url: str | None    # default None
  temperature: float | None  # default None (SDK decides)
  max_tokens: int | None     # default None (SDK decides; Agno defaults to 8192)
```

### Host model override

`HostConfig` gains optional model fields: `model`, `api_key`, `base_url`, `temperature`, `max_tokens`. These override the discussion-level defaults per-field (not all-or-nothing). If a host field is `None`, the discussion-level value is used.

A helper `HostConfig.resolve_model(discussion_model: ModelConfig) -> ModelConfig` returns the effective model config for the host by merging host overrides onto the discussion defaults.

### Agent instantiation

A single factory function `build_claude(model_config: ModelConfig) -> Claude` creates the Agno `Claude` instance:

- `id=model_config.model`
- `api_key=model_config.api_key` (if set)
- `temperature=model_config.temperature` (if set)
- `max_tokens=model_config.max_tokens` (if set)
- `client_params={"base_url": model_config.base_url}` (if `base_url` set)

All four agent creation sites use this factory:
1. Discussion agents in `engine.py`
2. Host agent in `engine.py`
3. Summary agent in `engine.py`
4. Compressor agent in `context.py`

### What does NOT change

- Top-level YAML keys (`agents`, `host`, `tools`, `context`) stay the same.
- `AgentConfig` stays the same (no per-discussion-agent model override — YAGNI).
- Plugin / tool loading is unaffected.
- `Archiver`, `models.py`, `registry.py` are unchanged.

### Archiver secret redaction

`Archiver` currently serializes `DiscussionConfig` via `dataclasses.asdict()` into `config.yaml`. After this change, the resolved `api_key` (a real secret) would be written to disk. To prevent this, `ModelConfig` provides a `to_safe_dict()` method that replaces the `api_key` value with `"***"` if set. The `Archiver` (or `DiscussionConfig.to_safe_dict()`) uses this when persisting config to disk.

## Task Breakdown

### T1: `ModelConfig` dataclass + `env:` resolver

Add `ModelConfig` to `config.py`. Add `resolve_env(value: str) -> str` helper that handles the `env:` prefix.

### T2: Update `ConfigLoader` to parse model fields

Parse `discussion.api_key`, `discussion.base_url`, `discussion.temperature`, `discussion.max_tokens` into a `ModelConfig` on `DiscussionConfig`. Replace the bare `model: str` field with `model_config: ModelConfig`.

### T3: Host model override

Add optional model override fields to `HostConfig`. Implement `resolve_model()` merge logic.

### T4: `build_claude` factory

Create `build_claude(model_config: ModelConfig) -> Claude` in `config.py` (or a small helper module). Update `engine.py` and `context.py` to use it.

### T5: Update `context.py` compressor

`ContextManager` already receives the full `DiscussionConfig`. Update `_compress_round` to use `build_claude(self._config.model_config)` instead of `Claude(id=self._config.model)`.

### T6: Archiver secret redaction

Add `ModelConfig.to_safe_dict()` that masks `api_key`. Update `Archiver` to use safe serialization when writing `config.yaml`.

### T7: Update README

Add configuration examples showing `api_key: env:...`, `base_url`, `temperature`, `max_tokens`, and host model override. Update the configuration reference table.

### T8: Update tests

- `conftest.py` fixtures updated for new schema.
- Unit tests for `ModelConfig`, `resolve_env`, `env:` prefix validation.
- Unit tests for host override merge logic.
- Update existing `test_config.py`, `test_engine.py`, `test_context.py` for the changed `DiscussionConfig` shape.
- Ensure no test file contains real API keys.

## Acceptance Criteria

### ModelConfig & env: resolution (4)

- **AC-01:** `ModelConfig` dataclass exists with fields: `model` (str, required), `api_key` (str | None), `base_url` (str | None), `temperature` (float | None), `max_tokens` (int | None). All optional fields default to `None`.
- **AC-02:** `resolve_env("env:FOO")` returns `os.environ["FOO"]`; raises `ValueError` when `FOO` is not set.
- **AC-03:** `resolve_env("sk-literal-key")` returns the string as-is (no prefix stripping).
- **AC-04:** `resolve_env(None)` returns `None`.

### ConfigLoader (5)

- **AC-05:** `ConfigLoader.load()` returns a `DiscussionConfig` whose `model_config` is a `ModelConfig` populated from the `discussion` block.
- **AC-06:** `discussion.model` remains required; missing it raises `ValueError`.
- **AC-07:** `discussion.api_key`, `discussion.base_url`, `discussion.temperature`, `discussion.max_tokens` are all optional and default to `None`.
- **AC-08:** An `api_key` with `env:` prefix is resolved at load time for both `discussion.api_key` and `host.api_key`; missing env var raises `ValueError`.
- **AC-09:** Existing configs that only specify `discussion.model` continue to work without changes (backward compatible).

### Host model override (3)

- **AC-10:** `HostConfig` accepts optional fields: `model`, `api_key`, `base_url`, `temperature`, `max_tokens`.
- **AC-11:** `HostConfig.resolve_model(discussion_model)` returns a `ModelConfig` that uses host overrides where set and falls back to discussion defaults where not set.
- **AC-12:** If no host model fields are specified, `resolve_model()` returns a copy of the discussion `ModelConfig` unchanged.

### Agent instantiation (4)

- **AC-13:** A `build_claude(model_config) -> Claude` function exists and creates a Claude instance with all non-None fields applied.
- **AC-14:** Discussion agents in `engine.py` use `build_claude(config.model_config)`.
- **AC-15:** Host and summary agents in `engine.py` use `build_claude(host_resolved_model)`.
- **AC-16:** Compressor agent in `context.py` uses `build_claude(config.model_config)`.

### README (1)

- **AC-17:** README contains a configuration example showing `api_key: env:ANTHROPIC_API_KEY`, `base_url`, `temperature`, `max_tokens`, and host model override.

### Tests (4)

- **AC-18:** Unit tests cover `resolve_env` with env: prefix, literal value, None, and missing env var.
- **AC-19:** Unit tests cover `ModelConfig` construction with all-defaults and all-explicit values.
- **AC-20:** Unit tests cover host model override merge (partial override, full override, no override).
- **AC-21:** All existing tests pass after the config shape change (fixtures updated).

### Security (2)

- **AC-22:** No committed file contains a real API key or endpoint URL. Example configs use `env:` prefix or placeholder values.
- **AC-23:** `ModelConfig.to_safe_dict()` masks the `api_key` field (replaces with `"***"`) so archived configs never contain real secrets.
