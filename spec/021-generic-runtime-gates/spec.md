# Spec 021 — Generic Discuss Agent Runtime Gates

## Status

Implemented on `feat/runtime-gates-tool-audit`.

## Scope

Add generic runtime guarantees to the discussion framework without introducing domain-specific participants, topics, data sources, or business rules.

## Requirements

### R1 — `min_rounds` is an execution gate

- A host convergence judgment MUST NOT run while `round_num < min_rounds`.
- A run MUST NOT return `converged=True` while `rounds_completed < min_rounds`, even if no open claims remain early.
- Beginning at `min_rounds`, the existing host judgment and all-claims-closed convergence behavior applies through `max_rounds`.
- Reaching `max_rounds` with open claims returns `converged=False`, lists remaining disputes, and MUST NOT create or return a completion summary.
- Regression case: a host response that could close the discussion in round 2 MUST be ignored/not requested when `min_rounds=3`; round 3 may close normally.

### R2 — Strict generic tool loading

- Add `discussion.strict_tool_loading`, defaulting to `false` for compatibility.
- When false, global and per-agent extra toolkit import/initialization failures retain warning-and-skip behavior.
- When true, before agent execution, every configured global and per-agent extra toolkit MUST:
  1. import from its configured dotted path;
  2. instantiate successfully;
  3. expose at least one sync or async function whose entrypoint is callable.
- Any strict validation failure aborts the run before agent calls (fail closed), persists the normal error artifact, and provides an error naming the scope/path/reason.
- Loading remains path/package agnostic; no specific toolkit package or filesystem path is embedded.

### R3 — Runtime audit integration

- The existing `AuditLogger` MUST be injected by `DiscussionEngine` into every `AgentConversation`.
- Every conversation call records `call_start` and `call_end`; failures additionally record `error`.
- Every attempted tool call records agent, discussion round, tool name, compact argument summary, result size, error if any, UTC timestamp, and duration.
- Sync and async tool entrypoints share the same audited execution path.
- Anthropic and OpenAI conversation loops share the same audit contract, including invalid/unknown/failing tool calls.
- Host calls are also delimited in the audit archive.
- Audit JSONL remains under `<archive_path>/audit/`; the result exposes that path for upstream readers.

### R4 — Compatibility and boundaries

- Existing public constructor/call signatures remain usable; new parameters and result fields are optional/defaulted.
- Existing YAML without `strict_tool_loading` behaves as before.
- No domain-specific semantics are added.
- Changes are limited to controlled source, tests, and this spec/plan area in the feature worktree.

## Acceptance tests

1. Round 2 is closable but `min_rounds=3`: host is not called in round 2, the run continues, and may converge in round 3.
2. At max rounds with continuing claims: non-converged and no summary artifact call.
3. Config default/explicit parsing for `strict_tool_loading`.
4. Strict global/extra import, constructor, empty toolkit, and non-callable entrypoint failures fail closed with actionable errors; legacy mode skips failures.
5. Sync and async tool calls emit complete audit records.
6. Anthropic and OpenAI tool loops emit equivalent audit events, including failures.
7. Engine wires logger/round context and exposes archived audit location.
8. Full `pytest` passes.
