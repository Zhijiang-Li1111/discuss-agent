# Plan — Spec 021 Generic Runtime Gates

1. **Specify and characterize**
   - Record compatibility and fail-closed boundaries.
   - Add failing tests for the `min_rounds=3` round-2-close regression and max-round behavior.
2. **Configuration and strict loader (TDD)**
   - Add defaulted `strict_tool_loading` parsing/model field.
   - Centralize generic toolkit extraction/validation so global and extra tools use one implementation.
   - Test strict import/init/empty/non-callable failures and permissive compatibility.
3. **Audit integration (TDD)**
   - Extend existing audit event methods with optional round/call lifecycle metadata.
   - Inject logger plus round context into conversations.
   - Audit one common tool executor so sync/async and provider loops cannot drift.
   - Delimit host calls and expose the audit archive path on the result.
4. **Implement the round gate**
   - Skip convergence precondition/host judgment/all-closed completion before `min_rounds`.
   - Preserve natural convergence from min through max and non-converged/no-summary max behavior.
5. **Verification and serial self-review**
   - Run focused tests after each implementation slice, then full `pytest`.
   - Review in order: spec compliance → code quality → reuse → smoke.
   - Inspect diff/status, confirm unrelated pre-existing deletions are untouched, and commit only controlled source/tests/spec files.

## Plan review

Approved against the task's explicit instruction to begin and execute to completion. The plan preserves API compatibility, places strict validation before agent execution, reuses the existing audit archive, and avoids domain semantics.
