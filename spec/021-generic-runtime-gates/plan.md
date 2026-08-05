# Plan — Spec 021 Generic Runtime Gates

1. **Specify and characterize**
   - Record compatibility, natural-convergence, and fail-closed boundaries.
   - Preserve the existing per-round precondition/Host flow and test max-round behavior.
2. **Configuration and strict loader (TDD)**
   - Add defaulted `strict_tool_loading` parsing/model field.
   - Centralize generic toolkit extraction/validation so global and extra tools use one implementation.
   - Test strict import/init/empty/non-callable failures and permissive compatibility.
3. **Audit integration (TDD)**
   - Extend existing audit event methods with optional round/call lifecycle metadata.
   - Inject logger plus round context into conversations.
   - Audit one common tool executor so sync/async and provider loops cannot drift.
   - Delimit host calls and expose the audit archive path on the result.
4. **Preserve natural convergence and enforce the ceiling**
   - Run convergence precondition/Host judgment every response round.
   - Preserve immediate natural convergence and non-converged/no-summary max behavior.
5. **Verification and serial self-review**
   - Run focused tests after each implementation slice, then full `pytest`.
   - Review in order: spec compliance → code quality → reuse → smoke.
   - Inspect diff/status, confirm unrelated pre-existing deletions are untouched, and commit only controlled source/tests/spec files.

## Plan review

Approved against the corrected task scope. The plan preserves API compatibility and natural convergence, places strict validation before agent execution, reuses the existing audit archive, and avoids domain semantics.

## Final review

- **Spec compliance:** verified that `DiscussionEngine` contains no `min_rounds` runtime check; a regression test uses `min_rounds=99` and converges in round 2.
- **Code quality:** natural convergence continues through the existing precondition and Host paths without a parallel decision branch.
- **Reuse:** the existing claim merge, precondition, Host judgment, and close logic are reused unchanged.
- **Smoke:** full test suite and diff checks pass; unrelated pre-existing worktree deletions remain outside the feature commits.
