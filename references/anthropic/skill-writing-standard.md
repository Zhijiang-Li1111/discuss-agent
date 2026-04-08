# Skill Writing Standard

> Anthropic's official guidance on writing agent skills — content structure, description triggers, body format, anti-patterns.

Sources:
- [Agent Skills Spec](https://agentskills.io/specification)
- [Skill Best Practices](https://agentskills.io/skill-creation/best-practices)
- [Optimizing Descriptions](https://agentskills.io/skill-creation/optimizing-descriptions)
- [Anthropic Skills Repo](https://github.com/anthropics/skills) — example skills + skill-creator

---

## 1. Architecture: Progressive Disclosure (3 Tiers)

- **Metadata (~100 tokens):** name + description — loaded at startup for ALL skills, used to decide whether to activate
- **Instructions (< 5000 tokens):** full skill body — loaded only when skill is activated
- **Resources (on demand):** scripts, references, assets — loaded only when needed during execution

The description carries the entire burden of triggering. The body is never seen until after the trigger decision.

## 2. Description (Trigger Text)

Max 1024 chars. Must convey what the skill does AND when to use it.

- Use imperative phrasing: "Use this skill when..." not "This skill does..."
- Focus on user intent, not implementation
- Include explicit TRIGGER and DO NOT TRIGGER conditions
- Err on the side of being pushy — list contexts where the skill applies, including indirect cases

**Good example (claude-api skill):**
```
"Build apps with the Claude API or Anthropic SDK. TRIGGER when: code imports
anthropic/@anthropic-ai/sdk/claude_agent_sdk, or user asks to use Claude API.
DO NOT TRIGGER when: code imports openai/other AI SDK, general programming."
```

## 3. Body Content: 5 Principles

1. **Add what the agent lacks, omit what it knows.** Don't explain common concepts. Focus on project-specific conventions, domain procedures, non-obvious edge cases.

2. **Teach processes, not results.** "How to approach this class of problems" not "what to output for this specific instance."

3. **Give defaults, not menus.** Recommend one approach; mention alternatives briefly.

4. **Match specificity to fragility.** Be prescriptive for critical/dangerous operations; give freedom where multiple approaches are valid.

5. **Explain WHY, not rigid ALL-CAPS rules.** An agent that understands purpose makes better context-dependent decisions.

## 4. Highest-Value Content Types

1. **Gotchas** — Concrete corrections to mistakes the agent WILL make. Not "handle errors appropriately" but "The users table uses soft deletes. Queries must include WHERE deleted_at IS NULL."

2. **Output templates** — Concrete format examples are more reliable than prose descriptions. Agents pattern-match well.

3. **Multi-step checklists** — Explicit step lists to prevent skipping steps.

4. **Validation loops** — Do → validate → fix → repeat until passing.

5. **Bundled scripts** — When the agent reinvents the same logic each run, write a tested script.

## 5. Format

- Markdown with headers, code blocks, lists — all fine (unlike tool descriptions which must be plaintext)
- No mandatory section naming — "Write whatever helps agents perform the task effectively"
- < 500 lines, < 5000 tokens for the body
- Move detailed reference material to separate files

## 6. Anti-Patterns

- Generic LLM-generated content without real domain expertise
- Explaining common knowledge the agent already has
- Exhaustive documentation covering every edge case (concise + working example beats this)
- Menus of equal options instead of clear defaults
- Over-broad or under-specified descriptions
- Deeply nested reference chains (keep one level deep)

## 7. Iteration

Eval-driven approach:
1. Write draft from real expertise (not LLM-generated)
2. Test against real tasks, read execution traces
3. For descriptions: 20 trigger eval queries (half should/half should-not), run 3x each, train/validation split
4. Iterate up to ~5 times; if not improving, try structurally different approach

**How to apply:** When writing or reviewing a skill, check: Does it teach processes (not just list facts)? Does it have gotchas (concrete mistakes to avoid)? Is the description trigger-focused? Is the body under 5000 tokens?
