# Tool Description & Design Standard

> Anthropic's official guidance on tool description writing, tool design, parameter naming, return values, and error messages.

Sources:
- [Tool Use Implementation Guide](https://docs.anthropic.com/en/docs/build-with-claude/tool-use/implement-tool-use)
- [Writing Tools for Agents](https://www.anthropic.com/engineering/writing-tools-for-agents)
- [Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents)
- [Advanced Tool Use](https://www.anthropic.com/engineering/advanced-tool-use)
- [Anthropic Skills Repo](https://github.com/anthropics/skills) — eval-driven description optimization loop

---

## 1. Description Format

**Plaintext prose.** No markdown, no bullet lists, no `Actions:` headers, no code blocks. The API defines it as "a detailed plaintext description."

**Length:** At least 3-4 sentences per tool, more for complex tools.

**Must include 5 elements:**
1. What the tool does
2. When to use it (and when NOT to)
3. What each parameter means and how it affects behavior
4. What it returns (format and constraints)
5. Caveats/limitations (what it does NOT do)

**Good example:**
```
"Retrieves the current stock price for a given ticker symbol. The ticker symbol must be a valid symbol for a publicly traded company on a major US stock exchange like NYSE or NASDAQ. The tool will return the latest trade price in USD. It should be used when the user asks about the current or most recent price of a specific stock. It will not provide any other information about the stock or company."
```

**Bad example:** `"Gets the stock price for a ticker."`

## 2. Examples: Use `input_examples` Field

Use the separate `input_examples` field (schema-validated JSON), not inline code in the description string. 1-5 examples per tool, show minimal/partial/full patterns, use realistic data.

> "Tool use examples improved accuracy from 72% to 90% on complex parameter handling."

If the framework doesn't support `input_examples`, keep examples in description but as natural language.

## 3. Tool Design: High-Level Operations

Tools should consolidate functionality, not mirror low-level CRUD APIs.

**Bad:** `list_users` + `list_events` + `create_event` (low-level CRUD)
**Good:** `schedule_event` (finds availability + creates in one call)

**Bad:** `get_customer_by_id` + `list_transactions` + `list_notes`
**Good:** `get_customer_context` (compiles recent relevant info)

> "Tools can consolidate functionality, handling potentially multiple discrete operations (or API calls) under the hood."

## 4. Parameter Naming

Unambiguous, semantically clear names. Make implicit context explicit.

**Bad:** `user` (is it an ID? a name? an object?)
**Good:** `user_id`

> "When writing tool descriptions and specs, think of how you would describe your tool to a new hire. Consider the context you might implicitly bring and make it explicit."

## 5. Return Values

Return human-interpretable fields, not internal identifiers.

**Remove:** `uuid`, `256px_image_url`, `mime_type`
**Include:** `name`, `image_url`, `file_type`

> "Resolving arbitrary alphanumeric UUIDs to semantically meaningful language significantly improves precision by reducing hallucinations."

Consider a `response_format` enum (concise vs detailed) to let agents control verbosity.

## 6. Error Messages as Prompt Engineering

Error responses should guide the agent's next action, not just report failure.

**Bad:** Generic error code or traceback
**Good:** Specific message steering toward the right approach (e.g., "No customer found with ID X. Try searching by name instead.")

## 7. Description Optimization

> "Prompt-engineering tool descriptions represents one of the most effective methods for improving tools."
> "Even small refinements can yield dramatic improvements." (Claude Sonnet 3.5 hit SOTA on SWE-bench Verified via tool description tuning.)

Anthropic's `skills/skill-creator` provides an automated eval-driven loop: write eval set (should_trigger true/false queries) → run eval → analyze failures → Claude rewrites description → repeat up to 5 iterations with 60/40 train/test split.

**How to apply:** Every time writing or reviewing a tool, check description against the 5 elements, use prose format, design as high-level operation (not CRUD wrapper), return interpretable values, write instructive error messages.
