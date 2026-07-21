# Discuss-agent tool media bridge complete — 2026-07-21

## Commits

- Provider-neutral media bridge: `a53d9c78b0401bfadacb4af842a6906781538937`
- Integrity validation: `cfca0c213dd680adc193fc63bb359350144c8ce7`
- Review hardening: `628e927`

## Behavior

- Agno `ToolResult` text and local images are preserved internally while legacy string/dict tools retain their text payloads.
- Anthropic receives base64 PNG/JPEG image blocks within `tool_result` content.
- OpenAI receives every paired `tool` message first, followed by one user multimodal data-URI message; history remains provider-valid across later turns.
- Local content/filepath PNG and JPEG are supported. Remote URLs, corrupt/truncated media, files over 5 MiB, decoded images over 25 million pixels, and more than three images per model turn produce visible tool errors.
- Pillow is an explicit dependency and validates real image fixtures; PNG IEND and JPEG EOI checks reject incomplete streams.

## Verification evidence

- Targeted conversation suite: `27 passed in 1.64s`.
- Full suite command: `uv run --with anthropic --with openai pytest -q`.
- Full suite result: `168 passed in 1.85s`.
- Independent spec review: pass.
- Independent code-quality review: pass after fixing empty Anthropic text blocks, pre-decode pixel limits, and awaitable callable objects.
- Media history was intentionally retained because multi-turn preservation is an explicit requirement; pruning it would violate the requested behavior.

## Actual Maestro E2E

The E2E instantiated `AgentConversation` against `http://localhost:23333/api/anthropic`, registered a local `get_page_68` tool returning:

```python
ToolResult(
    content='{"page": 68, "instruction": "Read the chart image independently."}',
    images=[Image(filepath="/home/zhijiang/.openclaw/workspace-research/.tmp/openclaw-spikes/ondemand-pdf-vision/out/page-68.png")],
)
```

No credential values were printed or recorded. Output:

```text
MODEL_OUTPUT=2022, 12.3%
ASSERT_2022=True
ASSERT_12_3=True
```

The raw PNG proves lossless bridge compatibility, but it is not the desired production encoding. Separate cost evidence shows the default 900px JPEG quality 45 remains accurate at approximately 1,010 input tokens and is preferred over the larger full-PNG payload.

No push was performed.
