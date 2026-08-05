from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from discuss_agent.audit import AuditLogger
from discuss_agent.config import AgentConfig, DiscussionConfig, HostConfig, ModelConfig, ToolConfig
from discuss_agent.conversation import AgentConversation


def _config(*, min_rounds=1, max_rounds=3, strict=False, tools=None, extra_tools=None):
    return DiscussionConfig(
        min_rounds=min_rounds,
        max_rounds=max_rounds,
        model_config=ModelConfig(model="claude-sonnet-4-20250514", api_key="test"),
        agents=[
            AgentConfig(name="A", system_prompt="A", extra_tools=extra_tools or []),
            AgentConfig(name="B", system_prompt="B"),
        ],
        host=HostConfig(convergence_prompt="judge", summary_prompt="summary", skip_summary=False),
        tools=tools or [],
        context={},
        strict_tool_loading=strict,
    )


def test_config_strict_tool_loading_defaults_false_and_parses_true(tmp_path):
    from discuss_agent.config import ConfigLoader

    raw = {
        "discussion": {"model": "model"},
        "agents": [{"name": "A", "system_prompt": "A"}],
        "host": {"convergence_prompt": "judge", "summary_prompt": "summary"},
        "tools": [],
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    assert ConfigLoader.load(str(path)).strict_tool_loading is False
    raw["discussion"]["strict_tool_loading"] = True
    path.write_text(yaml.safe_dump(raw))
    assert ConfigLoader.load(str(path)).strict_tool_loading is True


@patch("discuss_agent.engine.generate_usage_summary")
@patch("discuss_agent.engine.AuditLogger")
@patch("discuss_agent.engine.Archiver")
@patch("discuss_agent.engine.ContextManager")
@patch("discuss_agent.engine.AgentConversation")
async def test_natural_convergence_ignores_legacy_min_rounds(
    MockConversation, MockContext, MockArchiver, MockAudit, _summary,
):
    """Legacy min_rounds must not delay a closable second round."""
    from discuss_agent.engine import DiscussionEngine

    config = _config(min_rounds=99, max_rounds=4)
    config.host.skip_summary = True
    archiver = MagicMock()
    archiver.start_session.return_value = "/tmp/runtime-natural-convergence"
    MockArchiver.return_value = archiver
    MockAudit.return_value = MagicMock()
    MockContext.return_value.build_initial_context = AsyncMock(return_value="generic topic")
    counts = {"A": 0, "B": 0}

    def conversation(**kwargs):
        name = kwargs["agent_name"]
        conv = MagicMock(messages=[])

        async def send(_prompt):
            counts[name] += 1
            if counts[name] == 1:
                return f"[NEW_CLAIM:item-{name}] proposal"
            other = "B" if name == "A" else "A"
            return f"[ACCEPT TO:item-{other}] accepted"

        conv.send = AsyncMock(side_effect=send)
        return conv

    MockConversation.side_effect = conversation
    engine = DiscussionEngine(config)
    engine._host_judge = AsyncMock(return_value=[
        {"claim": "item-A", "verdict": "CLOSED:共识", "reason": "ok"},
        {"claim": "item-B", "verdict": "CLOSED:共识", "reason": "ok"},
    ])

    result = await engine.run()

    assert result.converged is True
    assert result.rounds_completed == 2
    assert counts == {"A": 2, "B": 2}
    engine._host_judge.assert_awaited_once()
    assert engine._host_judge.await_args.args[1] == 2


@pytest.mark.parametrize("strict", [True, False])
def test_strict_tool_loading_controls_fail_closed(strict):
    from discuss_agent.engine import DiscussionEngine

    engine = DiscussionEngine(_config(strict=strict, tools=[ToolConfig("missing.module.Toolkit")]))
    if strict:
        with pytest.raises(RuntimeError, match=r"global.*missing\.module\.Toolkit"):
            engine._create_conversations()
    else:
        engine._create_conversations()
        assert set(engine._conversations) == {"A", "B"}


def test_strict_extra_toolkit_requires_callable_function():
    from discuss_agent.engine import DiscussionEngine

    empty = SimpleNamespace(functions={}, async_functions={})
    config = _config(strict=True, extra_tools=[ToolConfig("pkg.Empty")])
    engine = DiscussionEngine(config)
    with patch("discuss_agent.engine.import_from_path", return_value=lambda: empty):
        with pytest.raises(RuntimeError, match=r"extra tool.*pkg\.Empty.*callable"):
            engine._create_conversations()


@pytest.mark.parametrize(
    ("loaded", "message"),
    [
        (lambda: (_ for _ in ()).throw(RuntimeError("constructor failed")), "constructor failed"),
        (
            lambda: SimpleNamespace(
                functions={"bad": SimpleNamespace(entrypoint=None)},
                async_functions={},
            ),
            "no callable entrypoint",
        ),
    ],
)
def test_strict_toolkit_initialization_and_entrypoint_failures(loaded, message):
    from discuss_agent.engine import DiscussionEngine

    config = _config(strict=True, tools=[ToolConfig("pkg.Toolkit")])
    engine = DiscussionEngine(config)
    with patch("discuss_agent.engine.import_from_path", return_value=loaded):
        with pytest.raises(RuntimeError, match=message):
            engine._create_conversations()


@patch("discuss_agent.engine.generate_usage_summary")
@patch("discuss_agent.engine.AuditLogger")
@patch("discuss_agent.engine.Archiver")
@patch("discuss_agent.engine.ContextManager")
async def test_strict_loading_run_fails_before_context_or_agent_calls(
    MockContext, MockArchiver, MockAudit, _summary,
):
    from discuss_agent.engine import DiscussionEngine

    archiver = MagicMock()
    archiver.start_session.return_value = "/tmp/runtime-gate-strict"
    MockArchiver.return_value = archiver
    context = MagicMock()
    context.build_initial_context = AsyncMock(return_value="must not run")
    MockContext.return_value = context
    result = await DiscussionEngine(
        _config(strict=True, tools=[ToolConfig("missing.module.Toolkit")])
    ).run()
    assert result.terminated_by_error is True
    context.build_initial_context.assert_not_awaited()
    error = archiver.save_error_log.call_args.args[0]
    assert "global tool" in error and "missing.module.Toolkit" in error


async def test_anthropic_sync_and_async_tools_are_audited(tmp_path):
    audit = AuditLogger(str(tmp_path))

    def sync_tool(value):
        return value * 2

    async def async_tool(value):
        return value + 1

    conv = AgentConversation(
        agent_name="agent",
        system_prompt="system",
        api_key="test",
        tools=[{"name": "sync", "input_schema": {}}, {"name": "async", "input_schema": {}}],
        tool_callables={"sync": sync_tool, "async": async_tool},
        audit_logger=audit,
    )
    conv.set_round(2)
    conv._client = MagicMock()
    blocks = lambda *xs: SimpleNamespace(content=list(xs), stop_reason="end_turn")
    use = lambda ident, name, args: SimpleNamespace(type="tool_use", id=ident, name=name, input=args)
    text = lambda value: SimpleNamespace(type="text", text=value)
    conv._client.messages.create = AsyncMock(side_effect=[
        blocks(use("1", "sync", {"value": 2}), use("2", "async", {"value": 2})),
        blocks(text("complete")),
    ])

    assert await conv.send("go") == "complete"
    audit.close()
    events = [json.loads(line) for line in (tmp_path / "audit" / "agent.jsonl").read_text().splitlines()]
    assert events[0]["event"] == "call_start" and events[0]["round"] == 2
    tools = [event for event in events if event["event"] == "tool_call"]
    assert [event["tool"] for event in tools] == ["sync", "async"]
    assert all(event["round"] == 2 and event["response_size"] > 0 for event in tools)
    assert events[-1]["event"] == "call_end" and events[-1]["round"] == 2


async def test_openai_invalid_and_failing_tools_are_audited(tmp_path):
    audit = AuditLogger(str(tmp_path))

    def failing():
        raise RuntimeError("failed safely")

    conv = AgentConversation(
        agent_name="openai-agent",
        system_prompt="system",
        model="agent-maestro-openai/model",
        api_key="test",
        tools=[{"name": "failing", "input_schema": {}}],
        tool_callables={"failing": failing},
        audit_logger=audit,
    )
    conv.set_round(4)
    conv._client = MagicMock()

    def call(ident, name, args):
        return SimpleNamespace(id=ident, type="function", function=SimpleNamespace(name=name, arguments=args))

    conv._client.chat.completions.create = AsyncMock(side_effect=[
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=[
            call("1", "failing", "{}"), call("2", "failing", "not-json")
        ]))]),
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="complete", tool_calls=[]))]),
    ])

    assert await conv.send("go") == "complete"
    audit.close()
    events = [json.loads(line) for line in (tmp_path / "audit" / "openai-agent.jsonl").read_text().splitlines()]
    tools = [event for event in events if event["event"] == "tool_call"]
    assert len(tools) == 2
    assert "failed safely" in tools[0]["error"]
    assert "Invalid tool arguments" in tools[1]["error"]
    assert all(event["round"] == 4 and "duration_ms" in event for event in tools)


async def test_failed_model_call_still_has_audited_start_error_and_end(tmp_path):
    audit = AuditLogger(str(tmp_path))
    conv = AgentConversation(
        agent_name="failed-agent", system_prompt="system", api_key="test",
        audit_logger=audit,
    )
    conv.set_round(3)
    conv._client = MagicMock()
    conv._client.messages.create = AsyncMock(side_effect=RuntimeError("model unavailable"))

    with pytest.raises(RuntimeError, match="model unavailable"):
        await conv.send("go")
    audit.close()
    events = [
        json.loads(line)
        for line in (tmp_path / "audit" / "failed-agent.jsonl").read_text().splitlines()
    ]
    assert [event["event"] for event in events] == ["call_start", "error", "call_end"]
    assert all(event["round"] == 3 for event in events)
    assert events[-1]["stop_reason"] == "error"
