"""Per-agent audit logger (JSONL).

Writes structured events to ``<output_dir>/audit/<agent_name>.jsonl``.
Each line is a self-contained JSON object describing a lifecycle event
of an agent call (start, tool_call, api_request, call_end, error).

Spec: research-pipeline/spec/020-audit-log/spec.md
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _truncate(text: str | None, max_len: int = 100) -> str:
    if text is None:
        return ""
    s = str(text)
    return s[:max_len] + "..." if len(s) > max_len else s


def _summarize_args(args: dict | None, max_len: int = 200) -> dict | str:
    """Return a compact summary of tool arguments."""
    if args is None:
        return {}
    try:
        dumped = json.dumps(args, ensure_ascii=False, default=str)
        if len(dumped) <= max_len:
            return args
        return dumped[:max_len] + "..."
    except Exception:
        return str(args)[:max_len]


class AuditLogger:
    """Manages per-agent JSONL audit log files.

    Usage::

        audit = AuditLogger("/path/to/session")
        audit.log_call_start("Author", "请写一篇文章...")
        # ... after agent.arun() ...
        audit.log_from_run_output("Author", run_output, duration_ms)
        audit.log_call_end("Author", duration_ms, output_preview, stop_reason)
    """

    def __init__(self, session_dir: str) -> None:
        self._audit_dir = Path(session_dir) / "audit"
        os.makedirs(self._audit_dir, exist_ok=True)
        self._files: dict[str, Any] = {}

    def _get_file(self, agent_name: str):
        if agent_name not in self._files:
            path = self._audit_dir / f"{agent_name}.jsonl"
            self._files[agent_name] = open(path, "a", encoding="utf-8")
        return self._files[agent_name]

    def _write(self, agent_name: str, event: dict) -> None:
        try:
            f = self._get_file(agent_name)
            f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
            f.flush()
        except Exception:
            logger.warning("Failed to write audit event for %s", agent_name, exc_info=True)

    def close(self) -> None:
        for f in self._files.values():
            try:
                f.close()
            except Exception:
                pass
        self._files.clear()

    # ------------------------------------------------------------------
    # Event writers
    # ------------------------------------------------------------------

    def log_call_start(
        self,
        agent_name: str,
        prompt: str,
        *,
        system_prompt_size_chars: int | None = None,
        system_prompt_size_tokens_est: int | None = None,
        skill_tools_loaded: list[str] | None = None,
    ) -> None:
        event: dict = {
            "ts": _now_iso(),
            "agent": agent_name,
            "event": "call_start",
            "prompt_preview": _truncate(prompt),
        }
        if system_prompt_size_chars is not None:
            event["system_prompt_size_chars"] = system_prompt_size_chars
        if system_prompt_size_tokens_est is not None:
            event["system_prompt_size_tokens_est"] = system_prompt_size_tokens_est
        if skill_tools_loaded is not None:
            event["skill_tools_loaded"] = skill_tools_loaded
        self._write(agent_name, event)

    def log_tool_call(
        self,
        agent_name: str,
        tool_name: str,
        args: dict | None = None,
        response_size: int | None = None,
        duration_ms: float | None = None,
        error: str | None = None,
    ) -> None:
        event: dict = {
            "ts": _now_iso(),
            "agent": agent_name,
            "event": "tool_call",
            "tool": tool_name,
            "args": _summarize_args(args),
        }
        if response_size is not None:
            event["response_size"] = response_size
        if duration_ms is not None:
            event["duration_ms"] = round(duration_ms, 1)
        if error is not None:
            event["error"] = error
        self._write(agent_name, event)

    def log_api_request(
        self,
        agent_name: str,
        model: str | None = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        duration_ms: float | None = None,
    ) -> None:
        event: dict = {
            "ts": _now_iso(),
            "agent": agent_name,
            "event": "api_request",
        }
        if model:
            event["model"] = model
        if input_tokens:
            event["input_tokens"] = input_tokens
        if output_tokens:
            event["output_tokens"] = output_tokens
        if duration_ms is not None:
            event["duration_ms"] = round(duration_ms, 1)
        self._write(agent_name, event)

    def log_call_end(
        self,
        agent_name: str,
        duration_ms: float,
        output_preview: str | None = None,
        stop_reason: str | None = None,
        *,
        messages_count: int | None = None,
        total_tool_response_chars: int | None = None,
    ) -> None:
        event: dict = {
            "ts": _now_iso(),
            "agent": agent_name,
            "event": "call_end",
            "duration_ms": round(duration_ms, 1),
        }
        if output_preview is not None:
            event["output_preview"] = _truncate(output_preview)
        if stop_reason is not None:
            event["stop_reason"] = stop_reason
        if messages_count is not None:
            event["messages_count"] = messages_count
        if total_tool_response_chars is not None:
            event["total_tool_response_chars"] = total_tool_response_chars
        self._write(agent_name, event)

    def log_error(
        self,
        agent_name: str,
        error: str,
        duration_ms: float | None = None,
    ) -> None:
        event: dict = {
            "ts": _now_iso(),
            "agent": agent_name,
            "event": "error",
            "error": error,
        }
        if duration_ms is not None:
            event["duration_ms"] = round(duration_ms, 1)
        self._write(agent_name, event)

    # ------------------------------------------------------------------
    # High-level: extract events from agno RunOutput
    # ------------------------------------------------------------------

    def log_from_run_output(self, agent_name: str, run_output) -> None:
        """Extract tool call and API metrics from an agno RunOutput object."""
        if run_output is None:
            return

        # Log tool calls from run_output.tools
        if getattr(run_output, "tools", None):
            for tool_exec in run_output.tools:
                tool_name = getattr(tool_exec, "tool_name", None) or "unknown"
                args = getattr(tool_exec, "tool_args", None)
                result_str = getattr(tool_exec, "result", None)
                response_size = len(result_str) if result_str else None
                error = None
                if getattr(tool_exec, "tool_call_error", False):
                    error = _truncate(result_str, 500)
                duration_ms = None
                metrics = getattr(tool_exec, "metrics", None)
                if metrics and getattr(metrics, "duration", None) is not None:
                    duration_ms = metrics.duration * 1000
                self.log_tool_call(
                    agent_name, tool_name, args, response_size, duration_ms, error
                )

        # ------------------------------------------------------------------
        # Messages summary: count, total chars, system message size
        # ------------------------------------------------------------------
        messages = getattr(run_output, "messages", None)
        messages_count = len(messages) if messages else 0
        messages_total_chars = 0
        system_message_chars = 0
        role_counts: dict[str, int] = {}

        if messages:
            for msg in messages:
                content = getattr(msg, "content", None)
                content_len = len(str(content)) if content is not None else 0
                messages_total_chars += content_len
                role = getattr(msg, "role", None) or "unknown"
                role_counts[role] = role_counts.get(role, 0) + 1
                if role == "system":
                    system_message_chars += content_len

        self._write(agent_name, {
            "ts": _now_iso(),
            "agent": agent_name,
            "event": "messages_summary",
            "messages_count": messages_count,
            "messages_total_chars": messages_total_chars,
            "system_message_chars": system_message_chars,
            "role_counts": role_counts,
        })

        # ------------------------------------------------------------------
        # Per-roundtrip details from assistant messages
        # ------------------------------------------------------------------
        if messages:
            roundtrip_idx = 0
            for msg in messages:
                role = getattr(msg, "role", None)
                if role != "assistant":
                    continue
                msg_metrics = getattr(msg, "metrics", None)
                if msg_metrics is None:
                    continue
                in_tok = getattr(msg_metrics, "input_tokens", 0) or 0
                out_tok = getattr(msg_metrics, "output_tokens", 0) or 0
                reasoning_tok = getattr(msg_metrics, "reasoning_tokens", 0) or 0
                cache_read = getattr(msg_metrics, "cache_read_tokens", 0) or 0
                cache_write = getattr(msg_metrics, "cache_write_tokens", 0) or 0
                cost = getattr(msg_metrics, "cost", None)
                duration = getattr(msg_metrics, "duration", None)
                ttft = getattr(msg_metrics, "time_to_first_token", None)
                content = getattr(msg, "content", None)
                content_len = len(str(content)) if content is not None else 0

                if in_tok or out_tok or reasoning_tok or content_len:
                    event: dict = {
                        "ts": _now_iso(),
                        "agent": agent_name,
                        "event": "roundtrip_tokens",
                        "roundtrip_idx": roundtrip_idx,
                        "input_tokens": in_tok,
                        "output_tokens": out_tok,
                        "content_chars": content_len,
                    }
                    if reasoning_tok:
                        event["reasoning_tokens"] = reasoning_tok
                    if cache_read:
                        event["cache_read_tokens"] = cache_read
                    if cache_write:
                        event["cache_write_tokens"] = cache_write
                    if cost is not None:
                        event["cost"] = cost
                    if duration is not None:
                        event["duration_ms"] = round(duration * 1000, 1)
                    if ttft is not None:
                        event["time_to_first_token_ms"] = round(ttft * 1000, 1)
                    self._write(agent_name, event)
                roundtrip_idx += 1

        # ------------------------------------------------------------------
        # Aggregate API metrics from run_output.metrics
        # ------------------------------------------------------------------
        metrics = getattr(run_output, "metrics", None)
        if metrics:
            model = getattr(run_output, "model", None)
            input_tokens = getattr(metrics, "input_tokens", 0) or 0
            output_tokens = getattr(metrics, "output_tokens", 0) or 0
            reasoning_tokens = getattr(metrics, "reasoning_tokens", 0) or 0
            cache_read_tokens = getattr(metrics, "cache_read_tokens", 0) or 0
            cache_write_tokens = getattr(metrics, "cache_write_tokens", 0) or 0
            total_tokens = getattr(metrics, "total_tokens", 0) or 0
            cost = getattr(metrics, "cost", None)
            duration_ms = None
            if getattr(metrics, "duration", None) is not None:
                duration_ms = metrics.duration * 1000
            if input_tokens or output_tokens or duration_ms:
                self.log_api_request(
                    agent_name, model, input_tokens, output_tokens, duration_ms
                )
            # Write extended metrics event with all available fields
            ext: dict = {
                "ts": _now_iso(),
                "agent": agent_name,
                "event": "run_metrics",
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
            }
            if reasoning_tokens:
                ext["reasoning_tokens"] = reasoning_tokens
            if cache_read_tokens:
                ext["cache_read_tokens"] = cache_read_tokens
            if cache_write_tokens:
                ext["cache_write_tokens"] = cache_write_tokens
            if cost is not None:
                ext["cost"] = cost
            if duration_ms is not None:
                ext["duration_ms"] = round(duration_ms, 1)
            self._write(agent_name, ext)

    @staticmethod
    def extract_call_start_extras(agent) -> dict:
        """Extract system_prompt_size and skill_tools_loaded from an agno Agent.

        Returns a dict of keyword arguments for ``log_call_start``.
        """
        extras: dict = {}

        # system_prompt_size
        sys_msg = getattr(agent, "system_message", None)
        if sys_msg is not None:
            chars = len(str(sys_msg))
            extras["system_prompt_size_chars"] = chars
            extras["system_prompt_size_tokens_est"] = chars // 4

        # skill_tools_loaded
        tools = getattr(agent, "tools", None)
        if tools:
            names: list[str] = []
            for t in tools:
                name = getattr(t, "name", None) or getattr(t, "__name__", None)
                if name:
                    names.append(name)
                else:
                    names.append(type(t).__name__)
            extras["skill_tools_loaded"] = names

        return extras

    @staticmethod
    def extract_call_end_extras(run_output) -> dict:
        """Extract messages_count and total_tool_response_chars from RunOutput.

        Returns a dict of keyword arguments for ``log_call_end``.
        """
        extras: dict = {}
        if run_output is None:
            return extras

        # messages_count
        messages = getattr(run_output, "messages", None)
        if messages is not None:
            extras["messages_count"] = len(messages)

        # total_tool_response_chars
        tools = getattr(run_output, "tools", None)
        if tools:
            total = 0
            for tool_exec in tools:
                result_str = getattr(tool_exec, "result", None)
                if result_str:
                    total += len(result_str)
            if total > 0:
                extras["total_tool_response_chars"] = total

        return extras


# ---------------------------------------------------------------------------
# Usage summary generation
# ---------------------------------------------------------------------------


def _format_number(n: int | float) -> str:
    """Format a number with thousands separators (1,234,567)."""
    if isinstance(n, float):
        return f"{n:,.2f}"
    return f"{n:,}"


def _format_duration_ms(ms: float) -> str:
    """Format milliseconds to human-readable duration (e.g. 18m 32s)."""
    if ms <= 0:
        return "0s"
    total_seconds = int(ms / 1000)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    parts = []
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    if seconds or not parts:
        parts.append(f"{seconds}s")
    return " ".join(parts)


def _parse_audit_files(audit_dir: Path) -> dict[str, list[dict]]:
    """Read all .jsonl files from audit_dir and return {agent_name: [events]}."""
    result: dict[str, list[dict]] = {}
    if not audit_dir.is_dir():
        return result
    for jsonl_file in sorted(audit_dir.glob("*.jsonl")):
        agent_name = jsonl_file.stem
        events: list[dict] = []
        try:
            with open(jsonl_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except OSError:
            continue
        if events:
            result[agent_name] = events
    return result


def generate_usage_summary(
    session_dir: str,
    *,
    run_name: str | None = None,
    model_name: str | None = None,
    total_rounds: int | None = None,
) -> str | None:
    """Generate a usage_summary.md from audit JSONL files in session_dir.

    Returns the path to the generated file, or None if no audit data found.
    """
    audit_dir = Path(session_dir) / "audit"
    agent_events = _parse_audit_files(audit_dir)
    if not agent_events:
        logger.info("No audit data found in %s, skipping usage summary", audit_dir)
        return None

    # Derive run_name from session dir name if not provided
    if run_name is None:
        run_name = Path(session_dir).name

    # ---------------------------------------------------------------------------
    # Per-agent aggregation
    # ---------------------------------------------------------------------------
    agent_stats: dict[str, dict] = {}

    for agent_name, events in agent_events.items():
        stats = {
            "calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "reasoning_tokens": 0,
            "duration_ms": 0.0,
            "cost": 0.0,
        }

        for ev in events:
            ev_type = ev.get("event", "")

            if ev_type == "roundtrip_tokens":
                stats["input_tokens"] += ev.get("input_tokens", 0)
                stats["output_tokens"] += ev.get("output_tokens", 0)
                stats["cache_read_tokens"] += ev.get("cache_read_tokens", 0)
                stats["cache_write_tokens"] += ev.get("cache_write_tokens", 0)
                stats["reasoning_tokens"] += ev.get("reasoning_tokens", 0)
                if ev.get("cost") is not None:
                    stats["cost"] += ev["cost"]

            elif ev_type == "call_end":
                stats["calls"] += 1
                stats["duration_ms"] += ev.get("duration_ms", 0)

        agent_stats[agent_name] = stats

    # ---------------------------------------------------------------------------
    # Per-round aggregation (based on call_start/call_end ordering)
    # ---------------------------------------------------------------------------
    agent_calls: dict[str, list[dict]] = {}
    for agent_name, events in agent_events.items():
        calls: list[dict] = []
        current_call: dict | None = None
        for ev in events:
            ev_type = ev.get("event", "")
            if ev_type == "call_start":
                current_call = {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "duration_ms": 0.0,
                    "cost": 0.0,
                }
            elif ev_type == "roundtrip_tokens" and current_call is not None:
                current_call["input_tokens"] += ev.get("input_tokens", 0)
                current_call["output_tokens"] += ev.get("output_tokens", 0)
                if ev.get("cost") is not None:
                    current_call["cost"] += ev["cost"]
            elif ev_type == "call_end" and current_call is not None:
                current_call["duration_ms"] += ev.get("duration_ms", 0)
                calls.append(current_call)
                current_call = None
        agent_calls[agent_name] = calls

    # Determine max number of calls across agents
    max_calls = max((len(c) for c in agent_calls.values()), default=0)

    round_stats: list[dict] = []
    for i in range(max_calls):
        rs = {"input_tokens": 0, "output_tokens": 0, "duration_ms": 0.0, "cost": 0.0}
        for calls in agent_calls.values():
            if i < len(calls):
                rs["input_tokens"] += calls[i]["input_tokens"]
                rs["output_tokens"] += calls[i]["output_tokens"]
                rs["duration_ms"] += calls[i]["duration_ms"]
                rs["cost"] += calls[i]["cost"]
        round_stats.append(rs)

    # ---------------------------------------------------------------------------
    # Totals
    # ---------------------------------------------------------------------------
    totals = {
        "calls": sum(s["calls"] for s in agent_stats.values()),
        "input_tokens": sum(s["input_tokens"] for s in agent_stats.values()),
        "output_tokens": sum(s["output_tokens"] for s in agent_stats.values()),
        "cache_read_tokens": sum(s["cache_read_tokens"] for s in agent_stats.values()),
        "cache_write_tokens": sum(s["cache_write_tokens"] for s in agent_stats.values()),
        "reasoning_tokens": sum(s["reasoning_tokens"] for s in agent_stats.values()),
        "duration_ms": sum(s["duration_ms"] for s in agent_stats.values()),
        "cost": sum(s["cost"] for s in agent_stats.values()),
    }

    rounds_display = total_rounds if total_rounds is not None else max_calls

    # ---------------------------------------------------------------------------
    # Build markdown
    # ---------------------------------------------------------------------------
    lines: list[str] = []
    lines.append("# Token Usage Summary\n")
    lines.append(f"Run: {run_name}")
    if model_name:
        lines.append(f"Model: {model_name}")
    lines.append(f"Rounds: {rounds_display}")
    lines.append("")

    # Per Agent table
    lines.append("## Per Agent\n")
    lines.append(
        "| Agent | Calls | Input Tokens | Output Tokens "
        "| Cache Read | Cache Write | Reasoning | Duration |"
    )
    lines.append(
        "|-------|-------|-------------|--------------|"
        "------------|-------------|-----------|----------|"
    )
    for agent_name, stats in agent_stats.items():
        lines.append(
            f"| {agent_name} "
            f"| {_format_number(stats['calls'])} "
            f"| {_format_number(stats['input_tokens'])} "
            f"| {_format_number(stats['output_tokens'])} "
            f"| {_format_number(stats['cache_read_tokens'])} "
            f"| {_format_number(stats['cache_write_tokens'])} "
            f"| {_format_number(stats['reasoning_tokens'])} "
            f"| {_format_duration_ms(stats['duration_ms'])} |"
        )
    lines.append(
        f"| **Total** "
        f"| **{_format_number(totals['calls'])}** "
        f"| **{_format_number(totals['input_tokens'])}** "
        f"| **{_format_number(totals['output_tokens'])}** "
        f"| **{_format_number(totals['cache_read_tokens'])}** "
        f"| **{_format_number(totals['cache_write_tokens'])}** "
        f"| **{_format_number(totals['reasoning_tokens'])}** "
        f"| **{_format_duration_ms(totals['duration_ms'])}** |"
    )
    lines.append("")

    # Cost section (only if there's cost data)
    if totals["cost"] > 0:
        lines.append("## Cost\n")
        lines.append("| Agent | Cost |")
        lines.append("|-------|------|")
        for agent_name, stats in agent_stats.items():
            if stats["cost"] > 0:
                lines.append(f"| {agent_name} | ${stats['cost']:.4f} |")
        lines.append(f"| **Total** | **${totals['cost']:.4f}** |")
        lines.append("")

    # Per Round table
    if round_stats:
        lines.append("## Per Round\n")
        has_cost = any(rs["cost"] > 0 for rs in round_stats)
        if has_cost:
            lines.append(
                "| Round | Input Tokens | Output Tokens | Duration | Cost |"
            )
            lines.append(
                "|-------|-------------|--------------|----------|------|"
            )
        else:
            lines.append("| Round | Input Tokens | Output Tokens | Duration |")
            lines.append("|-------|-------------|--------------|----------|")
        for i, rs in enumerate(round_stats, 1):
            row = (
                f"| {i} "
                f"| {_format_number(rs['input_tokens'])} "
                f"| {_format_number(rs['output_tokens'])} "
                f"| {_format_duration_ms(rs['duration_ms'])} |"
            )
            if has_cost:
                row = row[:-1] + f" ${rs['cost']:.4f} |"
            lines.append(row)
        lines.append("")

    content = "\n".join(lines)

    # Write to file
    output_path = Path(session_dir) / "usage_summary.md"
    try:
        output_path.write_text(content, encoding="utf-8")
        logger.info("Usage summary written to %s", output_path)
        return str(output_path)
    except OSError:
        logger.warning(
            "Failed to write usage summary to %s", output_path, exc_info=True
        )
        return None
