"""DiscussionEngine — shared claims.md + multi-turn conversations."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time as _time
from dataclasses import asdict

import anthropic
import openai

from discuss_agent.claims import AgentOutput, ClaimsManager
from discuss_agent.config import (
    DiscussionConfig,
    infer_provider,
    normalize_base_url,
)
from discuss_agent.context import ContextManager
from discuss_agent.conversation import AgentConversation
from discuss_agent.models import DiscussionResult
from discuss_agent.audit import AuditLogger, generate_usage_summary
from discuss_agent.persistence import Archiver
from discuss_agent.registry import import_from_path

logger = logging.getLogger(__name__)


class DiscussionEngine:
    """Run a structured multi-agent discussion using shared claims.md architecture.

    - Each agent maintains a multi-turn conversation (message history)
    - A shared claims.md file is the single source of truth
    - The program merges agent outputs into claims.md with FROM tags
    - Incremental updates are pushed to agents (OPEN claims full text, CLOSED status only)
    - Convergence is checked by program preconditions + host LLM
    """

    def __init__(self, config: DiscussionConfig):
        self._config = config
        self._archiver = Archiver()
        self._audit: AuditLogger | None = None

        # Load context builder
        context_builder = None
        if config.context_builder:
            context_builder = import_from_path(config.context_builder)
        self._context_mgr = ContextManager(config, context_builder=context_builder)

        # We'll create conversations in run() once we have the session path
        self._conversations: dict[str, AgentConversation] = {}

        # Host model config
        self._host_model_config = config.host.resolve_model(config.model_config)

    def _load_tools(self) -> tuple[list[dict], dict[str, callable]]:
        """Load tools from config and convert to Anthropic format.

        Returns (tool_definitions, tool_callables) where:
        - tool_definitions: list of dicts in Anthropic tool format
        - tool_callables: mapping from tool name to callable
        """
        tool_defs: list[dict] = []
        tool_callables: dict[str, callable] = {}

        for tc in self._config.tools:
            try:
                defs, callables = self._load_toolkit(
                    tc.path, "global tool", strict=self._config.strict_tool_loading,
                )
                self._merge_tools(tool_defs, tool_callables, defs, callables)
            except Exception as exc:
                if self._config.strict_tool_loading:
                    raise RuntimeError(
                        f"Strict tool loading failed for global tool '{tc.path}': {exc}"
                    ) from exc
                logger.warning("Failed to load tool %s", tc.path, exc_info=True)

        return tool_defs, tool_callables

    @staticmethod
    def _load_toolkit(
        path: str, scope: str, *, strict: bool = False,
    ) -> tuple[list[dict], dict[str, callable]]:
        """Import one configured toolkit and normalize its callable functions."""
        cls = import_from_path(path)
        toolkit = cls()
        functions = getattr(toolkit, "functions", {}) or {}
        async_functions = getattr(toolkit, "async_functions", {}) or {}
        defs: list[dict] = []
        callables: dict[str, callable] = {}
        for collection in (functions, async_functions):
            for name, func in collection.items():
                entrypoint = getattr(func, "entrypoint", None)
                if strict and not callable(entrypoint):
                    raise TypeError(f"{scope} '{path}' function '{name}' has no callable entrypoint")
                if name not in callables:
                    defs.append({
                        "name": name,
                        "description": getattr(func, "description", None) or "",
                        "input_schema": getattr(func, "parameters", None)
                        or {"type": "object", "properties": {}},
                    })
                callables[name] = entrypoint
        if strict and not callables:
            raise ValueError(f"{scope} '{path}' exposes no callable functions")
        return defs, callables

    @staticmethod
    def _merge_tools(
        target_defs: list[dict], target_callables: dict[str, callable],
        new_defs: list[dict], new_callables: dict[str, callable],
    ) -> None:
        known = {item["name"] for item in target_defs}
        target_defs.extend(item for item in new_defs if item["name"] not in known)
        target_callables.update(new_callables)

    def _resolve_agent_tools(
        self, ac, global_defs: list[dict], global_callables: dict[str, callable],
    ) -> tuple[list[dict], dict[str, callable]]:
        """Resolve per-agent tool set (global + extra - disabled)."""
        defs = list(global_defs)
        callables = dict(global_callables)

        # Add extra tools
        for tc in ac.extra_tools:
            try:
                extra_defs, extra_callables = self._load_toolkit(
                    tc.path, "extra tool", strict=self._config.strict_tool_loading,
                )
                self._merge_tools(defs, callables, extra_defs, extra_callables)
            except Exception as exc:
                if self._config.strict_tool_loading:
                    raise RuntimeError(
                        f"Strict tool loading failed for agent '{ac.name}' extra tool "
                        f"'{tc.path}': {exc}"
                    ) from exc
                logger.warning("Failed to load extra tool %s", tc.path, exc_info=True)

        # Remove disabled tools
        if ac.disable_tools:
            disable_names = set()
            for path in ac.disable_tools:
                try:
                    cls = import_from_path(path)
                    toolkit = cls()
                    disable_names.update(toolkit.functions.keys())
                    disable_names.update(toolkit.async_functions.keys())
                except Exception:
                    pass
            defs = [d for d in defs if d["name"] not in disable_names]
            for name in disable_names:
                callables.pop(name, None)

        return defs, callables

    def _create_conversations(self) -> None:
        """Create an AgentConversation for each configured agent."""
        global_defs, global_callables = self._load_tools()
        logger.info(
            "Loaded %d global tools: %s",
            len(global_defs), list(global_callables.keys()),
        )

        for ac in self._config.agents:
            agent_defs, agent_callables = self._resolve_agent_tools(
                ac, global_defs, global_callables,
            )
            logger.info(
                "Agent '%s': %d tools available",
                ac.name, len(agent_defs),
            )
            self._conversations[ac.name] = AgentConversation(
                agent_name=ac.name,
                system_prompt=ac.system_prompt,
                model=self._config.model_config.model,
                api_key=self._config.model_config.api_key,
                base_url=self._config.model_config.base_url,
                max_tokens=self._config.model_config.max_tokens or 4096,
                temperature=self._config.model_config.temperature,
                tools=agent_defs if agent_defs else None,
                tool_callables=agent_callables if agent_callables else None,
                audit_logger=self._audit,
            )

    async def _call_agent(self, agent_name: str, prompt: str) -> str | None:
        """Send a message to an agent, with retry on failure."""
        conv = self._conversations[agent_name]
        # _round_1/_round_n set this before dispatch; keep compatibility with
        # conversation doubles that do not implement round context.
        logger.info("  -> Calling agent '%s' (prompt %d chars)...", agent_name, len(prompt))
        t0 = _time.monotonic()
        for attempt in range(2):
            try:
                result = await conv.send(prompt)
                if result:
                    elapsed = (_time.monotonic() - t0) * 1000
                    logger.info(
                        "  <- Agent '%s' returned %d chars in %.1fs",
                        agent_name, len(result), elapsed / 1000,
                    )
                    return result
                if attempt == 0:
                    continue
                return None
            except Exception:
                if attempt == 0:
                    # Remove the failed user message so retry can re-add it
                    if conv.messages and conv.messages[-1]["role"] == "user":
                        conv.messages.pop()
                    continue
                return None
        return None

    async def _round_1(
        self, claims_mgr: ClaimsManager, topic: str, context: str = "",
    ) -> list[AgentOutput]:
        """Round 1: each agent proposes initial claims."""
        logger.info("=== Round 1: Initial claims ===")
        prompt = claims_mgr.generate_initial_prompt(
            topic,
            limitation=self._config.limitation,
            context=context,
        )
        for conv in self._conversations.values():
            if hasattr(conv, "set_round"):
                conv.set_round(1)

        async def call_one(name: str) -> AgentOutput | None:
            text = await self._call_agent(name, prompt)
            if text:
                return AgentOutput(agent_name=name, round_num=1, raw_text=text)
            return None

        results = await asyncio.gather(
            *[call_one(ac.name) for ac in self._config.agents]
        )
        outputs = [r for r in results if r is not None]
        if not outputs:
            raise RuntimeError("All agents failed in Round 1")
        return outputs

    async def _round_n(
        self, claims_mgr: ClaimsManager, round_num: int, prev_round: int,
    ) -> list[AgentOutput]:
        """Round N: push incremental update and collect responses."""
        logger.info("=== Round %d: Incremental update ===", round_num)
        for conv in self._conversations.values():
            if hasattr(conv, "set_round"):
                conv.set_round(round_num)

        async def call_one(name: str) -> AgentOutput | None:
            prompt = claims_mgr.generate_update_prompt(
                prev_round,
                agent_name=name,
                all_agent_names={ac.name for ac in self._config.agents},
            )
            text = await self._call_agent(name, prompt)
            if text:
                return AgentOutput(agent_name=name, round_num=round_num, raw_text=text)
            return None

        results = await asyncio.gather(
            *[call_one(ac.name) for ac in self._config.agents]
        )
        outputs = [r for r in results if r is not None]
        if not outputs:
            raise RuntimeError(f"All agents failed in Round {round_num}")
        return outputs

    def _check_convergence_precondition(
        self, claims_mgr: ClaimsManager, round_num: int,
    ) -> bool:
        """Return whether at least one OPEN claim is mature for Host judgment."""
        all_agent_names = {ac.name for ac in self._config.agents}
        return bool(claims_mgr.get_mature_claims(all_agent_names))

    def _create_host_client(self):
        """Create a host client using the configured model's wire protocol."""
        provider = infer_provider(self._host_model_config.model)
        client_kwargs: dict = {"timeout": 600.0}
        if self._host_model_config.api_key:
            client_kwargs["api_key"] = self._host_model_config.api_key
        base_url = normalize_base_url(self._host_model_config.base_url, provider)
        if base_url:
            client_kwargs["base_url"] = base_url
        if provider == "openai":
            return openai.AsyncOpenAI(**client_kwargs)
        return anthropic.AsyncAnthropic(**client_kwargs)

    async def _call_host(
        self, system_prompt: str, prompt: str, *, round_num: int | None = None,
    ) -> str:
        """Run one host request and extract text from either protocol."""
        started = _time.monotonic()
        if self._audit:
            self._audit.log_call_start(
                "host", prompt, round_num=round_num, call_type="host"
            )
        provider = infer_provider(self._host_model_config.model)
        client = self._create_host_client()
        try:
            if provider == "openai":
                kwargs: dict = {
                    "model": self._host_model_config.model,
                    "max_tokens": self._host_model_config.max_tokens or 4096,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                }
                if self._host_model_config.temperature is not None:
                    kwargs["temperature"] = self._host_model_config.temperature
                response = await client.chat.completions.create(**kwargs)
                result = response.choices[0].message.content or ""
            else:
                kwargs = {
                    "model": self._host_model_config.model,
                    "max_tokens": self._host_model_config.max_tokens or 4096,
                    "system": system_prompt,
                    "messages": [{"role": "user", "content": prompt}],
                }
                if self._host_model_config.temperature is not None:
                    kwargs["temperature"] = self._host_model_config.temperature
                response = await client.messages.create(**kwargs)
                result = "".join(
                    block.text for block in response.content if block.type == "text"
                )
        except Exception as exc:
            if self._audit:
                self._audit.log_error(
                    "host", str(exc), (_time.monotonic() - started) * 1000,
                    round_num=round_num,
                )
                self._audit.log_call_end(
                    "host", (_time.monotonic() - started) * 1000,
                    stop_reason="error", round_num=round_num,
                    call_type="host",
                )
            raise
        if self._audit:
            self._audit.log_call_end(
                "host", (_time.monotonic() - started) * 1000, result,
                round_num=round_num, call_type="host",
            )
        return result

    async def _host_judge(self, claims_mgr: ClaimsManager, round_num: int) -> list[dict]:
        """Host LLM judges each mature OPEN claim."""
        logger.info("=== HOST JUDGE (Round %d) ===", round_num)
        mature_claims = claims_mgr.get_mature_claims(
            {ac.name for ac in self._config.agents},
        )
        if not mature_claims:
            return []

        claims_text = "\n\n".join(c.format() for c in mature_claims)
        prompt = (
            f"以下是已完成各方回应、等待裁决的成熟 claims：\n\n{claims_text}\n\n"
            f"请对每个 claim 裁决：\n"
            f"- CLOSED:共识 — 各方达成一致\n"
            f"- CLOSED:分歧 — 讨论充分但立场不同，记录分歧\n"
            f"- CONTINUE — 仍需讨论\n\n"
            f'输出 JSON 数组: [{{"claim": "关键词", "verdict": "CLOSED:共识", "reason": "..."}}]'
        )
        for attempt in range(2):
            try:
                text = await self._call_host(
                    self._config.host.convergence_prompt, prompt,
                    round_num=round_num,
                )
                match = re.search(r"\[.*\]", text, re.DOTALL)
                if match:
                    return json.loads(match.group())
            except Exception:
                if attempt == 0:
                    continue
        return []

    def _apply_host_verdicts(
        self,
        claims_mgr: ClaimsManager,
        verdicts: list[dict],
        mature_keywords: set[str],
        round_num: int,
    ) -> tuple[list[dict], list[dict]]:
        """Apply only one valid verdict for each claim offered to the Host."""
        accepted: list[dict] = []
        rejected: list[dict] = []
        seen: set[str] = set()
        valid_verdicts = {"CLOSED:共识", "CLOSED:分歧", "CONTINUE"}

        for item in verdicts:
            verdict = item if isinstance(item, dict) else {}
            keyword = verdict.get("claim", "")
            decision = verdict.get("verdict", "")
            rejection_reason = ""
            if not isinstance(keyword, str):
                rejection_reason = "invalid claim field"
            elif not isinstance(decision, str):
                rejection_reason = "invalid verdict field"
            elif keyword in seen:
                rejection_reason = "duplicate verdict"
            elif keyword not in mature_keywords:
                rejection_reason = "claim was not in the mature set"
            elif decision not in valid_verdicts:
                rejection_reason = "invalid verdict"

            if rejection_reason:
                rejected.append({
                    **verdict,
                    "host_reason": verdict.get("reason", ""),
                    "reason": rejection_reason,
                })
                continue

            seen.add(keyword)
            reason = verdict.get("reason", "")
            if decision == "CONTINUE":
                claims_mgr.continue_claim(keyword, reason, round_num)
            else:
                claims_mgr.close_claim(
                    keyword, decision.removeprefix("CLOSED:"), reason, round_num,
                )
            accepted.append(verdict)

        for keyword in sorted(mature_keywords - seen):
            rejected.append({
                "claim": keyword,
                "verdict": None,
                "host_reason": "",
                "reason": "missing verdict",
            })

        return accepted, rejected

    async def _host_summarize(
        self, claims_mgr: ClaimsManager, round_num: int | None = None,
    ) -> str:
        """Host generates final summary using its configured protocol."""
        logger.info("=== HOST SUMMARIZE ===")
        prompt = (
            f"以下是完整的讨论记录：\n\n{claims_mgr.format_file()}\n\n"
            f"讨论已经结束。请基于各方达成的共识和记录的分歧输出总结。"
        )
        return await self._call_host(
            self._config.host.summary_prompt, prompt, round_num=round_num,
        )

    async def run(self) -> DiscussionResult:
        """Run the full shared-file discussion loop."""
        session_path = self._archiver.start_session(self._config)
        self._audit = AuditLogger(session_path)

        max_rounds = self._config.max_rounds

        try:
            # Strict tool validation/initialization happens before context or
            # any agent model call. Permissive mode preserves warning-and-skip.
            self._create_conversations()

            # Build initial context
            context = await self._context_mgr.build_initial_context()
            self._archiver.save_context(context)

            # Setup claims.md
            claims_file = os.path.join(session_path, "claims.md")
            claims_mgr = ClaimsManager(claims_file)
            # Extract topic from context (first non-empty line or config)
            topic = context.strip().split("\n")[0] if context.strip() else "讨论议题"
            claims_mgr.topic = topic
            # Round 1: agents propose initial claims
            outputs = await self._round_1(claims_mgr, topic, context)
            claims_mgr.merge_round(outputs)
            self._archiver.save_round(1, "agents", {
                "outputs": [
                    {"agent_name": o.agent_name, "raw_text": o.raw_text}
                    for o in outputs
                ]
            })

            rounds_completed = 1

            # Rounds 2..N: agents respond, then host judges
            for round_num in range(2, max_rounds + 1):
                # Send incremental update to agents
                outputs = await self._round_n(claims_mgr, round_num, round_num - 1)
                claims_mgr.merge_round(outputs)
                self._archiver.save_round(round_num, "agents", {
                    "outputs": [
                        {"agent_name": o.agent_name, "raw_text": o.raw_text}
                        for o in outputs
                    ]
                })
                rounds_completed = round_num

                # Check convergence precondition for this round
                precondition_met = self._check_convergence_precondition(
                    claims_mgr, round_num,
                )

                if precondition_met:
                    # Ask host to judge
                    mature_keywords = {
                        claim.keyword
                        for claim in claims_mgr.get_mature_claims(
                            {ac.name for ac in self._config.agents},
                        )
                    }
                    verdicts = await self._host_judge(claims_mgr, round_num)
                    accepted, rejected = self._apply_host_verdicts(
                        claims_mgr, verdicts, mature_keywords, round_num,
                    )
                    self._archiver.save_round(round_num, "host", {
                        "verdicts": verdicts,
                        "accepted_verdicts": accepted,
                        "rejected_verdicts": rejected,
                    })

                # Check if all claims are closed
                if not claims_mgr.get_open_claims():
                    logger.info("All claims closed. Generating summary.")
                    if self._config.host.skip_summary:
                        summary = None
                    else:
                        summary = await self._host_summarize(
                            claims_mgr, round_num=round_num,
                        )
                        self._archiver.save_summary(summary)
                    return DiscussionResult(
                        converged=True,
                        rounds_completed=rounds_completed,
                        archive_path=session_path,
                        summary=summary,
                        remaining_disputes=[],
                    )

            # Max rounds without full convergence
            open_keywords = [c.keyword for c in claims_mgr.get_open_claims()]
            return DiscussionResult(
                converged=False,
                rounds_completed=rounds_completed,
                archive_path=session_path,
                summary=None,
                remaining_disputes=open_keywords,
            )

        except Exception as exc:
            self._archiver.save_error_log(str(exc))
            return DiscussionResult(
                converged=False,
                rounds_completed=0,
                archive_path=session_path,
                summary=None,
                remaining_disputes=[],
                terminated_by_error=True,
            )

        finally:
            if self._audit:
                self._audit.close()
            try:
                generate_usage_summary(
                    session_path,
                    model_name=self._config.model_config.model,
                    total_rounds=max_rounds,
                )
            except Exception:
                logger.warning("Failed to generate usage summary", exc_info=True)
