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

from discuss_agent.claims import AgentOutput, ClaimsManager
from discuss_agent.config import DiscussionConfig, ModelConfig, build_claude
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
                cls = import_from_path(tc.path)
                toolkit = cls()  # instantiate the Toolkit
                # Collect sync functions
                for name, func in toolkit.functions.items():
                    tool_defs.append({
                        "name": name,
                        "description": func.description or "",
                        "input_schema": func.parameters or {"type": "object", "properties": {}},
                    })
                    tool_callables[name] = func.entrypoint
                # Collect async functions
                for name, func in toolkit.async_functions.items():
                    if name not in tool_callables:
                        tool_defs.append({
                            "name": name,
                            "description": func.description or "",
                            "input_schema": func.parameters or {"type": "object", "properties": {}},
                        })
                    # Prefer async over sync
                    tool_callables[name] = func.entrypoint
            except Exception:
                logger.warning("Failed to load tool %s", tc.path, exc_info=True)

        return tool_defs, tool_callables

    def _resolve_agent_tools(
        self, ac, global_defs: list[dict], global_callables: dict[str, callable],
    ) -> tuple[list[dict], dict[str, callable]]:
        """Resolve per-agent tool set (global + extra - disabled)."""
        defs = list(global_defs)
        callables = dict(global_callables)

        # Add extra tools
        for tc in ac.extra_tools:
            try:
                cls = import_from_path(tc.path)
                toolkit = cls()
                for name, func in toolkit.functions.items():
                    defs.append({
                        "name": name,
                        "description": func.description or "",
                        "input_schema": func.parameters or {"type": "object", "properties": {}},
                    })
                    callables[name] = func.entrypoint
                for name, func in toolkit.async_functions.items():
                    if name not in callables or name not in [d["name"] for d in defs]:
                        defs.append({
                            "name": name,
                            "description": func.description or "",
                            "input_schema": func.parameters or {"type": "object", "properties": {}},
                        })
                    callables[name] = func.entrypoint
            except Exception:
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
            )

    async def _call_agent(self, agent_name: str, prompt: str) -> str | None:
        """Send a message to an agent, with retry on failure."""
        conv = self._conversations[agent_name]
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
        self, claims_mgr: ClaimsManager, topic: str,
    ) -> list[AgentOutput]:
        """Round 1: each agent proposes initial claims."""
        logger.info("=== Round 1: Initial claims ===")
        prompt = claims_mgr.generate_initial_prompt(
            topic, limitation=self._config.limitation,
        )

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
        prompt = claims_mgr.generate_update_prompt(prev_round)

        async def call_one(name: str) -> AgentOutput | None:
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
        """Check if all OPEN claims have been responded to by all agents."""
        open_claims = claims_mgr.get_open_claims()
        all_agent_names = {ac.name for ac in self._config.agents}

        for claim in open_claims:
            responding_agents = {
                e.agent_name
                for e in claim.entries
                if e.round_num == round_num and e.entry_type != "FROM"
            }
            # The original proposer doesn't need to respond to their own claim
            proposer = None
            for e in claim.entries:
                if e.entry_type == "FROM":
                    proposer = e.agent_name
                    break
            expected = all_agent_names - {proposer} if proposer else all_agent_names
            if not expected.issubset(responding_agents | {proposer}):
                return False
        return True

    async def _host_judge(self, claims_mgr: ClaimsManager, round_num: int) -> list[dict]:
        """Host LLM judges each OPEN claim.

        Returns list of {claim, verdict, reason} dicts.
        """
        logger.info("=== HOST JUDGE (Round %d) ===", round_num)
        open_claims = claims_mgr.get_open_claims()
        if not open_claims:
            return []

        claims_text = "\n\n".join(c.format() for c in open_claims)
        prompt = (
            f"以下是当前所有 OPEN claims 的讨论记录：\n\n{claims_text}\n\n"
            f"请对每个 claim 裁决：\n"
            f"- CLOSED:共识 — 各方达成一致\n"
            f"- CLOSED:分歧 — 讨论充分但立场不同，记录分歧\n"
            f"- CONTINUE — 仍需讨论\n\n"
            f'输出 JSON 数组: [{{"claim": "关键词", "verdict": "CLOSED:共识", "reason": "..."}}]'
        )

        # Use a one-shot Anthropic call for host (no multi-turn needed)
        client_kwargs: dict = {"timeout": 600.0}
        if self._host_model_config.api_key:
            client_kwargs["api_key"] = self._host_model_config.api_key
        if self._host_model_config.base_url:
            client_kwargs["base_url"] = self._host_model_config.base_url
        client = anthropic.AsyncAnthropic(**client_kwargs)

        kwargs: dict = {
            "model": self._host_model_config.model,
            "max_tokens": self._host_model_config.max_tokens or 4096,
            "system": self._config.host.convergence_prompt,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self._host_model_config.temperature is not None:
            kwargs["temperature"] = self._host_model_config.temperature

        for attempt in range(2):
            try:
                response = await client.messages.create(**kwargs)
                text = ""
                for block in response.content:
                    if block.type == "text":
                        text += block.text

                # Extract JSON array from response
                match = re.search(r"\[.*\]", text, re.DOTALL)
                if match:
                    verdicts = json.loads(match.group())
                    return verdicts
            except Exception:
                if attempt == 0:
                    continue
        return []

    async def _host_summarize(self, claims_mgr: ClaimsManager) -> str:
        """Host generates final summary."""
        logger.info("=== HOST SUMMARIZE ===")
        all_claims_text = claims_mgr.format_file()

        client_kwargs: dict = {"timeout": 600.0}
        if self._host_model_config.api_key:
            client_kwargs["api_key"] = self._host_model_config.api_key
        if self._host_model_config.base_url:
            client_kwargs["base_url"] = self._host_model_config.base_url
        client = anthropic.AsyncAnthropic(**client_kwargs)

        prompt = (
            f"以下是完整的讨论记录：\n\n{all_claims_text}\n\n"
            f"讨论已经结束。请基于各方达成的共识和记录的分歧输出总结。"
        )

        response = await client.messages.create(
            model=self._host_model_config.model,
            max_tokens=self._host_model_config.max_tokens or 4096,
            system=self._config.host.summary_prompt,
            messages=[{"role": "user", "content": prompt}],
        )
        text_parts = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
        return "\n".join(text_parts)

    async def run(self) -> DiscussionResult:
        """Run the full shared-file discussion loop."""
        session_path = self._archiver.start_session(self._config)
        self._audit = AuditLogger(session_path)

        # Build initial context
        context = await self._context_mgr.build_initial_context()
        self._archiver.save_context(context)

        # Setup claims.md
        claims_file = os.path.join(session_path, "claims.md")
        claims_mgr = ClaimsManager(claims_file)
        # Extract topic from context (first non-empty line or config)
        topic = context.strip().split("\n")[0] if context.strip() else "讨论议题"
        claims_mgr.topic = topic

        # Create agent conversations
        self._create_conversations()

        max_rounds = self._config.max_rounds

        try:
            # Round 1: agents propose initial claims
            outputs = await self._round_1(claims_mgr, topic)
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
                    verdicts = await self._host_judge(claims_mgr, round_num)
                    for v in verdicts:
                        kw = v.get("claim", "")
                        verdict = v.get("verdict", "CONTINUE")
                        reason = v.get("reason", "")
                        if verdict.startswith("CLOSED"):
                            verdict_type = verdict.replace("CLOSED:", "")
                            claims_mgr.close_claim(kw, verdict_type, reason, round_num)
                    self._archiver.save_round(round_num, "host", {"verdicts": verdicts})

                # Check if all claims are closed
                if not claims_mgr.get_open_claims():
                    logger.info("All claims closed. Generating summary.")
                    if self._config.host.skip_summary:
                        summary = None
                    else:
                        summary = await self._host_summarize(claims_mgr)
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
