"""DiscussionEngine — orchestrates multi-agent adversarial discussion rounds."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import asdict

from agno.agent import Agent

from discuss_agent.config import DiscussionConfig, build_claude
from discuss_agent.context import ContextManager
from discuss_agent.models import AgentUtterance, DiscussionResult, RoundRecord
from discuss_agent.persistence import Archiver
from discuss_agent.registry import import_from_path

logger = logging.getLogger(__name__)


class AllAgentsFailedError(Exception):
    """Raised when every agent fails during a discussion step."""


class DiscussionEngine:
    """Run a structured multi-agent discussion to convergence or max rounds."""

    def __init__(self, config: DiscussionConfig):
        self._config = config
        self._archiver = Archiver()

        # Import global tool classes
        global_tool_entries: list[tuple[str, type]] = []
        for tc in config.tools:
            cls = import_from_path(tc.path)
            global_tool_entries.append((tc.path, cls))

        # Load context builder from config
        context_builder = None
        if config.context_builder:
            context_builder = import_from_path(config.context_builder)

        self._context_mgr = ContextManager(
            config, context_builder=context_builder
        )

        discussion_model = build_claude(config.model_config)

        # Create N discussion agents with per-agent tool sets
        self._agents: list[Agent] = []
        for ac in config.agents:
            # Compute per-agent tool set: global + extra - disabled
            agent_tool_entries = list(global_tool_entries)

            # Add extra tools
            for tc in ac.extra_tools:
                cls = import_from_path(tc.path)
                agent_tool_entries.append((tc.path, cls))

            # Deduplicate by path (keep first occurrence)
            seen: set[str] = set()
            deduped: list[tuple[str, type]] = []
            for path, cls in agent_tool_entries:
                if path not in seen:
                    seen.add(path)
                    deduped.append((path, cls))
            agent_tool_entries = deduped

            # Warn about disable_tools that don't match anything
            available_paths = {p for p, _ in agent_tool_entries}
            for dp in ac.disable_tools:
                if dp not in available_paths:
                    logger.warning(
                        "Agent '%s': disable_tools path '%s' does not match "
                        "any available tool; ignoring.",
                        ac.name, dp,
                    )

            # Remove disabled tools
            disabled = set(ac.disable_tools)
            agent_tool_entries = [
                (p, cls) for p, cls in agent_tool_entries if p not in disabled
            ]

            # Instantiate tools
            tool_instances = [
                cls(context=config.context) for _, cls in agent_tool_entries
            ]

            agent = Agent(
                name=ac.name,
                model=discussion_model,
                system_message=ac.system_prompt,
                tools=tool_instances if tool_instances else None,
                # Prevent unbounded memory growth from accumulated tool results
                store_tool_messages=False,
                add_history_to_context=False,
            )
            self._agents.append(agent)

        # Create Host agent (no tools)
        host_model_config = config.host.resolve_model(config.model_config)
        self._host = Agent(
            name="Host",
            model=build_claude(host_model_config),
            system_message=config.host.convergence_prompt,
            store_tool_messages=False,
            add_history_to_context=False,
        )

    # ------------------------------------------------------------------
    # Agent call with retry
    # ------------------------------------------------------------------

    async def _safe_agent_call(self, agent: Agent, prompt: str) -> str | None:
        """Call *agent* with retry. Returns content or ``None`` on failure."""
        for attempt in range(2):  # 1 retry
            try:
                result = await agent.arun(input=prompt, stream=False)
                if result.content:
                    return result.content
                if attempt == 0:
                    continue  # retry on empty content
                return None
            except Exception:
                if attempt == 0:
                    continue
                return None
        return None

    # ------------------------------------------------------------------
    # History formatting
    # ------------------------------------------------------------------

    def _format_history(self, history: list[RoundRecord]) -> str:
        """Format discussion history for agent context."""
        parts: list[str] = []
        for record in history:
            if record.is_summary:
                parts.append(f"[第{record.round_num}轮摘要] {record.summary_text}")
            else:
                parts.append(f"--- 第{record.round_num}轮 表达 ---")
                for u in record.expressions:
                    parts.append(f"[{u.agent_name}] {u.content}")
                parts.append(f"--- 第{record.round_num}轮 反驳 ---")
                for u in record.challenges:
                    parts.append(f"[{u.agent_name}] {u.content}")
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Step 1: Express
    # ------------------------------------------------------------------

    async def _express(
        self, round_num: int, context: str, history: list[RoundRecord]
    ) -> list[AgentUtterance]:
        """All agents express opinions in parallel."""
        history_text = self._format_history(history)

        limitation_prefix = ""
        if self._config.limitation:
            limitation_prefix = f"⚠️ 本次讨论范围仅限于：{self._config.limitation}\n\n"

        prompt = (
            f"{limitation_prefix}"
            f"{context}\n\n"
            f"{history_text}\n\n"
            f"这是第{round_num}轮讨论。请基于上述背景资料和此前的讨论记录，"
            f"提出你的分析和观点。\n\n"
            f"好的发言应该做到：\n"
            f"- 引用具体的数据、来源或事实来支撑你的论点\n"
            f"- 提出明确的立场，而不是面面俱到的概述\n"
            f"- 如果前几轮讨论中有你认同或反对的观点，直接回应它们\n\n"
            f"如果你需要查阅更多资料来支撑你的观点，请使用可用的工具。"
        )

        async def call_agent(agent: Agent) -> AgentUtterance | None:
            content = await self._safe_agent_call(agent, prompt)
            if content:
                return AgentUtterance(agent_name=agent.name, content=content)
            return None

        results = await asyncio.gather(*[call_agent(a) for a in self._agents])
        utterances = [r for r in results if r is not None]

        if not utterances:
            raise AllAgentsFailedError("All agents failed during express step")
        return utterances

    # ------------------------------------------------------------------
    # Step 2: Challenge
    # ------------------------------------------------------------------

    async def _challenge(
        self, round_num: int, expressions: list[AgentUtterance]
    ) -> list[AgentUtterance]:
        """Each agent challenges OTHER agents' expressions."""

        async def call_agent(agent: Agent) -> AgentUtterance | None:
            # Only show other agents' expressions
            others = [e for e in expressions if e.agent_name != agent.name]
            others_text = "\n\n".join(
                f"[{e.agent_name}] {e.content}" for e in others
            )

            limitation_prefix = ""
            if self._config.limitation:
                limitation_prefix = f"⚠️ 本次讨论范围仅限于：{self._config.limitation}\n\n"

            prompt = (
                f"{limitation_prefix}"
                f"以下是其他讨论者在第{round_num}轮的观点：\n\n"
                f"{others_text}\n\n"
                f"请仔细审视上述观点，找出其中的薄弱环节并提出有针对性的质疑。"
                f"有价值的质疑应该做到：\n"
                f"- 指出论证中的逻辑漏洞、数据缺失或隐含假设\n"
                f"- 提供反面证据或替代解释\n"
                f"- 追问关键细节：具体数据来源、时间窗口、适用范围\n\n"
                f"如果对方的某个论点确实有说服力，也可以明确认可并说明原因——"
                f"承认好的论据比勉强反驳更有建设性。"
            )
            content = await self._safe_agent_call(agent, prompt)
            if content:
                return AgentUtterance(agent_name=agent.name, content=content)
            return None

        results = await asyncio.gather(*[call_agent(a) for a in self._agents])
        utterances = [r for r in results if r is not None]

        if not utterances:
            raise AllAgentsFailedError("All agents failed during challenge step")
        return utterances

    # ------------------------------------------------------------------
    # Host: Judgment
    # ------------------------------------------------------------------

    async def _host_judge(self, history: list[RoundRecord]) -> dict:
        """Host judges convergence. Returns parsed JSON or default not-converged."""
        history_text = self._format_history(history)
        prompt = (
            f"以下是到目前为止的完整讨论记录：\n\n{history_text}\n\n"
            f"请判断这场讨论是否已经收敛。收敛意味着各方在核心问题上已经达成基本共识，"
            f"最近一轮不再出现实质性的新挑战或新论据。\n\n"
            f"判断时请注意：\n"
            f"- 重复已有论点或仅做措辞调整不算新质疑\n"
            f"- 各方观点不必完全一致，只要核心分歧已被充分讨论即可\n"
            f"- 一方明确接受对方论据并调整立场是收敛的强信号\n\n"
            f"请返回以下 JSON 格式：\n"
            f'{{"converged": true/false, "reason": "你的判断理由", '
            f'"remaining_disputes": ["尚未解决的分歧点，如果有"]}}'
        )

        for attempt in range(2):
            try:
                result = await self._host.arun(input=prompt, stream=False)
                content = result.content.strip()
                # Try to extract JSON from the response using regex
                match = re.search(r"\{[^{}]*\"converged\"[^{}]*\}", content)
                if match:
                    judgment = json.loads(match.group())
                    if "converged" in judgment:
                        judgment.setdefault("reason", "")
                        judgment.setdefault("remaining_disputes", [])
                        return judgment
            except Exception:
                pass
            if attempt == 0:
                continue

        # Default: not converged
        return {
            "converged": False,
            "reason": "Host judgment unclear",
            "remaining_disputes": [],
        }

    # ------------------------------------------------------------------
    # Host: Summary
    # ------------------------------------------------------------------

    async def _host_summarize(self, history: list[RoundRecord]) -> str:
        """Host generates final summary after convergence."""
        host_model_config = self._config.host.resolve_model(self._config.model_config)
        summary_agent = Agent(
            name="Host-Summary",
            model=build_claude(host_model_config),
            system_message=self._config.host.summary_prompt,
            store_tool_messages=False,
            add_history_to_context=False,
        )
        history_text = self._format_history(history)
        prompt = (
            f"以下是完整的讨论记录：\n\n{history_text}\n\n"
            f"讨论已经结束。请基于上述记录生成一份结构化的总结报告。\n\n"
            f"你的总结应当包含：\n"
            f"- 核心结论：讨论各方最终达成的共识是什么\n"
            f"- 关键论据：支撑结论的主要证据和推理链条\n"
            f"- 多元视角：不同讨论者带来的差异化观点和贡献\n"
            f"- 未决事项：如果存在未完全解决的分歧，如实记录\n\n"
            f"总结应当忠于讨论内容，不添加讨论中未出现的信息。"
        )
        result = await summary_agent.arun(input=prompt, stream=False)
        return result.content

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self) -> DiscussionResult:
        """Run the full discussion loop."""
        session_path = self._archiver.start_session(self._config)
        context = await self._context_mgr.build_initial_context()
        self._archiver.save_context(context)

        history: list[RoundRecord] = []

        try:
            for round_num in range(1, self._config.max_rounds + 1):
                # Step 1: Express
                expressions = await self._express(round_num, context, history)
                self._archiver.save_round(
                    round_num,
                    "express",
                    {"utterances": [asdict(u) for u in expressions]},
                )

                # Step 2: Challenge
                challenges = await self._challenge(round_num, expressions)
                self._archiver.save_round(
                    round_num,
                    "challenge",
                    {"utterances": [asdict(u) for u in challenges]},
                )

                # Build round record
                record = RoundRecord(
                    round_num=round_num,
                    expressions=expressions,
                    challenges=challenges,
                )

                # Host judgment
                judgment = await self._host_judge(history + [record])
                record.host_judgment = judgment
                self._archiver.save_round(round_num, "host", judgment)

                history.append(record)

                # Compress history
                history = await self._context_mgr.compress(history, round_num)

                # Check convergence (only after min_rounds satisfied)
                if (
                    judgment.get("converged", False)
                    and round_num >= self._config.min_rounds
                ):
                    summary = await self._host_summarize(history)
                    self._archiver.save_summary(summary)
                    return DiscussionResult(
                        converged=True,
                        rounds_completed=round_num,
                        archive_path=session_path,
                        summary=summary,
                        remaining_disputes=judgment.get(
                            "remaining_disputes", []
                        ),
                    )

            # Max rounds reached without convergence
            last_judgment = history[-1].host_judgment if history else {}
            return DiscussionResult(
                converged=False,
                rounds_completed=self._config.max_rounds,
                archive_path=session_path,
                summary=None,
                remaining_disputes=(
                    last_judgment.get("remaining_disputes", [])
                    if last_judgment
                    else []
                ),
            )

        except AllAgentsFailedError as exc:
            self._archiver.save_error_log(str(exc))
            return DiscussionResult(
                converged=False,
                rounds_completed=len(history),
                archive_path=session_path,
                summary=None,
                remaining_disputes=[],
                terminated_by_error=True,
            )
