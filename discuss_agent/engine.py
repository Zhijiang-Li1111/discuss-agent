"""DiscussionEngine — orchestrates multi-agent adversarial discussion rounds."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time as _time
from dataclasses import asdict

from agno.agent import Agent
from agno.skills import Skills
from agno.skills.loaders.local import LocalSkills

from discuss_agent.config import DiscussionConfig, SkillConfig, build_claude
from discuss_agent.context import ContextManager
from discuss_agent.models import AgentUtterance, DiscussionResult, RoundRecord
from discuss_agent.audit import AuditLogger, generate_usage_summary
from discuss_agent.persistence import Archiver
from discuss_agent.registry import import_from_path

logger = logging.getLogger(__name__)


def _build_skills(
    global_skills: list[SkillConfig] | None,
    local_skills: list[SkillConfig] | None,
) -> Skills | None:
    """Build an agno Skills object from global + per-agent skill configs.

    Returns None if no skills are configured.
    """
    all_configs: list[SkillConfig] = []
    if global_skills:
        all_configs.extend(global_skills)
    if local_skills:
        all_configs.extend(local_skills)
    if not all_configs:
        return None
    loaders = [LocalSkills(path=sc.path, validate=False) for sc in all_configs]
    return Skills(loaders=loaders)


class AllAgentsFailedError(Exception):
    """Raised when every agent fails during a discussion step."""


class DiscussionEngine:
    """Run a structured multi-agent discussion to convergence or max rounds."""

    def __init__(self, config: DiscussionConfig):
        self._config = config
        self._archiver = Archiver()
        self._audit: AuditLogger | None = None

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

            # Load skills for this agent (global + per-agent)
            agent_skills = _build_skills(
                config.skills, ac.skills if ac.skills else None
            )

            agent = Agent(
                name=ac.name,
                model=discussion_model,
                system_message=ac.system_prompt,
                tools=tool_instances if tool_instances else None,
                skills=agent_skills,
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
        logger.info("  -> Calling agent '%s' (prompt %d chars)...", agent.name, len(prompt))
        if self._audit:
            start_extras = AuditLogger.extract_call_start_extras(agent)
            self._audit.log_call_start(agent.name, prompt, **start_extras)
        t0 = _time.monotonic()
        for attempt in range(2):  # 1 retry
            try:
                result = await agent.arun(input=prompt, stream=False)
                if result.content:
                    elapsed = (_time.monotonic() - t0) * 1000
                    logger.info("  <- Agent '%s' returned %d chars in %.1fs", agent.name, len(result.content), elapsed/1000)
                    if self._audit:
                        self._audit.log_from_run_output(agent.name, result)
                        end_extras = AuditLogger.extract_call_end_extras(result)
                        self._audit.log_call_end(agent.name, elapsed, result.content, "end_turn", **end_extras)
                    return result.content
                if attempt == 0:
                    continue  # retry on empty content
                elapsed = (_time.monotonic() - t0) * 1000
                if self._audit:
                    self._audit.log_call_end(agent.name, elapsed, None, "empty_content")
                return None
            except Exception as exc:
                if attempt == 0:
                    continue
                elapsed = (_time.monotonic() - t0) * 1000
                if self._audit:
                    self._audit.log_error(agent.name, str(exc), elapsed)
                return None
        elapsed = (_time.monotonic() - t0) * 1000
        if self._audit:
            self._audit.log_call_end(agent.name, elapsed, None, "exhausted_retries")
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
        self, round_num: int, context: str, history: list[RoundRecord],
        guidance: str | None = None,
    ) -> list[AgentUtterance]:
        """All agents express opinions in parallel."""
        logger.info("=== Round %d EXPRESS start (%d agents) ===", round_num, len(self._agents))
        history_text = self._format_history(history)

        limitation_prefix = ""
        if self._config.limitation:
            limitation_prefix = f"⚠️ 本次讨论范围仅限于：{self._config.limitation}\n\n"

        guidance_prefix = ""
        if guidance:
            guidance_prefix = f"📋 主编指导意见（请在讨论中优先回应）：{guidance}\n\n"

        prompt = (
            f"{limitation_prefix}"
            f"{guidance_prefix}"
            f"{context}\n\n"
            f"{history_text}\n\n"
            f"这是第{round_num}轮讨论。请基于上述背景资料和此前的讨论记录，"
            f"提出你的分析和观点。\n\n"
            f"好的发言应该做到：\n"
            f"- 引用具体的数据、来源或事实来支撑你的论点\n"
            f"- 提出明确的立场，而不是面面俱到的概述\n"
            f"- 如果前几轮讨论中有你认同或反对的观点，直接回应它们\n\n"
            f"**格式要求：每个核心论点必须用 ##CLAIM:关键词 [OPEN]## 标记开头。** 例如：\n"
            f"##CLAIM:能繁去化进度 [OPEN]## 当前能繁母猪3904万头...\n"
            f"##CLAIM:牧原成本优势 [OPEN]## 头均完全成本14.5元...\n"
            f"所有新提出的论点初始状态为 OPEN。在后续轮次中，状态会更新为：\n"
            f"- [CHALLENGED] — 已被质疑，等待回应\n"
            f"- [CLOSED:共识] — 各方达成一致\n"
            f"- [CLOSED:分歧] — 讨论充分但无法一致，记录分歧\n\n"
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
        self, round_num: int, expressions: list[AgentUtterance],
        guidance: str | None = None,
    ) -> list[AgentUtterance]:
        """Each agent challenges OTHER agents' expressions."""
        logger.info("=== Round %d CHALLENGE start (%d agents) ===", round_num, len(self._agents))

        async def call_agent(agent: Agent) -> AgentUtterance | None:
            # Extract CLAIM tags as index, fall back to first 800 chars
            others = [e for e in expressions if e.agent_name != agent.name]

            def _extract_claims(content: str) -> str:
                """Extract ##CLAIM:xxx## lines as index."""
                claims = re.findall(r'(##CLAIM:.*?##.*?)(?=\n##CLAIM:|\n\n|$)', content, re.DOTALL)
                if claims:
                    # Return first 200 chars of each claim as index
                    return "\n".join(c[:200] + ("..." if len(c) > 200 else "") for c in claims)
                # Fallback: first 800 chars
                return content[:800] + ("..." if len(content) > 800 else "")

            others_index = "\n\n".join(
                f"[{e.agent_name}]\n{_extract_claims(e.content)}"
                for e in others
            )

            # Tell agent where to find full text
            archive_hint = ""
            if hasattr(self._archiver, '_session_path') and self._archiver._session_path:
                express_file = os.path.join(self._archiver._session_path, "rounds", f"round_{round_num}_express.json")
                if os.path.isfile(express_file):
                    archive_hint = (
                        f"\n\n💡 以上是各方核心论点索引（##CLAIM:关键词##）。"
                        f"完整论证在 {express_file}。"
                        f"你可以用 grep_file('{express_file}', '##CLAIM:') 查看所有论点，"
                        f"然后用 read_file 读取需要反驳的具体段落。\n"
                    )

            limitation_prefix = ""
            if self._config.limitation:
                limitation_prefix = f"⚠️ 本次讨论范围仅限于：{self._config.limitation}\n\n"

            guidance_prefix = ""
            if guidance:
                guidance_prefix = f"📋 主编指导意见（请在质疑中优先关注）：{guidance}\n\n"

            prompt = (
                f"{limitation_prefix}"
                f"{guidance_prefix}"
                f"以下是其他讨论者在第{round_num}轮的核心论点索引：\n\n"
                f"{others_index}\n\n"
                f"{archive_hint}"
                f"## 你的工作流程\n\n"
                f"**第一步：浏览索引。** 上面列出了每个讨论者的核心论点（##CLAIM:关键词##）。选择你最想反驳或质疑的2-3个论点。\n\n"
                f"**第二步：读完整论证。** 对每个要反驳的论点：\n"
                f"  1. 用 grep_file 搜关键词，拿到行号：grep_file('<文件路径>', '##CLAIM:关键词')\n"
                f"  2. 用 read_file 从该行号开始读完整段落：read_file('<文件路径>', offset=行号, limit=30)\n\n"
                f"**第三步：用 research_search 或 web_search 搜索反驳证据。** 不要凭空反驳，必须有数据支撑。\n\n"
                f"**第四步：写出你的质疑。** 针对每个论点：\n"
                f"  - 如果你反驳了它，将其标记为 ##CLAIM:关键词 [CHALLENGED]##\n"
                f"  - 如果你认可它，将其标记为 ##CLAIM:关键词 [CLOSED:共识]##\n"
                f"  - 指出具体的逻辑漏洞、数据缺失或隐含假设，并给出你的替代解释或反面证据。\n\n"
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
        logger.info("=== HOST JUDGE (convergence check) ===")
        history_text = self._format_history(history)
        prompt = (
            f"以下是到目前为止的完整讨论记录：\n\n{history_text}\n\n"
            f"请判断这场讨论是否已经收敛。\n\n"
            f"**关键：检查 ##CLAIM 标签的状态。**\n"
            f"- [OPEN] = 尚未被质疑\n"
            f"- [CHALLENGED] = 已被质疑，等待回应\n"
            f"- [CLOSED:共识] = 已达成一致\n"
            f"- [CLOSED:分歧] = 讨论充分，各方保留分歧\n\n"
            f"收敛条件：所有重要 CLAIM 的状态都是 CLOSED（共识或分歧均可），"
            f"没有 OPEN 或 CHALLENGED 的核心论点还悬而未决。\n\n"
            f"判断时请注意：\n"
            f"- 重复已有论点或仅做措辞调整不算新质疑\n"
            f"- 各方观点不必完全一致，只要核心分歧已被充分讨论即可\n"
            f"- 一方明确接受对方论据并调整立场是收敛的强信号\n\n"
            f"请返回以下 JSON 格式：\n"
            f'{{"converged": true/false, "reason": "你的判断理由", '
            f'"open_claims": ["仍为OPEN/CHALLENGED状态的CLAIM"], '
            f'"remaining_disputes": ["已CLOSED但标记为分歧的论点"]}}'
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
        logger.info("=== HOST SUMMARIZE (generating final report) ===")
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
            f"讨论已经结束。请基于最后一轮各方达成的共识输出总结。\n\n"
            f"关键原则：\n"
            f"- 如果最后一轮中已有完整的结构化输出（如 chapter_memo、分析报告等），"
            f"直接采用该版本的内容，不要混合之前轮次的不同版本\n"
            f"- 如果最后一轮有多个角色输出了不同版本，以最后发言的共识版本为准\n"
            f"- 如果最后一轮没有完整结构化输出，按照 system 指令的格式要求，"
            f"基于全部讨论记录综合提炼\n"
            f"- 如果最后一轮内容不完整或仅有部分角色发言，"
            f"以讨论中最接近共识的最新完整版本为准\n"
            f"- 忠于讨论内容，不添加讨论中未出现的信息"
        )
        result = await summary_agent.arun(input=prompt, stream=False)
        return result.content

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(
        self,
        resume_path: str | None = None,
        extra_rounds: int | None = None,
        guidance: str | None = None,
    ) -> DiscussionResult:
        """Run the full discussion loop.

        Parameters
        ----------
        resume_path:
            Path to an existing archive directory to resume from.
            When set, history and context are loaded from disk instead
            of being generated fresh.
        extra_rounds:
            Number of additional rounds to run after the loaded history.
            Required when *resume_path* is provided.
        guidance:
            Editorial guidance injected into agent prompts to steer
            the direction of the discussion.
        """
        if resume_path is not None:
            if not extra_rounds or extra_rounds < 1:
                raise ValueError("extra_rounds must be a positive integer when resuming")
            session_path = self._archiver.resume_session(resume_path)
            self._audit = AuditLogger(session_path)
            history = self._archiver.load_history()
            context = self._archiver.load_context()
            start_round = len(history) + 1
            max_rounds = len(history) + extra_rounds
        else:
            session_path = self._archiver.start_session(self._config)
            self._audit = AuditLogger(session_path)
            context = await self._context_mgr.build_initial_context()
            self._archiver.save_context(context)
            history = []
            start_round = 1
            max_rounds = self._config.max_rounds

        try:
            for round_num in range(start_round, max_rounds + 1):
                # Step 1: Express
                expressions = await self._express(round_num, context, history, guidance=guidance)
                self._archiver.save_round(
                    round_num,
                    "express",
                    {"utterances": [asdict(u) for u in expressions]},
                )

                # Step 2: Challenge
                challenges = await self._challenge(round_num, expressions, guidance=guidance)
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
                    if self._config.host.skip_summary:
                        summary = None
                    else:
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
                rounds_completed=max_rounds,
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

        finally:
            if self._audit:
                self._audit.close()
            # Generate usage summary after all rounds complete
            try:
                generate_usage_summary(
                    session_path,
                    model_name=self._config.model_config.model,
                    total_rounds=len(history),
                )
            except Exception:
                logger.warning(
                    "Failed to generate usage summary", exc_info=True
                )
