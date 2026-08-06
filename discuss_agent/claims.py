"""ClaimsManager — parse, merge, and manage the shared claims.md file."""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ClaimEntry:
    """A single entry (statement, rebuttal, accept, host verdict) within a claim."""

    entry_type: str  # "FROM", "REBUTTAL", "ACCEPT", "RESPONSE", "HOST"
    agent_name: str  # agent name or "HOST"
    round_num: int
    content: str

    def format(self) -> str:
        """Render this entry as a tagged line for claims.md."""
        if self.entry_type == "HOST":
            tag = f"[HOST @R{self.round_num}]"
        elif self.entry_type == "FROM":
            tag = f"[FROM:{self.agent_name} @R{self.round_num}]"
        else:
            tag = f"[{self.entry_type} FROM:{self.agent_name} @R{self.round_num}]"
        lines = self.content.split("\n")
        return f"{tag} {lines[0]}" + "".join(
            f"\n    {line}" for line in lines[1:]
        )


@dataclass
class Claim:
    """A single CLAIM with its status and history of entries."""

    keyword: str
    status: str  # "OPEN", "CLOSED:共识", "CLOSED:分歧"
    entries: list[ClaimEntry] = field(default_factory=list)

    def add_entry(self, entry: ClaimEntry) -> None:
        self.entries.append(entry)

    def format(self) -> str:
        """Render this claim as a block for claims.md."""
        lines = [f"##CLAIM:{self.keyword} [{self.status}]##"]
        for entry in self.entries:
            formatted = entry.format()
            # Indent non-initial entries for readability
            if entry.entry_type != "FROM":
                formatted = "  " + formatted
            lines.append(formatted)
        return "\n".join(lines)


# Regex patterns for parsing claims.md
_CLAIM_HEADER_RE = re.compile(r"^##CLAIM:(.+?)\s*\[(.+?)\]##\s*$")
_ENTRY_RE = re.compile(
    r"^\s*\[(?:(\w+)\s+)?FROM:(.+?)\s+@R(\d+)\]\s*(.*)$"
)
_HOST_ENTRY_RE = re.compile(r"^\s*\[HOST\s+@R(\d+)\]\s*(.*)$")
_UNMATCHED_HEADER = "##UNMATCHED_RESPONSES##"

# Regex for parsing agent outputs
_REBUTTAL_RE = re.compile(r"\[REBUTTAL\s+TO:(.+?)\]\s*(.*)", re.DOTALL)
_ACCEPT_RE = re.compile(r"\[ACCEPT\s+TO:(.+?)\]\s*(.*)", re.DOTALL)
_NEW_CLAIM_RE = re.compile(r"\[NEW_CLAIM:(.+?)\]\s*(.*)", re.DOTALL)


@dataclass
class ParsedResponse:
    """A single parsed response from an agent's output."""

    response_type: str  # "REBUTTAL", "ACCEPT", "NEW_CLAIM"
    target: str  # claim keyword (for REBUTTAL/ACCEPT) or new keyword
    content: str


def parse_agent_output(text: str) -> list[ParsedResponse]:
    """Parse an agent's raw output into structured responses."""
    responses: list[ParsedResponse] = []
    # Split by response markers
    parts = re.split(r"(?=\[(?:REBUTTAL TO|ACCEPT TO|NEW_CLAIM):)", text)
    for part in parts:
        part = part.strip()
        if not part:
            continue
        m = _REBUTTAL_RE.match(part)
        if m:
            responses.append(ParsedResponse("REBUTTAL", m.group(1).strip(), m.group(2).strip()))
            continue
        m = _ACCEPT_RE.match(part)
        if m:
            responses.append(ParsedResponse("ACCEPT", m.group(1).strip(), m.group(2).strip()))
            continue
        m = _NEW_CLAIM_RE.match(part)
        if m:
            responses.append(ParsedResponse("NEW_CLAIM", m.group(1).strip(), m.group(2).strip()))
            continue
    return responses


@dataclass
class AgentOutput:
    """Container for one agent's output in a round."""

    agent_name: str
    round_num: int
    raw_text: str
    parsed: list[ParsedResponse] = field(default_factory=list)

    def __post_init__(self):
        if not self.parsed:
            self.parsed = parse_agent_output(self.raw_text)


class ClaimsManager:
    """Manage the shared claims.md file — the single source of truth."""

    def __init__(self, claims_file: str | None = None):
        self.claims_file = claims_file
        self.claims: dict[str, Claim] = {}
        self.topic: str = ""
        self.current_round: int = 0
        self.unmatched_responses: list[dict[str, object]] = []

    @staticmethod
    def _resolve_response_targets(target: str, known_targets: set[str]) -> tuple[list[str], list[str]]:
        """Resolve one response marker to exact claim keywords.

        Exact matches win so punctuation inside a legitimate keyword is kept.
        Otherwise split the separators agents commonly use for compact targets.
        Unresolved targets are returned for an audit trail instead of vanishing.
        """
        normalized = target.strip()
        if normalized in known_targets:
            return [normalized], []

        separators = "、,，;；"
        ordered_targets = sorted(known_targets, key=len, reverse=True)
        matched: list[str] = []
        unmatched: list[str] = []
        buffer: list[str] = []
        position = 0
        at_boundary = True

        def flush_buffer() -> None:
            value = "".join(buffer).strip()
            if value:
                unmatched.append(value)
            buffer.clear()

        while position < len(normalized):
            candidate = None
            if at_boundary:
                candidate = next(
                    (
                        keyword for keyword in ordered_targets
                        if normalized.startswith(keyword, position)
                        and (
                            position + len(keyword) == len(normalized)
                            or normalized[position + len(keyword)] in separators
                        )
                    ),
                    None,
                )
            if candidate:
                flush_buffer()
                matched.append(candidate)
                position += len(candidate)
                at_boundary = False
                continue
            char = normalized[position]
            if char in separators:
                flush_buffer()
                at_boundary = True
            else:
                buffer.append(char)
                if not char.isspace():
                    at_boundary = False
            position += 1
        flush_buffer()
        return list(dict.fromkeys(matched)), list(dict.fromkeys(unmatched))

    def parse_claims_file(self) -> None:
        """Parse an existing claims.md file into memory."""
        if not self.claims_file or not Path(self.claims_file).exists():
            return
        text = Path(self.claims_file).read_text(encoding="utf-8")
        self._parse_text(text)

    def _parse_text(self, text: str) -> None:
        """Parse claims.md content into structured data."""
        self.claims.clear()
        self.unmatched_responses.clear()
        current_claim: Claim | None = None
        multiline_buffer: list[str] = []
        pending_entry_meta: tuple | None = None  # (entry_type, agent, round)
        parsing_unmatched = False

        def _flush_multiline():
            nonlocal pending_entry_meta, multiline_buffer
            if pending_entry_meta and current_claim is not None:
                entry_type, agent, rnd = pending_entry_meta
                content = "\n".join(multiline_buffer).strip()
                if content:
                    current_claim.add_entry(
                        ClaimEntry(entry_type, agent, rnd, content)
                    )
            multiline_buffer = []
            pending_entry_meta = None

        for line in text.split("\n"):
            if line == _UNMATCHED_HEADER:
                _flush_multiline()
                current_claim = None
                parsing_unmatched = True
                continue
            if parsing_unmatched:
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(item, dict):
                    self.unmatched_responses.append(item)
                continue
            if pending_entry_meta and line.startswith("    "):
                multiline_buffer.append(line[4:])
                continue

            # Check for topic
            if line.startswith("## 议题"):
                continue
            # Extract topic from line after "## 议题"
            if self.topic == "" and line.strip() and not line.startswith("#") and not line.startswith("---"):
                # Could be topic line
                pass

            # Check for claim header
            m = _CLAIM_HEADER_RE.match(line)
            if m:
                _flush_multiline()
                keyword = m.group(1).strip()
                status = m.group(2).strip()
                current_claim = Claim(keyword=keyword, status=status)
                self.claims[keyword] = current_claim
                continue

            if current_claim is None:
                # Check for topic
                if line.strip() and not line.startswith("#") and line.strip() != "---":
                    if not self.topic:
                        self.topic = line.strip()
                continue

            # Check for HOST entry
            m_host = _HOST_ENTRY_RE.match(line)
            if m_host:
                _flush_multiline()
                rnd = int(m_host.group(1))
                pending_entry_meta = ("HOST", "HOST", rnd)
                multiline_buffer = [m_host.group(2)]
                continue

            # Check for regular entry
            m_entry = _ENTRY_RE.match(line)
            if m_entry:
                _flush_multiline()
                entry_type = m_entry.group(1) or "FROM"
                agent = m_entry.group(2).strip()
                rnd = int(m_entry.group(3))
                content_start = m_entry.group(4)
                pending_entry_meta = (entry_type, agent, rnd)
                multiline_buffer = [content_start]
                continue

            # Continuation line for current entry
            if pending_entry_meta:
                multiline_buffer.append(line)

        _flush_multiline()
        claim_round = max(
            (
                entry.round_num
                for claim in self.claims.values()
                for entry in claim.entries
            ),
            default=0,
        )
        unmatched_round = max(
            (int(item["round_num"]) for item in self.unmatched_responses),
            default=0,
        )
        self.current_round = max(claim_round, unmatched_round)

    def merge_round(self, agent_outputs: list[AgentOutput]) -> None:
        """Merge all agent outputs from a round into the claims structure.

        The program automatically adds FROM tags — agents don't need to include them.
        """
        for output in agent_outputs:
            for response in output.parsed:
                if response.response_type == "NEW_CLAIM":
                    if response.target in self.claims:
                        self.unmatched_responses.append({
                            "agent_name": output.agent_name,
                            "round_num": output.round_num,
                            "response_type": "DUPLICATE_NEW_CLAIM",
                            "target": response.target,
                            "content": response.content,
                        })
                        continue
                    # Create new claim with FROM tag
                    claim = Claim(
                        keyword=response.target,
                        status="OPEN",
                    )
                    claim.add_entry(ClaimEntry(
                        entry_type="FROM",
                        agent_name=output.agent_name,
                        round_num=output.round_num,
                        content=response.content,
                    ))
                    self.claims[response.target] = claim
                elif response.response_type in {"REBUTTAL", "ACCEPT"}:
                    targets, unmatched = self._resolve_response_targets(
                        response.target, set(self.claims),
                    )
                    for target in targets:
                        self.claims[target].add_entry(ClaimEntry(
                            entry_type=response.response_type,
                            agent_name=output.agent_name,
                            round_num=output.round_num,
                            content=response.content,
                        ))
                    for target in unmatched:
                        self.unmatched_responses.append({
                            "agent_name": output.agent_name,
                            "round_num": output.round_num,
                            "response_type": response.response_type,
                            "target": target,
                        })
        self.current_round = max(
            (o.round_num for o in agent_outputs), default=self.current_round
        )
        self.save()

    def get_open_claims(self) -> list[Claim]:
        """Return all claims with OPEN status."""
        return [c for c in self.claims.values() if c.status == "OPEN"]

    @staticmethod
    def proposer_for(claim: Claim) -> str | None:
        """Return the original proposer for *claim*, if present."""
        return next(
            (entry.agent_name for entry in claim.entries if entry.entry_type == "FROM"),
            None,
        )

    def get_mature_claims(self, all_agent_names: set[str]) -> list[Claim]:
        """Return OPEN claims with every currently required response present."""
        return [
            claim for claim in self.get_open_claims()
            if not self.agents_needing_response(claim, all_agent_names)
        ]

    def agents_needing_response(
        self, claim: Claim, all_agent_names: set[str],
    ) -> set[str]:
        """Return agents that must respond after the latest substantive event."""
        proposer = self.proposer_for(claim)
        initial = next(
            (entry for entry in claim.entries if entry.entry_type == "FROM"),
            None,
        )
        responses = [
            entry for entry in claim.entries
            if entry.entry_type in {"ACCEPT", "REBUTTAL", "RESPONSE"}
        ]
        triggers = [
            entry for entry in claim.entries
            if entry.entry_type == "REBUTTAL"
            or (
                entry.entry_type == "HOST"
                and entry.content.startswith("CONTINUE：")
            )
        ]

        if not triggers:
            required = all_agent_names - ({proposer} if proposer else set())
            initial_round = initial.round_num if initial else -1
            responded = {
                entry.agent_name for entry in responses
                if entry.round_num > initial_round
            }
            return required - responded

        latest_round = max(entry.round_num for entry in triggers)
        latest = [entry for entry in triggers if entry.round_num == latest_round]
        if any(entry.entry_type == "HOST" for entry in latest):
            required = set(all_agent_names)
            responded = {
                entry.agent_name for entry in responses
                if entry.round_num > latest_round
            }
        else:
            rebutters = {entry.agent_name for entry in latest}
            required = set(all_agent_names)
            responded = (rebutters if len(rebutters) == 1 else set()) | {
                entry.agent_name for entry in responses
                if entry.round_num > latest_round
            }
        return required - responded

    @staticmethod
    def _format_response_context(claim: Claim) -> str:
        """Render the original claim plus entries in the latest dispute epoch."""
        initial = next(
            (entry for entry in claim.entries if entry.entry_type == "FROM"),
            None,
        )
        triggers = [
            entry for entry in claim.entries
            if entry.entry_type == "REBUTTAL"
            or (
                entry.entry_type == "HOST"
                and entry.content.startswith("CONTINUE：")
            )
        ]
        entries = [initial] if initial else []
        if triggers:
            latest_round = max(entry.round_num for entry in triggers)
            entries.extend(
                entry for entry in claim.entries
                if entry is not initial and entry.round_num >= latest_round
            )
        return Claim(claim.keyword, claim.status, entries).format()

    def close_claim(self, keyword: str, verdict: str, reason: str, round_num: int) -> None:
        """Host closes a claim with a verdict."""
        if keyword not in self.claims:
            return
        claim = self.claims[keyword]
        claim.status = f"CLOSED:{verdict}"
        claim.add_entry(ClaimEntry(
            entry_type="HOST",
            agent_name="HOST",
            round_num=round_num,
            content=f"裁决：{reason}",
        ))
        self.save()

    def continue_claim(self, keyword: str, reason: str, round_num: int) -> None:
        """Record a Host CONTINUE decision and reopen responses for all agents."""
        claim = self.claims.get(keyword)
        if claim is None or claim.status != "OPEN":
            return
        claim.add_entry(ClaimEntry(
            entry_type="HOST",
            agent_name="HOST",
            round_num=round_num,
            content=f"CONTINUE：{reason}",
        ))
        self.current_round = max(self.current_round, round_num)
        self.save()

    def generate_update_prompt(
        self,
        prev_round: int,
        *,
        agent_name: str | None = None,
        all_agent_names: set[str] | None = None,
    ) -> str:
        """Generate incremental update prompt for agents.

        - OPEN claims: full content included (with FROM tags)
        - CLOSED/new claims since prev_round: only status change notification
        """
        status_changes: list[str] = []
        needs_response_text: list[str] = []
        awaiting_host: list[str] = []
        mature_keywords = (
            {claim.keyword for claim in self.get_mature_claims(all_agent_names)}
            if all_agent_names
            else set()
        )

        for claim in self.claims.values():
            # Check if status changed since prev_round
            has_recent_host = any(
                e.entry_type == "HOST" and e.round_num > prev_round
                for e in claim.entries
            )
            has_recent_new = any(
                e.entry_type == "FROM" and e.round_num > prev_round
                for e in claim.entries
            ) and claim.status == "OPEN"

            if claim.status == "OPEN":
                proposer = self.proposer_for(claim)
                needs_response = (
                    agent_name in self.agents_needing_response(
                        claim, all_agent_names,
                    )
                    if agent_name is not None and all_agent_names is not None
                    else False
                )
                if agent_name is None:
                    needs_response_text.append(claim.format())
                elif needs_response:
                    needs_response_text.append(self._format_response_context(claim))
                elif claim.keyword in mature_keywords:
                    awaiting_host.append(f"- CLAIM:{claim.keyword}")
                elif agent_name == proposer and not any(
                    entry.entry_type == "REBUTTAL" for entry in claim.entries
                ):
                    awaiting_host.append(f"- CLAIM:{claim.keyword}（你提出；暂无反驳）")
                else:
                    awaiting_host.append(f"- CLAIM:{claim.keyword}（你已回应）")
                if has_recent_new and len(claim.entries) == 1:
                    status_changes.append(f"- [新增] CLAIM:{claim.keyword}")
                if has_recent_host:
                    status_changes.append(f"- [继续讨论] CLAIM:{claim.keyword}")
            elif has_recent_host:
                status_changes.append(
                    f"- [已关闭] CLAIM:{claim.keyword} — {claim.status}"
                )

        parts = [f"## 第{self.current_round}轮更新"]

        if status_changes:
            parts.append("\n状态变化：")
            parts.extend(status_changes)

        if needs_response_text:
            parts.append("\n## NEEDS_YOUR_RESPONSE\n")
            parts.extend(needs_response_text)
        if awaiting_host:
            parts.append("\n## AWAITING_HOST（无需重复回应）")
            parts.extend(awaiting_host)

        recent_unmatched = [
            item for item in self.unmatched_responses
            if int(item["round_num"]) > prev_round
            and (agent_name is None or item["agent_name"] == agent_name)
        ]
        if recent_unmatched:
            parts.append("\n## 上轮未匹配的target（请改用单个精确关键词）")
            parts.extend(f"- {item['target']}" for item in recent_unmatched)

        parts.append(
            "\n## 你的任务\n"
            "只回应 NEEDS_YOUR_RESPONSE；AWAITING_HOST 不要重复回应。\n"
            "- [REBUTTAL TO:关键词] 反驳 + 证据\n"
            "- [ACCEPT TO:关键词] 接受 + 理由\n"
            "- 每个 ACCEPT/REBUTTAL marker 只能写一个精确关键词，不得批量列出。\n"
            "- [NEW_CLAIM:关键词] 仅限新证据、新反证或新的实质UNKNOWN；不要为状态改名另开claim。\n"
            "可以随时用 research_search 或 web_search 搜索证据。"
        )

        return "\n".join(parts)

    def generate_initial_prompt(
        self,
        topic: str,
        limitation: str | None = None,
        context: str | None = None,
    ) -> str:
        """Generate the initial prompt for Round 1."""
        parts = []
        if limitation:
            parts.append(f"⚠️ 本次讨论范围仅限于：{limitation}\n")
        parts.append(f"议题：{topic}\n")
        if context:
            parts.append(f"## 已确认背景材料\n\n{context.strip()}\n")
        parts.append(
            "请基于你的专业分析提出观点。\n\n"
            "**输出格式要求：**\n"
            "- [NEW_CLAIM:关键词] 你的论点和证据\n"
            "- 每个独立论点用一个 NEW_CLAIM 标记\n"
            "- 可以随时用 research_search 或 web_search 搜索证据。"
        )
        return "\n".join(parts)

    def save(self) -> None:
        """Write the current state to claims.md."""
        if not self.claims_file:
            return
        p = Path(self.claims_file)
        p.parent.mkdir(parents=True, exist_ok=True)
        text = self.format_file()
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=p.parent,
            prefix=f".{p.name}.",
            delete=False,
        ) as temp_file:
            temp_file.write(text)
            temp_file.flush()
            os.fsync(temp_file.fileno())
            temp_name = temp_file.name
        os.replace(temp_name, p)

    def format_file(self) -> str:
        """Render the full claims.md content."""
        parts = ["# 讨论主文件\n"]
        if self.topic:
            parts.append(f"## 议题\n{self.topic}\n")
        parts.append("---\n")
        for claim in self.claims.values():
            parts.append(claim.format())
        if self.unmatched_responses:
            parts.append(_UNMATCHED_HEADER)
            parts.extend(
                json.dumps(item, ensure_ascii=False, sort_keys=True)
                for item in self.unmatched_responses
            )
        return "\n".join(parts)
