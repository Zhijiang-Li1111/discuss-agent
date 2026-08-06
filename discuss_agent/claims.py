"""ClaimsManager — parse, merge, and manage the shared claims.md file."""

from __future__ import annotations

import re
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
            return f"[HOST @R{self.round_num}] {self.content}"
        prefix = self.entry_type
        if prefix == "FROM":
            return f"[FROM:{self.agent_name} @R{self.round_num}] {self.content}"
        # REBUTTAL, ACCEPT, RESPONSE
        return f"[{prefix} FROM:{self.agent_name} @R{self.round_num}] {self.content}"


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
        parts = [part.strip() for part in re.split(r"[、,，;；]", normalized) if part.strip()]
        if len(parts) <= 1:
            return [], [normalized]
        return (
            [part for part in parts if part in known_targets],
            [part for part in parts if part not in known_targets],
        )

    def parse_claims_file(self) -> None:
        """Parse an existing claims.md file into memory."""
        if not self.claims_file or not Path(self.claims_file).exists():
            return
        text = Path(self.claims_file).read_text(encoding="utf-8")
        self._parse_text(text)

    def _parse_text(self, text: str) -> None:
        """Parse claims.md content into structured data."""
        self.claims.clear()
        current_claim: Claim | None = None
        multiline_buffer: list[str] = []
        pending_entry_meta: tuple | None = None  # (entry_type, agent, round)

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

    def merge_round(self, agent_outputs: list[AgentOutput]) -> None:
        """Merge all agent outputs from a round into the claims structure.

        The program automatically adds FROM tags — agents don't need to include them.
        """
        for output in agent_outputs:
            for response in output.parsed:
                if response.response_type == "NEW_CLAIM":
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
        """Return OPEN claims every expected peer has answered at least once."""
        mature: list[Claim] = []
        for claim in self.get_open_claims():
            proposer = self.proposer_for(claim)
            expected = all_agent_names - {proposer} if proposer else all_agent_names
            responding = {
                entry.agent_name
                for entry in claim.entries
                if entry.entry_type in {"ACCEPT", "REBUTTAL", "RESPONSE"}
            }
            if expected.issubset(responding):
                mature.append(claim)
        return mature

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
                has_responded = agent_name is not None and any(
                    entry.agent_name == agent_name
                    and entry.entry_type in {"ACCEPT", "REBUTTAL", "RESPONSE"}
                    for entry in claim.entries
                )
                if agent_name is None:
                    needs_response_text.append(claim.format())
                elif claim.keyword in mature_keywords:
                    awaiting_host.append(f"- CLAIM:{claim.keyword}")
                elif agent_name == proposer and not any(
                    entry.entry_type == "REBUTTAL" for entry in claim.entries
                ):
                    awaiting_host.append(f"- CLAIM:{claim.keyword}（你提出；暂无反驳）")
                elif has_responded:
                    awaiting_host.append(f"- CLAIM:{claim.keyword}（你已回应）")
                else:
                    needs_response_text.append(claim.format())
                if has_recent_new and len(claim.entries) == 1:
                    status_changes.append(f"- [新增] CLAIM:{claim.keyword}")
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
        p.write_text(text, encoding="utf-8")

    def format_file(self) -> str:
        """Render the full claims.md content."""
        parts = ["# 讨论主文件\n"]
        if self.topic:
            parts.append(f"## 议题\n{self.topic}\n")
        parts.append("---\n")
        for claim in self.claims.values():
            parts.append(claim.format())
        return "\n".join(parts)
