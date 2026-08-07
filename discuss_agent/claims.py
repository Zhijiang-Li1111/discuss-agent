"""ClaimsManager — parse, merge, and manage the shared claims.md file."""

from __future__ import annotations

import json
import hashlib
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ClaimEntry:
    """A single entry (statement, rebuttal, accept, host verdict) within a claim."""

    entry_type: str  # FROM, participant response type, or HOST
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
_CLAIM_HEADER_RE = re.compile(r"^##CLAIM:(.+)\s+\[([^\[\]]+)\]##\s*$")
_ENTRY_RE = re.compile(
    r"^\s*\[(?:(\w+)\s+)?FROM:(.+?)\s+@R(\d+)\][ ]?(.*)$"
)
_HOST_ENTRY_RE = re.compile(r"^\s*\[HOST\s+@R(\d+)\][ ]?(.*)$")
_UNMATCHED_HEADER = "##UNMATCHED_RESPONSES##"
_NEEDS_AGENTS_PREFIX = "NEEDS_AGENTS："
_MISSING_PREFIX = "MISSING："
_ROUTING_PREFIX = "ROUTING_JSON："
_CRITICAL_HISTORY_TYPES = (
    "REBUTTAL",
    "REVISE",
    "PARTIAL_ACCEPT",
    "REOPEN_REQUEST",
)
_STRUCTURED_RESPONSE_TYPES = ("REBUTTAL", "ACCEPT", "PARTIAL_ACCEPT", "REVISE")

# Regex for parsing agent outputs
_MARKER_TARGET = r"((?:[^\[\]\n]|\[[^\[\]\n]*\])*)"
_REBUTTAL_RE = re.compile(
    rf"\[REBUTTAL\s+TO:{_MARKER_TARGET}\]\s*(.*)", re.DOTALL,
)
_ACCEPT_RE = re.compile(
    rf"\[ACCEPT\s+TO:{_MARKER_TARGET}\]\s*(.*)", re.DOTALL,
)
_PARTIAL_ACCEPT_RE = re.compile(
    rf"\[PARTIAL_ACCEPT\s+TO:{_MARKER_TARGET}\]\s*(.*)", re.DOTALL,
)
_REVISE_RE = re.compile(
    rf"\[REVISE\s+TO:{_MARKER_TARGET}\]\s*(.*)", re.DOTALL,
)
_NEW_CLAIM_RE = re.compile(
    rf"\[NEW_CLAIM:{_MARKER_TARGET}\]\s*(.*)", re.DOTALL,
)
_REOPEN_REQUEST_RE = re.compile(
    rf"\[REOPEN_REQUEST\s+TO:{_MARKER_TARGET}\]\s*(.*)", re.DOTALL,
)
_INDENTED_MARKER_RE = re.compile(
    r"^[ \t]+\[(?:REBUTTAL TO|ACCEPT TO|PARTIAL_ACCEPT TO|REVISE TO|"
    r"REOPEN_REQUEST TO|NEW_CLAIM):",
)


@dataclass
class ParsedResponse:
    """A single parsed response from an agent's output."""

    response_type: str
    target: str  # existing claim keyword, or new keyword for NEW_CLAIM
    content: str


@dataclass(frozen=True)
class ReopenRequest:
    """A participant request to semantically re-examine one closed claim."""

    request_id: str
    claim_keyword: str
    agent_name: str
    round_num: int
    kind: str
    fact: str
    source: str
    reason: str


def parse_agent_output(text: str) -> list[ParsedResponse]:
    """Parse an agent's raw output into structured responses."""
    responses: list[ParsedResponse] = []
    text = "\n".join(
        line
        for line in text.splitlines()
        if not _INDENTED_MARKER_RE.match(line)
    )
    parts = re.split(
        r"(?m)(?=^\[(?:REBUTTAL TO|ACCEPT TO|PARTIAL_ACCEPT TO|REVISE TO|"
        r"REOPEN_REQUEST TO|NEW_CLAIM):)",
        text,
    )
    for part in parts:
        part = part.rstrip()
        if not part.strip():
            continue
        m = _REBUTTAL_RE.match(part)
        if m:
            responses.append(ParsedResponse("REBUTTAL", m.group(1).strip(), m.group(2).strip()))
            continue
        m = _ACCEPT_RE.match(part)
        if m:
            responses.append(ParsedResponse("ACCEPT", m.group(1).strip(), m.group(2).strip()))
            continue
        m = _PARTIAL_ACCEPT_RE.match(part)
        if m:
            responses.append(ParsedResponse(
                "PARTIAL_ACCEPT", m.group(1).strip(), m.group(2).strip(),
            ))
            continue
        m = _REVISE_RE.match(part)
        if m:
            responses.append(ParsedResponse("REVISE", m.group(1).strip(), m.group(2).strip()))
            continue
        m = _REOPEN_REQUEST_RE.match(part)
        if m:
            responses.append(ParsedResponse(
                "REOPEN_REQUEST", m.group(1).strip(), m.group(2).strip(),
            ))
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
        self.host_global_context_truncated = False

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

        unresolved: list[str] = []
        for value in unmatched:
            if "/" not in value:
                unresolved.append(value)
                continue
            slash_targets = [part.strip() for part in value.split("/")]
            if (
                all(slash_targets)
                and all(part in known_targets for part in slash_targets)
            ):
                matched.extend(slash_targets)
            else:
                unresolved.append(value)
        return list(dict.fromkeys(matched)), list(dict.fromkeys(unresolved))

    def parse_claims_file(self) -> None:
        """Parse an existing claims.md file into memory."""
        if not self.claims_file or not Path(self.claims_file).exists():
            return
        with Path(self.claims_file).open(
            "r",
            encoding="utf-8",
            newline="",
        ) as claims_stream:
            text = claims_stream.read()
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
                content = "\n".join(multiline_buffer)
                current_claim.add_entry(
                    ClaimEntry(entry_type, agent, rnd, content)
                )
            multiline_buffer = []
            pending_entry_meta = None

        lines = text.split("\n")
        if lines and lines[-1] == "":
            lines.pop()
        for line in lines:
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
                if keyword in self.claims:
                    raise ValueError(f"Duplicate persisted claim: {keyword}")
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
            for line in output.raw_text.splitlines():
                if _INDENTED_MARKER_RE.match(line):
                    violation = {
                        "agent_name": output.agent_name,
                        "round_num": output.round_num,
                        "response_type": "PROTOCOL_VIOLATION",
                        "content": line,
                    }
                    parsed_marker = parse_agent_output(line.lstrip())
                    if parsed_marker:
                        violation["target"] = parsed_marker[0].target
                    self.unmatched_responses.append(violation)
            for response in output.parsed:
                if not response.target:
                    self.unmatched_responses.append({
                        "agent_name": output.agent_name,
                        "round_num": output.round_num,
                        "response_type": "INVALID_TARGET",
                        "target": "",
                        "content": response.content,
                    })
                    continue
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
                elif response.response_type == "REOPEN_REQUEST":
                    self._merge_reopen_request(output, response)
                elif response.response_type in {
                    "REBUTTAL",
                    "ACCEPT",
                    "PARTIAL_ACCEPT",
                    "REVISE",
                }:
                    open_targets = {
                        keyword
                        for keyword, claim in self.claims.items()
                        if claim.status == "OPEN"
                    }
                    if response.target in open_targets:
                        response_target = response.target
                    else:
                        keyword_to_reference = self.build_host_references(
                            list(self.claims.values()),
                        )
                        reference_to_keyword = {
                            reference: keyword
                            for keyword, reference in keyword_to_reference.items()
                            if keyword in open_targets
                        }
                        response_target = reference_to_keyword.get(
                            response.target,
                            response.target,
                        )
                    targets, unmatched = self._resolve_response_targets(
                        response_target, open_targets,
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
                            "content": response.content,
                        })
        self.current_round = max(
            (o.round_num for o in agent_outputs), default=self.current_round
        )
        self.save()

    @staticmethod
    def _parse_reopen_payload(content: str) -> dict[str, str] | None:
        """Parse the strict evidence payload without judging its materiality."""
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            return None
        required = {"kind", "fact", "source", "reason"}
        if (
            not isinstance(payload, dict)
            or set(payload) != required
            or payload.get("kind") not in {
                "HARD_FACT",
                "MATERIAL_COUNTEREXAMPLE",
            }
            or any(
                type(payload[field]) is not str or not payload[field].strip()
                for field in required
            )
        ):
            return None
        return {
            field: payload[field].strip()
            for field in ("kind", "fact", "source", "reason")
        }

    @staticmethod
    def _reopen_request_id(
        claim_keyword: str,
        agent_name: str,
        round_num: int,
        payload: dict[str, str],
    ) -> str:
        identity = json.dumps(
            {
                "claim": claim_keyword,
                "agent": agent_name,
                "round": round_num,
                **payload,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:32]
        return f"reopen:{digest}"

    def _merge_reopen_request(
        self,
        output: AgentOutput,
        response: ParsedResponse,
    ) -> None:
        closed_targets = {
            keyword
            for keyword, claim in self.claims.items()
            if claim.status.startswith("CLOSED:")
        }
        keyword_to_reference = self.build_host_references(
            list(self.claims.values()),
        )
        reference_to_keyword = {
            reference: keyword
            for keyword, reference in keyword_to_reference.items()
            if keyword in closed_targets
        }
        response_target = reference_to_keyword.get(
            response.target,
            response.target,
        )
        targets, unmatched = self._resolve_response_targets(
            response_target,
            closed_targets,
        )
        payload = self._parse_reopen_payload(response.content)
        if payload is None:
            self.unmatched_responses.append({
                "agent_name": output.agent_name,
                "round_num": output.round_num,
                "response_type": "INVALID_REOPEN_REQUEST",
                "target": response.target,
                "content": response.content,
            })
            return
        for target in targets:
            request = ReopenRequest(
                request_id=self._reopen_request_id(
                    target,
                    output.agent_name,
                    output.round_num,
                    payload,
                ),
                claim_keyword=target,
                agent_name=output.agent_name,
                round_num=output.round_num,
                **payload,
            )
            if any(
                existing.request_id == request.request_id
                for existing in self.get_reopen_requests()
            ):
                self.unmatched_responses.append({
                    "agent_name": output.agent_name,
                    "round_num": output.round_num,
                    "response_type": "DUPLICATE_REOPEN_REQUEST",
                    "target": target,
                    "content": response.content,
                    "request_id": request.request_id,
                })
                continue
            self.add_reopen_request(request, persist=False)
        for target in unmatched:
            self.unmatched_responses.append({
                "agent_name": output.agent_name,
                "round_num": output.round_num,
                "response_type": "REOPEN_REQUEST",
                "target": target,
                "content": response.content,
            })

    @staticmethod
    def _reopen_request_from_entry(
        claim: Claim,
        entry: ClaimEntry,
    ) -> ReopenRequest | None:
        if entry.entry_type != "REOPEN_REQUEST":
            return None
        try:
            payload = json.loads(entry.content)
        except json.JSONDecodeError:
            return None
        required = {"request_id", "kind", "fact", "source", "reason"}
        if (
            not isinstance(payload, dict)
            or set(payload) != required
            or any(type(payload[field]) is not str for field in required)
        ):
            return None
        return ReopenRequest(
            request_id=payload["request_id"],
            claim_keyword=claim.keyword,
            agent_name=entry.agent_name,
            round_num=entry.round_num,
            kind=payload["kind"],
            fact=payload["fact"],
            source=payload["source"],
            reason=payload["reason"],
        )

    @staticmethod
    def _resolved_reopen_ids(claim: Claim) -> set[str]:
        resolved: set[str] = set()
        for entry in claim.entries:
            if entry.entry_type != "HOST" or not entry.content.startswith(
                ("REOPEN_APPROVED：", "REOPEN_REJECTED：")
            ):
                continue
            _, _, encoded = entry.content.partition("：")
            try:
                payload = json.loads(encoded)
            except json.JSONDecodeError:
                continue
            request_id = payload.get("request_id") if isinstance(payload, dict) else None
            if isinstance(request_id, str):
                resolved.add(request_id)
        return resolved

    @classmethod
    def _pending_reopen_entry_ids(cls, claim: Claim) -> set[int]:
        resolved = cls._resolved_reopen_ids(claim)
        return {
            id(entry)
            for entry in claim.entries
            if (
                (request := cls._reopen_request_from_entry(claim, entry))
                is not None
                and request.request_id not in resolved
            )
        }

    def get_reopen_requests(self) -> list[ReopenRequest]:
        """Return every persisted reopen request in stable claim order."""
        requests: list[ReopenRequest] = []
        for claim in self.claims.values():
            for entry in claim.entries:
                request = self._reopen_request_from_entry(claim, entry)
                if request is not None:
                    requests.append(request)
        return requests

    def get_pending_reopen_requests(self) -> list[ReopenRequest]:
        """Return requests that do not yet have a durable Host decision."""
        pending: list[ReopenRequest] = []
        for claim in self.claims.values():
            resolved = self._resolved_reopen_ids(claim)
            pending.extend(
                request
                for entry in claim.entries
                if (
                    (request := self._reopen_request_from_entry(claim, entry))
                    is not None
                    and request.request_id not in resolved
                )
            )
        return pending

    def add_reopen_request(
        self,
        request: ReopenRequest,
        *,
        persist: bool = True,
    ) -> None:
        """Persist a structurally valid request without changing claim status."""
        claim = self.claims.get(request.claim_keyword)
        if claim is None or not claim.status.startswith("CLOSED:"):
            return
        payload = {
            "request_id": request.request_id,
            "kind": request.kind,
            "fact": request.fact,
            "source": request.source,
            "reason": request.reason,
        }
        claim.add_entry(ClaimEntry(
            "REOPEN_REQUEST",
            request.agent_name,
            request.round_num,
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        ))
        self.current_round = max(self.current_round, request.round_num)
        if persist:
            self.save()

    def resolve_reopen_request(
        self,
        request_id: str,
        *,
        approved: bool,
        reason: str,
        round_num: int,
        persist: bool = True,
    ) -> bool:
        """Persist a Host decision and reopen only after semantic approval."""
        request = next(
            (
                item for item in self.get_pending_reopen_requests()
                if item.request_id == request_id
            ),
            None,
        )
        if request is None:
            return False
        claim = self.claims[request.claim_keyword]
        decision = "REOPEN_APPROVED" if approved else "REOPEN_REJECTED"
        claim.add_entry(ClaimEntry(
            "HOST",
            "HOST",
            round_num,
            decision + "：" + json.dumps(
                {"request_id": request_id, "reason": reason},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        ))
        if approved:
            claim.status = "OPEN"
        self.current_round = max(self.current_round, round_num)
        if persist:
            self.save()
        return True

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

    def get_host_candidates(self) -> list[Claim]:
        """Return OPEN claims plus CLOSED claims with pending reopen requests."""
        pending_keywords = {
            request.claim_keyword
            for request in self.get_pending_reopen_requests()
        }
        return [
            claim
            for claim in self.claims.values()
            if claim.status == "OPEN" or claim.keyword in pending_keywords
        ]

    def has_protocol_violation(self, keyword: str, round_num: int) -> bool:
        """Return whether this round contains a malformed marker for a claim."""
        return any(
            item.get("response_type") == "PROTOCOL_VIOLATION"
            and item.get("target") == keyword
            and item.get("round_num") == round_num
            for item in self.unmatched_responses
        )

    @staticmethod
    def claim_keywords_from_formatted(text: str) -> set[str]:
        """Extract claim keywords using the persisted header grammar."""
        return {
            match.group(1).strip()
            for line in text.split("\n")
            if (match := _CLAIM_HEADER_RE.match(line))
        }

    @staticmethod
    def host_reference(keyword: str) -> str:
        """Return a bounded Host protocol identity for a claim keyword."""
        if len(f"##CLAIM:{keyword} [OPEN]##") <= 512:
            return keyword
        digest = hashlib.sha256(keyword.encode("utf-8")).hexdigest()[:24]
        return f"@sha256:{digest}"

    @classmethod
    def build_host_references(cls, claims: list[Claim]) -> dict[str, str]:
        """Assign unique bounded Host identities without colliding with keywords."""
        keywords = {claim.keyword for claim in claims}
        references = {
            claim.keyword: claim.keyword
            for claim in claims
            if cls.host_reference(claim.keyword) == claim.keyword
        }
        used = set(references.values())
        for claim in claims:
            if claim.keyword in references:
                continue
            base = cls.host_reference(claim.keyword)
            reference = base
            suffix = 1
            while reference in keywords or reference in used:
                reference = f"{base}:{suffix}"
                suffix += 1
            references[claim.keyword] = reference
            used.add(reference)
        return references

    @staticmethod
    def truncated_claim_references_from_formatted(text: str) -> set[str]:
        """Return claim references whose bounded blocks lost required context."""
        truncated: set[str] = set()
        current: str | None = None
        before_entries = False
        for line in text.splitlines():
            match = _CLAIM_HEADER_RE.match(line)
            if match:
                current = match.group(1).strip()
                before_entries = True
            elif current is not None and (
                _ENTRY_RE.match(line) or _HOST_ENTRY_RE.match(line)
            ):
                before_entries = False
            elif (
                current is not None
                and before_entries
                and line == "[MUST_CONTINUE:TRUNCATED]"
            ):
                truncated.add(current)
        return truncated

    @staticmethod
    def global_context_is_truncated(text: str) -> bool:
        """Report whether global Host context is explicitly incomplete."""
        lines = text.splitlines()
        return bool(
            lines
            and lines[0] == "[GLOBAL_CONTEXT_TRUNCATED]"
        )

    @staticmethod
    def _host_request(claim: Claim) -> tuple[set[str], str]:
        """Read the latest persisted Host routing request for a claim."""
        routing = ClaimsManager._host_routing(claim)
        return set(routing["needs_agents"]), routing["missing"]

    @staticmethod
    def _host_routing(claim: Claim) -> dict[str, object]:
        """Read the latest persisted Host routing context for a claim."""
        host_entry = None
        for entry in reversed(claim.entries):
            if entry.entry_type != "HOST":
                continue
            if entry.content.startswith("CONTINUE："):
                host_entry = entry
            # A close/reopen decision starts a new lifecycle. Routing from the
            # previous lifecycle must not leak into the reopened context.
            break
        if host_entry is None:
            return {
                "needs_agents": [],
                "missing": "",
                "allow_unknown_progress": None,
            }
        agents: set[str] = set()
        missing = ""
        allow_unknown_progress: bool | None = None
        for line in host_entry.content.splitlines()[1:]:
            if line.startswith(_ROUTING_PREFIX):
                try:
                    routing = json.loads(line.removeprefix(_ROUTING_PREFIX))
                    agents = set(routing.get("needs_agents", []))
                    missing = routing.get("missing", "")
                    allow_unknown_progress = routing.get(
                        "allow_unknown_progress",
                    )
                except (json.JSONDecodeError, TypeError):
                    continue
            elif line.startswith(_NEEDS_AGENTS_PREFIX):
                agents = {
                    name.strip()
                    for name in line.removeprefix(_NEEDS_AGENTS_PREFIX).split(",")
                    if name.strip()
                }
            elif line.startswith(_MISSING_PREFIX):
                missing = line.removeprefix(_MISSING_PREFIX).strip()
        return {
            "needs_agents": sorted(agents),
            "missing": missing,
            "allow_unknown_progress": allow_unknown_progress,
        }

    @staticmethod
    def _truncate(text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        marker = "\n...[内容已省略/截断，信息可能不完整]"
        if limit <= len(marker):
            return marker[:max(0, limit)]
        return text[: max(0, limit - len(marker))].rstrip() + marker

    @staticmethod
    def _truncate_ends(text: str, limit: int) -> str:
        """Bound text while retaining both the claim identity and latest context."""
        if len(text) <= limit:
            return text
        marker = "\n...[中间内容已省略/截断，首尾信息保留]...\n"
        if limit <= len(marker):
            return marker[:max(0, limit)]
        available = max(0, limit - len(marker))
        head_size = available // 2
        tail_size = available - head_size
        return (
            text[:head_size].rstrip()
            + marker
            + text[-tail_size:].lstrip()
        )

    @classmethod
    def _compact_claim(
        cls,
        claim: Claim,
        limit: int = 800,
        *,
        reference: str | None = None,
    ) -> str:
        """Keep every claim visible without replaying its full history."""
        if not claim.entries:
            reference = reference or cls.host_reference(claim.keyword)
            return f"##CLAIM:{reference} [{claim.status}]##"
        selected_ids = {
            id(claim.entries[0]),
            *(
                id(entry)
                for entry in claim.entries[-2:]
            ),
        }
        selected_ids.update(cls._pending_reopen_entry_ids(claim))
        for entry_type in _CRITICAL_HISTORY_TYPES:
            response_rounds = [
                entry.round_num
                for entry in claim.entries
                if entry.entry_type == entry_type
            ]
            if response_rounds:
                latest_response_round = max(response_rounds)
                selected_ids.update(
                    id(entry)
                    for entry in claim.entries
                    if entry.entry_type == entry_type
                    and entry.round_num == latest_response_round
                )
        latest_host = next(
            (
                entry for entry in reversed(claim.entries)
                if entry.entry_type == "HOST"
            ),
            None,
        )
        if latest_host is not None:
            selected_ids.add(id(latest_host))
        selected = [
            entry for entry in claim.entries if id(entry) in selected_ids
        ]
        omitted = len(claim.entries) - len({id(entry) for entry in selected})
        reference = reference or cls.host_reference(claim.keyword)
        header = f"##CLAIM:{reference} [{claim.status}]##"
        keyword_preview = ""
        keyword_context_truncated = False
        if reference != claim.keyword:
            keyword_context_truncated = len(claim.keyword) > 256
            keyword_preview = (
                "\nKEYWORD_PREVIEW:"
                + cls._truncate_ends(claim.keyword, 256)
            )
        omission_marker = ""
        if omitted > 0:
            omission_marker = (
                f"\n...[中间{omitted}条记录已省略；"
                "现有信息不足时选择CONTINUE]"
            )
        prefixes: list[str] = []
        for entry in selected:
            if entry.entry_type == "HOST":
                tag = f"[HOST @R{entry.round_num}]"
            elif entry.entry_type == "FROM":
                tag = f"[FROM:{entry.agent_name} @R{entry.round_num}]"
            else:
                tag = (
                    f"[{entry.entry_type} FROM:{entry.agent_name} "
                    f"@R{entry.round_num}]"
                )
            indent = "" if entry.entry_type == "FROM" else "  "
            prefixes.append(f"{indent}{tag}")
        separators = len(selected)
        identity_size = (
            len(header)
            + len(keyword_preview)
            + separators
            + sum(len(prefix) for prefix in prefixes)
        )
        if identity_size + len(omission_marker) > limit:
            marker = "[MUST_CONTINUE:TRUNCATED]"
            required_prefix = header + "\n" + marker
            if len(required_prefix) > limit:
                return ""
            detail = (
                "...[关键回应身份截断：超出上下文安全上限；"
                "部分身份未展开，必须CONTINUE，不得据此收敛]..."
            )
            details = "\n".join(
                part
                for part in (
                    keyword_preview.removeprefix("\n"),
                    detail,
                    "\n".join(prefixes) + omission_marker,
                )
                if part
            )
            if not details:
                return required_prefix
            return (
                required_prefix
                + "\n"
                + cls._truncate_ends(
                    details,
                    limit - len(required_prefix) - 1,
                )
            )
        context_marker = (
            "\n[MUST_CONTINUE:TRUNCATED]"
            if keyword_context_truncated
            else ""
        )
        fixed_size = (
            len(header)
            + len(keyword_preview)
            + len(omission_marker)
            + len(context_marker)
            + separators
            + sum(len(prefix) for prefix in prefixes)
            + len(selected)
        )
        content_budget = max(
            0,
            limit - fixed_size,
        )
        body_limit, body_extra = divmod(
            content_budget, max(1, len(selected)),
        )
        entry_limits = [
            body_limit + (1 if index < body_extra else 0)
            for index in range(len(selected))
        ]
        if any(
            len(entry.content) > entry_limit
            for entry, entry_limit in zip(selected, entry_limits)
        ) and not context_marker:
            context_marker = "\n[MUST_CONTINUE:TRUNCATED]"
            content_budget = max(
                0,
                limit - fixed_size - len(context_marker),
            )
            body_limit, body_extra = divmod(
                content_budget, max(1, len(selected)),
            )
            entry_limits = [
                body_limit + (1 if index < body_extra else 0)
                for index in range(len(selected))
            ]
        blocks: list[str] = []
        for entry, prefix, entry_limit in zip(
            selected,
            prefixes,
            entry_limits,
        ):
            body = cls._truncate_ends(entry.content, entry_limit)
            blocks.append(prefix + (f" {body}" if body else ""))
        rendered = (
            header + keyword_preview + context_marker + "\n"
            + "\n".join(blocks)
            + omission_marker
        )
        return cls._truncate_ends(rendered, limit)

    def format_host_candidate_batches(
        self, max_chars: int = 80_000, max_claims: int = 40,
    ) -> list[str]:
        """Render every OPEN claim in independently bounded Host batches."""
        batches: list[str] = []
        current: list[str] = []
        current_size = 0
        candidates = self.get_host_candidates()
        references = self.build_host_references(candidates)
        for claim in candidates:
            block = self._compact_claim(
                claim,
                min(4_000, max_chars),
                reference=references[claim.keyword],
            )
            if not block:
                continue
            if self.claim_keywords_from_formatted(block) != {
                references[claim.keyword],
            }:
                continue
            block_size = len(block) + (2 if current else 0)
            if current and (
                current_size + block_size > max_chars
                or len(current) >= max_claims
            ):
                batches.append("\n\n".join(current))
                current = []
                current_size = 0
                block_size = len(block)
            current.append(block)
            current_size += block_size
        if current:
            batches.append("\n\n".join(current))
        return batches

    def format_host_round_provenance(
        self,
        round_num: int,
        max_chars: int = 12_000,
    ) -> str:
        """Render authoritative topology and structured-response counts."""
        recorded_rounds = {
            entry.round_num
            for claim in self.claims.values()
            for entry in claim.entries
            if 1 <= entry.round_num <= round_num
        }
        recorded_rounds.update({1, round_num})

        lines = [
            "ROUND_TOPOLOGY:",
            "- R1=PARALLEL_INDEPENDENT_INITIAL_OUTPUTS; "
            "same-round peers were not visible.",
            "- R2+=PARALLEL_UPDATES_TO_PRIOR_PERSISTED_RECORD; "
            "same-round peers remain invisible.",
            "STRUCTURED_RESPONSE_COUNTS:",
        ]
        for recorded_round in sorted(recorded_rounds):
            counts = {
                response_type: 0
                for response_type in _STRUCTURED_RESPONSE_TYPES
            }
            for claim in self.claims.values():
                for entry in claim.entries:
                    if (
                        entry.round_num == recorded_round
                        and entry.entry_type in counts
                    ):
                        counts[entry.entry_type] += 1
            lines.append(
                f"- R{recorded_round} "
                + " ".join(
                    f"{response_type}={counts[response_type]}"
                    for response_type in _STRUCTURED_RESPONSE_TYPES
                )
            )

        return self._truncate("\n".join(lines), max_chars)

    def format_host_global_context(self, max_chars: int = 40_000) -> str:
        """Summarize every OPEN claim for cross-batch semantic consistency.

        The completeness marker may exceed an impossibly small ``max_chars`` so
        truncation remains machine-recognizable.
        """
        self.host_global_context_truncated = False
        claims = self.get_host_candidates()
        if not claims:
            return ""
        references = self.build_host_references(claims)
        prefixes = [
            f"- {references[claim.keyword]}: "
            for claim in claims
        ]
        fixed_size = sum(len(prefix) for prefix in prefixes) + len(claims) - 1
        if fixed_size > max_chars:
            self.host_global_context_truncated = True
        available = max(0, max_chars - fixed_size)
        base_content_limit, extra = divmod(available, len(claims))
        lines: list[str] = []
        for index, (claim, prefix) in enumerate(zip(claims, prefixes)):
            content_limit = base_content_limit + (1 if index < extra else 0)
            critical_ids: set[int] = set()
            critical_ids.update(self._pending_reopen_entry_ids(claim))
            for entry_type in _CRITICAL_HISTORY_TYPES:
                response_rounds = [
                    entry.round_num
                    for entry in claim.entries
                    if entry.entry_type == entry_type
                ]
                if not response_rounds:
                    continue
                latest_response_round = max(response_rounds)
                critical_ids.update(
                    id(entry)
                    for entry in claim.entries
                    if entry.entry_type == entry_type
                    and entry.round_num == latest_response_round
                )
            for entry_type in ("HOST",):
                latest = next(
                    (
                        entry for entry in reversed(claim.entries)
                        if entry.entry_type == entry_type
                    ),
                    None,
                )
                if latest is not None:
                    critical_ids.add(id(latest))
            if claim.entries:
                critical_ids.add(id(claim.entries[-1]))
            critical = [
                entry for entry in claim.entries
                if id(entry) in critical_ids
            ]
            labels = [
                (
                    f"{entry.entry_type} FROM:{entry.agent_name} "
                    f"@R{entry.round_num}: "
                    if entry.entry_type in _CRITICAL_HISTORY_TYPES
                    else f"{entry.entry_type}: "
                )
                for entry in critical
            ]
            summary_overhead = (
                sum(len(label) for label in labels)
                + max(0, len(labels) - 1) * 3
            )
            if content_limit <= summary_overhead:
                if critical:
                    self.host_global_context_truncated = True
                summary = self._truncate_ends(
                    " | ".join(labels),
                    content_limit,
                )
            else:
                body_total = content_limit - summary_overhead
                body_limit, body_extra = divmod(
                    body_total,
                    max(1, len(critical)),
                )
                summaries: list[str] = []
                for entry_index, (entry, label) in enumerate(
                    zip(critical, labels)
                ):
                    limit = body_limit + (
                        1 if entry_index < body_extra else 0
                    )
                    content = entry.content.replace("\n", " ")
                    if len(content) > limit:
                        self.host_global_context_truncated = True
                    summaries.append(
                        label + self._truncate_ends(
                            content,
                            limit,
                        )
                    )
                summary = " | ".join(summaries)
            lines.append(prefix + summary)
        rendered = "\n".join(lines)
        if (
            len(rendered) <= max_chars
            and not self.host_global_context_truncated
        ):
            return rendered
        self.host_global_context_truncated = True
        marker = "[GLOBAL_CONTEXT_TRUNCATED]"
        if max_chars < len(marker):
            return marker
        remaining = max_chars - len(marker)
        if remaining <= 1:
            return marker
        return marker + "\n" + self._truncate_ends(
            rendered,
            remaining - 1,
        )

    @classmethod
    def _bounded_blocks(
        cls, blocks: list[str], budget: int, omitted_label: str,
    ) -> list[str]:
        """Fit whole blocks into a prompt section and disclose omissions."""
        selected: list[str] = []
        used = 0
        for block in blocks:
            separator_size = 1 if selected else 0
            if used + separator_size + len(block) > budget:
                break
            selected.append(block)
            used += separator_size + len(block)
        omitted = len(blocks) - len(selected)
        if omitted:
            marker_prefix = "[MUST_CONTINUE:TRUNCATED_CONTEXT_BLOCKS]\n"
            while selected:
                omitted = len(blocks) - len(selected)
                minimum_marker = (
                    marker_prefix
                    + f"...[{omitted} {omitted_label}因上下文安全上限未展开]"
                )
                if used + 1 + len(minimum_marker) <= budget:
                    break
                removed = selected.pop()
                used -= len(removed) + (1 if selected else 0)

            omitted = len(blocks) - len(selected)
            base_marker = (
                marker_prefix
                + f"...[{omitted} {omitted_label}因上下文安全上限未展开]"
            )
            omitted_keywords = [
                match.group(1)
                for block in blocks[len(selected):]
                if (match := re.search(r"##CLAIM:(.+?)\s+\[", block))
            ]
            separator_size = 1 if selected else 0
            marker_budget = max(0, budget - used - separator_size)
            marker = cls._truncate(base_marker, marker_budget)
            if omitted_keywords and len(base_marker) <= marker_budget:
                preview = ", ".join(omitted_keywords[:10])
                suffix = "..." if len(omitted_keywords) > 10 else ""
                preview_budget = marker_budget - len(base_marker)
                preview = cls._truncate_ends(
                    f"：{preview}{suffix}",
                    preview_budget,
                )
                marker = base_marker[:-1] + preview + "]"
            if marker:
                selected.append(marker)
        return selected

    @classmethod
    def _truncate_response_context(cls, block: str, budget: int) -> str:
        """Bound one routed claim block and disclose secondary truncation."""
        if len(block) <= budget:
            return block
        header, separator, body = block.partition("\n")
        marker = "[MUST_CONTINUE:TRUNCATED_RESPONSE_CONTEXT]"
        prefix = header + "\n" + marker
        if not separator or len(prefix) >= budget:
            return cls._truncate(prefix, budget)
        return (
            prefix
            + "\n"
            + cls._truncate_ends(
                body,
                budget - len(prefix) - 1,
            )
        )

    @classmethod
    def _bounded_all_blocks(cls, blocks: list[str], budget: int) -> list[str]:
        """Keep every routed block visible by sharing the section budget."""
        if not blocks:
            return []
        separators = len(blocks) - 1
        if separators > budget:
            return [cls._truncate(
                f"[MUST_CONTINUE:TRUNCATED_TARGETED_REQUESTS] "
                f"{len(blocks)}个定向请求超出上下文安全上限",
                budget,
            )]
        block_budget, extra = divmod(
            budget - separators,
            len(blocks),
        )
        if block_budget < 64:
            return [cls._truncate(
                f"[MUST_CONTINUE:TRUNCATED_TARGETED_REQUESTS] "
                f"{len(blocks)}个定向请求无法在预算内保留身份",
                budget,
            )]
        if any(
            len(block.partition("\n")[0]) + 64 > block_budget
            for block in blocks
        ):
            return [cls._truncate(
                f"[MUST_CONTINUE:TRUNCATED_TARGETED_REQUESTS] "
                f"{len(blocks)}个定向请求的claim身份超出预算",
                budget,
            )]
        return [
            cls._truncate_response_context(
                block,
                block_budget + (index < extra),
            )
            for index, block in enumerate(blocks)
        ]

    @classmethod
    def _format_response_context(
        cls,
        claim: Claim,
        limit: int = 7_000,
        *,
        reference: str | None = None,
    ) -> str:
        """Render the original claim plus the latest substantive context."""
        return cls._compact_claim(claim, limit, reference=reference)

    def close_claim(
        self,
        keyword: str,
        verdict: str,
        reason: str,
        round_num: int,
        *,
        persist: bool = True,
    ) -> None:
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
        if persist:
            self.save()

    def continue_claim(
        self,
        keyword: str,
        reason: str,
        round_num: int,
        *,
        needs_agents: list[str] | None = None,
        missing: str = "",
        allow_unknown_progress: bool | None = None,
        persist: bool = True,
    ) -> None:
        """Persist a Host CONTINUE decision and optional semantic routing."""
        claim = self.claims.get(keyword)
        if claim is None or claim.status != "OPEN":
            return
        content = f"CONTINUE：{reason}"
        routing = {
            "needs_agents": needs_agents or [],
            "missing": missing,
            "allow_unknown_progress": allow_unknown_progress,
        }
        content += (
            f"\n{_ROUTING_PREFIX}"
            f"{json.dumps(routing, ensure_ascii=False, separators=(',', ':'))}"
        )
        claim.add_entry(ClaimEntry(
            entry_type="HOST",
            agent_name="HOST",
            round_num=round_num,
            content=content,
        ))
        self.current_round = max(self.current_round, round_num)
        if persist:
            self.save()

    def generate_update_prompt(
        self,
        prev_round: int,
        *,
        agent_name: str | None = None,
    ) -> str:
        """Generate incremental update prompt for agents.

        - OPEN claims: full content included (with FROM tags)
        - CLOSED/new claims since prev_round: only status change notification
        """
        status_changes: list[str] = []
        directed_requests: list[str] = []
        open_claims: list[str] = []
        claim_references = self.build_host_references(
            list(self.claims.values()),
        )

        for claim in self.claims.values():
            # Check if status changed since prev_round
            has_recent_host = any(
                e.entry_type == "HOST" and e.round_num >= prev_round
                for e in claim.entries
            )
            has_recent_new = any(
                e.entry_type == "FROM" and e.round_num >= prev_round
                for e in claim.entries
            ) and claim.status == "OPEN"

            if claim.status == "OPEN":
                if agent_name is None:
                    open_claims.append(self._compact_claim(
                        claim,
                        8_000,
                        reference=claim_references[claim.keyword],
                    ))
                else:
                    routing = self._host_routing(claim)
                    requested_agents = set(routing["needs_agents"])
                    missing = routing["missing"]
                    if agent_name in requested_agents:
                        suffix_lines: list[str] = []
                        if missing:
                            suffix_lines.append(
                                "HOST指出的缺口："
                                + self._truncate_ends(str(missing), 1_000)
                            )
                        allow_unknown = routing["allow_unknown_progress"]
                        if allow_unknown is not None:
                            policy = "允许" if allow_unknown else "不允许"
                            suffix_lines.append(
                                f"HOST判断：{policy}带 UNKNOWN 推进"
                            )
                        suffix = "\n".join(suffix_lines)
                        context_budget = 8_000 - len(suffix) - (
                            1 if suffix else 0
                        )
                        request = self._format_response_context(
                            claim,
                            context_budget,
                            reference=claim_references[claim.keyword],
                        )
                        if suffix:
                            request += "\n" + suffix
                        directed_requests.append(request)
                    else:
                        open_claims.append(self._compact_claim(
                            claim,
                            8_000,
                            reference=claim_references[claim.keyword],
                        ))
                if has_recent_new and len(claim.entries) == 1:
                    status_changes.append(f"- [新增] CLAIM:{claim.keyword}")
                if has_recent_host:
                    status_changes.append(f"- [继续讨论] CLAIM:{claim.keyword}")
            elif has_recent_host:
                status_changes.append(
                    f"- [已关闭] CLAIM:{claim_references[claim.keyword]} "
                    f"— {claim.status}"
                )

        parts = [f"## 第{self.current_round}轮更新"]

        if status_changes:
            parts.append("\n状态变化：")
            parts.extend(self._bounded_blocks(
                status_changes, 8_000, "条状态变化",
            ))

        if directed_requests:
            parts.append("\n## HOST定向请求\n")
            parts.extend(self._bounded_all_blocks(directed_requests, 32_000))
        if open_claims:
            parts.append("\n## OPEN CLAIMS（语义审阅）\n")
            parts.extend(self._bounded_blocks(
                open_claims, 40_000, "claims",
            ))

        recent_unmatched = [
            item for item in self.unmatched_responses
            if int(item["round_num"]) >= prev_round
            and (agent_name is None or item["agent_name"] == agent_name)
        ]
        if recent_unmatched:
            parts.append("\n## 上轮协议冲突（请按精确关键词修正）")
            audit_blocks: list[str] = []
            for item in recent_unmatched:
                response_type = item.get("response_type")
                if response_type == "PROTOCOL_VIOLATION":
                    content = json.dumps(
                        item.get("content", ""),
                        ensure_ascii=False,
                    )
                    audit_blocks.append(
                        f"- PROTOCOL_VIOLATION：缩进 marker 未执行；"
                        f"原始内容={content}；请将协议 marker 放在 column 0。"
                    )
                    continue
                target = str(item["target"])
                if response_type == "DUPLICATE_NEW_CLAIM":
                    content = json.dumps(
                        item.get("content", ""),
                        ensure_ascii=False,
                    )
                    audit_blocks.append(
                        f"- 重复 NEW_CLAIM CLAIM:{target} 未写入；"
                        f"冲突内容={content}；"
                        f"若要质疑现有 claim，改用 [REBUTTAL TO:{target}]。"
                    )
                else:
                    audit_blocks.append(
                        f"- 未匹配 target：{target}；"
                        "请改用单个精确关键词。"
                    )
            parts.extend(self._bounded_blocks(
                audit_blocks,
                8_000,
                "个协议冲突",
            ))

        parts.append(
            "\n## 你的任务\n"
            "根据你的职责和专业判断审阅 OPEN CLAIMS；不要用固定数量、固定轮次或全员回应代替实质判断。\n"
            "优先处理 HOST定向请求。其他 claim 仅在与你的职责或证据相关时回应；没有实质增量时不要机械重复。\n"
            "每位 Agent 都会看到其他人的 OPEN claims，并可主动接受、部分接受、反驳或提出修订；"
            "这些能力不受角色限制。Challenger 是独立、高强度红队，但无否决权，"
            "也不是唯一可反驳的角色。\n"
            "- 所有协议 marker 必须从 column 0 开始；缩进 marker 仅记录为协议 warning，不会执行。\n"
            "- [REBUTTAL TO:关键词] 反驳 + 证据\n"
            "- [ACCEPT TO:关键词] 接受 + 理由\n"
            "- [PARTIAL_ACCEPT TO:关键词] 部分接受 + 接受边界/保留条件\n"
            "- [REVISE TO:关键词] 对现有 claim 提出修订文本 + 理由/证据\n"
            "- CLOSED claim 不接受上述普通响应；只有发现新的硬事实或 material 反例时，"
            "可用 [REOPEN_REQUEST TO:关键词] "
            '{"kind":"HARD_FACT|MATERIAL_COUNTEREXAMPLE","fact":"新事实或反例",'
            '"source":"可追溯来源","reason":"为何可能改变原裁决"} 请求重开。'
            "runtime 只校验结构、身份和 exactly-once，不判断业务 materiality；"
            "请求保持 pending，直到下一次 Host 语义 APPROVE/REJECT。\n"
            "- 每个响应 marker 只能写一个精确关键词，不得批量列出。\n"
            "- [NEW_CLAIM:关键词] 仅限新的实质主张、证据、反证或 UNKNOWN；不要为状态改名另开claim。\n"
            "- 证据不足时明确 UNKNOWN、缺什么、应由谁补证；不要把沉默当作同意。\n"
            "可以随时用 research_search 或 web_search 搜索证据。\n"
            "若你的职责是 Challenger：对可验证且 material 的争议，凡涉及二级摘要、"
            "尚未核对的原始页、完整卖方方法、可复现冲突数字，或截止日前可获得的证据，"
            "应独立使用原文件、计算或研究工具核查；若对某一此类争议零调用合理，"
            "必须逐项明确说明理由。没有固定调用次数，Host/runtime 也不得以调用次数"
            "作为质量标准。"
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
            parts.append(
                f"⚠️ 本次讨论范围仅限于：{self._truncate(limitation, 8_000)}\n"
            )
        parts.append(f"议题：{self._truncate(topic, 4_000)}\n")
        if context:
            parts.append(
                "## 已确认背景材料\n\n"
                f"{self._truncate(context.strip(), 80_000)}\n"
            )
        parts.append(
            "请基于你的职责和专业判断提出观点；只对有能力核查的部分下结论。\n"
            "证据标准：事实必须可追溯，区分事实、推断和 UNKNOWN，并说明适用边界。\n"
            "主动寻找反例、反证和失败条件；不得把无人反驳当作共识，"
            "也不得用固定回应数或固定轮次代替判断。\n"
            "证据不足时明确缺口及最适合补证的角色，不要猜测或机械附和。\n\n"
            "**输出格式要求：**\n"
            "- 所有协议 marker 必须从 column 0 开始；缩进 marker 仅记录为协议 warning，不会执行。\n"
            "- [NEW_CLAIM:关键词] 你的论点和证据\n"
            "- 每个独立论点用一个 NEW_CLAIM 标记\n"
            "- REOPEN_REQUEST 仅用于后续轮次对 CLOSED claim 提交新的硬事实或 material "
            "反例，并必须附可追溯 source 与可能改变原裁决的 reason。\n"
            "- 可以随时用 research_search 或 web_search 搜索证据。\n"
            "若你的职责是 Challenger：对可验证且 material 的争议，凡涉及二级摘要、"
            "尚未核对的原始页、完整卖方方法、可复现冲突数字，或截止日前可获得的证据，"
            "应独立使用原文件、计算或研究工具核查；若对某一此类争议零调用合理，"
            "必须逐项明确说明理由。没有固定调用次数，Host/runtime 也不得以调用次数"
            "作为质量标准。"
        )
        return "\n".join(parts)

    def save(self) -> None:
        """Write the current state to claims.md."""
        if not self.claims_file:
            return
        p = Path(self.claims_file)
        p.parent.mkdir(parents=True, exist_ok=True)
        text = self.format_file()
        temp_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=p.parent,
                prefix=f".{p.name}.",
                delete=False,
            ) as temp_file:
                temp_name = temp_file.name
                temp_file.write(text)
                temp_file.flush()
                os.fsync(temp_file.fileno())
            os.replace(temp_name, p)
            directory_fd = os.open(
                p.parent,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
            )
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if temp_name is not None:
                try:
                    os.unlink(temp_name)
                except FileNotFoundError:
                    pass

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
