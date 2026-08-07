"""AgentConversation — provider-aware multi-turn conversation manager."""

from __future__ import annotations

import asyncio
import base64
from collections import OrderedDict
from dataclasses import dataclass
import inspect
from io import BytesIO
import json
import logging
from pathlib import Path
import time
from typing import Any, Callable

import anthropic
from agno.tools.function import ToolResult
import openai
from PIL import Image as PILImage, UnidentifiedImageError

from discuss_agent.config import infer_provider, normalize_base_url
from discuss_agent.audit import AuditLogger

logger = logging.getLogger(__name__)

_MAX_TOOL_ITERATIONS = 20
_MAX_TOOL_IMAGES = 3
_MAX_TOOL_IMAGE_BYTES = 5 * 1024 * 1024
_MAX_TOOL_IMAGE_PIXELS = 25_000_000
_RESEARCH_SEARCH_MODULE = "discuss_tools.research_search"
_RESEARCH_SUCCESS_PREFIXES = ("关于「", "未找到与「")


@dataclass(frozen=True)
class _ToolImage:
    data: bytes
    media_type: str


@dataclass(frozen=True)
class _ExecutedToolResult:
    text: str
    images: tuple[_ToolImage, ...] = ()


def _is_research_search_callable(fn: Callable[..., Any]) -> bool:
    owner = getattr(fn, "__self__", None)
    if owner is None:
        return False
    owner_type = type(owner)
    return (
        owner_type.__module__ == _RESEARCH_SEARCH_MODULE
        and owner_type.__name__ == "ResearchSearchTools"
        and getattr(fn, "__name__", None) == "search_research"
    )


class _ToolResultCache:
    """Bounded run-scoped cache for immutable text-only tool results."""

    def __init__(
        self,
        *,
        max_entries: int = 32,
        max_bytes: int = 2 * 1024 * 1024,
    ):
        if max_entries <= 0 or max_bytes <= 0:
            raise ValueError("tool result cache limits must be positive")
        self._max_entries = max_entries
        self._max_bytes = max_bytes
        self._size_bytes = 0
        self._results: OrderedDict[
            tuple[int, str, str], tuple[_ExecutedToolResult, int]
        ] = OrderedDict()
        self._inflight: dict[
            tuple[int, str, str], asyncio.Future[_ExecutedToolResult | None]
        ] = {}
        self._lock = asyncio.Lock()

    @staticmethod
    def _key(
        namespace: int, name: str, input_args: dict
    ) -> tuple[int, str, str] | None:
        try:
            encoded = json.dumps(
                input_args,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (TypeError, ValueError):
            return None
        return namespace, name, encoded

    async def execute(
        self, namespace: int, name: str, input_args: dict, operation
    ):
        key = self._key(namespace, name, input_args)
        if key is None:
            result, _ = await operation()
            return result

        while True:
            async with self._lock:
                cached = self._results.get(key)
                if cached is not None:
                    self._results.move_to_end(key)
                    return cached[0]

                future = self._inflight.get(key)
                leader = future is None
                if leader:
                    future = asyncio.get_running_loop().create_future()
                    self._inflight[key] = future

            if not leader:
                shared = await asyncio.shield(future)
                if shared is not None:
                    return shared
                continue

            try:
                result, cacheable = await operation()
            except BaseException:
                async with self._lock:
                    self._inflight.pop(key, None)
                    if not future.done():
                        future.set_result(None)
                raise

            async with self._lock:
                self._inflight.pop(key, None)
                shared_result = result if cacheable else None
                if cacheable:
                    result_size = len(result.text.encode("utf-8"))
                    if result_size <= self._max_bytes:
                        while (
                            len(self._results) >= self._max_entries
                            or self._size_bytes + result_size > self._max_bytes
                        ):
                            _, (_, removed_size) = self._results.popitem(last=False)
                            self._size_bytes -= removed_size
                        self._results[key] = (result, result_size)
                        self._size_bytes += result_size
                if not future.done():
                    future.set_result(shared_result)
            return result


class AgentConversation:
    """Manage a single agent's multi-turn conversation and tool-use loop."""

    def __init__(
        self,
        agent_name: str,
        system_prompt: str,
        model: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 4096,
        temperature: float | None = None,
        tools: list[dict] | None = None,
        tool_callables: dict[str, Callable[..., Any]] | None = None,
        audit_logger: AuditLogger | None = None,
        tool_result_cache: _ToolResultCache | None = None,
        cacheable_tool_callables: tuple[Callable[..., Any], ...] | None = None,
    ):
        self.agent_name = agent_name
        self.system_prompt = system_prompt
        self.model = model
        self.provider = infer_provider(model)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.tools = tools or []
        self._tool_callables = tool_callables or {}
        self.messages: list[dict[str, Any]] = []
        self._audit = audit_logger
        self._round_num: int | None = None
        self._tool_result_cache = tool_result_cache
        self._cacheable_tool_callables = cacheable_tool_callables or ()

        client_kwargs: dict[str, Any] = {"timeout": 600.0}
        if api_key:
            client_kwargs["api_key"] = api_key
        normalized_url = normalize_base_url(base_url, self.provider)
        if normalized_url:
            client_kwargs["base_url"] = normalized_url
        if self.provider == "openai":
            self._client = openai.AsyncOpenAI(**client_kwargs)
        else:
            self._client = anthropic.AsyncAnthropic(**client_kwargs)

    def set_round(self, round_num: int) -> None:
        """Associate subsequent lifecycle/tool events with a discussion round."""
        self._round_num = round_num

    def _build_anthropic_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": self.system_prompt,
            "messages": self.messages,
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.tools:
            kwargs["tools"] = self.tools
        return kwargs

    def _build_openai_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "system", "content": self.system_prompt}, *self.messages],
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.tools:
            kwargs["tools"] = [self._to_openai_tool(tool) for tool in self.tools]
        return kwargs

    def _build_api_kwargs(self) -> dict[str, Any]:
        return self._build_openai_kwargs() if self.provider == "openai" else self._build_anthropic_kwargs()

    @staticmethod
    def _to_openai_tool(tool: dict) -> dict:
        return {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {"type": "object", "properties": {}}),
            },
        }

    @staticmethod
    def _load_image(image: Any) -> _ToolImage:
        if getattr(image, "url", None):
            raise ValueError("remote image URLs are not allowed in tool results")
        data = getattr(image, "content", None)
        if data is None and getattr(image, "filepath", None):
            path = Path(image.filepath)
            if path.stat().st_size > _MAX_TOOL_IMAGE_BYTES:
                raise ValueError("tool image exceeds 5 MiB limit")
            with path.open("rb") as image_file:
                data = image_file.read(_MAX_TOOL_IMAGE_BYTES + 1)
        if not isinstance(data, bytes) or not data:
            raise ValueError("tool image has no local bytes")
        if len(data) > _MAX_TOOL_IMAGE_BYTES:
            raise ValueError("tool image exceeds 5 MiB limit")
        try:
            with PILImage.open(BytesIO(data)) as parsed:
                fmt = (parsed.format or "").upper()
                if parsed.width <= 0 or parsed.height <= 0:
                    raise ValueError("tool image dimensions are invalid")
                if parsed.width * parsed.height > _MAX_TOOL_IMAGE_PIXELS:
                    raise ValueError("tool image exceeds safe pixel limit")
                parsed.load()
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError("image must be a complete valid PNG or JPEG") from exc
        if fmt == "PNG":
            # Pillow may decode a truncated PNG without its terminal IEND chunk.
            if not data.endswith(b"\x00\x00\x00\x00IEND\xaeB`\x82"):
                raise ValueError("image must be a complete valid PNG or JPEG")
            media_type = "image/png"
        elif fmt in {"JPEG", "JPG"}:
            # A complete JPEG stream terminates with the EOI marker.
            if not data.endswith(b"\xff\xd9"):
                raise ValueError("image must be a complete valid PNG or JPEG")
            media_type = "image/jpeg"
        else:
            raise ValueError("only PNG and JPEG tool images are supported")
        return _ToolImage(data=data, media_type=media_type)

    @classmethod
    def _normalize_tool_result(cls, value: Any) -> _ExecutedToolResult:
        if isinstance(value, str):
            return _ExecutedToolResult(value)
        if isinstance(value, ToolResult):
            images = value.images or []
            if len(images) > _MAX_TOOL_IMAGES:
                raise ValueError("tool result exceeds 3 image limit")
            return _ExecutedToolResult(value.content, tuple(cls._load_image(image) for image in images))
        return _ExecutedToolResult(json.dumps(value, ensure_ascii=False, default=str))

    async def _execute_tool(self, name: str, input_args: dict) -> _ExecutedToolResult:
        started = time.monotonic()
        fn = self._tool_callables.get(name)
        if fn is None:
            error = f"Unknown tool: {name}"
            result = _ExecutedToolResult(json.dumps({"error": error}))
            self._log_tool(name, input_args, result, started, error)
            return result
        error: str | None = None

        async def invoke() -> tuple[_ExecutedToolResult, bool]:
            nonlocal error
            try:
                value = fn(**input_args)
                if inspect.isawaitable(value):
                    value = await value
                result = self._normalize_tool_result(value)
                explicitly_cacheable = any(
                    fn is candidate
                    for candidate in self._cacheable_tool_callables
                )
                cacheable = explicitly_cacheable and isinstance(value, str)
                if cacheable and _is_research_search_callable(fn):
                    cacheable = value.startswith(_RESEARCH_SUCCESS_PREFIXES)
                return result, cacheable
            except Exception as exc:
                error = str(exc)
                logger.warning(
                    "Tool '%s' failed for agent '%s': %s",
                    name, self.agent_name, exc,
                )
                return _ExecutedToolResult(json.dumps({"error": error})), False

        if self._tool_result_cache is None:
            result, _ = await invoke()
        else:
            result = await self._tool_result_cache.execute(
                id(fn), name, input_args, invoke
            )
        self._log_tool(name, input_args, result, started, error)
        return result

    def _log_tool(
        self, name: str, args: dict, result: _ExecutedToolResult,
        started: float, error: str | None = None,
    ) -> None:
        if self._audit:
            self._audit.log_tool_call(
                self.agent_name, name, args,
                response_size=len(result.text) + sum(len(image.data) for image in result.images),
                duration_ms=(time.monotonic() - started) * 1000,
                error=error,
                round_num=self._round_num,
            )

    async def send(self, user_message: str) -> str:
        started = time.monotonic()
        if self._audit:
            self._audit.log_call_start(
                self.agent_name, user_message, round_num=self._round_num,
                system_prompt_size_chars=len(self.system_prompt),
                system_prompt_size_tokens_est=len(self.system_prompt) // 4,
                skill_tools_loaded=[tool.get("name", "") for tool in self.tools],
            )
        self.messages.append({"role": "user", "content": user_message})
        try:
            result = await self._send_openai() if self.provider == "openai" else await self._send_anthropic()
        except Exception as exc:
            if self._audit:
                self._audit.log_error(
                    self.agent_name, str(exc),
                    duration_ms=(time.monotonic() - started) * 1000,
                    round_num=self._round_num,
                )
                self._audit.log_call_end(
                    self.agent_name, (time.monotonic() - started) * 1000,
                    stop_reason="error", messages_count=len(self.messages),
                    round_num=self._round_num,
                )
            raise
        if self._audit:
            self._audit.log_call_end(
                self.agent_name, (time.monotonic() - started) * 1000,
                result, messages_count=len(self.messages), round_num=self._round_num,
            )
        return result

    async def _send_anthropic(self) -> str:
        for iteration in range(_MAX_TOOL_ITERATIONS):
            response = await self._client.messages.create(**self._build_anthropic_kwargs())
            self.messages.append({"role": "assistant", "content": [self._block_to_dict(b) for b in response.content]})
            tool_uses = [b for b in response.content if b.type == "tool_use"]
            if not tool_uses:
                text = "\n".join(b.text for b in response.content if b.type == "text")
                self._log_response(text, iteration)
                return text
            results = []
            image_count = 0
            for block in tool_uses:
                logger.info("Agent '%s' calling tool '%s' with args: %s", self.agent_name, block.name, block.input)
                result = await self._execute_tool(block.name, block.input)
                if image_count + len(result.images) > _MAX_TOOL_IMAGES:
                    result = _ExecutedToolResult(json.dumps({
                        "error": "tool results exceed 3 image limit for one model turn"
                    }))
                image_count += len(result.images)
                if result.images:
                    content: str | list[dict[str, Any]] = []
                    if result.text:
                        content.append({"type": "text", "text": result.text})
                    for image in result.images:
                        content.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": image.media_type,
                                "data": base64.b64encode(image.data).decode("ascii"),
                            },
                        })
                else:
                    # Preserve the legacy text-only payload shape exactly.
                    content = result.text
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": content})
            self.messages.append({"role": "user", "content": results})
        return self._max_iterations_text()

    async def _send_openai(self) -> str:
        for iteration in range(_MAX_TOOL_ITERATIONS):
            response = await self._client.chat.completions.create(**self._build_openai_kwargs())
            message = response.choices[0].message
            calls = message.tool_calls or []
            assistant_message: dict[str, Any] = {"role": "assistant", "content": message.content}
            if calls:
                assistant_message["tool_calls"] = [self._openai_tool_call_to_dict(call) for call in calls]
            self.messages.append(assistant_message)
            if not calls:
                text = message.content or ""
                self._log_response(text, iteration)
                return text

            pending_images: list[_ToolImage] = []
            for call in calls:
                name = call.function.name
                try:
                    args = json.loads(call.function.arguments or "{}")
                    if not isinstance(args, dict):
                        raise ValueError("tool arguments must be a JSON object")
                    result = await self._execute_tool(name, args)
                except (json.JSONDecodeError, ValueError) as exc:
                    error = f"Invalid tool arguments: {exc}"
                    result = _ExecutedToolResult(json.dumps({"error": error}))
                    # Argument parsing failed before the common executor.
                    if self._audit:
                        self._audit.log_tool_call(
                            self.agent_name, name, {"raw": call.function.arguments},
                            response_size=len(result.text), duration_ms=0.0,
                            error=error, round_num=self._round_num,
                        )
                if len(pending_images) + len(result.images) > _MAX_TOOL_IMAGES:
                    result = _ExecutedToolResult(json.dumps({
                        "error": "tool results exceed 3 image limit for one model turn"
                    }))
                self.messages.append({"role": "tool", "tool_call_id": call.id, "content": result.text})
                pending_images.extend(result.images)
            if pending_images:
                content: list[dict[str, Any]] = [{"type": "text", "text": "Images returned by the preceding tool calls."}]
                for image in pending_images:
                    data_uri = f"data:{image.media_type};base64,{base64.b64encode(image.data).decode('ascii')}"
                    content.append({"type": "image_url", "image_url": {"url": data_uri, "detail": "high"}})
                self.messages.append({"role": "user", "content": content})
        return self._max_iterations_text()

    def _max_iterations_text(self) -> str:
        logger.warning("Agent '%s' hit max tool iterations (%d)", self.agent_name, _MAX_TOOL_ITERATIONS)
        if not self.messages or self.messages[-1]["role"] != "assistant":
            return ""
        content = self.messages[-1].get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(x["text"] for x in content if isinstance(x, dict) and x.get("type") == "text")
        return ""

    def _log_response(self, text: str, iteration: int) -> None:
        logger.info("Agent '%s' responded (%d chars, %d messages, %d tool iterations)", self.agent_name, len(text), len(self.messages), iteration)

    @staticmethod
    def _openai_tool_call_to_dict(call) -> dict:
        return {
            "id": call.id,
            "type": getattr(call, "type", "function"),
            "function": {"name": call.function.name, "arguments": call.function.arguments},
        }

    @staticmethod
    def _block_to_dict(block) -> dict:
        if block.type == "text":
            return {"type": "text", "text": block.text}
        if block.type == "tool_use":
            return {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
        return {"type": block.type}

    def get_history_length(self) -> int:
        return len(self.messages)
