"""
llm.py — LLM Provider Implementations

Concrete providers that satisfy weave.py's needs:

    # simple text completion (backward compat)
    async def complete(messages, **kwargs) -> str

    # full response with tool calls
    async def chat(messages, tools=None, **kwargs) -> LLMResponse

Supported:
  - OpenRouter (most Moltbook bots use this, huge model selection)
  - Ollama (local, free, no API key needed)
  - Anthropic (Claude API)
  - OpenAI-compatible (vLLM, LM Studio, any OpenAI-format endpoint)

Zero external dependencies. Pure stdlib.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


# ── response types ──────────────────────────────────────────────────

@dataclass
class ToolCall:
    """A tool call from the LLM."""
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    """LLM response that may contain text, tool calls, or both."""
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0

    @property
    def text(self) -> str:
        return self.content or ""


# ── base HTTP ───────────────────────────────────────────────────────

def _post_json(
    url: str,
    body: dict,
    headers: dict[str, str] | None = None,
    timeout: int = 120,
) -> dict:
    """Synchronous JSON POST. Pure stdlib."""
    data = json.dumps(body).encode()
    hdrs = {"Content-Type": "application/json"}
    if headers:
        hdrs.update(headers)

    req = urllib.request.Request(url, data=data, headers=hdrs, method="POST")

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        raw = e.read().decode() if e.fp else ""
        log.error("LLM API error %d: %s", e.code, raw[:500])
        raise LLMError(f"HTTP {e.code}: {raw[:200]}")
    except urllib.error.URLError as e:
        raise LLMError(f"Connection failed: {e.reason}")


class LLMError(Exception):
    """LLM provider error."""
    pass


# ── response parsing helpers ────────────────────────────────────────

def _parse_openai_tool_calls(message: dict) -> list[ToolCall]:
    """Parse tool calls from OpenAI-format response message."""
    calls = []
    for tc in message.get("tool_calls", []):
        func = tc.get("function", {})
        args_raw = func.get("arguments", "{}")
        if isinstance(args_raw, str):
            try:
                args = json.loads(args_raw)
            except json.JSONDecodeError:
                args = {"raw": args_raw}
        else:
            args = args_raw
        calls.append(ToolCall(
            id=tc.get("id", ""),
            name=func.get("name", ""),
            arguments=args,
        ))
    return calls


def _parse_openai_response(resp: dict) -> LLMResponse:
    """Parse OpenAI-format response into LLMResponse."""
    choices = resp.get("choices", [])
    if not choices:
        raise LLMError(f"No choices in response: {resp}")
    choice = choices[0]
    message = choice.get("message", {})
    return LLMResponse(
        content=message.get("content"),
        tool_calls=_parse_openai_tool_calls(message),
        finish_reason=choice.get("finish_reason", "stop"),
    )


# ── OpenRouter ──────────────────────────────────────────────────────

class OpenRouterLLM:
    """OpenRouter — access hundreds of models through one API.
    Env: OPENROUTER_API_KEY
    """

    API_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(
        self,
        api_key: str = "",
        model: str = "meta-llama/llama-3.1-8b-instruct",
        *,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        timeout: int = 120,
        app_name: str = "valkyrie",
    ):
        self._api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._timeout = timeout
        self._app_name = app_name
        if not self._api_key:
            raise LLMError("OpenRouter API key required. Set OPENROUTER_API_KEY or pass api_key.")

    @property
    def model(self) -> str:
        return self._model

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "HTTP-Referer": f"https://github.com/awakenedarchitect/{self._app_name}",
            "X-Title": self._app_name,
        }

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> LLMResponse:
        body: dict[str, Any] = {
            "model": kwargs.get("model", self._model),
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", self._max_tokens),
            "temperature": kwargs.get("temperature", self._temperature),
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        loop = asyncio.get_event_loop()
        resp = await loop.run_in_executor(
            None, lambda: _post_json(self.API_URL, body, self._headers(), self._timeout)
        )
        return _parse_openai_response(resp)

    async def complete(self, messages: list[dict[str, str]], **kwargs) -> str:
        result = await self.chat(messages, **kwargs)
        if not result.text:
            raise LLMError("Empty response from OpenRouter")
        return result.text


# ── Ollama ──────────────────────────────────────────────────────────

class OllamaLLM:
    """Ollama — run models locally. Free, no API key.
    Default endpoint: http://localhost:11434
    """

    def __init__(
        self,
        model: str = "llama3.1:8b",
        *,
        host: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.7,
        timeout: int = 180,
    ):
        self._model = model
        self._host = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._timeout = timeout

    @property
    def model(self) -> str:
        return self._model

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> LLMResponse:
        url = f"{self._host.rstrip('/')}/api/chat"
        body: dict[str, Any] = {
            "model": kwargs.get("model", self._model),
            "messages": messages,
            "stream": False,
            "options": {
                "num_predict": kwargs.get("max_tokens", self._max_tokens),
                "temperature": kwargs.get("temperature", self._temperature),
            },
        }
        if tools:
            body["tools"] = tools
        loop = asyncio.get_event_loop()
        resp = await loop.run_in_executor(
            None, lambda: _post_json(url, body, timeout=self._timeout)
        )
        msg = resp.get("message", {})
        content = msg.get("content", "") or None
        # parse Ollama tool calls
        calls = []
        for i, tc in enumerate(msg.get("tool_calls", [])):
            func = tc.get("function", {})
            args = func.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"raw": args}
            calls.append(ToolCall(id=f"call_{i}", name=func.get("name", ""), arguments=args))
        return LLMResponse(content=content, tool_calls=calls, finish_reason="tool_calls" if calls else "stop")

    async def complete(self, messages: list[dict[str, str]], **kwargs) -> str:
        result = await self.chat(messages, **kwargs)
        if not result.text:
            raise LLMError("Empty response from Ollama")
        return result.text


# ── Anthropic ───────────────────────────────────────────────────────

class AnthropicLLM:
    """Anthropic Claude API.
    Env: ANTHROPIC_API_KEY
    """

    API_URL = "https://api.anthropic.com/v1/messages"

    def __init__(
        self,
        api_key: str = "",
        model: str = "claude-sonnet-4-20250514",
        *,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        timeout: int = 120,
    ):
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._timeout = timeout
        if not self._api_key:
            raise LLMError("Anthropic API key required. Set ANTHROPIC_API_KEY or pass api_key.")

    @property
    def model(self) -> str:
        return self._model

    def _headers(self) -> dict[str, str]:
        return {"x-api-key": self._api_key, "anthropic-version": "2023-06-01"}

    def _convert_messages(self, messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
        """Convert OpenAI-style messages to Anthropic format."""
        system_parts: list[str] = []
        chat: list[dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                system_parts.append(content)
            elif role == "tool":
                chat.append({
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": msg.get("tool_call_id", ""), "content": content}],
                })
            elif role == "assistant" and "tool_calls" in msg:
                blocks: list[dict[str, Any]] = []
                if content:
                    blocks.append({"type": "text", "text": content})
                for tc in msg["tool_calls"]:
                    func = tc.get("function", {})
                    args = func.get("arguments", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {}
                    blocks.append({"type": "tool_use", "id": tc.get("id", ""), "name": func.get("name", ""), "input": args})
                chat.append({"role": "assistant", "content": blocks})
            else:
                chat.append({"role": role, "content": content})
        if not chat or chat[0]["role"] != "user":
            chat.insert(0, {"role": "user", "content": "Hello."})
        return "\n\n".join(system_parts), chat

    def _convert_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert OpenAI tool schemas to Anthropic format."""
        return [
            {"name": t["function"]["name"], "description": t["function"]["description"], "input_schema": t["function"].get("parameters", {"type": "object"})}
            for t in tools
        ]

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> LLMResponse:
        system, chat_msgs = self._convert_messages(messages)
        body: dict[str, Any] = {
            "model": kwargs.get("model", self._model),
            "messages": chat_msgs,
            "max_tokens": kwargs.get("max_tokens", self._max_tokens),
            "temperature": kwargs.get("temperature", self._temperature),
        }
        if system:
            body["system"] = system
        if tools:
            body["tools"] = self._convert_tools(tools)
        loop = asyncio.get_event_loop()
        resp = await loop.run_in_executor(
            None, lambda: _post_json(self.API_URL, body, self._headers(), self._timeout)
        )
        # parse Anthropic response
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in resp.get("content", []):
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif block.get("type") == "tool_use":
                tool_calls.append(ToolCall(id=block.get("id", ""), name=block.get("name", ""), arguments=block.get("input", {})))
        content = "\n".join(text_parts) if text_parts else None
        stop = resp.get("stop_reason", "end_turn")
        return LLMResponse(content=content, tool_calls=tool_calls, finish_reason="tool_calls" if stop == "tool_use" else "stop")

    async def complete(self, messages: list[dict[str, str]], **kwargs) -> str:
        result = await self.chat(messages, **kwargs)
        if not result.text:
            raise LLMError("Empty response from Anthropic")
        return result.text


# ── OpenAI-compatible ───────────────────────────────────────────────

class OpenAICompatibleLLM:
    """Any OpenAI-compatible endpoint (vLLM, LM Studio, Together, etc.)"""

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "",
        model: str = "default",
        *,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        timeout: int = 120,
    ):
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._timeout = timeout

    @property
    def model(self) -> str:
        return self._model

    def _headers(self) -> dict[str, str]:
        hdrs: dict[str, str] = {}
        if self._api_key:
            hdrs["Authorization"] = f"Bearer {self._api_key}"
        return hdrs

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> LLMResponse:
        url = f"{self._base_url}/chat/completions"
        body: dict[str, Any] = {
            "model": kwargs.get("model", self._model),
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", self._max_tokens),
            "temperature": kwargs.get("temperature", self._temperature),
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        loop = asyncio.get_event_loop()
        resp = await loop.run_in_executor(
            None, lambda: _post_json(url, body, self._headers(), self._timeout)
        )
        return _parse_openai_response(resp)

    async def complete(self, messages: list[dict[str, str]], **kwargs) -> str:
        result = await self.chat(messages, **kwargs)
        if not result.text:
            raise LLMError("Empty response from LLM")
        return result.text


# ── factory ─────────────────────────────────────────────────────────

def create_llm(config: dict) -> Any:
    """Create an LLM provider from a config dict."""
    provider = config.get("provider", "").lower()
    model = config.get("model", "")
    api_key = config.get("api_key", "")
    max_tokens = config.get("max_tokens", 1024)
    temperature = config.get("temperature", 0.7)

    if provider == "openrouter":
        return OpenRouterLLM(api_key=api_key, model=model or "meta-llama/llama-3.1-8b-instruct", max_tokens=max_tokens, temperature=temperature)
    elif provider == "ollama":
        return OllamaLLM(model=model or "llama3.1:8b", host=config.get("base_url", ""), max_tokens=max_tokens, temperature=temperature)
    elif provider == "anthropic":
        return AnthropicLLM(api_key=api_key, model=model or "claude-sonnet-4-20250514", max_tokens=max_tokens, temperature=temperature)
    elif provider in ("openai", "vllm", "lmstudio", "together"):
        return OpenAICompatibleLLM(base_url=config.get("base_url", "https://api.openai.com/v1"), api_key=api_key, model=model or "gpt-4o-mini", max_tokens=max_tokens, temperature=temperature)
    else:
        raise LLMError(f"Unknown provider '{provider}'. Options: openrouter, ollama, anthropic, openai")