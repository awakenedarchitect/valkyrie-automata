"""
llm.py — LLM Provider Implementations

Concrete providers that satisfy weave.py's LLM Protocol:

    class LLM(Protocol):
        async def complete(self, messages: list[dict[str, str]], **kwargs) -> str: ...

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
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)


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


# ── OpenRouter ──────────────────────────────────────────────────────

class OpenRouterLLM:
    """OpenRouter — access hundreds of models through one API.

    Most Moltbook bots use OpenRouter. Cheapest path to deployment.

    Usage:
        llm = OpenRouterLLM(
            api_key="sk-or-...",
            model="meta-llama/llama-3.1-8b-instruct",
        )
        response = await llm.complete(messages)

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
            raise LLMError(
                "OpenRouter API key required. Set OPENROUTER_API_KEY or pass api_key."
            )

    @property
    def model(self) -> str:
        return self._model

    async def complete(self, messages: list[dict[str, str]], **kwargs) -> str:
        """Send messages, get response text."""
        body = {
            "model": kwargs.get("model", self._model),
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", self._max_tokens),
            "temperature": kwargs.get("temperature", self._temperature),
        }

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "HTTP-Referer": f"https://github.com/awakenedarchitect/{self._app_name}",
            "X-Title": self._app_name,
        }

        # run sync HTTP in executor to not block async loop
        loop = asyncio.get_event_loop()
        resp = await loop.run_in_executor(
            None, lambda: _post_json(self.API_URL, body, headers, self._timeout)
        )

        return _extract_content(resp)


# ── Ollama ──────────────────────────────────────────────────────────

class OllamaLLM:
    """Ollama — run models locally. Free, no API key.

    Usage:
        llm = OllamaLLM(model="llama3.1:8b")
        response = await llm.complete(messages)

    Requires Ollama running locally: https://ollama.ai
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

    async def complete(self, messages: list[dict[str, str]], **kwargs) -> str:
        """Send messages, get response text."""
        url = f"{self._host.rstrip('/')}/api/chat"

        body = {
            "model": kwargs.get("model", self._model),
            "messages": messages,
            "stream": False,
            "options": {
                "num_predict": kwargs.get("max_tokens", self._max_tokens),
                "temperature": kwargs.get("temperature", self._temperature),
            },
        }

        loop = asyncio.get_event_loop()
        resp = await loop.run_in_executor(
            None, lambda: _post_json(url, body, timeout=self._timeout)
        )

        # Ollama format: {"message": {"content": "..."}}
        msg = resp.get("message", {})
        content = msg.get("content", "")
        if not content:
            raise LLMError(f"Empty response from Ollama: {resp}")
        return content


# ── Anthropic ───────────────────────────────────────────────────────

class AnthropicLLM:
    """Anthropic Claude API.

    Usage:
        llm = AnthropicLLM(api_key="sk-ant-...", model="claude-sonnet-4-20250514")
        response = await llm.complete(messages)

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
            raise LLMError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY or pass api_key."
            )

    @property
    def model(self) -> str:
        return self._model

    async def complete(self, messages: list[dict[str, str]], **kwargs) -> str:
        """Send messages, get response text.

        Converts from OpenAI-style messages to Anthropic format:
        - system messages → top-level "system" param
        - user/assistant messages → "messages" array
        """
        system_parts = []
        chat_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_parts.append(msg["content"])
            else:
                chat_messages.append({
                    "role": msg["role"],
                    "content": msg["content"],
                })

        # Anthropic requires alternating user/assistant
        # If chat_messages is empty or doesn't start with user, add a stub
        if not chat_messages or chat_messages[0]["role"] != "user":
            chat_messages.insert(0, {"role": "user", "content": "Hello."})

        body: dict[str, Any] = {
            "model": kwargs.get("model", self._model),
            "messages": chat_messages,
            "max_tokens": kwargs.get("max_tokens", self._max_tokens),
            "temperature": kwargs.get("temperature", self._temperature),
        }

        if system_parts:
            body["system"] = "\n\n".join(system_parts)

        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
        }

        loop = asyncio.get_event_loop()
        resp = await loop.run_in_executor(
            None, lambda: _post_json(self.API_URL, body, headers, self._timeout)
        )

        # Anthropic format: {"content": [{"type": "text", "text": "..."}]}
        content_blocks = resp.get("content", [])
        text_parts = [
            b.get("text", "") for b in content_blocks if b.get("type") == "text"
        ]
        result = "\n".join(text_parts)
        if not result:
            raise LLMError(f"Empty response from Anthropic: {resp}")
        return result


# ── OpenAI-compatible ───────────────────────────────────────────────

class OpenAICompatibleLLM:
    """Any OpenAI-compatible endpoint (vLLM, LM Studio, Together, etc.)

    Usage:
        llm = OpenAICompatibleLLM(
            base_url="http://localhost:8000/v1",
            model="meta-llama/Llama-3.1-8B-Instruct",
        )
        response = await llm.complete(messages)

    Also works with actual OpenAI:
        llm = OpenAICompatibleLLM(
            base_url="https://api.openai.com/v1",
            api_key="sk-...",
            model="gpt-4o-mini",
        )
    """

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

    async def complete(self, messages: list[dict[str, str]], **kwargs) -> str:
        """Send messages, get response text."""
        url = f"{self._base_url}/chat/completions"

        body = {
            "model": kwargs.get("model", self._model),
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", self._max_tokens),
            "temperature": kwargs.get("temperature", self._temperature),
        }

        headers = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        loop = asyncio.get_event_loop()
        resp = await loop.run_in_executor(
            None, lambda: _post_json(url, body, headers, self._timeout)
        )

        return _extract_content(resp)


# ── helpers ─────────────────────────────────────────────────────────

def _extract_content(resp: dict) -> str:
    """Extract text content from OpenAI-format response."""
    choices = resp.get("choices", [])
    if not choices:
        raise LLMError(f"No choices in response: {resp}")
    msg = choices[0].get("message", {})
    content = msg.get("content", "")
    if not content:
        raise LLMError(f"Empty content in response: {resp}")
    return content


# ── factory ─────────────────────────────────────────────────────────

def create_llm(config: dict) -> Any:
    """Create an LLM provider from a config dict.

    Config format:
    {
        "provider": "openrouter" | "ollama" | "anthropic" | "openai",
        "model": "meta-llama/llama-3.1-8b-instruct",
        "api_key": "...",          # optional, falls back to env var
        "base_url": "...",         # for openai-compatible
        "max_tokens": 1024,
        "temperature": 0.7,
    }
    """
    provider = config.get("provider", "").lower()
    model = config.get("model", "")
    api_key = config.get("api_key", "")
    max_tokens = config.get("max_tokens", 1024)
    temperature = config.get("temperature", 0.7)

    if provider == "openrouter":
        return OpenRouterLLM(
            api_key=api_key,
            model=model or "meta-llama/llama-3.1-8b-instruct",
            max_tokens=max_tokens,
            temperature=temperature,
        )

    elif provider == "ollama":
        return OllamaLLM(
            model=model or "llama3.1:8b",
            host=config.get("base_url", ""),
            max_tokens=max_tokens,
            temperature=temperature,
        )

    elif provider == "anthropic":
        return AnthropicLLM(
            api_key=api_key,
            model=model or "claude-sonnet-4-20250514",
            max_tokens=max_tokens,
            temperature=temperature,
        )

    elif provider in ("openai", "vllm", "lmstudio", "together"):
        return OpenAICompatibleLLM(
            base_url=config.get("base_url", "https://api.openai.com/v1"),
            api_key=api_key,
            model=model or "gpt-4o-mini",
            max_tokens=max_tokens,
            temperature=temperature,
        )

    else:
        raise LLMError(
            f"Unknown provider '{provider}'. "
            f"Options: openrouter, ollama, anthropic, openai"
        )