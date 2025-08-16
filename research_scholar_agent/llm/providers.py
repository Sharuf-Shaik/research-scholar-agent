from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional, Protocol, TypedDict

import httpx
from loguru import logger

from ..config import get_settings

try:
	from openai import AsyncOpenAI  # type: ignore
except Exception:  # pragma: no cover - optional dependency resolution at runtime
	AsyncOpenAI = None  # type: ignore


class ChatMessage(TypedDict):
	role: str
	content: str


class LLMProvider(Protocol):
	async def generate(self, messages: List[ChatMessage], model: Optional[str] = None, temperature: float = 0.2, max_tokens: Optional[int] = None) -> str:  # noqa: D401
		"""Generate a chat completion."""


class OpenAIProvider:
	def __init__(self) -> None:
		settings = get_settings()
		if not settings.openai_api_key:
			raise RuntimeError("OPENAI_API_KEY is not set")
		if AsyncOpenAI is None:
			raise RuntimeError("openai package not available")
		self._client = AsyncOpenAI(api_key=settings.openai_api_key)
		self._default_model = "gpt-4o-mini"

	async def generate(self, messages: List[ChatMessage], model: Optional[str] = None, temperature: float = 0.2, max_tokens: Optional[int] = None) -> str:
		chosen_model = model or self._default_model
		resp = await self._client.chat.completions.create(
			model=chosen_model,
			messages=[{"role": m["role"], "content": m["content"]} for m in messages],
			temperature=temperature,
			max_tokens=max_tokens,
		)
		return resp.choices[0].message.content or ""


class TogetherProvider:
	def __init__(self) -> None:
		self._settings = get_settings()
		if not self._settings.together_api_key:
			raise RuntimeError("TOGETHER_API_KEY is not set")
		self._default_model = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
		self._base_url = "https://api.together.xyz/v1/chat/completions"

	async def generate(self, messages: List[ChatMessage], model: Optional[str] = None, temperature: float = 0.2, max_tokens: Optional[int] = None) -> str:
		chosen_model = model or self._default_model
		headers = {
			"Authorization": f"Bearer {self._settings.together_api_key}",
			"Content-Type": "application/json",
		}
		payload = {
			"model": chosen_model,
			"messages": messages,
			"temperature": temperature,
		}
		if max_tokens is not None:
			payload["max_tokens"] = max_tokens
		async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
			resp = await client.post(self._base_url, headers=headers, content=json.dumps(payload))
			resp.raise_for_status()
			data = resp.json()
			return data["choices"][0]["message"]["content"]


class OllamaProvider:
	def __init__(self) -> None:
		self._settings = get_settings()
		self._default_model = "llama3.1"
		self._base_url = self._settings.ollama_base_url.rstrip("/")

	async def generate(self, messages: List[ChatMessage], model: Optional[str] = None, temperature: float = 0.2, max_tokens: Optional[int] = None) -> str:
		chosen_model = model or self._default_model
		url = f"{self._base_url}/api/chat"
		payload: Dict[str, Any] = {
			"model": chosen_model,
			"messages": messages,
			"options": {"temperature": temperature},
		}
		if max_tokens is not None:
			payload["options"]["num_predict"] = max_tokens
		async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
			resp = await client.post(url, json=payload)
			resp.raise_for_status()
			data = resp.json()
			# Ollama returns a streaming-like array if stream=false not set; try to handle final content
			if isinstance(data, dict) and "message" in data:
				return data["message"].get("content", "")
			# If it's a stream of events, concatenate
			if isinstance(data, list):
				parts = []
				for chunk in data:
					msg = chunk.get("message", {})
					if "content" in msg:
						parts.append(msg["content"]) 
				return "".join(parts)
			return ""


def get_default_llm_provider(preferred: Optional[str] = None) -> LLMProvider:
	settings = get_settings()
	choice = preferred or settings.default_llm_provider
	if choice:
		choice = choice.lower()
		if choice == "openai":
			return OpenAIProvider()
		if choice == "together":
			return TogetherProvider()
		if choice == "ollama":
			return OllamaProvider()
	# Auto-detect order: OpenAI -> Together -> Ollama
	try:
		return OpenAIProvider()
	except Exception as e:
		logger.debug(f"OpenAI provider not available: {e}")
	try:
		return TogetherProvider()
	except Exception as e:
		logger.debug(f"Together provider not available: {e}")
	return OllamaProvider()