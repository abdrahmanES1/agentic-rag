# -*- coding: utf-8 -*-
"""
Shared infrastructure for all benchmarking baselines.

Provides:
  - OllamaClient: thin OpenAI-SDK wrapper pointing at Ollama
  - api_retrieve(): calls /api/retrieve and returns contexts
  - api_ask(): calls /api/ask and returns the full response dict
  - RETRIEVE_URL, ASK_URL: default endpoint URLs
"""

import logging
import time
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

log = logging.getLogger("benchmarking")

# Default endpoints — override with env vars or constructor args
RETRIEVE_URL = "http://localhost:5000/api/retrieve"
ASK_URL = "http://localhost:5000/api/ask"
OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_MODEL = "gemma4:e4b"
REQUEST_TIMEOUT = 300


class OllamaClient:
    """
    Thin OpenAI-SDK wrapper for Ollama.
    Shared by all baselines so configuration is centralized.
    """

    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        model: str = OLLAMA_MODEL,
        api_key: str = "ollama",
    ):
        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=REQUEST_TIMEOUT)
        self.model = model

    def generate(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.3,
        max_tokens: int = 2048,
        repetition_penalty: float = 1.0,
    ) -> str:
        """Simple prompt → response. Returns empty string on failure."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
                extra_body={
                    "options": {
                        "num_ctx": 8192,
                        "repeat_penalty": repetition_penalty,  # Ollama param name
                    },
                    "think": False,
                    "repetition_penalty": repetition_penalty,  # LM Studio param name
                },
            )
            content = resp.choices[0].message.content
            if not content:
                reasoning = getattr(resp.choices[0].message, "reasoning_content", None)
                if reasoning:
                    content = reasoning
            return (content or "").strip()
        except Exception as exc:
            log.warning(f"[OllamaClient] generate failed: {exc}")
            return ""

    def generate_messages(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 2048,
        fmt: Any = None,
    ) -> str:
        """Full messages list for structured-output calls."""
        try:
            kwargs: Dict[str, Any] = dict(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
                extra_body={"options": {"num_ctx": 8192}, "think": False},
            )
            if fmt is not None:
                kwargs["response_format"] = {"type": "json_object"} if fmt == "json" else fmt
            resp = self._client.chat.completions.create(**kwargs)
            content = resp.choices[0].message.content
            if not content:
                reasoning = getattr(resp.choices[0].message, "reasoning_content", None)
                if reasoning:
                    content = reasoning
            return (content or "").strip()
        except Exception as exc:
            log.warning(f"[OllamaClient] generate_messages failed: {exc}")
            return ""


def api_retrieve(
    question: str,
    top_k: int = 5,
    url: str = RETRIEVE_URL,
) -> Dict:
    """
    Call /api/retrieve and return the full response dict.

    Returns:
        {
            "chunks": [...],      # List[{text, source, page, ...}]
            "contexts": [...],    # List[str]  — chunk texts
            "context": str,       # formatted context string
            "is_outscope": bool,
            ...
        }
    """
    try:
        resp = requests.post(url, json={"question": question, "top_k": top_k}, timeout=REQUEST_TIMEOUT)
        if resp.status_code == 200:
            return resp.json()
        log.warning(f"[api_retrieve] HTTP {resp.status_code}")
        return {}
    except Exception as exc:
        log.warning(f"[api_retrieve] failed: {exc}")
        return {}


def api_ask(
    question: str,
    mode: str = "research",
    url: str = ASK_URL,
) -> Dict:
    """Call /api/ask (mode=research) and return the full response dict."""
    try:
        resp = requests.post(url, json={"question": question, "mode": mode}, timeout=REQUEST_TIMEOUT)
        if resp.status_code == 200:
            return resp.json()
        log.warning(f"[api_ask] HTTP {resp.status_code}")
        return {}
    except Exception as exc:
        log.warning(f"[api_ask] failed: {exc}")
        return {}


def format_context(chunks: List[Dict]) -> str:
    """Format a list of chunk dicts into a context string for prompts."""
    parts = [
        f"[Source: {c.get('source', '?')} | Page: {c.get('page', '?')}]\n{c.get('text', '')}"
        for c in chunks
    ]
    return "\n\n".join(parts)


def get_contexts_from_response(response: Dict) -> List[str]:
    """Extract List[str] contexts from an /api/retrieve or /api/ask response."""
    if "contexts" in response:
        return response["contexts"]
    if "chunks" in response:
        return [c.get("text", "") for c in response["chunks"]]
    return []
