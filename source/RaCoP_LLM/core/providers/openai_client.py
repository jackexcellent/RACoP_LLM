"""OpenAI Client Wrapper for Planner Stage 2.

Handles loading environment variables (via python-dotenv if available) and
provides a thin generate() method for unified interface.
"""
from __future__ import annotations

from typing import Optional
import os

try:  # optional .env loading
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()  # load from .env if present
except Exception:
    pass


try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


class OpenAIClient:
    """Minimal wrapper around OpenAI SDK.

    If the SDK or API key is not available, generate() returns an empty string.
    Caller (planner) will treat empty string as failure and fallback.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self._client = None
        if OpenAI is not None and self.api_key:
            try:
                self._client = OpenAI(api_key=self.api_key)
            except Exception:
                self._client = None

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str = "gpt-4o-mini",
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> str:
        """Send a chat completion style request and return raw text.

        Returns empty string on failure to allow graceful fallback.
        """
        if not self._client:
            return ""
        try:
            # Prefer new Responses API if available; else fallback to chat.completions
            if hasattr(self._client, "responses"):
                resp = self._client.responses.create(
                    model=model,
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    input=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                # Attempt to extract text
                for out in getattr(resp, "output", []) or []:
                    if getattr(out, "type", None) == "message":
                        for c in getattr(out, "content", []) or []:
                            if getattr(c, "type", None) == "output_text":
                                return getattr(c, "text", "")
                # Fallback generic str
                return getattr(resp, "output_text", "") or ""
            else:  # legacy style
                chat = self._client.chat.completions.create(
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                choice = (chat.choices or [None])[0]
                if choice and getattr(choice, "message", None):
                    return getattr(choice.message, "content", "") or ""
                return ""
        except Exception:
            return ""


__all__ = ["OpenAIClient"]
