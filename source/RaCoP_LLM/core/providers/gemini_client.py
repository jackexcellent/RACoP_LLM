"""Gemini Client Wrapper (Google Generative AI) for Planner/Responder integration.

Usage is intentionally similar to OpenAIClient.generate to allow easy swapping.
Reads API key from environment variable GEMINI_API_KEY (or GOOGLE_API_KEY fallback).
"""
from __future__ import annotations

from typing import Optional
import os

try:  # optional dotenv
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

try:
    import google.generativeai as genai  # type: ignore
except Exception:  # pragma: no cover
    genai = None  # type: ignore


class GeminiClient:
    """Minimal wrapper around google-generativeai Gemini models.

    generate() returns empty string on failure (caller handles fallback).
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.0-flash"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", "")
        self.model_name = model
        self._model = None
        if genai and self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self._model = genai.GenerativeModel(self.model_name)
            except Exception:
                self._model = None

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> str:
        if not self._model:
            print("GeminiClient: model not initialized.")
            return ""
        try:
            # Compose a single prompt: emulate system+user separation
            full_prompt = f"[SYSTEM]\n{system_prompt}\n[USER]\n{user_prompt}"
            response = self._model.generate_content(
                full_prompt,
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                },
            )
            if not response:
                return ""
            # google-generativeai returns .text for aggregated text
            return getattr(response, "text", "") or ""
        except Exception:
            return ""


__all__ = ["GeminiClient"]
