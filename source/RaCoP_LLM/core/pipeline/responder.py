"""Responder 模組 (Stage 3)
LLM-B (Gemini) 生成：基於計畫槽位 + 模板 + system prompt。
若金鑰或模型失敗 → 規則式 fallback (starter + validation + question)。
"""
from __future__ import annotations

from typing import Dict, Any, Optional
import os
import re

from core.providers.gemini_client import GeminiClient

FALLBACKS = {
    "starter": "I’m here with you as you share this.",
    "validation": "It makes sense that you’d feel this way.",
    "question": "What would be most supportive for you right now?",
}

BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # core/
PROMPTS_DIR = os.path.join(BASE_DIR, "prompts")
SYSTEM_RESP_PATH = os.path.join(PROMPTS_DIR, "system_resp.txt")
TEMPLATE_PCT_PATH = os.path.join(PROMPTS_DIR, "templates", "pct.md")


def load_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


def safe_snippet(text: str, max_len: int = 60) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"[^A-Za-z0-9 ,.;:'?!-]", "", text).strip()
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[:max_len].rstrip() + "..."


def _assemble_user_prompt(plan: Dict[str, Any], user_msg: str, template: str) -> str:
    slots = plan.get("template_slots", {}).get("pct", {}) if isinstance(plan, dict) else {}
    starter = slots.get("starter") or FALLBACKS["starter"]
    validation = slots.get("validation") or FALLBACKS["validation"]
    question = slots.get("question") or FALLBACKS["question"]
    snippet = safe_snippet(user_msg, 60)
    prompt = template.replace("{{starter}}", starter).replace("{{validation}}", validation).replace("{{question}}", question).replace("{{snippet}}", snippet or "this")
    return prompt


def _rule_based_fallback(plan: Dict[str, Any], user_msg: str) -> str:
    slots = plan.get("template_slots", {}).get("pct", {}) if isinstance(plan, dict) else {}
    starter = slots.get("starter") or FALLBACKS["starter"]
    validation = slots.get("validation") or FALLBACKS["validation"]
    question = slots.get("question") or FALLBACKS["question"]
    snippet = safe_snippet(user_msg, 60)
    mention = f" about \"{snippet}\"" if snippet else ""
    sentences = [f"{starter}{mention}.", validation, question]
    return " ".join(s.strip() for s in sentences if s).strip()


def generate_response(plan: Dict[str, Any], user_msg: str, short_ctx: Optional[str] = None) -> str:
    system_prompt = load_text(SYSTEM_RESP_PATH)
    template = load_text(TEMPLATE_PCT_PATH)
    if not template:
        return _rule_based_fallback(plan, user_msg)

    user_prompt = _assemble_user_prompt(plan, user_msg, template)
    if short_ctx:
        user_prompt += f"\nContext: {safe_snippet(short_ctx, 80)}"

    client = GeminiClient()
    # model parameters (Stage 3 spec): gemini-2.0-flash, temp=0.7, max_tokens ~512
    raw = client.generate(system_prompt=system_prompt, user_prompt=user_prompt, temperature=0.7, max_tokens=512)
    if not raw:
        return _rule_based_fallback(plan, user_msg)

    # Basic post-processing: strip & ensure no JSON remnants (not expected here)
    cleaned = raw.strip()
    # If model accidentally returned fenced code or JSON, attempt to extract plain text lines
    if cleaned.startswith("{") and cleaned.endswith("}"):
        return _rule_based_fallback(plan, user_msg)
    cleaned = re.sub(r"```.*?```", "", cleaned, flags=re.DOTALL).strip()
    return cleaned or _rule_based_fallback(plan, user_msg)


__all__ = ["generate_response", "safe_snippet", "load_text"]
