"""Responder 模組 (Stage 1)
根據 planner 的硬編計畫，產生簡單 PCT 同理回覆。
"""
from __future__ import annotations

from typing import Dict, Any
import re


FALLBACKS = {
    "starter": "I’m here with you as you share this.",
    "validation": "It makes sense that you’d feel this way.",
    "question": "What would be most supportive for you right now?",
}


def safe_snippet(text: str, max_len: int = 40) -> str:
    """取一段安全片語：
    - 僅保留英數與常見標點與空白
    - 去頭尾空白
    - 截斷後加上 …
    """
    if not text:
        return ""
    cleaned = re.sub(r"[^A-Za-z0-9 ,.;:'?!-]", "", text).strip()
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[:max_len].rstrip() + "…"


def respond(plan: Dict[str, Any], user_msg: str) -> str:
    slots = plan.get("template_slots", {}).get("pct", {}) if isinstance(plan, dict) else {}

    starter = slots.get("starter") or FALLBACKS["starter"]
    validation = slots.get("validation") or FALLBACKS["validation"]
    question = slots.get("question") or FALLBACKS["question"]

    snippet = safe_snippet(user_msg, 50)
    snippet_part = f" about \"{snippet}\"" if snippet else ""

    sentences = [
        f"{starter}{snippet_part}.",
        validation,
        question,
    ]
    return " " .join(s.strip() for s in sentences if s).strip()


__all__ = ["respond", "safe_snippet"]
