"""Safety Gate v1 (Stage 4)
Minimal high-risk detection & escalation messaging.
"""
from __future__ import annotations

from typing import List

HIGH_RISK_KEYWORDS: List[str] = [
    "自殺", "結束生命", "想死", "不想活", "殺了我", "傷害自己", "割腕", "跳樓", "服藥過量",
    "suicide", "kill myself", "end my life", "self-harm", "hurt myself", "overdose",
]


def contains_high_risk(text: str) -> bool:
    if not text:
        return False
    lower = text.lower()
    for kw in HIGH_RISK_KEYWORDS:
        if kw.lower() in lower:
            return True
    return False


def assess(plan: dict, user_msg: str) -> str:
    # Priority 1: planner risk flag
    try:
        if isinstance(plan, dict) and plan.get("risk", {}).get("level") == "high":
            return "high"
    except Exception:
        pass
    # Priority 2: keyword scan
    if contains_high_risk(user_msg or ""):
        return "high"
    return "low"  # v1 only distinguishes low/high


def escalation_message(user_msg: str) -> str:
    return (
        "I hear how intense and difficult this feels, and your safety is very important. "
        "If you may be in immediate danger or at risk of harming yourself, please contact local emergency services right now. "
        "Consider reaching out to a trusted friend, family member, or a qualified mental health professional for support. "
        "This system can offer emotional support but it cannot replace professional or emergency care. "
        "You deserve help and you do not have to face this alone."
    )


__all__ = ["assess", "escalation_message", "contains_high_risk", "HIGH_RISK_KEYWORDS"]
