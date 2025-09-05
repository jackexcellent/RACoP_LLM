"""Planner 模組 (Stage 1)
提供硬編的心理支持 (PCT-only) 規劃。
"""
from __future__ import annotations

from typing import Dict, Any


def fake_plan(user_msg: str) -> Dict[str, Any]:
    """忽略 user_msg，回傳固定規劃 dict。

    Returns
    -------
    dict : 預先定義的 CoP 規劃
    """
    return {
        "risk": {"level": "low", "signals": []},
        "emotions": ["anxiety"],
        "distortions": [],
        "plan": [
            {"therapy": "PCT", "goal": "情緒承接與驗證（先同理、先陪伴）", "weight": 1.0}
        ],
        "tone": "warm, validating, non-judgmental",
        "reading_level": "B1",
        "template_slots": {
            "pct": {
                "starter": "I sense the deep emotions you're experiencing",
                "validation": "Your feelings are completely valid and worth exploring.",
                "question": "What do you need most in this moment?",
            }
        },
        "retrieval_queries": [],
        "final_prompt": "",
        "notes": "Stage-1 hardcoded plan focusing on PCT only.",
    }


__all__ = ["fake_plan"]
