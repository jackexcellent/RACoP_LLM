"""Conversation coordinator (Stage 4)
Orchestrates Planner -> Safety Gate -> Responder.
"""
from __future__ import annotations

from typing import Dict, Any

from core.pipeline.planner import generate_plan  # type: ignore
from core.pipeline import safety  # type: ignore
from core.pipeline.responder import generate_response  # type: ignore


APOLOGY_MSG = (
    "Sorry, something went wrong generating a response. Please try again or rephrase your message."
)


def run_once(user_msg: str) -> str:
    """Run one full pipeline turn.

    Steps:
    1. Planner generate_plan -> plan (dict)
    2. Safety assess(plan, user_msg) -> risk
       - If high -> escalation message
    3. Responder generate_response(plan, user_msg)
    Any exception returns a safe apology message.
    """
    try:
        plan: Dict[str, Any]
        try:
            plan = generate_plan(user_msg)
        except Exception:
            return APOLOGY_MSG

        try:
            risk = safety.assess(plan, user_msg)
        except Exception:
            risk = "low"

        if risk == "high":
            try:
                return safety.escalation_message(user_msg)
            except Exception:
                return APOLOGY_MSG

        try:
            return generate_response(plan, user_msg)
        except Exception:
            return APOLOGY_MSG
    except Exception:
        return APOLOGY_MSG


__all__ = ["run_once"]
