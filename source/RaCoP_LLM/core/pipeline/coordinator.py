"""Conversation coordinator (Stage 6)
Orchestrates: Memory -> Planner -> Safety/DBT Gate -> Responder + persistence.
"""
from __future__ import annotations

from typing import Dict, Any

from core.pipeline.planner import generate_plan  # type: ignore
from core.pipeline import safety  # type: ignore
from core.pipeline.responder import generate_response  # type: ignore
from core.pipeline import memory  # type: ignore


APOLOGY_MSG = (
    "Sorry, something went wrong generating a response. Please try again or rephrase your message."
)


def run_once(user_msg: str, session_id: str = "default") -> str:
    """Run one full pipeline turn with short-term memory persistence.

    1. Load short context summary (last N turns) for session.
    2. Planner generate_plan(user_msg, recent_ctx=short_ctx)
    3. Safety assess(plan, user_msg)
       - If high -> escalation message
    4. Responder generate_response(plan, user_msg, short_ctx=short_ctx)
    5. Append user + assistant turns to JSONL with plan summary meta.
    """
    # Always attempt to store user message even if later steps fail.
    short_ctx = ""
    try:
        short_ctx = memory.get_short_context(session_id, turns=12)
    except Exception:
        short_ctx = ""

    try:
        try:
            plan: Dict[str, Any] = generate_plan(user_msg, recent_ctx=short_ctx)
        except Exception:
            plan = {"risk": {"level": "low"}}  # minimal placeholder

        try:
            risk = safety.assess(plan, user_msg)
        except Exception:
            risk = "low"

        import os
        rag_enabled = os.getenv("RAG_ENABLED", "1") != "0"
        kb_snippets = []

        if risk == "high":
            try:
                assistant_text = safety.escalation_message(user_msg)
            except Exception:
                assistant_text = APOLOGY_MSG
        elif safety.requires_professional_for_dbt(plan):
            try:
                assistant_text = safety.refer_to_professional(user_msg)
            except Exception:
                assistant_text = APOLOGY_MSG
        else:
            try:
                if rag_enabled:
                    try:
                        # Lazy import to avoid cost when disabled
                        from core.pipeline import retriever  # type: ignore
                        retriever.ensure_index()
                        queries = plan.get("retrieval_queries") or []
                        if isinstance(queries, list) and queries:
                            kb_snippets = retriever.search(queries, top_k=5)
                    except Exception:
                        kb_snippets = []
                assistant_text = generate_response(
                    plan,
                    user_msg,
                    short_ctx=short_ctx,
                    kb_snippets=kb_snippets,
                )
            except Exception:
                assistant_text = APOLOGY_MSG

        # Persist turns
        try:
            memory.append_session(session_id, "user", user_msg)
        except Exception:
            pass
        try:
            memory.append_session(
                session_id,
                "assistant",
                assistant_text,
                meta={"plan": memory.plan_summary(plan)},
            )
        except Exception:
            pass
        return assistant_text
    except Exception:
        # Attempt to store user even on catastrophic failure
        try:
            memory.append_session(session_id, "user", user_msg)
        except Exception:
            pass
        return APOLOGY_MSG


__all__ = ["run_once"]
