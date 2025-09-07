"""Conversation coordinator (Stage 8)
Orchestrates: Profile feedback update -> Memory -> Planner (with profile) -> Safety/DBT Gate -> RAG -> Responder + persistence + profile save.
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
    """Run one pipeline turn with profile personalization + RAG (Stage 8)."""
    short_ctx = ""
    try:
        short_ctx = memory.get_short_context(session_id, turns=12)
    except Exception:
        short_ctx = ""

    try:
        # Update profile based on feedback words in current user message about previous reply
        try:
            memory.update_profile_from_feedback(session_id, user_msg)
        except Exception:
            pass

        # Load profile
        try:
            profile = memory.load_profile(session_id)
        except Exception:
            profile = {"effective_skills": [], "ineffective_skills": [], "tone_preference": "warm"}

        # Planner context with profile summary
        try:
            profile_text = memory.profile_summary_text(profile)
        except Exception:
            profile_text = "tone_preference: warm"
        planner_ctx = short_ctx + ("\n\nProfile:\n" + profile_text if short_ctx or profile_text else "")

        # Generate plan
        try:
            plan: Dict[str, Any] = generate_plan(user_msg, recent_ctx=planner_ctx)
        except Exception:
            plan = {"risk": {"level": "low"}}

        try:
            risk = safety.assess(plan, user_msg)
        except Exception:
            risk = "low"

        import os
        rag_enabled = os.getenv("RAG_ENABLED", "1") != "0"
        kb_snippets: list[str] = []

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
                queries = plan.get("retrieval_queries") or []
                if rag_enabled:
                    try:
                        from core.pipeline import retriever  # type: ignore
                        retriever.ensure_index()
                        if isinstance(queries, list) and queries:
                            kb_snippets = retriever.search(queries, top_k=5)
                    except Exception:
                        kb_snippets = []
                assistant_text = generate_response(
                    plan,
                    user_msg,
                    short_ctx=short_ctx,
                    kb_snippets=kb_snippets,
                    profile=profile,
                )
            except Exception:
                assistant_text = APOLOGY_MSG

        # Persist
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
        try:
            memory.save_profile(session_id, profile)
        except Exception:
            pass
        return assistant_text
    except Exception:
        try:
            memory.append_session(session_id, "user", user_msg)
        except Exception:
            pass
        return APOLOGY_MSG
