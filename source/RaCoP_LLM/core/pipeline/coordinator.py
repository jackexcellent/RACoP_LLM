"""Conversation coordinator (Planner-only)
Flow: Profile feedback update -> Profile load -> Pre-RAG (History + KB) -> Planner(JSON) -> Safety/DBT/Off-topic routing -> Short reply assembly -> Persistence.
"""
from __future__ import annotations

from typing import Dict, Any, List, Tuple

from core.pipeline.planner import generate_plan  # type: ignore
from core.pipeline import safety  # type: ignore
from core.pipeline import memory  # type: ignore
from core.pipeline import history_retriever, retriever  # type: ignore
import os
import re


APOLOGY_MSG = (
    "Sorry, something went wrong generating a response. Please try again or rephrase your message."
)


def _dedup_sentences(sentences: List[str], user_msg: str) -> List[str]:
    seen = set()
    out: List[str] = []
    um = re.sub(r"\s+", " ", (user_msg or "").strip().lower())
    for s in sentences:
        t = re.sub(r"\s+", " ", (s or "").strip())
        if not t:
            continue
        key = t.lower()
        if key in seen:
            continue
        # drop if near-identical to user message
        if len(t) > 0 and (t.lower() == um or (len(t) > 12 and t.lower() in um)):
            continue
        seen.add(key)
        out.append(t)
    return out


def _limit_sentences(sentences: List[str]) -> List[str]:
    try:
        limit = int(os.getenv("OUTPUT_MAX_SENTENCES", "3") or 3)
    except Exception:
        limit = 3
    return sentences[: max(1, limit)]


def _assemble_short_reply(plan: Dict[str, Any], user_msg: str) -> str:
    ts = (plan.get("template_slots") or {}) if isinstance(plan, dict) else {}
    mode = (ts.get("mode") or "counsel").strip()
    sentences: List[str] = []

    def add(s: str):
        s = (s or "").strip()
        if s:
            sentences.append(s)

    if mode == "greet":
        add(ts.get("greeting") or ts.get("pct", {}).get("starter", "Hi, I’m glad you reached out."))
        add(ts.get("light_question") or ts.get("pct", {}).get("question", "What would you like to talk about right now?"))
    elif mode == "gather":
        pct = ts.get("pct", {}) if isinstance(ts.get("pct"), dict) else {}
        add(pct.get("starter", "I’m here with you."))
        add(pct.get("validation", "It’s already a lot that you’re putting this into words."))
        gqs = ts.get("gather_questions") or []
        if isinstance(gqs, list) and gqs:
            add(gqs[0])
            if len(gqs) > 1:
                add(gqs[1])
    else:  # counsel
        pct = ts.get("pct", {}) if isinstance(ts.get("pct"), dict) else {}
        add(pct.get("starter", "I hear how much you’re carrying; thank you for sharing."))
        add(pct.get("validation", "Your feelings deserve to be seen."))
        add(pct.get("question", "What feels most important to you right now?"))

        # Weights
        weights = { (p.get("therapy") or ""): float(p.get("weight", 0)) for p in (plan.get("plan") or []) if isinstance(p, dict) }
        cbt_w = float(weights.get("CBT", 0.0))
        sfbt_w = float(weights.get("SFBT", 0.0))

        # optional CBT alt_thought
        cbt = ts.get("cbt", {}) if isinstance(ts.get("cbt"), dict) else {}
        if cbt_w >= 0.4:
            alt = (cbt.get("alt_thought") or "").strip()
            if alt:
                add(alt)

        # optional SFBT one_step
        sfbt = ts.get("sfbt", {}) if isinstance(ts.get("sfbt"), dict) else {}
        if sfbt_w >= 0.4 and len(sentences) < 4:  # allow maybe one more before truncation
            step = (sfbt.get("one_step") or "").strip()
            if step:
                add(step)

    # Dedup and trim
    sentences = _dedup_sentences(sentences, user_msg)

    # Enforce max sentences with priority: keep first three where PCT parts come first
    sentences = _limit_sentences(sentences)
    return " ".join(sentences).strip()


def run_once(user_msg: str, session_id: str = "default") -> Tuple[str, Dict[str, Any]] | str:
    """Run one pipeline turn with Planner-only and pre-RAG. Returns (assistant, plan) for debug printing."""
    try:
        # 1) Update profile from feedback
        try:
            memory.update_profile_from_feedback(session_id, user_msg)
        except Exception:
            pass

        # 2) Load profile and summary
        try:
            profile = memory.load_profile(session_id)
            print("=== PROFILE ===")
            print(f"{profile}\n")  # Debug log
        except Exception:
            profile = {"effective_skills": [], "ineffective_skills": [], "tone_preference": "warm"}
        try:
            profile_text = memory.profile_summary_text(profile)
        except Exception:
            profile_text = "tone_preference: warm"

        # 3) Pre-RAG blocks
        recent_ctx_parts: List[str] = []
        if os.getenv("HISTORY_RAG_ENABLED", "1") != "0":
            try:
                hist_block = history_retriever.build_history_block(session_id, user_msg)
                if hist_block:
                    recent_ctx_parts.append(hist_block)
            except Exception:
                pass
        if os.getenv("RAG_ENABLED", "1") != "0":
            try:
                kb_block = retriever.build_kb_block(user_msg)
                if kb_block:
                    recent_ctx_parts.append(kb_block)
            except Exception:
                pass
        recent_ctx_parts.append(f"<profile>\n{profile_text}\n</profile>")
        recent_ctx = "\n\n".join([p for p in recent_ctx_parts if p])
        
        # print("=== RECENT CONTEXT ===")
        # print(f"{recent_ctx}\n---")  # Debug log

        # 4) Planner
        try:
            plan: Dict[str, Any] = generate_plan(user_msg, recent_ctx=recent_ctx)
            print("=== PLAN ===")
            print(f"{plan}\n")
        except Exception:
            plan = {
                "risk": {"level": "low"},
                "template_slots": {
                    "mode": "greet",
                    "greeting": "Hi, I’m glad you’re here.",
                    "light_question": "Where would you like to start?",
                    "pct": {
                        "starter": "I’m here with you.",
                        "validation": "Your feelings matter.",
                        "question": "What’s most on your mind right now?",
                    },
                },
            }

        # 5) Routing
        try:
            risk = safety.assess(plan, user_msg)
        except Exception:
            risk = "low"
        mode = ((plan.get("template_slots") or {}).get("mode") or "counsel").strip()

        if risk == "high":
            assistant_text = safety.escalation_message(user_msg)
        elif safety.requires_professional_for_dbt(plan):
            assistant_text = safety.refer_to_professional(user_msg)
        elif mode == "off_topic":
            assistant_text = "This is outside the scope of this assistant. Let’s focus on mental and emotional support."
        else:
            print("=== ASSEMBLING SHORT REPLY ===") # Debug log
            assistant_text = _assemble_short_reply(plan, user_msg)
            

        # 6) Persistence
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

        return assistant_text, plan
    except Exception:
        try:
            memory.append_session(session_id, "user", user_msg)
        except Exception:
            pass
        return APOLOGY_MSG
