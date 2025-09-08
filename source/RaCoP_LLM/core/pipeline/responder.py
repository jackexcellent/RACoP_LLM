"""Responder 模組 (Mode-aware)
Modes: greet / gather / counsel (off_topic handled upstream).
greet: 1–2 sentences (greeting + light question)
gather: 2–3 sentences (PCT empathic line + 1–2 open questions, no advice)
counsel: multi-therapy paragraph (PCT + optional CBT + optional SFBT) with personalization & optional RAG.
Fallbacks per mode if LLM fails.
DBT responses are never generated here (handled by safety referral upstream).
"""
from __future__ import annotations

from typing import Dict, Any, Optional, List
import os
import re

from core.providers.gemini_client import GeminiClient

PCT_FALLBACK = {
    "starter": "I’m here with you as you share this.",
    "validation": "It makes sense that you’d feel this way.",
    "question": "What would be most supportive for you right now?",
}

CBT_FALLBACK = {
    "thought": "I might be missing something important.",
    "evidence_for": "a few moments that seem to confirm it",
    "evidence_against": "several details that don’t fully fit",
    "alt_thought": "There may be another way to see this without blaming myself.",
}

SFBT_FALLBACK = {
    "scale": "3",
    "one_step": "Set a 10-minute timer and do one tiny task that moves you an inch forward.",
}

BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # core/
PROMPTS_DIR = os.path.join(BASE_DIR, "prompts")
SYSTEM_RESP_PATH = os.path.join(PROMPTS_DIR, "system_resp.txt")
TEMPLATE_PCT_PATH = os.path.join(PROMPTS_DIR, "templates", "pct.md")
TEMPLATE_CBT_PATH = os.path.join(PROMPTS_DIR, "templates", "cbt.md")
TEMPLATE_SFBT_PATH = os.path.join(PROMPTS_DIR, "templates", "sfbt.md")


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


def _gather_slots(plan: Dict[str, Any], key: str) -> Dict[str, Any]:
    return (plan.get("template_slots", {}) or {}).get(key, {}) if isinstance(plan, dict) else {}


def _fill_template(tmpl: str, mapping: Dict[str, str]) -> str:
    out = tmpl
    for k, v in mapping.items():
        out = out.replace(f"{{{{{k}}}}}", v)
    return out


def _assemble_multi_prompt(plan: Dict[str, Any], user_msg: str, profile: Optional[Dict[str, Any]] = None) -> str:
    snippet = safe_snippet(user_msg, 80) or "this"
    pct_slots = {**PCT_FALLBACK, **_gather_slots(plan, "pct")}
    pct_map = {
        "starter": pct_slots.get("starter", PCT_FALLBACK["starter"]),
        "validation": pct_slots.get("validation", PCT_FALLBACK["validation"]),
        "question": pct_slots.get("question", PCT_FALLBACK.get("question", "")),
        "snippet": snippet,
    }
    pct_tmpl = load_text(TEMPLATE_PCT_PATH) or "{{starter}} — I’m hearing {{snippet}}.\n{{validation}}\n{{question}}"
    parts: List[str] = [_fill_template(pct_tmpl, pct_map).strip()]

    # Determine order from plan array
    sequence = []
    try:
        for item in plan.get("plan", []) or []:
            t = item.get("therapy")
            if t in {"CBT", "SFBT"} and t not in sequence:
                sequence.append(t)
    except Exception:
        pass

    ineffective = set()
    if profile and isinstance(profile.get("ineffective_skills"), list):
        ineffective = {s for s in profile["ineffective_skills"] if isinstance(s, str)}

    if "CBT" in sequence and "CBT" not in ineffective:
        cbt_slots = {**CBT_FALLBACK, **_gather_slots(plan, "cbt")}
        cbt_map = {k: cbt_slots.get(k, CBT_FALLBACK[k]) for k in CBT_FALLBACK}
        cbt_tmpl = load_text(TEMPLATE_CBT_PATH) or (
            "When that thought comes up — “{{thought}}” — it makes sense you’d feel tense. "
            "Looking closer we see pieces that support it ({{evidence_for}}) and pieces that don’t ({{evidence_against}}). A kinder line might be: “{{alt_thought}}.”"
        )
        parts.append(_fill_template(cbt_tmpl, cbt_map).strip())

    if "SFBT" in sequence and "SFBT" not in ineffective:
        sfbt_slots = {**SFBT_FALLBACK, **_gather_slots(plan, "sfbt")}
        sfbt_map = {k: sfbt_slots.get(k, SFBT_FALLBACK[k]) for k in SFBT_FALLBACK}
        sfbt_tmpl = load_text(TEMPLATE_SFBT_PATH) or (
            "If today sits around {{scale}} out of 10, try one tiny thing soon: {{one_step}}."
        )
        parts.append(_fill_template(sfbt_tmpl, sfbt_map).strip())

    # Combine into single user prompt for model
    combined = "\n\n".join(p for p in parts if p)
    return combined


def _rule_based_fallback(plan: Dict[str, Any], user_msg: str) -> str:
    mode = (plan.get("template_slots", {}) or {}).get("mode", "counsel")
    slots = plan.get("template_slots", {}) or {}
    if mode == "greet":
        greeting = slots.get("greeting") or "Hey — good to connect."
        light_q = slots.get("light_question") or "What’s on your mind today?"
        return f"{greeting} {light_q}".strip()
    if mode == "gather":
        gather_qs = slots.get("gather_questions") or [
            "What feels most present for you right now?",
            "If you could change one tiny thing this week, what would it be?",
        ]
        empathic = "I’m hearing there’s something weighing on you and I’d like to understand more."  # 1 sentence
        qs_text = " " .join(q.strip().rstrip("?") + "?" for q in gather_qs[:2])
        return f"{empathic} {qs_text}".strip()
    # counsel fallback (PCT minimal)
    pct_slots = _gather_slots(plan, "pct")
    starter = pct_slots.get("starter") or PCT_FALLBACK["starter"]
    validation = pct_slots.get("validation") or PCT_FALLBACK["validation"]
    question = pct_slots.get("question") or PCT_FALLBACK["question"]
    snippet = safe_snippet(user_msg, 60)
    mention = f" about \"{snippet}\"" if snippet else ""
    sentences = [f"{starter}{mention}.", validation, question]
    return " ".join(s.strip() for s in sentences if s).strip()


def generate_response(
    plan: Dict[str, Any],
    user_msg: str,
    short_ctx: Optional[str] = None,
    kb_snippets: Optional[List[str]] = None,
    history_snippets: Optional[List[str]] = None,
    profile: Optional[Dict[str, Any]] = None,
) -> str:
    mode = (plan.get("template_slots", {}) or {}).get("mode", "counsel")
    system_prompt = load_text(SYSTEM_RESP_PATH)

    # Build user prompt depending on mode
    slots = plan.get("template_slots", {}) or {}
    if mode == "greet":
        greeting = slots.get("greeting") or "Hey — good to connect."
        light_q = slots.get("light_question") or "What’s on your mind today?"
        user_prompt = f"<mode:greet>\nUser snippet: {safe_snippet(user_msg,80)}\nGreeting: {greeting}\nLightQuestion: {light_q}"
    elif mode == "gather":
        gather_qs = slots.get("gather_questions") or [
            "What feels most present for you right now?",
            "If you could change one tiny thing this week, what would it be?",
        ]
        empathic_seed = _gather_slots(plan, "pct").get("starter") or "I’m hearing there’s something on your mind."
        qs_join = " | ".join(gather_qs[:2])
        user_prompt = (
            f"<mode:gather>\nSnippet: {safe_snippet(user_msg,80)}\nEmpathic: {empathic_seed}\nQuestions: {qs_join}\nInstruction: produce 2–3 sentences: reflection + open questions only."
        )
    else:  # counsel
        user_prompt = _assemble_multi_prompt(plan, user_msg, profile=profile)
        if kb_snippets:
            block = "\n---\n".join(kb_snippets)
            user_prompt += f"\n\n<kb_snippets>\n{block}\n</kb_snippets>\n\nInstructions: You MAY paraphrase only helpful ideas above. Do NOT quote verbatim, do NOT list bullets, keep one compact paragraph."
        if history_snippets:
            hblock = "\n---\n".join(history_snippets)
            user_prompt += (
                f"\n\n<history_snippets>\n{hblock}\n</history_snippets>\n"
                "Guidance: absorb only what helps current concern; paraphrase naturally; no verbatim quotes, no lists, one single paragraph (2–6 sentences)."
            )
        if short_ctx:
            user_prompt += f"\n\nRecent context:\n{safe_snippet(short_ctx, 120)}"
        if profile:
            try:
                eff = profile.get("effective_skills") or []
                ineff = profile.get("ineffective_skills") or []
                tone_pref = profile.get("tone_preference", "warm")
                prof_summary = [f"tone_preference={tone_pref}"]
                if eff:
                    prof_summary.append("effective=" + ", ".join([str(x) for x in eff][:8]))
                if ineff:
                    prof_summary.append("ineffective=" + ", ".join([str(x) for x in ineff][:8]))
                user_prompt += "\n\n<profile>\n" + " | ".join(prof_summary) + "\nGuidance: prefer effective, avoid ineffective.\n</profile>"
            except Exception:
                pass

    client = GeminiClient()
    raw = client.generate(system_prompt=system_prompt, user_prompt=user_prompt, temperature=0.7, max_tokens=420)
    if not raw:
        return _rule_based_fallback(plan, user_msg)
    cleaned = raw.strip()
    if cleaned.startswith("{") and cleaned.endswith("}"):
        return _rule_based_fallback(plan, user_msg)
    cleaned = re.sub(r"```.*?```", "", cleaned, flags=re.DOTALL).strip()
    return cleaned or _rule_based_fallback(plan, user_msg)


__all__ = ["generate_response", "safe_snippet", "load_text"]
