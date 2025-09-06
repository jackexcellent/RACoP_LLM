"""Responder 模組 (Stage 6)
LLM-B (Gemini) 生成多療法回覆 (PCT / CBT / SFBT)。
若金鑰或模型失敗 → 規則式 fallback (僅 PCT 三句)。
DBT 由 Safety 層轉介，不在此輸出技巧。
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


def _assemble_multi_prompt(plan: Dict[str, Any], user_msg: str) -> str:
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

    if "CBT" in sequence:
        cbt_slots = {**CBT_FALLBACK, **_gather_slots(plan, "cbt")}
        cbt_map = {k: cbt_slots.get(k, CBT_FALLBACK[k]) for k in CBT_FALLBACK}
        cbt_tmpl = load_text(TEMPLATE_CBT_PATH) or (
            "When that thought comes up — “{{thought}}” — it makes sense you’d feel tense. "
            "Looking closer we see pieces that support it ({{evidence_for}}) and pieces that don’t ({{evidence_against}}). A kinder line might be: “{{alt_thought}}.”"
        )
        parts.append(_fill_template(cbt_tmpl, cbt_map).strip())

    if "SFBT" in sequence:
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
    slots = _gather_slots(plan, "pct")
    starter = slots.get("starter") or PCT_FALLBACK["starter"]
    validation = slots.get("validation") or PCT_FALLBACK["validation"]
    question = slots.get("question") or PCT_FALLBACK["question"]
    snippet = safe_snippet(user_msg, 60)
    mention = f" about \"{snippet}\"" if snippet else ""
    sentences = [f"{starter}{mention}.", validation, question]
    return " ".join(s.strip() for s in sentences if s).strip()


def generate_response(
    plan: Dict[str, Any],
    user_msg: str,
    short_ctx: Optional[str] = None,
    kb_snippets: Optional[List[str]] = None,
) -> str:
    system_prompt = load_text(SYSTEM_RESP_PATH)
    user_prompt = _assemble_multi_prompt(plan, user_msg)
    if kb_snippets:
        block = "\n---\n".join(kb_snippets)
        user_prompt += f"\n\n<kb_snippets>\n{block}\n</kb_snippets>\n\nInstructions: You MAY paraphrase only helpful ideas above. Do NOT quote verbatim, do NOT list bullets, keep one compact paragraph."
    if short_ctx:
        user_prompt += f"\n\nRecent context:\n{safe_snippet(short_ctx, 120)}"

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
