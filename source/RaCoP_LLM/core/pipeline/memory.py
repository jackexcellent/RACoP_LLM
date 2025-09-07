"""Memory & Profile management (Stage 8)

Short-term:
    - Store conversation turns per session JSONL
    - Provide compact context snippet

Long-term profile (per session):
    {
        "effective_skills": [],
        "ineffective_skills": [],
        "tone_preference": "warm|neutral|direct"
    }

Update rule: user feedback keywords applied to previous assistant plan therapies & suggested action.
"""
from __future__ import annotations

from typing import List, Dict, Any, Optional
import time

from core.utils.io import session_path, append_jsonl, read_jsonl, PROFILES_DIR, read_json, write_json, ensure_dir
from pathlib import Path


def load_session(session_id: str) -> List[Dict[str, Any]]:
    return read_jsonl(session_path(session_id))


def append_session(session_id: str, role: str, text: str, meta: Optional[Dict[str, Any]] = None) -> None:
    obj = {
        "ts": int(time.time()),
        "role": role,
        "text": text,
        "meta": meta,
    }
    append_jsonl(session_path(session_id), obj)


def _clip(s: str, max_len: int = 160) -> str:
    if len(s) <= max_len:
        return s
    return s[: max_len - 3].rstrip() + "..."


def summarize_short(history: List[Dict[str, Any]], turns: int = 5) -> str:
    # Filter only user / assistant roles
    filtered = [h for h in history if h.get("role") in {"user", "assistant"}]
    # Take last N entries
    tail = filtered[-turns:]
    lines: List[str] = []
    for item in tail:
        role = item.get("role", "?")
        text = _clip(str(item.get("text", "")).replace("\n", " "))
        lines.append(f"{role}: {text}")
    return "\n".join(lines)


def get_short_context(session_id: str, turns: int = 5) -> str:
    history = load_session(session_id)
    if not history:
        return ""
    return summarize_short(history, turns=turns)


def plan_summary(plan: Dict[str, Any]) -> Dict[str, Any]:
    therapies = []
    try:
        for p in plan.get("plan", []) or []:
            t = p.get("therapy")
            if t and t not in therapies:
                therapies.append(t)
    except Exception:
        pass
    # suggested action: prefer sfbt slot one_step
    suggested = None
    try:
        slots = plan.get("template_slots", {}) or {}
        sfbt = slots.get("sfbt", {}) if isinstance(slots, dict) else {}
        one_step = sfbt.get("one_step") if isinstance(sfbt, dict) else None
        if one_step and isinstance(one_step, str) and one_step.strip():
            suggested = one_step.strip()
    except Exception:
        pass
    return {
        "risk": plan.get("risk", {}).get("level"),
        "therapies": therapies,
        "tone": plan.get("tone"),
        "suggested_action": suggested,
    }

# ---------------- Profile Handling ---------------- #

DEFAULT_PROFILE = {
    "effective_skills": [],
    "ineffective_skills": [],
    "tone_preference": "warm",
}


def profile_path(session_id: str) -> Path:
    ensure_dir(PROFILES_DIR)
    safe_id = session_id.replace("/", "_").replace("..", "_")
    return Path(PROFILES_DIR) / f"{safe_id}.json"


def load_profile(session_id: str) -> Dict[str, Any]:
    path = profile_path(session_id)
    data = read_json(path, default={})
    # merge defaults
    profile = DEFAULT_PROFILE.copy()
    if isinstance(data, dict):
        for k in profile.keys():  # only known keys
            if k in data and isinstance(data[k], type(profile[k])):
                profile[k] = data[k]
    return profile


def save_profile(session_id: str, profile: Dict[str, Any]) -> None:
    # clip list sizes
    for key in ("effective_skills", "ineffective_skills"):
        if isinstance(profile.get(key), list):
            profile[key] = profile[key][-20:]
    write_json(profile_path(session_id), profile)


POS_KEYWORDS = [
    "有幫助", "幫到", "有效", "舒服些", "useful", "helped", "worked", "better"
]
NEG_KEYWORDS = [
    "沒幫助", "沒用", "無效", "更糟", "not help", "didn't help", "worse"
]


def _contains_any(text: str, keywords: List[str]) -> bool:
    tl = text.lower()
    for kw in keywords:
        if kw.lower() in tl:
            return True
    return False


def update_profile_from_feedback(session_id: str, user_msg: str) -> None:
    try:
        history = load_session(session_id)
        # find last assistant turn
        last_assistant = None
        for item in reversed(history):
            if item.get("role") == "assistant":
                last_assistant = item
                break
        if not last_assistant:
            return
        plan_meta = (last_assistant.get("meta") or {}).get("plan") or {}
        therapies = plan_meta.get("therapies") or []
        suggested = plan_meta.get("suggested_action")

        profile = load_profile(session_id)
        pos = _contains_any(user_msg, POS_KEYWORDS)
        neg = _contains_any(user_msg, NEG_KEYWORDS)
        targets: List[str] = []
        if isinstance(therapies, list):
            for t in therapies:
                if isinstance(t, str):
                    targets.append(t)
        if isinstance(suggested, str) and suggested.strip():
            targets.append(suggested.strip())
        if not targets:
            return
        if pos and not neg:
            for t in targets:
                if t not in profile["effective_skills"]:
                    profile["effective_skills"].append(t)
            # remove from ineffective if now marked effective
            profile["ineffective_skills"] = [x for x in profile["ineffective_skills"] if x not in targets]
        elif neg and not pos:
            for t in targets:
                if t not in profile["ineffective_skills"]:
                    profile["ineffective_skills"].append(t)
            profile["effective_skills"] = [x for x in profile["effective_skills"] if x not in targets]
        else:
            return  # ambiguous or none
        save_profile(session_id, profile)
    except Exception:
        return


def profile_summary_text(profile: Dict[str, Any]) -> str:
    eff = ", ".join(profile.get("effective_skills") or []) or "(none)"
    ineff = ", ".join(profile.get("ineffective_skills") or []) or "(none)"
    tone = profile.get("tone_preference", "warm")
    return (
        f"tone_preference: {tone}\n"
        f"effective: {eff}\n"
        f"ineffective: {ineff}"
    )


__all__ = [
    "load_session",
    "append_session",
    "summarize_short",
    "get_short_context",
    "plan_summary",
    # profile
    "profile_path",
    "load_profile",
    "save_profile",
    "update_profile_from_feedback",
    "profile_summary_text",
]
