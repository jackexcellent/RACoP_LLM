"""Short-term memory management (Stage 6).

Stores conversation turns in JSONL per session and provides compact context.
"""
from __future__ import annotations

from typing import List, Dict, Any, Optional
import time

from core.utils.io import session_path, append_jsonl, read_jsonl


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
    return {
        "risk": plan.get("risk", {}).get("level"),
        "therapies": therapies,
        "tone": plan.get("tone"),
    }


__all__ = [
    "load_session",
    "append_session",
    "summarize_short",
    "get_short_context",
    "plan_summary",
]
