"""Planner-only module
Generate a JSON plan only; no natural language reply here.
"""
from __future__ import annotations

from typing import Dict, Any, Optional
import json
import os
import re

from jsonschema import validate, ValidationError  # type: ignore

from core.providers.openai_client import OpenAIClient
from core.providers.gemini_client import GeminiClient


SCHEMA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "schemas", "cop_plan.schema.json")
SYSTEM_PROMPT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts", "system_plan.txt")


def _load_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


def _load_schema() -> Dict[str, Any]:
    try:
        with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def fake_plan(user_msg: str) -> Dict[str, Any]:
    print("Using fallback fake_plan.")
    return {
        "risk": {"level": "low", "signals": []},
        "emotions": ["unsure"],
        "plan": [
            {"therapy": "PCT", "goal": "Initial rapport and validation", "weight": 1.0}
        ],
        "tone": "warm, validating, non-judgmental",
        "template_slots": {
            "mode": "greet",
            "greeting": "Hi, glad you reached out.",
            "light_question": "What would you like to share today?",
            "pct": {
                "starter": "I’m here with you",
                "validation": "It’s okay to take a moment as we start.",
                "question": "Anything on your mind you’d like to explore?",
            },
        },
        "retrieval_queries": [],
    }


def _extract_json_block(text: str) -> Optional[str]:
    if not text:
        return None
    # Find first balanced {...}
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0)
    return None

def _strip_code_fence(s: str) -> str:
    """移除可能包住 JSON 的 ``` 或 ```json 樣式 code fence。

    支援：
    ```json\n{...}\n```
    ```\n{...}\n```
    以及前後多餘空白。若格式不匹配則原樣返回。
    """
    if not s:
        return s
    stripped = s.strip()
    # 檢查起始 ``` (可帶語言標籤) 與結尾 ```
    if stripped.startswith("```") and stripped.endswith("```"):
        # 去掉開頭 ```lang? 與結尾 ```
        # 取第一行之後直到最後一行之前
        lines = stripped.splitlines()
        if len(lines) >= 2:
            first = lines[0].strip()
            last = lines[-1].strip()
            if first.startswith("```") and last == "```":
                inner = "\n".join(lines[1:-1])
                return inner.strip()
    return s

def _coerce_for_schema(data: Dict[str, Any]) -> Dict[str, Any]:
    # risk.signals: allow string -> [string]
    try:
        if isinstance(data.get("risk"), dict):
            sig = data["risk"].get("signals")
            if isinstance(sig, str):
                cleaned = [s.strip() for s in sig.split(",") if s.strip()] or [sig.strip()]
                data["risk"]["signals"] = cleaned
    except Exception:  # pragma: no cover
        pass

    # plan: allow list of strings -> objects; ensure each has therapy
    plan = data.get("plan")
    if isinstance(plan, list):
        new_plan = []
        therapy_key_map = {"pct": "PCT", "cbt": "CBT", "sfbt": "SFBT", "dbt": "DBT"}
        for item in plan:
            if isinstance(item, str):
                new_plan.append({"therapy": item, "weight": 1.0 / max(len(plan), 1)})
            elif isinstance(item, dict) and "therapy" in item:
                if "weight" not in item:
                    item["weight"] = 0.5
                new_plan.append(item)
            elif isinstance(item, dict):
                # 支援 LLM 輸出形如 {"pct": {...}} 的結構，轉為 {"therapy": "PCT"}
                lowered_keys = {k.lower(): k for k in item.keys() if isinstance(k, str)}
                for lk, original_k in lowered_keys.items():
                    if lk in therapy_key_map:
                        new_plan.append({"therapy": therapy_key_map[lk], "weight": 1.0 / max(len(plan), 1)})
                        break  # 一個 item 只取第一個匹配的 therapy
        if new_plan:
            data["plan"] = new_plan

    # emotions: if object convert to array of distinct emotion terms (primary, secondary)
    emos = data.get("emotions")
    if isinstance(emos, dict):
        arr = []
        for key in ("primary", "secondary"):
            v = emos.get(key)
            if isinstance(v, str) and v.strip():
                for part in v.split(","):
                    p = part.strip()
                    if p and p not in arr:
                        arr.append(p)
        if arr:
            data["emotions"] = arr

    # template_slots: merge loose pct/cbt/sfbt/dbt objects into template_slots
    ts = data.get("template_slots")
    if not isinstance(ts, dict):
        ts = {}
    for key in ("pct", "cbt", "sfbt", "dbt"):
        if isinstance(data.get(key), dict):
            ts.setdefault(key, {}).update(data[key])  # layer in
    if ts:
        data["template_slots"] = ts

    # tone: allow object -> convert to descriptive string
    tone_val = data.get("tone")
    if isinstance(tone_val, dict):
        # e.g. {"warmth":0.6, "directness":0.4} -> "warmth=0.6, directness=0.4"
        try:
            parts = []
            for k, v in tone_val.items():
                parts.append(f"{k}={v}")
            if parts:
                data["tone"] = ", ".join(parts)
        except Exception:
            pass
    return data


def _parse_and_validate(raw: str, schema: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    candidate_str = _strip_code_fence(raw.strip())
    for attempt in range(3):  # raw, extracted, coerced
        try:
            data = json.loads(candidate_str)
        except json.JSONDecodeError:
            block = _extract_json_block(candidate_str)
            if block and block != candidate_str:
                candidate_str = _strip_code_fence(block)
                continue
            return None
        # First try direct validation
        try:
            validate(instance=data, schema=schema)
            return data
        except ValidationError:
            # Attempt coercion then re-validate once
            try:
                coerced = _coerce_for_schema(data)
                validate(instance=coerced, schema=schema)
                return coerced
            except ValidationError:
                return None
    return None


def generate_plan(user_msg: str, recent_ctx: Optional[str] = None, max_retries: int = 2) -> Dict[str, Any]:
    system_prompt = _load_text(SYSTEM_PROMPT_PATH)
    schema = _load_schema()
    
    if not system_prompt or not schema:
        return fake_plan(user_msg)

    ctx_part = recent_ctx or "(no recent context)"
    print("=== RECENT CONTEXT ===")
    print(f"{ctx_part}\n---")  # Debug log
    
    print("=== USER MESSAGE ===")
    print(f"{user_msg}\n---")  # Debug log
    
    user_prompt = json.dumps(
        {
            "user_message": user_msg,
            "recent_context": ctx_part,
            "instruction": "Return ONLY one JSON object per the system instructions and schema. Do NOT add any extra text especially like ```json ... ``` these kind of code fences. only generate JSON. If you cannot comply, return an empty JSON {} .",
        },
        ensure_ascii=False,
    )

    client = GeminiClient()

    for _ in range(max_retries + 1):
        raw = client.generate(system_prompt=system_prompt, user_prompt=user_prompt)
        print(f"Raw LLM output:\n{raw}\n---")  # Debug log
        parsed = _parse_and_validate(raw, schema)
        if parsed:
            return parsed
    return fake_plan(user_msg)


__all__ = ["fake_plan", "generate_plan"]
