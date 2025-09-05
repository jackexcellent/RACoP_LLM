"""Planner 模組 (Stage 2)
提供兩種方式：
1. generate_plan: 呼叫 OpenAI + JSON Schema 驗證 (失敗重試與 fallback)
2. fake_plan: 硬編 fallback 計畫 (PCT only)
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
SYSTEM_PROMPT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts", "system_cop.txt")


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
    ###
    print("Using fallback fake_plan.")
    ###
    
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
        "notes": "Stage-2 fallback: PCT only.",
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

def _parse_and_validate(raw: str, schema: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    candidate_str = _strip_code_fence(raw.strip())
    for attempt in range(2):  # first raw parse, then extracted block
        try:
            data = json.loads(candidate_str)
        except json.JSONDecodeError:
            block = _extract_json_block(candidate_str)
            if block and block != candidate_str:
                block = _strip_code_fence(block)
                candidate_str = block
                continue
            return None
        try:
            validate(instance=data, schema=schema)
            return data
        except ValidationError:
            return None
    return None


def generate_plan(user_msg: str, recent_ctx: Optional[str] = None, max_retries: int = 0) -> Dict[str, Any]:
    system_prompt = _load_text(SYSTEM_PROMPT_PATH)
    schema = _load_schema()
    
    if not system_prompt or not schema:
        return fake_plan(user_msg)

    ctx_part = recent_ctx or "(no recent context)"
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
