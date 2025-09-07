from __future__ import annotations
from typing import Dict, Any, Optional
import json
import os
import re

from jsonschema import validate, ValidationError  # type: ignore

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
    # print("Candidate string for JSON parsing:\n", candidate_str)  # Debug log
    for attempt in range(2):  # first raw parse, then extracted block
        try:
            data = json.loads(candidate_str)
            # print(data)
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
            print("\n\nValidation error for data:", data)  # Debug log
            print("\n\nSchema:", schema)  # Debug log
            print()
            return None
    return None

schema = {
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "required": ["risk", "emotions", "plan", "tone"],
  "properties": {
    "risk": {
      "type": "object",
      "required": ["level"],
      "properties": {
        "level": { "enum": ["low", "med", "high"] },
        "signals": { "type": "array", "items": { "type": "string" } }
      }
    },
    "emotions": { "type": "array", "items": { "type": "string" } },
    "distortions": { "type": "array", "items": { "type": "string" } },
    "plan": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["therapy", "weight"],
        "properties": {
          "therapy": { "enum": ["PCT", "CBT", "SFBT", "DBT"] },
          "goal": { "type": "string" },
          "weight": { "type": "number", "minimum": 0, "maximum": 1 }
        }
      }
    },
    "tone": { "type": "string" },
    "reading_level": { "type": "string" },
    "template_slots": { "type": "object" },
    "retrieval_queries": { "type": "array", "items": { "type": "string" } },
    "final_prompt": { "type": "string" },
    "notes": { "type": "string" }
  }
}

raw = """
```json
{
  "risk": {
    "level": "low",
    "signals": []
  },
  "emotions": {
    "primary": "hopelessness",
    "secondary": "sadness",
    "tertiary": "frustration"
  },
  "plan": [
    "PCT",
    "CBT",
    "SFBT"
  ],
  "pct": {
    "starter": "It sounds incredibly painful to feel constantly judged and treated meanly by your father. That's a really tough situation to be in.",
    "validation": "It makes perfect sense that you'd feel hopeless when faced with that kind of negativity regularly. Your feelings are valid and understandable.",
    "question": "What is one specific interaction with your father that stands out in your mind right now?"
  },
  "cbt": {
    "thought": "My father's behavior means I am unlovable and incapable.",
    "evidence_for": "He is consistently mean to me.",
    "evidence_against": "His behavior could be related to his own issues and doesn't reflect my worth. Other people in my life show me love and support.",
    "alt_thought": "My father's behavior is a reflection of his own struggles, and it doesn't define my worth as a person. I am worthy of love and respect."
  },
  "sfbt": {
    "scale": "On a scale of 0 to 10, where 0 is feeling completely hopeless and 10 is feeling completely capable of facing your father, where would you rate yourself right now?",
    "one_step": "Identify one small thing you can do in the next 24-48 hours to create a little bit ofy worth as a person. I am worthy of love and respect."
  },
  "sfbt": {
    "scale": "On a scale of 0 to 10, where 0 is feeling completely hopeless and 10 is feeling completely capable of facing your father, where would you rate yourself right now?",
    "one_step": "Identify one small thing you can do in the next 24-48 hours to create a little bit of  },
  "sfbt": {
    "scale": "On a scale of 0 to 10, where 0 is feeling completely hopeless and 10 is feeling completely capable of facing your father, where would you rate yourself right now?",
    "one_step": "Identify one small thing you can do in the next 24-48 hours to create a little bit of  "sfbt": {
    "scale": "On a scale of 0 to 10, where 0 is feeling completely hopeless and 10 is feeling completely capable of facing your father, where would you rate yourself right now?",
    "one_step": "Identify one small thing you can do in the next 24-48 hours to create a little bit of    "scale": "On a scale of 0 to 10, where 0 is feeling completely hopeless and 10 is feeling completely capable of facing your father, where would you rate yourself right now?",
    "one_step": "Identify one small thing you can do in the next 24-48 hours to create a little bit ofly capable of facing your father, where would you rate yourself right now?",
    "one_step": "Identify one small thing you can do in the next 24-48 hours to create a little bit of distance or self-care after an interaction with your father, like taking a short walk or listening to a favorite song."
    "one_step": "Identify one small thing you can do in the next 24-48 hours to create a little bit of distance or self-care after an interaction with your father, like taking a short walk or listening to a favorite song."
  },
 distance or self-care after an interaction with your father, like taking a short walk or listening to a favorite song."
  },
 a favorite song."
  },
  "dbt": {
    "skill_name": null,
  },
  "dbt": {
    "skill_name": null,
  "dbt": {
    "skill_name": null,
    "steps": []
  },
    "skill_name": null,
    "steps": []
  },
    "steps": []
  },
  },
  "tone": "warm, validating, non-judgmental",
  "weights": {
    "PCT": 0.5,
    "CBT": 0.3,
    "SFBT": 0.2,
    "DBT": 0.0
  }
}
```
"""

parsed = _parse_and_validate(raw, schema)
print("Parsed result:", parsed)