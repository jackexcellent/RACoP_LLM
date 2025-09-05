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
  "emotions": [
    "positive"
  ],
  "plan": [
    {
      "therapy": "PCT",
      "weight": 0.7
    },
    {
      "therapy": "SFBT",
      "weight": 0.3
    }
  ],
  "tone": "warm, validating, non-judgmental",
  "template_slots": {
    "PCT": {
      "starter": "It's wonderful to hear you're feeling good! Can you tell me more about what's contributing to this positive feeling?",
      "validation": "That sounds like a really positive experience, and it's great that you're recognizing and appreciating it.",     
      "question": "What are some things you're looking forward to?"
    },
    "SFBT": {
      "scale": "On a scale of 1 to 10, where 1 is the worst you've felt and 10 is the best, where would you rate your current feeling of goodness?",
      "one_step": "What's one small thing you could do today to maintain or even slightly improve that feeling?"
    }
  },
  "retrieval_queries": [
    "benefits of positive emotions",
    "activities that promote well-being"
  ],
  "final_prompt": "Continue exploring the positive feelings and identify small steps to maintain them.",
  "notes": "Focus on amplifying the positive feelings and identifying resources."
}
```"""

parsed = _parse_and_validate(raw, schema)
print("Parsed result:", parsed)