"""RaCoP 心理支持聊天機器人 - REPL 版本
Planner-only：User Input -> Coordinator (Profile update -> Pre-RAG -> Planner(JSON) -> Routing -> Short reply + Persistence)
執行方式：python cli/main.py

輸入 exit / quit 結束。
"""

from __future__ import annotations

import os
import sys

# 將專案根目錄 (含 core/) 加入 sys.path，允許 "from core..." 匯入
_CURRENT_DIR = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.dirname(_CURRENT_DIR)
if _PROJECT_ROOT not in sys.path:  # 避免重複插入
    sys.path.insert(0, _PROJECT_ROOT)

from core.pipeline.coordinator import run_once  # type: ignore


def main() -> None:
    session_id = os.getenv("SESSION_ID", "default")
    dbg = os.getenv("DEBUG_PLAN_JSON", "0") == "1"
    while True:
        try:
            user_msg = input("You: ")
        except (EOFError, KeyboardInterrupt):  # graceful exit
            print("\nAssistant: Bye!")
            break
        if user_msg.strip().lower() in {"exit", "quit"}:
            print("Assistant: Bye!")
            break
        result = run_once(user_msg, session_id=session_id)
        # result can be a tuple (assistant, plan) or plain string for safety
        if isinstance(result, tuple) and len(result) == 2:
            assistant, plan = result
            if dbg:
                try:
                    import json
                    print(json.dumps(plan, ensure_ascii=False, indent=2))
                except Exception:
                    pass
            print("Assistant: " + assistant)
        else:
            print("Assistant: " + str(result))


if __name__ == "__main__":  # pragma: no cover
    main()
