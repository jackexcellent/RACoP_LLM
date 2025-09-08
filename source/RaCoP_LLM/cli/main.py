"""RaCoP 心理支持聊天機器人 - REPL 版本
連續多輪：User Input -> Coordinator (Memory -> Planner -> Safety/DBT Gate -> Mode Routing -> Responder/Referral + Persistence)
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
    while True:
        try:
            user_msg = input("You: ")
        except (EOFError, KeyboardInterrupt):  # graceful exit
            print("\nAssistant: Bye!")
            break
        if user_msg.strip().lower() in {"exit", "quit"}:
            print("Assistant: Bye!")
            break
        reply = run_once(user_msg, session_id=session_id)
        print("Assistant: " + reply)


if __name__ == "__main__":  # pragma: no cover
    main()
