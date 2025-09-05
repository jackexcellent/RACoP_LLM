"""RaCoP 心理支持聊天機器人 - Stage 4
流程：User Input -> Coordinator (Planner -> Safety Gate -> Responder)
執行方式：python cli/main.py

注意：為了可直接以路徑執行 (而非 -m 套件方式)，此檔案在匯入前動態加入父層路徑。
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
    user_msg = input("You: ")
    reply = run_once(user_msg)
    print("Assistant: " + reply)


if __name__ == "__main__":  # pragma: no cover
    main()
