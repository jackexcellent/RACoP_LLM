# RaCoP 心理支持聊天機器人 (初始化階段)

此專案目前為最小可執行骨架，後續將加入兩段式 LLM (Planner → Responder)。

## 環境需求

- Python 3.11+

## 安裝

(目前無第三方套件)

```
# 可選：建立虛擬環境
python -m venv .venv
.venv\Scripts\activate  # Windows PowerShell

pip install -r requirements.txt
```

`requirements.txt` 目前內容：

```
# no deps yet
```

## 執行

在專案根目錄下：

```
python cli/main.py
```

看到提示後輸入任一文字，會得到固定回覆：

```
You: 你好
Assistant: 我在喔（初始化完成）
```

## 後續規劃 (摘要)

- 加入 Planner / Responder 模組
- 封裝對話狀態與上下文記憶
- 加入可設定的模型介面（OpenAI / 本地模型）
- 善用錯誤處理與日誌

---

本 README 將在各階段持續更新。
