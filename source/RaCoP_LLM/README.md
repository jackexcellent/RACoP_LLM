# RaCoP 心理支持聊天機器人 (Stage 1)

本階段：建立最小「兩段式管線」(Planner -> Responder)；Planner 回傳硬編 PCT 規劃，Responder 依規劃輸出 2–3 句英文同理回覆。

## 目標摘要

1. CLI 單輪：讀取一行使用者輸入
2. Planner: `fake_plan(user_msg)` 回傳固定計畫 (忽略輸入內容)
3. Responder: `respond(plan, user_msg)` 根據計畫組合同理 + 驗證 + 開放式問題
4. 僅使用標準函式庫，無任何外部依賴

## 專案結構

```
RACoP/source/RaCoP_LLM/
├─ README.md
├─ requirements.txt
├─ cli/
│  └─ main.py
└─ core/
	└─ pipeline/
		├─ planner.py
		└─ responder.py
```

## 環境需求

- Python 3.11+

## 安裝 (無第三方套件)

```
python -m venv .venv
.venv\Scripts\activate  # Windows PowerShell
pip install -r requirements.txt
```

`requirements.txt`：

```
# no deps yet
```

## 執行

在 `RACoP/source/RaCoP_LLM/` 目錄下：

```
python cli/main.py
```

輸入一句文字，例如：

```
You: I have been feeling nervous about tomorrow's presentation and can't sleep.
Assistant: I sense the deep emotions you're experiencing about "I have been feeling nervous about tomorrow's presentatio…". Your feelings are completely valid and worth exploring. What do you need most in this moment?
```

（實際輸出會依輸入長度在第一句中截斷顯示部份內容。）

## 模組說明

- `core/pipeline/planner.py`: `fake_plan` 回傳硬編 PCT-only 規劃 JSON (dict)
- `core/pipeline/responder.py`: `respond` 取計畫槽位 + 使用者輸入 snippet 組合 2–3 句英文

## 後續可能階段 (預告)

- 風險訊號抽取 / 情緒分類（假資料 → 模型）
- 記憶管理與多輪對話
- LLM 提示詞動態生成 (final_prompt)
- 多療法 (CBT, ACT, REBT...) 權重融合

---

本檔案會在後續階段持續更新。
