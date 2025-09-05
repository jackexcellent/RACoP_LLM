# RaCoP 心理支持聊天機器人 (Stage 3)

本階段：在 Stage 2 的 LLM Planner 基礎上，加入 LLM Responder（Gemini）與模板化回覆。若 Gemini 失敗或缺金鑰則退回規則式（PCT 三句）組裝。

## 功能摘要

1. CLI 單輪互動：讀一行輸入 → 產生計畫 → 生成回覆
2. Planner (`generate_plan`):
   - 讀取 `core/prompts/system_cop.txt` 作為 system prompt
   - 呼叫 OpenAI (模型: gpt-4o-mini, temperature=0.2)
   - 嚴格要求僅輸出 JSON；用 `core/schemas/cop_plan.schema.json` 驗證
   - 失敗重試最多 2 次；仍失敗 → `fake_plan` fallback
3. Responder (`generate_response`): 使用 Gemini (gemini-2.0-flash, temp=0.7) + system prompt + PCT 模板生成 2–5 句英文；失敗則規則式 fallback（2–3 句）
4. `.env` 讀取 `OPENAI_API_KEY` (Planner) 與 `GOOGLE_API_KEY` (Responder)；任一缺失即對應模組 fallback

## 專案結構

```
RACoP/source/RaCoP_LLM/
├─ README.md
├─ requirements.txt
├─ cli/
│  └─ main.py
└─ core/
	├─ pipeline/
	│  ├─ planner.py
	│  └─ responder.py
	├─ providers/
	│  ├─ openai_client.py
	│  └─ gemini_client.py
	├─ prompts/
	│  ├─ system_cop.txt
	│  ├─ system_resp.txt
	│  └─ templates/
	│     └─ pct.md
	└─ schemas/
		└─ cop_plan.schema.json
```

## 需求與安裝

- Python 3.11+
- 套件：OpenAI SDK、python-dotenv、jsonschema、google-generativeai

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` 內容：

```
openai>=1.0.0
python-dotenv>=1.0.0
jsonschema>=4.19.0
google-generativeai>=0.7.0
```

## 設定 API Key (.env)

在專案根目錄建立 `.env`：

```
OPENAI_API_KEY=sk-your-openai-key
GOOGLE_API_KEY=your-gemini-key
```

缺 `OPENAI_API_KEY` → Planner fallback (`fake_plan`)
缺 `GOOGLE_API_KEY` → Responder fallback (規則式三句)

## 執行

於 `RACoP/source/RaCoP_LLM/`：

```
python cli/main.py
```

範例（有金鑰且成功）：

```
You: I feel stuck and overthinking everything about my future.
Assistant: (LLM 回覆，2–5 句，含 PCT 反映 + 驗證 + 小步驟/開放式問題)
```

若 LLM 失敗或無金鑰，仍會得到 PCT fallback：

```
You: I feel overwhelmed.
Assistant: I’m here with you as you share this about "I feel overwhelmed.". It makes sense that you’d feel this way. What would be most supportive for you right now?
```

## 關鍵模組

- `core/pipeline/planner.py`
  - `generate_plan`: 呼叫 LLM + schema 驗證 + 重試 + fallback
  - `fake_plan`: 固定 PCT-only 計畫
- `core/providers/openai_client.py`: LLM-A (Planner) 呼叫
- `core/providers/gemini_client.py`: LLM-B (Responder) 呼叫
- `core/pipeline/responder.py`: `generate_response` 讀模板並呼叫 Gemini；失敗改用規則式 fallback

## Fallback 策略

任何以下情況觸發 fallback：

- 缺 `OPENAI_API_KEY` → Planner fallback
- Planner LLM 回傳非 JSON 或 schema 驗證失敗 (重試 >= 2 次) → Planner fallback
- 缺 `GOOGLE_API_KEY` 或 Gemini 失敗 → Responder 規則式 fallback

## 後續規劃 (展望)

- 風險/情緒抽取升級為模型鏈結
- 多療法權重動態調整與融合策略
- 對話記憶 (短期/長期) 與上下文維護
- 最終提示合成 (final_prompt) 與多輪生成

---

本階段完成：LLM Planner + Schema 驗證 + LLM Responder + 模板化 + Fallback。下一階段可擴展多輪記憶與多療法權重細化。
