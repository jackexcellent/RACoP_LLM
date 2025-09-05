# RaCoP 心理支持聊天機器人 (Stage 2)

本階段：Planner 由硬編升級為「LLM + JSON Schema 驗證 + Fallback」。若 LLM 輸出失敗或金鑰缺失，則自動退回 PCT-only 假資料計畫，再由 Responder 生成 2–3 句英文同理回覆。

## 功能摘要

1. CLI 單輪互動：讀一行輸入 → 產生計畫 → 生成回覆
2. Planner (`generate_plan`):
   - 讀取 `core/prompts/system_cop.txt` 作為 system prompt
   - 呼叫 OpenAI (模型: gpt-4o-mini, temperature=0.2)
   - 嚴格要求僅輸出 JSON；用 `core/schemas/cop_plan.schema.json` 驗證
   - 失敗重試最多 2 次；仍失敗 → `fake_plan` fallback
3. Responder (`respond`): 使用 PCT 槽位 (starter / validation / question) 組合回覆，缺失即填預設
4. `.env` 讀取 `OPENAI_API_KEY`（可缺：會 fallback）

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
	│  └─ openai_client.py
	├─ prompts/
	│  └─ system_cop.txt
	└─ schemas/
		└─ cop_plan.schema.json
```

## 需求與安裝

- Python 3.11+
- 套件：OpenAI SDK、python-dotenv、jsonschema

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
```

## 設定 API Key (.env)

在專案根目錄建立 `.env`：

```
OPENAI_API_KEY=sk-xxxx...
```

未設定或金鑰失效時，Planner 會直接使用 `fake_plan`。

## 執行

於 `RACoP/source/RaCoP_LLM/`：

```
python cli/main.py
```

範例（有金鑰且成功）：

```
You: I feel stuck and overthinking everything about my future.
Assistant: I sense the deep emotions you're experiencing about "I feel stuck and overthinking everything about my futu…". Your feelings are completely valid and worth exploring. What do you need most in this moment?
```

若 LLM 失敗或無金鑰，仍會得到 PCT fallback：

```
You: I feel overwhelmed.
Assistant: I sense the deep emotions you're experiencing about "I feel overwhelmed.". Your feelings are completely valid and worth exploring. What do you need most in this moment?
```

## 關鍵模組

- `core/pipeline/planner.py`
  - `generate_plan`: 呼叫 LLM + schema 驗證 + 重試 + fallback
  - `fake_plan`: 固定 PCT-only 計畫
- `core/providers/openai_client.py`: 輕量 OpenAI 包裝，失敗回空字串
- `core/pipeline/responder.py`: 取計畫 `template_slots.pct` 字段組合回覆

## Fallback 策略

任何以下情況觸發 fallback：

- 缺 `OPENAI_API_KEY`
- OpenAI SDK 初始化失敗
- 回傳非 JSON 或 schema 驗證失敗 (重試 >= 2 次後)

## 後續規劃 (展望)

- 風險/情緒抽取升級為模型鏈結
- 多療法權重動態調整與融合策略
- 對話記憶 (短期/長期) 與上下文維護
- 最終提示合成 (final_prompt) 與多輪生成

---

本階段完成：LLM Planner + Schema 驗證 + Responder + Fallback。下一階段可擴展多療法槽位與上下文管理。
