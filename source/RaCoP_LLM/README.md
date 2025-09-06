# RaCoP 心理支持聊天機器人 (Stage 6)

本階段：在 Stage 5（短期記憶 + Safety Gate v1）基礎上，升級為「多療法策略 (Multi‑Therapy) + DBT 專業轉介 (Professional Referral)」。新流程：

使用者輸入 → 讀取過去 N 輪摘要 (Short Context) → Planner (OpenAI + schema + fallback) 產生多療法計畫 → Safety 風險檢測 → 若 high risk → 升級訊息；若計畫包含 DBT → 專業轉介訊息；否則進入 Responder (Gemini 多模板組合或 fallback) → 寫入 JSONL 會話檔。

## 功能摘要 (Stage 6)

1. 單輪 CLI：讀一行輸入 → 生成多療法計畫 → 風險/轉介判斷 → 回覆/升級 → 寫入紀錄
2. Planner (`generate_plan`):
   - System prompt: `core/prompts/system_cop.txt`（已擴展為多療法 slot 指令）
   - 使用 OpenAI: gpt-4o-mini, temperature=0.2
   - 僅允許純 JSON；以 `core/schemas/cop_plan.schema.json` 驗證
   - 最多 2 次重試；失敗 → `fake_plan` (單一 PCT) fallback
   - 可能輸出 therapies（例如 PCT, CBT, SFBT, DBT）及對應 template_slots
3. Safety / Referral：
   - `safety.assess`：risk=high 或高風險關鍵詞 → 升級訊息 (不進入 Responder)
   - `safety.requires_professional_for_dbt(plan)`：若計畫中出現 DBT → 回覆專業轉介訊息（不生成 DBT 具體技巧內容）
4. Responder (`generate_response`):
   - 使用 Gemini (gemini-2.0-flash, temp=0.7)
   - 根據計畫 therapies 動態組合模板：
     - 永遠包含 PCT (核心同理/反映)
     - 若計畫含 CBT → 加入 CBT 認知重構 / 行為小步驟段落
     - 若計畫含 SFBT → 加入解決導向（例：例外、資源、下一小步）
     - DBT 不會在此生成（改為 referral）
   - 若 LLM 失敗或無金鑰 → 規則式 PCT 簡化 fallback
5. `.env` 讀取：
   - `OPENAI_API_KEY` (Planner) 缺失 → 使用 `fake_plan`
   - `GOOGLE_API_KEY` (Responder) 缺失 → 規則式 fallback
6. 記憶：讀取最近 N=12 turns 摘要作為 planner 與 responder 的短期上下文
7. 儲存：每輪將 user / assistant 追加 JSONL（含 plan 摘要 meta）

### 多療法邏輯摘要

| Therapy | 來源         | 生成方式              | 何時使用        | 備註                      |
| ------- | ------------ | --------------------- | --------------- | ------------------------- |
| PCT     | Planner plan | 模板 + LLM / fallback | 永遠 (基底同理) | 反映 + 驗證情緒           |
| CBT     | Planner plan | 模板 slot → LLM       | 計畫含 CBT      | 聚焦自動思考 / 認知再框架 |
| SFBT    | Planner plan | 模板 slot → LLM       | 計畫含 SFBT     | 資源、例外、下一個小步驟  |
| DBT     | Planner plan | 不進入 responder      | 計畫含 DBT      | 直接給「專業轉介」訊息    |

模板位置：`core/prompts/templates/` 目前包括：`pct.md`, `cbt.md`, `sfbt.md`（無 dbt.md by 設計）。

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
	│  ├─ responder.py
	│  ├─ coordinator.py
	│  ├─ safety.py
	│  └─ memory.py
	├─ providers/
	│  ├─ openai_client.py
	│  └─ gemini_client.py
	├─ prompts/
	│  ├─ system_cop.txt
	│  ├─ system_resp.txt
	│  └─ templates/
	│     ├─ pct.md
	│     ├─ cbt.md
	│     └─ sfbt.md
	├─ schemas/
	│  └─ cop_plan.schema.json
	└─ utils/
	   └─ io.py
└─ runs/
	└─ sessions/
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

## Safety Gate 與 DBT 專業轉介

觸發高風險的條件：

- Planner 產生的 `plan.risk.level == "high"`
- 或使用者輸入文字包含下列任一關鍵詞（中英）例如："自殺", "想死", "kill myself", "end my life", "self-harm", "overdose"。

高風險處理：

- 不呼叫 Responder；直接回覆升級訊息 (英文 3–6 句)

DBT 專業轉介：

- 若 planner 計畫中包含 therapy = "DBT"，系統視為需要更具結構之技能訓練 / 情緒調節支持 → 不生成具體 DBT 技巧段落 → 直接輸出專業轉介訊息（鼓勵尋求合格治療師、醫療資源、同時保持支持語氣）。

此設計目的：避免模型幻覺式產生複雜或潛在錯誤的 DBT 技巧指導，在無人工監督前以「安全轉介」替代。

- 內容包含：情緒承接、安全重視、鼓勵聯絡緊急服務或可信任他人、說明此系統不替代專業急救
- 不提供具體自傷方式或聯絡電話（此版本維持泛化）

限制：v1 僅區分 high / low，未提供中度層級 (med)。未來可拓展多層級與地區化資源連結。

## Sessions & Memory

環境變數 `SESSION_ID` 用於指定會話；未設定時預設為 `default`。每輪互動會寫入：
`runs/sessions/<session_id>.jsonl`，每行為一個 JSON 物件：

```
{ "ts": 1736160000, "role": "user"|"assistant", "text": "...", "meta": {"plan": {"risk": "low", "therapies": ["PCT"], "tone": "warm, validating, non-judgmental"}} }
```

短期記憶：在生成新計畫時取最近 N (預設 12) 筆 user/assistant 轉換為簡短文字：

```
user: ...
assistant: ...
...
```

此摘要作為 Planner 的 recent_ctx 與 Responder 的 short_ctx，以便回覆更貼近上下文。

若檔案不存在會自動建立；所有寫入為追加模式，避免資料覆蓋。

## Fallback 策略

順序性判斷（前一條件優先生效）：

1. 高風險 → 升級訊息（停止後續）
2. 計畫含 DBT → 專業轉介訊息（停止後續）
3. 缺 `OPENAI_API_KEY` → Planner fake_plan
4. Planner 回傳非 JSON / schema 失敗 (>=2 次) → fake_plan
5. 缺 `GOOGLE_API_KEY` 或 Responder LLM 失敗 → 規則式 PCT fallback
6. 其他未預期例外 → 簡短道歉訊息

## 後續規劃 (展望)

- 風險/情緒抽取升級為模型鏈結
- 多療法權重動態調整與融合策略
- 對話記憶 (短期/長期) 與上下文維護
- 最終提示合成 (final_prompt) 與多輪生成

---

Stage 6 完成：短期記憶 + LLM Planner + Schema 驗證 + Safety Gate v1 + 多療法 (PCT/CBT/SFBT) + DBT 專業轉介 + 多模板 LLM Responder + 多層 fallback。

## Stage 7 (RAG 檢索試驗版)

新增 TF-IDF 檢索 (`core/pipeline/retriever.py`)：

- KB 來源：`data/kb/*.txt`（預設三份：pct_reflection / cbt_reframing / sfbt_scaling）
- 首次執行或索引檔缺失時自動建立：`data/embeddings/tfidf_index.pkl`
- 向量化：TfidfVectorizer(ngram_range=(1,2), max_df=0.9, min_df=1, stop_words="english")
- Planner 可輸出 `retrieval_queries` (list[str])；Coordinator 在非 high risk、非 DBT 分支且 `RAG_ENABLED != 0` 時呼叫 search 取前 5 條 snippet
- Responder 以 `<kb_snippets> ... </kb_snippets>` 內嵌檢索片段，模型被指示「可吸收精華、不可逐字引用、不產生列表」

環境變數：

- `RAG_ENABLED=0` → 關閉檢索流程（不載入索引、不搜尋、不傳 kb_snippets）
- 其他值（預設未設）→ 啟用 RAG

更新檔案：

- `core/pipeline/coordinator.py`：整合 RAG 開關與 kb_snippets 注入
- `core/pipeline/responder.py`：新增 `kb_snippets` 參數與 prompt block
- `core/pipeline/retriever.py`：索引與檢索實作
- `core/utils/io.py`：新增 `KB_DIR`、`EMB_DIR`
- `requirements.txt`：加入 `scikit-learn`
- `data/kb/`：三份示例知識檔
- `core/prompts/system_resp.txt`：加入 CONTEXT 區段（kb_snippets 使用規則）

維護 / 替換 KB：

1. 將自訂 .txt 放入 `data/kb/`
2. 刪除 `data/embeddings/tfidf_index.pkl`
3. 再次執行程式會自動重建索引

限制：

- 僅簡單 TF-IDF；無向量語義查詢
- 無增量更新（需刪索引重建）
- snippet 為字串拼接（無高亮與排名解釋）

後續可擴展：BM25 / embedding 模型、段落切片、得分加權注入、檢索品質評估。

後續可擴展：

- 長期記憶 / 檢索式上下文
- 風險等級細緻化 (low/med/high) 與地區化資源建議
- 動態療法權重與融合（而非段落串接）
- 評估 / 追蹤情緒變化的指標化摘要
