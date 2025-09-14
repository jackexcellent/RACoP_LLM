# RaCoP 心理支持聊天機器人（Planner-only）

新架構：Input → (History RAG + KB RAG) → Planner(JSON) → Coordinator(路由+短回覆組裝) → Output

重點

- 前置 RAG：對會話歷史與 KB 做 TF‑IDF 檢索、去重合併後注入 Planner 的 recent_context。
- 單一 LLM：Planner 僅輸出 JSON（含 mode 與 PCT/CBT/SFBT 槽位）。
- 最終回覆：Coordinator 依 mode 與權重把槽位組成短句（最多 OUTPUT_MAX_SENTENCES 句）。
- 風險與轉介：risk=high → 升級；含 DBT → 轉介；off_topic → 「與此程式無關，不予回應。」

環境變數（預設值）

- RAG_ENABLED="1"（KB RAG 開關）
- HISTORY_RAG_ENABLED="1"（History RAG 開關）
- CTX_MAX_SNIPPETS="6"、CTX_MIN_SIM="0.18"、CTX_DEDUP_SIM="0.85"、HIST_RAG_NGRAM="2"
- OUTPUT_MAX_SENTENCES="3"、DEBUG_PLAN_JSON="0"

執行（REPL）

```
python cli/main.py
```

輸入 exit/quit 結束。DEBUG_PLAN_JSON=1 可於 CLI 顯示當輪 Planner JSON。

檔案結構（節錄）

```
RACoP/source/RaCoP_LLM/
├─ cli/main.py                 # REPL
├─ core/pipeline/
│  ├─ coordinator.py           # 主流程＋路由＋短回覆組裝
│  ├─ planner.py               # 單一 LLM（只回 JSON）
│  ├─ history_retriever.py     # 對話歷史檢索與合併，輸出 <history> 區塊
│  ├─ retriever.py             # KB RAG，輸出 <kb_snippets> 區塊
│  └─ safety.py                # 風險評估、轉介訊息
├─ core/prompts/system_plan.txt
├─ core/schemas/cop_plan.schema.json
└─ runs/sessions/, runs/profiles/
```

短回覆策略

- greet：greeting + light_question（1–2 句）
- gather：PCT starter + validation + 1–2 個開放式問題（不給建議；2–3 句）
- counsel：PCT 主幹（starter + validation + question）；
  - 若 CBT.weight ≥ 0.4 且有 alt_thought → 附上一句替代想法
  - 若 SFBT.weight ≥ 0.4 且有 one_step → 附上一句超小步
  - 合併後總句數 ≤ OUTPUT_MAX_SENTENCES；超出則優先保留 PCT 與擇一（CBT 或 SFBT）

品質檢查（手動驗收）

- 啟動後可多輪聊天；hi/how are you → greet；「最近壓力大但說不清」→ gather；
  明確壓力脈絡 → counsel（2–3 句）；「1+1?」→ off_topic 固定回覆。
- 有歷史相近主題時，Planner 的計畫會更貼近過去內容；無歷史時亦能正常。
- 觸發風險/DBT 規則時不給技巧；輸出升級或轉介訊息。
