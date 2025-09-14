"""History Conversation RAG (pre-Planner)
TF-IDF retrieval over session JSONL to produce a compact <history> block.

Env defaults (read in builder):
    CTX_MAX_SNIPPETS (6)
    CTX_MIN_SIM (0.18)
    CTX_DEDUP_SIM (0.85)
    HIST_RAG_NGRAM (2)

Public API:
    search_history(session_id: str, query: str, top_k: int = 12) -> list[str]
    clean_and_merge(snippets: list[str], max_snippets: int, min_sim: float, dedup_sim: float) -> list[str]
    build_history_block(session_id: str, user_msg: str) -> str
"""
from __future__ import annotations

from typing import List, Sequence
from pathlib import Path
import os
import json
import re

from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore

from core.utils.io import BASE_DIR


def _session_path(session_id: str) -> Path:
    safe_id = session_id.replace("/", "_").replace("..", "_")
    return BASE_DIR / "runs" / "sessions" / f"{safe_id}.jsonl"


def _read_lines(session_id: str) -> List[str]:
    path = _session_path(session_id)
    if not path.exists():
        return []
    lines: List[str] = []
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
            role = (obj.get("role") or "").strip()
            text = (obj.get("text") or "").strip()
            if not text:
                continue
            # Collapse whitespace
            text_clean = re.sub(r"\s+", " ", text)
            line = f"{role}: {text_clean}" if role else text_clean
            lines.append(line)
        except Exception:
            continue
    return lines


def _prepare_query(user_msg: str) -> str:
    q = (user_msg or "").strip()
    q = re.sub(r"\s+", " ", q)
    return q


def search_history(session_id: str, query: str, top_k: int = 12) -> List[str]:
    docs = _read_lines(session_id)
    query = _prepare_query(query)
    if not docs or not query:
        return []
    try:
        ngram = int(os.getenv("HIST_RAG_NGRAM", "2") or 2)
    except Exception:
        ngram = 2

    try:
        vectorizer = TfidfVectorizer(
            ngram_range=(1, max(1, ngram)),
            max_df=0.9,
            min_df=1,
            stop_words="english",
        )
        doc_matrix = vectorizer.fit_transform(docs)
        query_vec = vectorizer.transform([query])
        sims = cosine_similarity(doc_matrix, query_vec)[:, 0]
    except Exception:
        return []

    import numpy as np
    order = np.argsort(sims)[::-1]
    results: List[str] = []
    for i in order[: max(1, top_k)]:
        s = docs[int(i)]
        if s and float(sims[int(i)]) > 0:
            results.append(s)
    return results


def _cosine_sim_sentences(a: str, b: str) -> float:
    try:
        vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        X = vec.fit_transform([a, b])
        sim = cosine_similarity(X[0], X[1])[0][0]
        return float(sim)
    except Exception:
        return 0.0


def clean_and_merge(snippets: List[str], max_snippets: int, min_sim: float, dedup_sim: float) -> List[str]:
    # Normalize
    cleaned = []
    for s in snippets:
        t = re.sub(r"\s+", " ", (s or "").strip())
        if len(t) >= 8:
            cleaned.append(t)
    if not cleaned:
        return []

    # Dedup by similarity/hash
    unique: List[str] = []
    for s in cleaned:
        dup = False
        for u in unique:
            if _cosine_sim_sentences(s, u) >= dedup_sim:
                dup = True
                break
        if not dup:
            unique.append(s)

    # Truncate to max
    return unique[: max(1, max_snippets)]


def build_history_block(session_id: str, user_msg: str) -> str:
    try:
        max_snips = int(os.getenv("CTX_MAX_SNIPPETS", "6") or 6)
    except Exception:
        max_snips = 6
    try:
        min_sim = float(os.getenv("CTX_MIN_SIM", "0.18") or 0.18)
    except Exception:
        min_sim = 0.18
    try:
        dedup_sim = float(os.getenv("CTX_DEDUP_SIM", "0.85") or 0.85)
    except Exception:
        dedup_sim = 0.85

    raw = search_history(session_id, user_msg, top_k=max_snips * 2)
    merged = clean_and_merge(raw, max_snippets=max_snips, min_sim=min_sim, dedup_sim=dedup_sim)
    if not merged:
        return ""
    body = "\n---\n".join(merged)
    return f"<history>\n{body}\n</history>"


__all__ = ["search_history", "clean_and_merge", "build_history_block"]
