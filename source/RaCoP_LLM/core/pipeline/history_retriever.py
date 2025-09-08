"""History Conversation RAG
TF-IDF retrieval over the current session's JSONL conversation (no persistent index).

Environment variables (with defaults):
  HISTORY_RAG_ENABLED: "1" enables (checked in coordinator, not here)
  HIST_RAG_MAX: max merged history segments to return (default 5)
  HIST_RAG_MIN_SIM: cosine similarity threshold (default 0.18)
  HIST_RAG_MERGE_NEIGHBOR_RADIUS: line distance threshold for merging hits (default 1)
  HIST_RAG_NGRAM: ngram upper bound (default 2)

Public API:
  search_history(session_id, queries, user_msg, **overrides) -> list[str]

Implementation notes:
  - Reads runs/sessions/<session_id>.jsonl via memory/session log format.
  - Each line converted to a simple doc string: "role: text" (condensed whitespace).
  - Vectorizes documents + queries using TfidfVectorizer over (1, ngram).
  - Computes similarity for each document as max cosine over queries OR sum (here we use max for sharper selection).
  - Filters docs below min_sim.
  - Ranks remaining docs by similarity desc, pick top_k (default: min(8, 2*max_snippets)).
  - Merges adjacent hit lines whose indices distance <= merge_neighbor_radius into contiguous windows.
  - Each merged window concatenates original per-line "role: text" lines separated by space; collapses whitespace.
  - Returns up to max_snippets merged segments.
  - Pure in-memory; no disk writes.
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


def _prepare_queries(user_msg: str, queries: Sequence[str]) -> List[str]:
    all_q: List[str] = []
    candidates = [user_msg or ""] + list(queries or [])
    for q in candidates:
        q = (q or "").strip()
        if not q:
            continue
        # simple whitespace collapse
        q = re.sub(r"\s+", " ", q)
        if q and q.lower() not in {x.lower() for x in all_q}:
            all_q.append(q)
    return all_q[:12]  # sanity cap


def search_history(
    session_id: str,
    queries: Sequence[str] | None,
    user_msg: str,
    *,
    max_snippets: int | None = None,
    min_sim: float | None = None,
    merge_neighbor_radius: int | None = None,
    ngram: int | None = None,
    top_k: int | None = None,
) -> List[str]:
    # Load env defaults
    env_max = int(os.getenv("HIST_RAG_MAX", "5") or 5)
    env_min_sim = float(os.getenv("HIST_RAG_MIN_SIM", "0.18") or 0.18)
    env_radius = int(os.getenv("HIST_RAG_MERGE_NEIGHBOR_RADIUS", "1") or 1)
    env_ngram = int(os.getenv("HIST_RAG_NGRAM", "2") or 2)

    max_snippets = max_snippets if max_snippets is not None else env_max
    min_sim = min_sim if min_sim is not None else env_min_sim
    merge_neighbor_radius = (
        merge_neighbor_radius if merge_neighbor_radius is not None else env_radius
    )
    ngram = ngram if ngram is not None else env_ngram
    top_k = top_k if top_k is not None else max(8, max_snippets * 2)

    docs = _read_lines(session_id)
    if not docs or not user_msg.strip():
        return []
    q_list = _prepare_queries(user_msg, queries or [])
    if not q_list:
        return []

    try:
        vectorizer = TfidfVectorizer(
            ngram_range=(1, max(1, ngram)),
            max_df=0.9,
            min_df=1,
            stop_words="english",
        )
        doc_matrix = vectorizer.fit_transform(docs)
        query_matrix = vectorizer.transform(q_list)
        # cosine similarities: for each doc take max over queries for selectivity
        sims = cosine_similarity(doc_matrix, query_matrix)  # shape (D, Q)
    except Exception:
        return []

    import numpy as np  # local import
    max_sims = sims.max(axis=1) if sims.size else np.array([])
    if max_sims.size == 0:
        return []

    # Collect candidate indices meeting threshold
    candidate_indices = [i for i, s in enumerate(max_sims) if s >= min_sim]
    if not candidate_indices:
        return []

    # Rank by similarity descending
    ranked = sorted(candidate_indices, key=lambda i: float(max_sims[i]), reverse=True)
    ranked = ranked[:top_k]

    # Merge neighbors based on line distance
    ranked_sorted = sorted(ranked)
    windows = []  # list of (start, end)
    cur_start = cur_end = None
    for idx in ranked_sorted:
        if cur_start is None:
            cur_start = cur_end = idx
            continue
        if idx - cur_end <= merge_neighbor_radius:
            cur_end = idx
        else:
            windows.append((cur_start, cur_end))
            cur_start = cur_end = idx
    if cur_start is not None:
        windows.append((cur_start, cur_end))

    # Construct merged segments
    segments: List[str] = []
    for (s, e) in windows:
        merged_lines = docs[s : e + 1]
        merged_text = " ".join(merged_lines)
        merged_text = re.sub(r"\s+", " ", merged_text).strip()
        if merged_text:
            segments.append(merged_text)
        if len(segments) >= max_snippets:
            break

    return segments[:max_snippets]


__all__ = ["search_history"]
