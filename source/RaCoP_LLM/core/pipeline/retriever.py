"""RAG Retriever (Stage 7)
TF-IDF based retrieval over KB text files.

Public API:
    ensure_index() -> None
    search(queries: list[str], top_k: int = 5) -> list[str]

Environment-controlled toggle: RAG_ENABLED (handled in coordinator).
Index file: data/embeddings/tfidf_index.pkl

Design:
- Build index lazily on first ensure_index()/search call.
- Index format (pickle dict): {"vectorizer", "matrix", "docs", "paths"}.
- Each doc = dict(id, path, title, text, cleaned_text)
- Title extracted from first non-empty line (<=120 chars) else filename stem.
- Snippet selection: return a 300-500 char excerpt starting near first query term match; fallback to leading segment.

Safety: Content is static KB (controlled). No user data persisted here.
"""
from __future__ import annotations

from typing import List, Dict, Any
import os
import pickle
from pathlib import Path
import re
import sys

# Ensure parent directory (RaCoP_LLM root) is on sys.path so `core` package can be imported
_PKG_ROOT = Path(__file__).resolve().parents[1]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore

from core.utils.io import BASE_DIR, ensure_dir

KB_DIR = BASE_DIR / "data" / "kb"
EMB_DIR = BASE_DIR / "data" / "embeddings"
INDEX_PATH = EMB_DIR / "tfidf_index.pkl"

_vectorizer = None
_matrix = None
_docs: List[Dict[str, Any]] = []


def _iter_kb_files() -> List[Path]:
    if not KB_DIR.exists():
        return []
    return [p for p in KB_DIR.glob("*.txt") if p.is_file()]


def _read_file(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def _clean_text(t: str) -> str:
    return re.sub(r"\s+", " ", t).strip()


def _extract_title(text: str, fallback: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line[:120]
    return fallback


def _build_index() -> None:
    global _vectorizer, _matrix, _docs
    files = _iter_kb_files()
    docs: List[Dict[str, Any]] = []
    raw_texts: List[str] = []
    for idx, p in enumerate(files):
        raw = _read_file(p)
        cleaned = _clean_text(raw)
        title = _extract_title(raw, p.stem)
        docs.append({
            "id": idx,
            "path": str(p),
            "title": title,
            "text": raw,
            "cleaned_text": cleaned,
        })
        raw_texts.append(cleaned)

    if not docs:
        _vectorizer = None
        _matrix = None
        _docs = []
        return

    vectorizer = TfidfVectorizer(
        ngram_range=(1,2),
        max_df=0.9,
        min_df=1,
        stop_words="english"
    )
    matrix = vectorizer.fit_transform(raw_texts)

    ensure_dir(EMB_DIR)
    with open(INDEX_PATH, "wb") as f:
        pickle.dump({
            "vectorizer": vectorizer,
            "matrix": matrix,
            "docs": docs,
            "paths": [d["path"] for d in docs],
        }, f)

    _vectorizer = vectorizer
    _matrix = matrix
    _docs = docs


def ensure_index() -> None:
    """Ensure TF-IDF index exists in memory (and on disk)."""
    if INDEX_PATH.exists() and (_vectorizer is None or _matrix is None or not _docs):
        # try load
        try:
            with open(INDEX_PATH, "rb") as f:
                obj = pickle.load(f)
            _load_index_obj(obj)
            return
        except Exception:
            pass
    if not INDEX_PATH.exists():
        _build_index()
    elif _vectorizer is None or _matrix is None or not _docs:
        # rebuild if corrupted
        _build_index()


def _load_index_obj(obj: Dict[str, Any]) -> None:
    global _vectorizer, _matrix, _docs
    _vectorizer = obj.get("vectorizer")
    _matrix = obj.get("matrix")
    _docs = obj.get("docs") or []


def _score_queries(queries: List[str]) -> List[float]:
    if _vectorizer is None or _matrix is None or not _docs:
        return []
    # accumulate cosine scores
    import numpy as np  # local import to avoid global dependency surprise
    scores = None
    for q in queries:
        q = (q or "").strip()
        if not q:
            continue
        q_vec = _vectorizer.transform([q])
        sim = cosine_similarity(q_vec, _matrix)[0]
        if scores is None:
            scores = sim
        else:
            scores += sim
    if scores is None:
        return []
    return scores.tolist()


def _make_snippet(doc: Dict[str, Any], queries: List[str]) -> str:
    text = doc.get("cleaned_text", "")
    if not text:
        return doc.get("title", "(empty)")
    # find first query occurrence
    pos = len(text)  # default large
    for q in queries:
        if not q:
            continue
        i = text.lower().find(q.lower())
        if i != -1 and i < pos:
            pos = i
    # window extraction
    start = max(0, pos - 80)
    end = min(len(text), start + 480)
    snippet = text[start:end]
    # trim to nearest sentence boundary-ish
    snippet = re.sub(r"\s+", " ", snippet).strip()
    if len(snippet) > 500:
        snippet = snippet[:500].rsplit(" ", 1)[0]
    if len(snippet) < 300:
        # extend to 300 if possible
        end = min(len(text), start + 500)
        snippet = text[start:end]
        snippet = re.sub(r"\s+", " ", snippet).strip()
    title = doc.get("title", "")
    return f"{title}: {snippet}"


def search(queries: List[str], top_k: int = 5) -> List[str]:
    ensure_index()
    if _vectorizer is None or _matrix is None or not _docs:
        return []
    scores = _score_queries(queries)
    if not scores:
        return []
    # rank
    import numpy as np
    arr = np.array(scores)
    order = arr.argsort()[::-1]
    seen = set()
    results: List[str] = []
    for idx in order:
        if len(results) >= top_k:
            break
        doc = _docs[int(idx)]
        doc_id = doc.get("id")
        if doc_id in seen:
            continue
        seen.add(doc_id)
        results.append(_make_snippet(doc, queries))
    return results


__all__ = ["ensure_index", "search"]


def _simple_dedup(snippets: List[str], dedup_sim: float) -> List[str]:
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
    from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
    cleaned = [re.sub(r"\s+", " ", (s or "").strip()) for s in snippets if s and len(s.strip()) >= 8]
    out: List[str] = []
    for s in cleaned:
        is_dup = False
        for t in out:
            try:
                vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
                X = vec.fit_transform([s, t])
                sim = float(cosine_similarity(X[0], X[1])[0][0])
            except Exception:
                sim = 0.0
            if sim >= dedup_sim:
                is_dup = True
                break
        if not is_dup:
            out.append(s)
    return out


def build_kb_block(user_msg: str) -> str:
    if os.getenv("RAG_ENABLED", "1") == "0":
        return ""
    ensure_index()
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

    queries = [user_msg]
    cands = search(queries, top_k=max_snips * 2)
    # Filter by a weak similarity gate by rescoring with TF-IDF on candidates (self-sim)
    filtered: List[str] = []
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
    from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
    try:
        vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        for s in cands:
            X = vec.fit_transform([user_msg, s])
            sim = float(cosine_similarity(X[0], X[1])[0][0])
            if sim >= min_sim:
                filtered.append(s)
    except Exception:
        filtered = cands
    unique = _simple_dedup(filtered, dedup_sim)
    unique = unique[:max_snips]
    if not unique:
        return ""
    body = "\n---\n".join(unique)
    return f"<kb_snippets>\n{body}\n</kb_snippets>"


__all__.extend(["build_kb_block"]) 
