"""I/O utilities for RaCoP sessions (Stage 6)."""
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Union
import json
import os


def _detect_base_dir() -> Path:
    # This file path: <root>/core/utils/io.py
    here = Path(__file__).resolve()
    # ascend to core -> RaCoP_LLM root
    return here.parent.parent.parent


BASE_DIR: Path = _detect_base_dir()
SESS_DIR: Path = BASE_DIR / "runs" / "sessions"
KB_DIR: Path = BASE_DIR / "data" / "kb"
EMB_DIR: Path = BASE_DIR / "data" / "embeddings"
PROFILES_DIR: Path = BASE_DIR / "runs" / "profiles"


def ensure_dir(p: Union[str, Path]) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def session_path(session_id: str) -> Path:
    ensure_dir(SESS_DIR)
    safe_id = session_id.replace("/", "_").replace("..", "_")
    return SESS_DIR / f"{safe_id}.jsonl"


def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


# Ensure base directories at import
ensure_dir(SESS_DIR)
ensure_dir(KB_DIR)
ensure_dir(EMB_DIR)
ensure_dir(PROFILES_DIR)

def read_json(path: Path, default: dict | None = None) -> dict:
    if not path.exists():
        return {} if default is None else default
    try:
        import json
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {} if default is None else default

def write_json(path: Path, obj: dict) -> None:
    ensure_dir(path.parent)
    import json
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

__all__ = [
    "BASE_DIR",
    "SESS_DIR",
    "KB_DIR",
    "EMB_DIR",
    "ensure_dir",
    "session_path",
    "append_jsonl",
    "read_jsonl",
    "PROFILES_DIR",
    "read_json",
    "write_json",
]
