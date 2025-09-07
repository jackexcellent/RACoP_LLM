import os
from pathlib import Path
import sys

# Ensure parent directory (RaCoP_LLM root) is on sys.path so `core` package can be imported
_PKG_ROOT = Path(__file__).resolve().parents[1]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))
    
from core.pipeline import retriever
from core.utils.io import KB_DIR, EMB_DIR

TEST_SESSION = "unittest_session"

def test_retriever_build_and_search():
    # Ensure index builds
    retriever.ensure_index()
    index_file = EMB_DIR / "tfidf_index.pkl"
    assert index_file.exists(), "Index file should be created"

    results = retriever.search("reflection empathy feelings")
    assert isinstance(results, list)
    assert len(results) > 0, "Should retrieve at least one snippet"
    # Expect a tuple (title, snippet)
    title, snippet = results[0]
    assert isinstance(title, str) and isinstance(snippet, str)

if __name__ == "__main__":
    test_retriever_build_and_search()
    print("OK: test_retriever_build_and_search")
