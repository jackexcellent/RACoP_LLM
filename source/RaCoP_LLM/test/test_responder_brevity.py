from pathlib import Path
import sys

# Ensure parent directory (RaCoP_LLM root) is on sys.path so `core` package can be imported
_PKG_ROOT = Path(__file__).resolve().parents[1]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from core.pipeline import responder

class DummyPlan:
    pass

# We'll craft a small fake plan dict and call internal assembly logic if available.

def test_sentence_trimming():
    text = "This is sentence one. This is sentence two. This is sentence three. This is sentence four. This is sentence five."  # noqa: E501
    trimmed = responder._trim_sentences(text, max_sentences=3)  # type: ignore
    assert trimmed.count('.') <= 3, "Should keep at most three sentences"

if __name__ == "__main__":
    test_sentence_trimming()
    print("OK: test_responder_brevity")
