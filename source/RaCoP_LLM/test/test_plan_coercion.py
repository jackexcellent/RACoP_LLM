from pathlib import Path
import sys

# Ensure parent directory (RaCoP_LLM root) is on sys.path so `core` package can be imported
_PKG_ROOT = Path(__file__).resolve().parents[1]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from core.pipeline import planner

# We simulate malformed data structures already parsed into python objects and ensure coercion stabilizes them.

def test_coercion_handles_string_list_items():
    data = {"plan": ["PCT", "CBT"], "risk": {"signals": "crying,withdrawn"}, "emotions": {"primary": "sadness", "secondary": "anxiety"}}
    coerced = planner._coerce_for_schema(data)  # type: ignore
    assert isinstance(coerced["plan"], list)
    assert all(isinstance(p, dict) for p in coerced["plan"])
    assert isinstance(coerced["risk"]["signals"], list)
    assert isinstance(coerced["emotions"], list) and "sadness" in coerced["emotions"]


def test_coercion_merges_template_slots():
    data = {"pct": {"starter": "Hi"}, "template_slots": {"cbt": {"reframe": "Try X"}}}
    coerced = planner._coerce_for_schema(data)  # type: ignore
    assert "pct" in coerced["template_slots"], "pct should merge into template_slots"
    assert "cbt" in coerced["template_slots"], "existing template_slots preserved"

if __name__ == "__main__":
    test_coercion_handles_string_list_items()
    test_coercion_merges_template_slots()
    print("OK: test_plan_coercion")
