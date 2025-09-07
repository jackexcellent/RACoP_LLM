import os
import json
from pathlib import Path
import sys

# Ensure parent directory (RaCoP_LLM root) is on sys.path so `core` package can be imported
_PKG_ROOT = Path(__file__).resolve().parents[1]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from core.pipeline import memory  # type: ignore
from core.utils.io import BASE_DIR, read_json  # type: ignore

TEST_SESSION = "unittest_session"

def test_profile_feedback_cycle():
    # Clean old artifacts
    sess_path = BASE_DIR / "runs" / "sessions" / f"{TEST_SESSION}.jsonl"
    prof_path = BASE_DIR / "runs" / "profiles" / f"{TEST_SESSION}.json"
    if sess_path.exists():
        sess_path.unlink()
    if prof_path.exists():
        prof_path.unlink()

    # Append an assistant turn with plan meta (simulating previous response)
    plan_meta = {
        "plan": {
            "risk": {"level": "low"},
            "plan": [
                {"therapy": "PCT"},
                {"therapy": "CBT"}
            ],
            "template_slots": {"sfbt": {"one_step": "Take a 5-minute mindful break."}},
            "tone": "warm"
        }
    }
    # Use plan_summary to ensure consistent format
    dummy_plan = {
        "risk": {"level": "low"},
        "plan": [
            {"therapy": "PCT"},
            {"therapy": "CBT"}
        ],
        "template_slots": {"sfbt": {"one_step": "Take a 5-minute mindful break."}},
        "tone": "warm"
    }
    memory.append_session(TEST_SESSION, "assistant", "previous reply", meta={"plan": memory.plan_summary(dummy_plan)})

    # User gives positive feedback
    user_msg = "That helped, it was useful thank you"
    memory.update_profile_from_feedback(TEST_SESSION, user_msg)

    profile = memory.load_profile(TEST_SESSION)
    assert "PCT" in profile["effective_skills"] or "CBT" in profile["effective_skills"], "Therapies should be marked effective"
    assert any("Take a 5-minute mindful break" in s for s in profile["effective_skills"]), "Suggested action should be stored"

    # Next round negative feedback should move skills to ineffective
    neg_msg = "That didn't help and feels worse"
    memory.update_profile_from_feedback(TEST_SESSION, neg_msg)
    profile2 = memory.load_profile(TEST_SESSION)
    assert any(skill in profile2["ineffective_skills"] for skill in ["PCT", "CBT"]), "Therapies should move to ineffective"

if __name__ == "__main__":
    test_profile_feedback_cycle()
    print("OK: test_profile_feedback_cycle")
