from pathlib import Path
import sys

# Ensure parent directory (RaCoP_LLM root) is on sys.path so `core` package can be imported
_PKG_ROOT = Path(__file__).resolve().parents[1]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from core.pipeline import safety

def test_high_risk_detection():
    text = "I feel like ending it all tonight"
    result = safety.assess(text)
    assert result["risk_level"] in ("high", "medium", "low")
    assert result["high_risk"] is True, "Should flag high risk keywords"


def test_professional_referral_message():
    referral = safety.escalation_message()
    assert "help" in referral.lower()

if __name__ == "__main__":
    test_high_risk_detection()
    test_professional_referral_message()
    print("OK: test_safety")
