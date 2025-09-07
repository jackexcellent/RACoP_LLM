import sys
from pathlib import Path

# Add project src path so `import core...` works during tests
PROJECT_ROOT = Path(__file__).resolve().parents[2] / "source" / "RaCoP_LLM"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
