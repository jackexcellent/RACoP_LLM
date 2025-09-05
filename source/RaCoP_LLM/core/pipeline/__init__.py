"""Pipeline subpackage: includes planner & responder (Stage 1)."""
from .planner import fake_plan  # noqa: F401
from .responder import respond  # noqa: F401

__all__ = ["fake_plan", "respond"]
