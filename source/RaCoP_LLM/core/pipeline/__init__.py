"""Pipeline subpackage: includes planner & responder (Stage 1)."""
from .planner import fake_plan  # noqa: F401
from .responder import generate_response  # noqa: F401

__all__ = ["fake_plan", "generate_response"]
