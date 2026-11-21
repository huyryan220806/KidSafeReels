class ModelNotReadyError(RuntimeError):
    """Required local model/dependency is unavailable."""


class InvalidMediaError(ValueError):
    """Video cannot be decoded or exceeds supported limits."""
