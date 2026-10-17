"""SSI bridge-response surrogate package."""

from .schema import RESPONSE_COMPONENTS, GraphCase, ResponseCase

try:
    from .models import MODEL_NAMES, build_model
except ModuleNotFoundError as exc:  # Allows metadata/HDF5 tools without PyTorch installed.
    if exc.name != "torch":
        raise
    MODEL_NAMES = ()
    build_model = None

__all__ = ["RESPONSE_COMPONENTS", "GraphCase", "ResponseCase", "MODEL_NAMES", "build_model"]
