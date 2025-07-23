import os
from types import ModuleType

BACKEND = os.environ.get("YIRGACHEFFE_BACKEND", "NUMPY").upper()

match BACKEND:
    case "MLX":
        from . import mlx
        backend: ModuleType = mlx
    case "NUMPY":
        from . import numpy
        backend = numpy
    case _:
        raise NotImplementedError("Only NUMPY and MLX backends supported")
