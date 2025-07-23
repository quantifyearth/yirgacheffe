import os

BACKEND = os.environ.get("YIRGACHEFFE_BACKEND", "NUMPY").upper()

match BACKEND:
    case "MLX":
        from . import mlx
        backend = mlx
    case "NUMPY":
        from . import numpy
        backend = numpy # type: ignore[misc]
    case _:
        raise NotImplementedError("Only NUMPY and MLX backends supported")
