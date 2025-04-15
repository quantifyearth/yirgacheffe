import os

BACKEND = os.environ.get("YIRGACHEFFE_BACKEND", "NUMPY").upper()

match BACKEND:
    case "MLX":
        from . import mlx
        backend = mlx
    case _:
        from . import numpy
        backend = numpy
