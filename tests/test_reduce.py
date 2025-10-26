from functools import reduce
import operator

import numpy as np

import yirgacheffe as yg

def test_add_similar_layers() -> None:
    data = [
        np.array([[1, 2, 3, 4], [5, 6, 7, 8]]),
        np.array([[10, 20, 30, 40], [50, 60, 70, 80]]),
        np.array([[100, 200, 300, 400], [500, 600, 700, 800]]),
    ]

    origin = (0.0, 0.0)
    map_projection = yg.MapProjection("epsg:4326", 1.0, -1.0)
    layers = [yg.from_array(x, origin, map_projection) for x in data]

    summed_layers = reduce(operator.add, layers)
    actual = summed_layers.read_array(0, 0, 4, 2)

    expected = reduce(operator.add, data)

    assert (expected == actual).all()
