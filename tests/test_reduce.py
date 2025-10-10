from functools import reduce
import operator

import numpy as np

from yirgacheffe.layers import RasterLayer
from tests.helpers import gdal_dataset_with_data

def test_add_similar_layers() -> None:
    data = [
        np.array([[1, 2, 3, 4], [5, 6, 7, 8]]),
        np.array([[10, 20, 30, 40], [50, 60, 70, 80]]),
        np.array([[100, 200, 300, 400], [500, 600, 700, 800]]),
    ]

    layers = [RasterLayer(gdal_dataset_with_data((0,0), 1.0, x)) for x in data]

    summed_layers = reduce(operator.add, layers)
    actual = summed_layers.read_array(0, 0, 4, 2)

    expected = reduce(operator.add, data)

    assert (expected == actual).all()
