import os
import pickle
import tempfile

import pytest

from helpers import gdal_dataset_of_region, make_vectors_with_id
from yirgacheffe.window import Area, PixelScale, Window
from yirgacheffe.layers import RasterLayer, VectorLayer
from yirgacheffe import WGS_84_PROJECTION


def test_pickle_raster_layer() -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        path = os.path.join(tempdir, "test.tif")
        area = Area(-10, 10, 10, -10)
        layer = RasterLayer(gdal_dataset_of_region(area, 0.02, filename=path))

        p = pickle.dumps(layer)
        restore = pickle.loads(p)

        assert restore.area == area
        assert restore.pixel_scale == (0.02, -0.02)
        assert restore.geo_transform == (-10, 0.02, 0.0, 10, 0.0, -0.02)
        assert restore.window == Window(0, 0, 1000, 1000)

def test_pickle_raster_mem_layer_fails() -> None:
    area = Area(-10, 10, 10, -10)
    layer = RasterLayer(gdal_dataset_of_region(area, 0.02))

    with pytest.raises(ValueError):
        _ = pickle.dumps(layer)

def test_pickle_dyanamic_vector_layer() -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        path = os.path.join(tempdir, "test.gpkg")
        area = Area(-10.0, 10.0, 10.0, 0.0)
        make_vectors_with_id(42, {area}, path)

        layer = VectorLayer.layer_from_file(path, "id_no = 42", PixelScale(1.0, -1.0), WGS_84_PROJECTION)

        p = pickle.dumps(layer)
        restore = pickle.loads(p)

        assert restore.area == area
        assert restore.geo_transform == (area.left, 1.0, 0.0, area.top, 0.0, -1.0)
        assert restore.window == Window(0, 0, 20, 10)
        assert restore.projection == WGS_84_PROJECTION

        del layer
