
import math
import os
import pickle
import tempfile

import numpy as np
import pytest

from tests.helpers import gdal_dataset_of_region, make_vectors_with_id
from yirgacheffe.window import Area, MapProjection, PixelScale, Window
from yirgacheffe.layers import ConstantLayer, GroupLayer, RasterLayer, RescaledRasterLayer, \
    UniformAreaLayer, VectorLayer
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
    with RasterLayer(gdal_dataset_of_region(area, 0.02)) as layer:
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

def test_pickle_uniform_area_layer() -> None:
    pixel_scale = 0.2
    with tempfile.TemporaryDirectory() as tempdir:
        path = os.path.join(tempdir, "test.tif")
        area = Area(
            math.floor(-180 / pixel_scale) * pixel_scale,
            math.ceil(90 / pixel_scale) * pixel_scale,
            (math.floor(-180 / pixel_scale) * pixel_scale) + pixel_scale,
            math.floor(-90 / pixel_scale) * pixel_scale
        )
        dataset = gdal_dataset_of_region(area, pixel_scale, filename=path)
        assert dataset.RasterXSize == 1
        assert dataset.RasterYSize == math.ceil(180 / pixel_scale)
        dataset.Close()

        layer = UniformAreaLayer.layer_from_file(path)

        p = pickle.dumps(layer)
        restore = pickle.loads(p)

        assert restore.pixel_scale == (pixel_scale, -pixel_scale)
        assert restore.area == Area(
            math.floor(-180 / pixel_scale) * pixel_scale,
            math.ceil(90 / pixel_scale) * pixel_scale,
            math.ceil(180 / pixel_scale) * pixel_scale,
            math.floor(-90 / pixel_scale) * pixel_scale
        )
        assert restore.window == Window(
            0,
            0,
            math.ceil((restore.area.right - restore.area.left) / pixel_scale),
            math.ceil((restore.area.top - restore.area.bottom) / pixel_scale)
        )

def test_pickle_group_layer() -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        path = os.path.join(tempdir, "test.tif")
        area = Area(-10, 10, 10, -10)
        dataset = gdal_dataset_of_region(area, 0.2, filename=path)
        dataset.Close()

        group = GroupLayer.layer_from_directory(tempdir)
        expected = group.read_array(0, 0, 100, 100)
        assert expected.sum() != 0 # just check there is meaningful data

        p = pickle.dumps(group)
        restore = pickle.loads(p)

        assert restore.area == area
        assert restore.window == Window(0, 0, 100, 100)

        result = restore.read_array(0, 0, 100, 100)
        assert (expected == result).all()

@pytest.mark.parametrize("c", [
    (float(2.5)),
    (int(2)),
    (np.uint16(2)),
    (np.float32(2.5)),
])
def test_pickle_constant_layer(c) -> None:
    layer = ConstantLayer(c)

    p = pickle.dumps(layer)
    restore = pickle.loads(p)

    result = restore.read_array(0, 0, 1, 1)
    assert (result == np.array([[c]])).all()

@pytest.mark.parametrize("c", [
    (float(2.5)),
    (int(2)),
    (np.uint16(2)),
    (np.float32(2.5)),
])
def test_pickle_simple_calc(c) -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        path = os.path.join(tempdir, "test.tif")
        area = Area(-10, 10, 10, -10)
        layer = RasterLayer(gdal_dataset_of_region(area, 0.2, filename=path))

        calc = layer * c
        assert calc.sum() != 0
        assert calc.sum() == layer.sum() * c

        p = pickle.dumps(calc)
        restore = pickle.loads(p)
        assert calc.sum() == restore.sum()

def test_pickle_lambda_calc() -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        path = os.path.join(tempdir, "test.tif")
        area = Area(-10, 10, 10, -10)
        layer = RasterLayer(gdal_dataset_of_region(area, 0.2, filename=path))

        calc = layer.numpy_apply(lambda x: x * 2.0)
        assert calc.sum() != 0
        assert calc.sum() == layer.sum() * 2

        p = pickle.dumps(calc)
        restore = pickle.loads(p)

        assert calc.sum() == restore.sum()

def test_pickle_func_calc() -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        path = os.path.join(tempdir, "test.tif")
        area = Area(-10, 10, 10, -10)
        layer = RasterLayer(gdal_dataset_of_region(area, 0.2, filename=path))

        def mulex(x):
            return x * 2.0

        calc = layer.numpy_apply(mulex)
        assert calc.sum() != 0
        assert calc.sum() == layer.sum() * 2

        p = pickle.dumps(calc)
        restore = pickle.loads(p)

        assert calc.sum() == restore.sum()

def test_pickle_rescaled_raster_layer() -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        path = os.path.join(tempdir, "test.tif")
        area = Area(-10, 10, 10, -10)
        raster = RasterLayer(gdal_dataset_of_region(area, 0.02, filename=path))
        layer = RescaledRasterLayer(raster, MapProjection(WGS_84_PROJECTION, 0.01, -0.01))

        p = pickle.dumps(layer)
        restore = pickle.loads(p)

        assert restore.area == area
        assert restore.pixel_scale == (0.01, -0.01)
        assert restore.geo_transform == (-10, 0.01, 0.0, 10, 0.0, -0.01)
        assert restore.window == Window(0, 0, 2000, 2000)

        expected = restore.read_array(0, 0, 100, 100)
        assert expected.sum() != 0 # just check there is meaningful data
