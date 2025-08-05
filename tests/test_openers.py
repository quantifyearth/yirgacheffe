import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

import yirgacheffe.core as yg
from yirgacheffe import WGS_84_PROJECTION
from yirgacheffe.window import Area, PixelScale, Window
from tests.helpers import gdal_dataset_of_region, gdal_multiband_dataset_with_data, \
    make_vectors_with_id, make_vectors_with_mutlile_ids

def test_raster_from_nonexistent_file() -> None:
    with pytest.raises(FileNotFoundError):
        _ = yg.read_raster("this_file_does_not_exist.tif")

def test_open_raster_file() -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        path = os.path.join(tempdir, "test.tif")
        area = Area(-10, 10, 10, -10)
        dataset = gdal_dataset_of_region(area, 0.02, filename=path)
        dataset.Close()
        assert os.path.exists(path)

        with yg.read_raster(path) as layer:
            assert layer.area == area
            assert layer.pixel_scale == (0.02, -0.02)
            assert layer.geo_transform == (-10, 0.02, 0.0, 10, 0.0, -0.02)
            assert layer.window == Window(0, 0, 1000, 1000)

def test_open_raster_file_as_path() -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        path = Path(tempdir) / "test.tif"
        area = Area(-10, 10, 10, -10)
        dataset = gdal_dataset_of_region(area, 0.02, filename=path)
        dataset.Close()
        assert path.exists

        with yg.read_raster(path) as layer:
            assert layer.area == area
            assert layer.pixel_scale == (0.02, -0.02)
            assert layer.geo_transform == (-10, 0.02, 0.0, 10, 0.0, -0.02)
            assert layer.window == Window(0, 0, 1000, 1000)

def test_open_multiband_raster() -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        path = Path(tempdir) / "test.tif"
        data1 = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        data2 = np.array([[10.0, 20.0, 30.0, 40.0], [50.0, 60.0, 70.0, 80.0]])

        datas = [data1, data2]
        dataset = gdal_multiband_dataset_with_data((0.0, 0.0), 0.02, datas, filename=path)
        dataset.Close()

        for i in range(2):
            with yg.read_raster(path, i + 1) as layer:
                data = datas[i]
                actual = layer.read_array(0, 0, 4, 2)
                assert (data == actual).all()

def test_shape_from_nonexistent_file() -> None:
    with pytest.raises(FileNotFoundError):
        _ = yg.read_shape("this_file_does_not_exist.gpkg", (1.0, -1.0), WGS_84_PROJECTION)

def test_open_gpkg() -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        path = os.path.join(tempdir, "test.gpkg")
        area = Area(-10.0, 10.0, 10.0, 0.0)
        make_vectors_with_id(42, {area}, path)

        with yg.read_shape(path, PixelScale(1.0, -1.0), WGS_84_PROJECTION) as layer:
            assert layer.area == area
            assert layer.geo_transform == (area.left, 1.0, 0.0, area.top, 0.0, -1.0)
            assert layer.window == Window(0, 0, 20, 10)
            assert layer.projection == WGS_84_PROJECTION

def test_open_gpkg_direct_scale() -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        path = Path(tempdir) / "test.gpkg"
        area = Area(-10.0, 10.0, 10.0, 0.0)
        make_vectors_with_id(42, {area}, path)

        with yg.read_shape(path, (1.0, -1.0), WGS_84_PROJECTION) as layer:
            assert layer.area == area
            assert layer.geo_transform == (area.left, 1.0, 0.0, area.top, 0.0, -1.0)
            assert layer.window == Window(0, 0, 20, 10)
            assert layer.projection == WGS_84_PROJECTION

def test_open_gpkg_with_filter() -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        path = Path(tempdir) / "test.gpkg"
        areas = {
            (Area(-10.0, 10.0, 0.0, 0.0), 42),
            (Area(0.0, 0.0, 10, -10), 43)
        }
        make_vectors_with_mutlile_ids(areas, path)

        with yg.read_shape(path, (1.0, -1.0), WGS_84_PROJECTION, "id_no=42") as layer:
            assert layer.area == Area(-10.0, 10.0, 0.0, 0.0)
            assert layer.geo_transform == (-10.0, 1.0, 0.0, 10.0, 0.0, -1.0)
            assert layer.window == Window(0, 0, 10, 10)

            # Because we picked one later, all pixels should be burned
            total = layer.sum()
            assert total == (layer.window.xsize * layer.window.ysize)
