import os
import tempfile
from math import ceil, floor
from pathlib import Path

import numpy as np
import pytest

import yirgacheffe as yg
from yirgacheffe import WGS_84_PROJECTION
from yirgacheffe.layers import InvalidRasterBand, RasterLayer
from yirgacheffe.window import Area, MapProjection, Window
from yirgacheffe.operators import DataType
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

def test_open_multiband_raster_wrong_band() -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        path = Path(tempdir) / "test.tif"
        data1 = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        data2 = np.array([[10.0, 20.0, 30.0, 40.0], [50.0, 60.0, 70.0, 80.0]])

        datas = [data1, data2]
        dataset = gdal_multiband_dataset_with_data((0.0, 0.0), 0.02, datas, filename=path)
        dataset.Close()

        with pytest.raises(InvalidRasterBand):
            _ = yg.read_raster(path, 3)

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
        _ = yg.read_shape("this_file_does_not_exist.gpkg", (WGS_84_PROJECTION, (1.0, -1.0)))

def test_open_gpkg() -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        path = os.path.join(tempdir, "test.gpkg")
        area = Area(-10.0, 10.0, 10.0, 0.0)
        make_vectors_with_id(42, {area}, path)

        with yg.read_shape(path, (WGS_84_PROJECTION, (1.0, -1.0))) as layer:
            assert layer.area == area
            assert layer.geo_transform == (area.left, 1.0, 0.0, area.top, 0.0, -1.0)
            assert layer.window == Window(0, 0, 20, 10)
            assert layer.map_projection == MapProjection(WGS_84_PROJECTION, 1.0, -1.0)
            assert layer.projection == WGS_84_PROJECTION

def test_open_gpkg_with_mapprojection() -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        path = os.path.join(tempdir, "test.gpkg")
        area = Area(-10.0, 10.0, 10.0, 0.0)
        make_vectors_with_id(42, {area}, path)

        with yg.read_shape(path, MapProjection(WGS_84_PROJECTION, 1.0, -1.0)) as layer:
            assert layer.area == area
            assert layer.geo_transform == (area.left, 1.0, 0.0, area.top, 0.0, -1.0)
            assert layer.window == Window(0, 0, 20, 10)
            assert layer.map_projection == MapProjection(WGS_84_PROJECTION, 1.0, -1.0)
            assert layer.projection == WGS_84_PROJECTION

def test_open_gpkg_with_no_projection() -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        path = os.path.join(tempdir, "test.gpkg")
        area = Area(-10.0, 10.0, 10.0, 0.0)
        make_vectors_with_id(42, {area}, path)

        with yg.read_shape(path) as layer:
            assert layer.area == area
            assert layer.projection is None
            with pytest.raises(AttributeError):
                _ = layer.geo_transform
            with pytest.raises(AttributeError):
                _ = layer.window

def test_open_gpkg_direct_scale() -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        path = Path(tempdir) / "test.gpkg"
        area = Area(-10.0, 10.0, 10.0, 0.0)
        make_vectors_with_id(42, {area}, path)

        with yg.read_shape(path, (WGS_84_PROJECTION, (1.0, -1.0))) as layer:
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

        with yg.read_shape(path, (WGS_84_PROJECTION, (1.0, -1.0)), "id_no=42") as layer:
            assert layer.area == Area(-10.0, 10.0, 0.0, 0.0)
            assert layer.geo_transform == (-10.0, 1.0, 0.0, 10.0, 0.0, -1.0)
            assert layer.window == Window(0, 0, 10, 10)

            # Because we picked one later, all pixels should be burned
            total = layer.sum()
            assert total == (layer.window.xsize * layer.window.ysize)

def test_open_shape_like() -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        path = Path(tempdir) / "test.tif"
        area = Area(-10, 10, 10, -10)
        dataset = gdal_dataset_of_region(area, 1.0, filename=path)
        dataset.Close()
        assert os.path.exists(path)

        with yg.read_raster(path) as raster_layer:
            path = os.path.join(tempdir, "test.gpkg")
            area = Area(-10.0, 10.0, 10.0, 0.0)
            make_vectors_with_id(42, {area}, path)

            with yg.read_shape_like(path, raster_layer) as layer:
                assert layer.area == area
                assert layer.geo_transform == (area.left, 1.0, 0.0, area.top, 0.0, -1.0)
                assert layer.window == Window(0, 0, 20, 10)
                assert layer.projection == raster_layer.projection

@pytest.mark.parametrize("tiled", [False, True])
def test_empty_rasters_list(tiled):
    with pytest.raises(ValueError):
        _ = yg.read_rasters([], tiled=tiled)

@pytest.mark.parametrize("tiled", [False, True])
def test_open_two_raster_areas_side_by_side(tiled):
    with tempfile.TemporaryDirectory() as tempdir:
        path1 = Path(tempdir) / "test1.tif"
        area1 = Area(-10, 10, 10, -10)
        dataset1 = gdal_dataset_of_region(area1, 0.2, filename=path1)
        dataset1.Close()

        path2 = Path(tempdir) / "test2.tif"
        area2 = Area(10, 10, 30, -10)
        dataset2 = gdal_dataset_of_region(area2, 0.2, filename=path2)
        dataset2.Close()

        with yg.read_rasters([path1, path2], tiled=tiled) as group:
            assert group.area == Area(-10, 10, 30, -10)
            assert group.window == Window(0, 0, 200, 100)

            with yg.read_raster(path1) as raster1:
                with yg.read_raster(path2) as raster2:
                    assert group.sum() == raster1.sum() + raster2.sum()

@pytest.mark.parametrize("tiled", [False, True])
def test_open_two_raster_by_glob(tiled):
    with tempfile.TemporaryDirectory() as tempdir:
        temppath = Path(tempdir)
        path1 = temppath / "test1.tif"
        area1 = Area(-10, 10, 10, -10)
        dataset1 = gdal_dataset_of_region(area1, 0.2, filename=path1)
        dataset1.Close()

        path2 = temppath / "test2.tif"
        area2 = Area(10, 10, 30, -10)
        dataset2 = gdal_dataset_of_region(area2, 0.2, filename=path2)
        dataset2.Close()

        with yg.read_rasters(temppath.glob("*.tif"), tiled=tiled) as group:
            assert group.area == Area(-10, 10, 30, -10)
            assert group.window == Window(0, 0, 200, 100)

            with yg.read_raster(path1) as raster1:
                with yg.read_raster(path2) as raster2:
                    assert group.sum() == raster1.sum() + raster2.sum()

def test_open_uniform_area_layer() -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        path = os.path.join(tempdir, "test.tif")
        pixel_scale = 0.5
        area = Area(
            floor(-180 / pixel_scale) * pixel_scale,
            ceil(90 / pixel_scale) * pixel_scale,
            (floor(-180 / pixel_scale) * pixel_scale) + pixel_scale,
            floor(-90 / pixel_scale) * pixel_scale
        )
        dataset = gdal_dataset_of_region(area, pixel_scale, filename=path)
        assert dataset.RasterXSize == 1
        assert dataset.RasterYSize == ceil(180 / pixel_scale)
        dataset.Close()

        with yg.read_narrow_raster(path) as layer:
            assert layer.map_projection is not None
            assert layer.map_projection.scale == (pixel_scale, -pixel_scale)
            assert layer.area == Area(
                floor(-180 / pixel_scale) * pixel_scale,
                ceil(90 / pixel_scale) * pixel_scale,
                ceil(180 / pixel_scale) * pixel_scale,
                floor(-90 / pixel_scale) * pixel_scale
            )
            assert layer.window == Window(
                0,
                0,
                ceil((layer.area.right - layer.area.left) / pixel_scale),
                ceil((layer.area.top - layer.area.bottom) / pixel_scale)
            )

def test_incorrect_tiff_for_uniform_area() -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        path = Path(tempdir) / "test.tif"
        area = Area(-10, 10, 10, -10)
        gdal_dataset_of_region(area, 1.0, filename=path)
        assert path.exists()
        with pytest.raises(ValueError):
            _ = yg.read_narrow_raster(path)

def test_constant() -> None:
    with yg.constant(42.0) as layer:
        area = Area(left=-1.0, right=1.0, top=1.0, bottom=-1.0)
        projection = MapProjection(WGS_84_PROJECTION, 0.1, -0.1)
        with RasterLayer.empty_raster_layer(area, projection.scale, DataType.Float32) as result:
            layer.save(result)

            expected = np.full((20, 20), 42.0)
            actual = result.read_array(0, 0, 20, 20)
            assert (expected == actual).all()
