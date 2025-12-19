import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
from osgeo import gdal
from pyproj import Transformer

import yirgacheffe as yg
from tests.unit.helpers import gdal_dataset_of_region, gdal_dataset_with_data
from yirgacheffe import WGS_84_PROJECTION
from yirgacheffe.layers import RasterLayer, ReprojectedRasterLayer
from yirgacheffe.window import Area, MapProjection, Window


def test_simple_scale_down() -> None:
    area = Area(-10, 10, 10, -10)
    dataset = gdal_dataset_of_region(area, 0.02)
    with RasterLayer(dataset) as raster:
        target_projection = MapProjection(WGS_84_PROJECTION, 0.01, -0.01)
        with ReprojectedRasterLayer(raster, target_projection) as layer:
            assert layer.area == Area(-10, 10, 10, -10, target_projection)
            assert layer.map_projection == target_projection
            assert layer.pixel_scale == target_projection.scale
            assert layer.geo_transform == (-10, 0.01, 0.0, 10, 0.0, -0.01)
            assert layer.window == Window(0, 0, 2000, 2000)


def test_simple_scale_up() -> None:
    area = Area(-10, 10, 10, -10)
    dataset = gdal_dataset_of_region(area, 0.02)
    with RasterLayer(dataset) as raster:
        target_projection = MapProjection(WGS_84_PROJECTION, 0.04, -0.04)
        with ReprojectedRasterLayer(raster, target_projection) as layer:
            assert layer.area == Area(-10, 10, 10, -10, target_projection)
            assert layer.map_projection == target_projection
            assert layer.pixel_scale == target_projection.scale
            assert layer.geo_transform == (-10, 0.04, 0.0, 10, 0.0, -0.04)
            assert layer.window == Window(0, 0, 500, 500)


def test_scaling_up_pixels() -> None:
    # data has top left and bottom right quarters as 0
    # and the remaining quarters as 1
    data = np.zeros((4, 4))
    data[0:2, 2:4] = 1
    data[2:4, 0:2] = 1
    dataset = gdal_dataset_with_data((0, 0), 1.0, data)
    with RasterLayer(dataset) as raster:

        target_projection = MapProjection(WGS_84_PROJECTION, 0.5, -0.5)
        with ReprojectedRasterLayer(raster, target_projection) as layer:
            assert layer.area == Area(0, 0, 4, -4, target_projection)
            assert layer.map_projection == target_projection
            assert layer.pixel_scale == target_projection.scale
            assert layer.geo_transform == (0.0, 0.5, 0.0, 0.0, 0.0, -0.5)
            assert layer.window == Window(0, 0, 8, 8)

            actual_raster = layer.read_array(0, 0, 8, 8)
            expected_raster = np.zeros((8, 8))
            expected_raster[0:4, 4:8] = 1
            expected_raster[4:8, 0:4] = 1
            assert (expected_raster == actual_raster).all()

            # Try getting just the quads
            expected_quad_raster = np.zeros((4, 4))
            actual_raster = layer.read_array(0, 0, 4, 4)
            assert (expected_quad_raster == actual_raster).all()
            actual_raster = layer.read_array(4, 4, 4, 4)
            assert (expected_quad_raster == actual_raster).all()

            expected_quad_raster = np.ones((4, 4))
            actual_raster = layer.read_array(0, 4, 4, 4)
            assert (expected_quad_raster == actual_raster).all()
            actual_raster = layer.read_array(4, 0, 4, 4)
            assert (expected_quad_raster == actual_raster).all()

            for box in range(1, 7):
                # anchored top left
                actual_raster = layer.read_array(0, 0, box, box)
                exepected_sub_raster = expected_raster[0:box, 0:box]
                assert (exepected_sub_raster == actual_raster).all()

                # anchored bottom right
                actual_raster = layer.read_array(box, box, 8 - box, 8 - box)
                exepected_sub_raster = expected_raster[box:8, box:8]
                assert (exepected_sub_raster == actual_raster).all()

                # anchored top right
                actual_raster = layer.read_array(box, 0, 8 - box, box)
                exepected_sub_raster = expected_raster[0:box, box:8]
                assert (exepected_sub_raster == actual_raster).all()

                # anchored bottom left
                actual_raster = layer.read_array(0, box, box, 8 - box)
                exepected_sub_raster = expected_raster[box:8, 0:box]
                assert (exepected_sub_raster == actual_raster).all()

                # columns
                actual_raster = layer.read_array(box, 0, 1, 8)
                exepected_sub_raster = expected_raster[0:8, box : box + 1]
                assert (exepected_sub_raster == actual_raster).all()

                # rows
                actual_raster = layer.read_array(0, box, 8, 1)
                exepected_sub_raster = expected_raster[box : box + 1, 0:8]
                assert (exepected_sub_raster == actual_raster).all()


def test_scaling_down_pixels() -> None:
    # data has top left and bottom right quarters as 0
    # and the remaining quarters as 1
    data = np.zeros((8, 8))
    data[0:4, 4:8] = 1
    data[4:8, 0:4] = 1
    dataset = gdal_dataset_with_data((0, 0), 1.0, data)
    with RasterLayer(dataset) as raster:

        target_projection = MapProjection(WGS_84_PROJECTION, 2.0, -2.0)
        with ReprojectedRasterLayer(raster, target_projection) as layer:
            assert layer.area == Area(0, 0, 8, -8, target_projection)
            assert layer.map_projection == target_projection
            assert layer.pixel_scale == target_projection.scale
            assert layer.geo_transform == (0.0, 2.0, 0.0, 0.0, 0.0, -2.0)
            assert layer.window == Window(0, 0, 4, 4)

            actual_raster = layer.read_array(0, 0, 4, 4)
            expected_raster = np.zeros((4, 4))
            expected_raster[0:2, 2:4] = 1
            expected_raster[2:4, 0:2] = 1
            assert (expected_raster == actual_raster).all()

            # Try getting just the quads
            expected_quad_raster = np.zeros((2, 2))
            actual_raster = layer.read_array(0, 0, 2, 2)
            assert (expected_quad_raster == actual_raster).all()
            actual_raster = layer.read_array(2, 2, 2, 2)
            assert (expected_quad_raster == actual_raster).all()

            expected_quad_raster = np.ones((2, 2))
            actual_raster = layer.read_array(0, 2, 2, 2)
            assert (expected_quad_raster == actual_raster).all()
            actual_raster = layer.read_array(2, 0, 2, 2)
            assert (expected_quad_raster == actual_raster).all()

            for box in range(1, 3):
                # anchored top left
                actual_raster = layer.read_array(0, 0, box, box)
                exepected_sub_raster = expected_raster[0:box, 0:box]
                assert (exepected_sub_raster == actual_raster).all()

                # anchored bottom right
                actual_raster = layer.read_array(box, box, 4 - box, 4 - box)
                exepected_sub_raster = expected_raster[box:4, box:4]
                assert (exepected_sub_raster == actual_raster).all()

                # anchored top right
                actual_raster = layer.read_array(box, 0, 4 - box, box)
                exepected_sub_raster = expected_raster[0:box, box:4]
                assert (exepected_sub_raster == actual_raster).all()

                # anchored bottom left
                actual_raster = layer.read_array(0, box, box, 4 - box)
                exepected_sub_raster = expected_raster[box:4, 0:box]
                assert (exepected_sub_raster == actual_raster).all()

                # columns
                actual_raster = layer.read_array(box, 0, 1, 4)
                exepected_sub_raster = expected_raster[0:4, box : box + 1]
                assert (exepected_sub_raster == actual_raster).all()

                # rows
                actual_raster = layer.read_array(0, box, 4, 1)
                exepected_sub_raster = expected_raster[box : box + 1, 0:4]
                assert (exepected_sub_raster == actual_raster).all()


def test_reprojected_up_in_operation() -> None:
    data1 = np.zeros((8, 8))
    data1[0:4, 4:8] = 1
    data1[4:8, 0:4] = 1
    dataset1 = gdal_dataset_with_data((0, 0), 1.0, data1)
    raster1 = RasterLayer(dataset1)
    assert raster1.map_projection

    data2 = np.zeros((4, 4))
    data2[0:2, 0:2] = 1
    data2[2:4, 2:4] = 1
    dataset2 = gdal_dataset_with_data((0, 0), 2.0, data2)
    raster2 = RasterLayer(dataset2)

    rescaled = ReprojectedRasterLayer(raster2, raster1.map_projection)

    assert raster1.window == rescaled.window
    assert raster1.area == rescaled.area

    calc = raster1 + rescaled
    calc.ystep = 1
    assert calc.sum() == (8 * 8)


def test_reprojected_down_in_operation() -> None:
    data1 = np.zeros((8, 8))
    data1[0:4, 4:8] = 1
    data1[4:8, 0:4] = 1
    dataset1 = gdal_dataset_with_data((0, 0), 1.0, data1)
    raster1 = RasterLayer(dataset1)

    data2 = np.zeros((4, 4))
    data2[0:2, 0:2] = 1
    data2[2:4, 2:4] = 1
    dataset2 = gdal_dataset_with_data((0, 0), 2.0, data2)
    raster2 = RasterLayer(dataset2)
    assert raster2.map_projection

    rescaled = ReprojectedRasterLayer(raster1, raster2.map_projection)

    assert raster2.window == rescaled.window
    assert raster2.area == rescaled.area

    calc = rescaled + raster2
    calc.ystep = 1
    assert calc.sum() == (4 * 4)


def test_reprojected_up_with_window_set() -> None:
    # data has top left and bottom right quarters as 0
    # and the remaining quarters as 1
    data = np.zeros((4, 4))
    data[0:2, 2:4] = 1
    data[2:4, 0:2] = 1
    dataset = gdal_dataset_with_data((0, 0), 1.0, data)
    with RasterLayer(dataset) as raster:

        target_projection = MapProjection(WGS_84_PROJECTION, 0.5, -0.5)
        with ReprojectedRasterLayer(raster, target_projection) as layer:
            assert layer.area == Area(0.0, 0.0, 4.0, -4.0, target_projection)
            assert layer.window == Window(0, 0, 8, 8)

            layer.set_window_for_intersection(Area(1.0, -1.0, 3.0, -3.0))
            assert layer.area == Area(1.0, -1.0, 3.0, -3.0, target_projection)
            assert layer.window == Window(2, 2, 4, 4)

            actual_raster = layer.read_array(0, 0, 4, 4)
            expected_raster = np.zeros((4, 4))
            expected_raster[0:2, 2:4] = 1
            expected_raster[2:4, 0:2] = 1

            assert (actual_raster == expected_raster).all()


def test_reprojected_up_with_window_set_2() -> None:
    # data has top left and bottom right quarters as 0
    # and the remaining quarters as 1
    data = np.zeros((4, 4))
    data[0:2, 2:4] = 1
    data[2:4, 0:2] = 1
    dataset = gdal_dataset_with_data((0, 0), 1.0, data)
    with RasterLayer(dataset) as raster:

        target_projection = MapProjection(WGS_84_PROJECTION, 0.5, -0.5)
        with ReprojectedRasterLayer(raster, target_projection) as layer:
            assert layer.area == Area(0.0, 0.0, 4.0, -4.0, target_projection)

            expected_raster = np.zeros((6, 6))
            expected_raster[0:3, 3:6] = 1
            expected_raster[3:6, 0:3] = 1

            # Try get the intended data without the window offset first
            actual_raster = layer.read_array(1, 1, 6, 6)
            assert (actual_raster == expected_raster).all()

            layer.set_window_for_intersection(Area(0.5, -0.5, 3.5, -3.5))
            assert layer.area == Area(0.5, -0.5, 3.5, -3.5, target_projection)
            assert layer.window == Window(1, 1, 6, 6)

            actual_raster = layer.read_array(0, 0, 6, 6)
            assert (actual_raster == expected_raster).all()


def test_reprojected_down_with_window_set() -> None:
    # data has top left and bottom right quarters as 0
    # and the remaining quarters as 1
    data = np.zeros((8, 8))
    data[0:4, 4:8] = 1
    data[4:8, 0:4] = 1
    dataset = gdal_dataset_with_data((0, 0), 1.0, data)
    with RasterLayer(dataset) as raster:

        target_projection = MapProjection(WGS_84_PROJECTION, 2.0, -2.0)
        with ReprojectedRasterLayer(raster, target_projection) as layer:
            assert layer.area == Area(0.0, 0.0, 8.0, -8.0, target_projection)

            layer.set_window_for_intersection(Area(2.0, -2.0, 6.0, -6.0))
            assert layer.area == Area(2.0, -2.0, 6.0, -6.0, target_projection)
            assert layer.window == Window(1, 1, 2, 2)

            actual_raster = layer.read_array(0, 0, 2, 2)
            expected_raster = np.zeros((2, 2))
            expected_raster[0:1, 1:2] = 1
            expected_raster[1:2, 0:1] = 1

            assert (actual_raster == expected_raster).all()

def test_somewhat_aligned_rastered_polygons() -> None:
    # This is a test for the ReprojectedRasterLayer - other tests will cover ReprojectedVectorLayer

    points_4326 = [
        (0.0, 0.0),
        (10.0, 0.0),
        (10.0, 10.0),
        (0.0, 10.0),
    ]

    def _generate_geojson(projection: str, points: list[tuple[float,float]]) -> dict:
        return {
            "type": "FeatureCollection",
            "crs": {
                "type": "name",
                "properties": {"name": projection}
            },
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[
                            list(pt) for pt in points
                        ] + [list(points[0])]]  # Close the polygon
                    },
                    "properties": {"value": 1}
                }
            ]
        }

    geojson_4326 = _generate_geojson("EPSG:4326", points_4326)

    transformer = Transformer.from_crs("EPSG:4326", "ESRI:54009", always_xy=True)
    geojson_54009 = _generate_geojson(
        "ESRI:54009",
        [transformer.transform(*pt) for pt in points_4326],
    )

    with tempfile.TemporaryDirectory() as tmpdirstr:
        tmpdir = Path(tmpdirstr)

        geojson_4326_path = tmpdir / "points_4326.geojson"
        with open(geojson_4326_path, 'w', encoding="utf-8") as f:
            json.dump(geojson_4326, f)

        geojson_54009_path = tmpdir / "points_54009.geojson"
        with open(geojson_54009_path, 'w', encoding="utf-8") as f:
            json.dump(geojson_54009, f)

        raster_4326_path = tmpdir / "raster_4326.tif"
        raster_54009_path = tmpdir / "raster_54009.tif"

        projection_4326 = MapProjection("EPSG:4326", 0.1, -0.1)
        projection_54009 = MapProjection("ESRI:54009", 10000, -10000)
        with yg.read_shape(geojson_4326_path, projection_4326) as vector_4326:
            vector_4326.to_geotiff(raster_4326_path)
        with yg.read_shape(geojson_54009_path, projection_54009) as vector_54009:
            vector_54009.to_geotiff(raster_54009_path)

        with (
            yg.read_raster(raster_4326_path) as raster_4326,
            yg.read_raster(raster_54009_path) as raster_54009,
        ):
            assert raster_4326.map_projection == projection_4326
            assert raster_54009.map_projection == projection_54009

            with (
                ReprojectedRasterLayer(raster_54009, projection_4326) as reprojected_54009_to_4326,
                ReprojectedRasterLayer(raster_4326, projection_54009) as reprojected_4326_to_54009,
            ):
                # We do not expect perfect reproduction, so let's sum them pixels and see if they are
                # within a few percent:
                diff_4326 = abs(raster_4326.sum() - reprojected_54009_to_4326.sum()) / raster_4326.sum()
                assert diff_4326 < 0.02
                diff_54009 = abs(raster_54009.sum() - reprojected_4326_to_54009.sum()) / raster_54009.sum()
                assert diff_54009 < 0.02

@pytest.mark.parametrize("blocksize", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("src_projection,dst_projection", [
    (
        MapProjection("EPSG:4326", 0.1, -0.1),
        MapProjection("ESRI:54009", 10000, -10000),
    ),
    (
        MapProjection("ESRI:54009", 10000, -10000),
        MapProjection("EPSG:4326", 0.1, -0.1),
    ),
])
def test_vs_gdal_warp(monkeypatch, blocksize, src_projection, dst_projection) -> None:
    # This test is mostly to just check we've not done anything odd with chunking
    data = np.zeros((8, 8))
    data[0:4, 4:8] = 1
    data[4:8, 0:4] = 1

    with monkeypatch.context() as m:
        m.setattr(yg.constants, "YSTEP", blocksize)
        with tempfile.TemporaryDirectory() as tmpdirstr:
            tmpdir = Path("/tmp")

            with yg.from_array(data, (0, 0), src_projection) as original:
                og_raster_path = tmpdir /  "original.tif"
                warped_raster_path = tmpdir / "warped.tif"

                original.to_geotiff(og_raster_path)

                gdal.Warp(
                    warped_raster_path,
                    og_raster_path,
                    options=gdal.WarpOptions(
                        dstSRS=dst_projection._gdal_projection,
                        outputType=original.datatype.to_gdal(),
                        xRes=dst_projection.xstep,
                        yRes=dst_projection.ystep,
                        resampleAlg="nearest",
                        targetAlignedPixels=True,
                    )
                )

                with (
                    yg.read_raster(warped_raster_path) as warped,
                    ReprojectedRasterLayer(original, dst_projection) as reprojected,
                ):
                    reprojected.to_geotiff(tmpdir / "reprojected.tif")
                    assert reprojected.map_projection == dst_projection
                    assert warped.map_projection == dst_projection
                    assert warped.area == reprojected.area
                    assert warped.window == reprojected.window
                    warped_data = warped.read_array(0, 0, warped.window.xsize, warped.window.ysize)
                    reprojected_data = reprojected.read_array(0, 0, reprojected.window.xsize, reprojected.window.ysize)
                    assert (warped_data == reprojected_data).all()
