import numpy as np

from tests.helpers import gdal_dataset_of_region, gdal_dataset_with_data
from yirgacheffe import WGS_84_PROJECTION
from yirgacheffe.layers import RasterLayer, RescaledRasterLayer
from yirgacheffe.window import Area, MapProjection, Window


def test_simple_scale_down() -> None:
    area = Area(-10, 10, 10, -10)
    dataset = gdal_dataset_of_region(area, 0.02)
    with RasterLayer(dataset) as raster:
        target_projection = MapProjection(WGS_84_PROJECTION, 0.01, -0.01)
        with RescaledRasterLayer(raster, target_projection) as layer:
            assert layer.area == area
            assert layer.map_projection == target_projection
            assert layer.pixel_scale == target_projection.scale
            assert layer.geo_transform == (-10, 0.01, 0.0, 10, 0.0, -0.01)
            assert layer.window == Window(0, 0, 2000, 2000)

def test_simple_scale_up() -> None:
    area = Area(-10, 10, 10, -10)
    dataset = gdal_dataset_of_region(area, 0.02)
    with RasterLayer(dataset) as raster:
        target_projection = MapProjection(WGS_84_PROJECTION, 0.04, -0.04)
        with RescaledRasterLayer(raster, target_projection) as layer:
            assert layer.area == area
            assert layer.map_projection == target_projection
            assert layer.pixel_scale == target_projection.scale
            assert layer.geo_transform == (-10, 0.04, 0.0, 10, 0.0, -0.04)
            assert layer.window == Window(0, 0, 500, 500)

def test_scaling_up_pixels() -> None:
    # data has top left and bottom right quarters as 0
    # and the remaining quarters as 1
    data = np.zeros((4, 4))
    data[0:2,2:4] = 1
    data[2:4,0:2] = 1
    dataset = gdal_dataset_with_data((0, 0), 1.0, data)
    with RasterLayer(dataset) as raster:

        target_projection = MapProjection(WGS_84_PROJECTION, 0.5, -0.5)
        with RescaledRasterLayer(raster, target_projection) as layer:
            assert layer.area == Area(0, 0, 4, -4)
            assert layer.map_projection == target_projection
            assert layer.pixel_scale == target_projection.scale
            assert layer.geo_transform == (0.0, 0.5, 0.0, 0.0, 0.0, -0.5)
            assert layer.window == Window(0, 0, 8, 8)

            actual_raster = layer.read_array(0, 0, 8, 8)
            expected_raster = np.zeros((8, 8))
            expected_raster[0:4,4:8] = 1
            expected_raster[4:8,0:4] = 1
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
                actual_raster = layer.read_array(0, box,  box, 8 - box)
                exepected_sub_raster = expected_raster[box:8, 0:box]
                assert (exepected_sub_raster == actual_raster).all()

                # columns
                actual_raster = layer.read_array(box, 0, 1, 8)
                exepected_sub_raster = expected_raster[0:8, box:box+1]
                assert (exepected_sub_raster == actual_raster).all()

                # rows
                actual_raster = layer.read_array(0, box, 8, 1)
                exepected_sub_raster = expected_raster[box:box+1, 0:8]
                assert (exepected_sub_raster == actual_raster).all()

def test_scaling_down_pixels() -> None:
    # data has top left and bottom right quarters as 0
    # and the remaining quarters as 1
    data = np.zeros((8, 8))
    data[0:4,4:8] = 1
    data[4:8,0:4] = 1
    dataset = gdal_dataset_with_data((0, 0), 1.0, data)
    with RasterLayer(dataset) as raster:

        target_projection = MapProjection(WGS_84_PROJECTION, 2.0, -2.0)
        with RescaledRasterLayer(raster, target_projection) as layer:
            assert layer.area == Area(0, 0, 8, -8)
            assert layer.map_projection == target_projection
            assert layer.pixel_scale == target_projection.scale
            assert layer.geo_transform == (0.0, 2.0, 0.0, 0.0, 0.0, -2.0)
            assert layer.window == Window(0, 0, 4, 4)

            actual_raster = layer.read_array(0, 0, 4, 4)
            expected_raster = np.zeros((4, 4))
            expected_raster[0:2,2:4] = 1
            expected_raster[2:4,0:2] = 1
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
                actual_raster = layer.read_array(0, box,  box, 4 - box)
                exepected_sub_raster = expected_raster[box:4, 0:box]
                assert (exepected_sub_raster == actual_raster).all()

                # columns
                actual_raster = layer.read_array(box, 0, 1, 4)
                exepected_sub_raster = expected_raster[0:4, box:box+1]
                assert (exepected_sub_raster == actual_raster).all()

                # rows
                actual_raster = layer.read_array(0, box, 4, 1)
                exepected_sub_raster = expected_raster[box:box+1, 0:4]
                assert (exepected_sub_raster == actual_raster).all()

def test_rescaled_up_in_operation() -> None:
    data1 = np.zeros((8, 8))
    data1[0:4,4:8] = 1
    data1[4:8,0:4] = 1
    dataset1 = gdal_dataset_with_data((0, 0), 1.0, data1)
    raster1 = RasterLayer(dataset1)

    data2 = np.zeros((4, 4))
    data2[0:2,0:2] = 1
    data2[2:4,2:4] = 1
    dataset2 = gdal_dataset_with_data((0, 0), 2.0, data2)
    raster2 = RasterLayer(dataset2)

    rescaled = RescaledRasterLayer(raster2, raster1.map_projection)

    assert raster1.window == rescaled.window
    assert raster1.area == rescaled.area

    calc = raster1 + rescaled
    calc.ystep = 1
    assert calc.sum() == (8 * 8)

def test_rescaled_down_in_operation() -> None:
    data1 = np.zeros((8, 8))
    data1[0:4,4:8] = 1
    data1[4:8,0:4] = 1
    dataset1 = gdal_dataset_with_data((0, 0), 1.0, data1)
    raster1 = RasterLayer(dataset1)

    data2 = np.zeros((4, 4))
    data2[0:2,0:2] = 1
    data2[2:4,2:4] = 1
    dataset2 = gdal_dataset_with_data((0, 0), 2.0, data2)
    raster2 = RasterLayer(dataset2)

    rescaled = RescaledRasterLayer(raster1, raster2.map_projection)

    assert raster2.window == rescaled.window
    assert raster2.area == rescaled.area

    calc = rescaled + raster2
    calc.ystep = 1
    assert calc.sum() == (4 * 4)

def test_rescaled_up_with_window_set() -> None:
    # data has top left and bottom right quarters as 0
    # and the remaining quarters as 1
    data = np.zeros((4, 4))
    data[0:2,2:4] = 1
    data[2:4,0:2] = 1
    dataset = gdal_dataset_with_data((0, 0), 1.0, data)
    with RasterLayer(dataset) as raster:

        target_projection = MapProjection(WGS_84_PROJECTION, 0.5, -0.5)
        with RescaledRasterLayer(raster, target_projection) as layer:
            assert layer.area == Area(0.0, 0.0, 4.0, -4.0)

            layer.set_window_for_intersection(Area(1.0, -1.0, 3.0, -3.0))
            assert layer.area == Area(1.0, -1.0, 3.0, -3.0)
            assert layer.window == Window(2, 2, 4, 4)

            actual_raster = layer.read_array(0, 0, 4, 4)
            expected_raster = np.zeros((4, 4))
            expected_raster[0:2,2:4] = 1
            expected_raster[2:4,0:2] = 1

            assert (actual_raster == expected_raster).all()

def test_rescaled_up_with_window_set_2() -> None:
    # data has top left and bottom right quarters as 0
    # and the remaining quarters as 1
    data = np.zeros((4, 4))
    data[0:2,2:4] = 1
    data[2:4,0:2] = 1
    dataset = gdal_dataset_with_data((0, 0), 1.0, data)
    with RasterLayer(dataset) as raster:

        target_projection = MapProjection(WGS_84_PROJECTION, 0.5, -0.5)
        with RescaledRasterLayer(raster, target_projection) as layer:
            assert layer.area == Area(0.0, 0.0, 4.0, -4.0)

            expected_raster = np.zeros((6, 6))
            expected_raster[0:3,3:6] = 1
            expected_raster[3:6,0:3] = 1

            # Try get the intended data without the window offset first
            actual_raster = layer.read_array(1, 1, 6, 6)
            assert (actual_raster == expected_raster).all()

            layer.set_window_for_intersection(Area(0.5, -0.5, 3.5, -3.5))
            assert layer.area == Area(0.5, -0.5, 3.5, -3.5)
            assert layer.window == Window(1, 1, 6, 6)

            actual_raster = layer.read_array(0, 0, 6, 6)
            assert (actual_raster == expected_raster).all()

def test_rescaled_down_with_window_set() -> None:
    # data has top left and bottom right quarters as 0
    # and the remaining quarters as 1
    data = np.zeros((8, 8))
    data[0:4,4:8] = 1
    data[4:8,0:4] = 1
    dataset = gdal_dataset_with_data((0, 0), 1.0, data)
    with RasterLayer(dataset) as raster:

        target_projection = MapProjection(WGS_84_PROJECTION, 2.0, -2.0)
        with RescaledRasterLayer(raster, target_projection) as layer:
            assert layer.area == Area(0.0, 0.0, 8.0, -8.0)

            layer.set_window_for_intersection(Area(2.0, -2.0, 6.0, -6.0))
            assert layer.area == Area(2.0, -2.0, 6.0, -6.0)
            assert layer.window == Window(1, 1, 2, 2)

            actual_raster = layer.read_array(0, 0, 2, 2)
            expected_raster = np.zeros((2, 2))
            expected_raster[0:1,1:2] = 1
            expected_raster[1:2,0:1] = 1

            assert (actual_raster == expected_raster).all()
