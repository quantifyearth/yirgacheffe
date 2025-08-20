import os
import tempfile

import numpy as np
import pytest
from osgeo import gdal

from tests.helpers import gdal_dataset_of_region, gdal_multiband_dataset_with_data, gdal_dataset_with_data
from yirgacheffe.window import Area, PixelScale, Window
from yirgacheffe.layers import RasterLayer, InvalidRasterBand
from yirgacheffe.rounding import round_up_pixels
from yirgacheffe.operators import DataType


# There is a lot of "del" in this file, due to a combination of gdal having no way
# to explicitly close a file beyond forcing the gdal object's deletion, and Windows
# getting upset that it tries to clear up the TemporaryDirectory and there's an open
# file handle within that directory.

def test_make_basic_layer() -> None:
    area = Area(-10, 10, 10, -10)
    dataset = gdal_dataset_of_region(area, 0.02)

    # The context manager routines are just called on the base class, so ensure
    # they do call the GDAL close method, otherwise all that is for nothing
    original_ref = dataset.Close
    close_called = []
    def mocked_close():
        close_called.append(True)
        original_ref()
    dataset.Close = mocked_close

    with RasterLayer(dataset) as layer:
        assert layer.area == area
        assert layer.pixel_scale == (0.02, -0.02)
        assert layer.geo_transform == (-10, 0.02, 0.0, 10, 0.0, -0.02)
        assert layer.window == Window(0, 0, 1000, 1000)

    assert close_called

def test_layer_from_null() -> None:
    # Seems a petty test, but gdal doesn't throw exceptions
    # so you often get None datasets if you're not careful
    with pytest.raises(ValueError):
        with RasterLayer(None) as _layer:
            pass

def test_layer_from_nonexistent_file() -> None:
    with pytest.raises(FileNotFoundError):
        _ = RasterLayer.layer_from_file("this_file_does_not_exist.tif")

def test_open_file() -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        path = os.path.join(tempdir, "test.tif")
        area = Area(-10, 10, 10, -10)
        dataset = gdal_dataset_of_region(area, 0.02, filename=path)
        dataset.Close()
        assert os.path.exists(path)
        with RasterLayer.layer_from_file(path) as layer:
            assert layer.area == area
            assert layer.pixel_scale == (0.02, -0.02)
            assert layer.geo_transform == (-10, 0.02, 0.0, 10, 0.0, -0.02)
            assert layer.window == Window(0, 0, 1000, 1000)
            del layer

@pytest.mark.parametrize("initial_area",
    [
        Area(left=10.0, top=10.0, right=10.1, bottom=9.9),
        Area(left=-10.0, top=10.0, right=-9.9, bottom=9.9),
        Area(left=10.0, top=-10.0, right=10.1, bottom=-10.1),
        Area(left=-10.0, top=-10.0, right=-9.9, bottom=-10.1),
        Area(left=-0.1, top=0.1, right=0.1, bottom=-0.1),
    ]
)
def test_empty_layers_are_pixel_aligned(initial_area):
    scale = PixelScale(0.000898315284120,-0.000898315284120)

    expanded_area = initial_area.grow(0.1)

    initial_layer = RasterLayer.empty_raster_layer(initial_area, scale, gdal.GDT_Float64)

    pixel_width = (initial_layer.area.right - initial_layer.area.left) / scale.xstep
    assert round_up_pixels(pixel_width, scale.xstep) == initial_layer.window.xsize

    expanded_layer = RasterLayer.empty_raster_layer(expanded_area, scale, gdal.GDT_Float64)
    pixel_width = (expanded_layer.area.right - expanded_layer.area.left) / scale.xstep
    assert round_up_pixels(pixel_width, scale.xstep) == expanded_layer.window.xsize

def test_empty_layer_from_raster():
    source = RasterLayer(gdal_dataset_of_region(Area(-10, 10, 10, -10), 0.02))
    empty = RasterLayer.empty_raster_layer_like(source)
    assert empty.pixel_scale == source.pixel_scale
    assert empty.projection == source.projection
    assert empty.map_projection == source.map_projection
    assert empty.window == source.window
    assert empty.datatype == source.datatype
    assert empty.geo_transform == source.geo_transform
    assert empty._dataset.GetRasterBand(1).GetNoDataValue() is None

@pytest.mark.parametrize("nodata", [0, 0.0, 2, 2.0])
def test_empty_layer_from_raster_with_no_data_value(nodata):
    source = RasterLayer(gdal_dataset_of_region(Area(-10, 10, 10, -10), 0.02))
    empty = RasterLayer.empty_raster_layer_like(source, nodata=nodata)
    assert empty.pixel_scale == source.pixel_scale
    assert empty.projection == source.projection
    assert empty.map_projection == source.map_projection
    assert empty.window == source.window
    assert empty.datatype == source.datatype
    assert empty.geo_transform == source.geo_transform
    assert empty._dataset.GetRasterBand(1).GetNoDataValue() == nodata

def test_empty_layer_from_raster_with_new_smaller_area():
    source = RasterLayer(gdal_dataset_of_region(Area(-10, 10, 10, -10), 0.02))
    smaller_area = Area(-1, 1, 1, -1)
    empty = RasterLayer.empty_raster_layer_like(source, area=smaller_area)
    assert empty.pixel_scale == source.pixel_scale
    assert empty.projection == source.projection
    assert empty.map_projection == source.map_projection
    assert empty.window == Window(0, 0, 100, 100)
    assert empty.datatype == source.datatype
    assert empty.geo_transform == (-1.0, 0.02, 0.0, 1.0, 0.0, -0.02)

def test_empty_layer_from_raster_new_datatype():
    source = RasterLayer(gdal_dataset_of_region(Area(-10, 10, 10, -10), 0.02))
    assert source.datatype == DataType.Byte
    empty = RasterLayer.empty_raster_layer_like(source, datatype=gdal.GDT_Float64)
    assert empty.pixel_scale == source.pixel_scale
    assert empty.projection == source.projection
    assert empty.map_projection == source.map_projection
    assert empty.window == source.window
    assert empty.datatype == DataType.Float64

def test_empty_layer_from_raster_with_window():
    source = RasterLayer(gdal_dataset_of_region(Area(-10, 10, 10, -10), 0.02))
    original_window = source.window

    source.set_window_for_intersection(Area(-1, 1, 1, -1))
    assert source.window < original_window

    empty = RasterLayer.empty_raster_layer_like(source)
    assert empty.pixel_scale == source.pixel_scale
    assert empty.projection == source.projection
    assert empty.map_projection == source.map_projection
    assert empty.window.xoff == 0
    assert empty.window.yoff == 0
    assert empty.window.xsize == source.window.xsize
    assert empty.window.ysize == source.window.ysize

def test_layer_with_positive_offset():
    source = RasterLayer(gdal_dataset_of_region(Area(-10, 10, 10, -10), 0.02))
    assert source.area == Area(-10, 10, 10, -10)
    assert source.window == Window(0, 0, 20 / 0.02, 20 / 0.02)

    source.offset_window_by_pixels(5)
    assert source.area == Area(-10 - (5 * 0.02), 10 + (5 * 0.02), 10 + (5 * 0.02), -10 - (5 * 0.02))
    assert source.window == Window(-5, -5, int(20 / 0.02) + 10, int(20 / 0.02) + 10)

def test_layer_with_zero_offset():
    source = RasterLayer(gdal_dataset_of_region(Area(-10, 10, 10, -10), 0.02))
    assert source.area == Area(-10, 10, 10, -10)
    assert source.window == Window(0, 0, 20 / 0.02, 20 / 0.02)

    source.offset_window_by_pixels(0)
    assert source.area == Area(-10, 10, 10, -10)
    assert source.window == Window(0, 0, 20 / 0.02, 20 / 0.02)

def test_layer_with_negative_offset():
    source = RasterLayer(gdal_dataset_of_region(Area(-10, 10, 10, -10), 0.02))
    assert source.area == Area(-10, 10, 10, -10)
    assert source.window == Window(0, 0, 20 / 0.02, 20 / 0.02)

    source.offset_window_by_pixels(-5)
    assert source.area == Area(-10 + (5 * 0.02), 10 - (5 * 0.02), 10 - (5 * 0.02), -10 + (5 * 0.02))
    assert source.window == Window(5, 5, int(20 / 0.02) - 10, int(20 / 0.02) - 10)

def test_layer_with_excessive_negative_offset():
    source = RasterLayer(gdal_dataset_of_region(Area(-10, 10, 10, -10), 0.02))
    with pytest.raises(ValueError):
        source.offset_window_by_pixels(-9999)

def test_layer_offsets_accumulate():
    source = RasterLayer(gdal_dataset_of_region(Area(-10, 10, 10, -10), 0.02))
    assert source.area == Area(-10, 10, 10, -10)
    assert source.window == Window(0, 0, 20 / 0.02, 20 / 0.02)

    source.offset_window_by_pixels(5)
    source.offset_window_by_pixels(-5)

    assert source.area == Area(-10, 10, 10, -10)
    assert source.window == Window(0, 0, 20 / 0.02, 20 / 0.02)

def test_scale_up():
    source = RasterLayer(gdal_dataset_of_region(Area(-10, 10, 10, -10), 0.2))
    assert source.area == Area(-10, 10, 10, -10)
    assert source.window == Window(0, 0, 20 / 0.2, 20 / 0.2)

    new_pixel_scale = PixelScale(0.1, -0.1)
    scaled = RasterLayer.scaled_raster_from_raster(
        source,
        new_pixel_scale
    )
    assert scaled.area == Area(-10, 10, 10, -10)
    assert scaled.window == Window(0, 0, 20 / 0.1, 20 / 0.1)
    assert scaled.sum() == source.sum() * 4

def test_scale_down():
    source = RasterLayer(gdal_dataset_of_region(Area(-10, 10, 10, -10), 0.2))
    assert source.area == Area(-10, 10, 10, -10)
    assert source.window == Window(0, 0, 20 / 0.2, 20 / 0.2)

    new_pixel_scale = PixelScale(0.5, -0.5)
    scaled = RasterLayer.scaled_raster_from_raster(
        source,
        new_pixel_scale
    )
    assert scaled.area == Area(-10, 10, 10, -10)
    assert scaled.window == Window(0, 0, 20 / 0.5, 20 / 0.5)
    # because we're dropping pixels, it's not easy to do this comparison
    # but at least make sure its less
    assert scaled.sum() < source.sum()

@pytest.mark.parametrize("size,expect_success",
    [
        ((5, 5), True),
        ((5, 1), True),
        ((1, 5), True),
        ((5, 0), False),
        ((0, 5), False),
    ]
)
def test_read_array_size(size, expect_success):
    source = RasterLayer(gdal_dataset_of_region(Area(-10, 10, 10, -10), 0.2))
    assert source.area == Area(-10, 10, 10, -10)
    assert source.window == Window(0, 0, 20 / 0.2, 20 / 0.2)

    if expect_success:
        data = source.read_array(0, 0, size[0], size[1])
        assert data.shape == (size[1], size[0])
    else:
        with pytest.raises(ValueError):
            _ = source.read_array(0, 0, size[0], size[1])

def test_invalid_band() -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        path = os.path.join(tempdir, "test.tif")
        area = Area(-10, 10, 10, -10)
        dataset = gdal_dataset_of_region(area, 0.02, filename=path)
        dataset.Close()
        assert os.path.exists(path)
        with pytest.raises(InvalidRasterBand):
            _ = RasterLayer.layer_from_file(path, band=42)

def test_multiband_raster() -> None:
    data1 = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    data2 = np.array([[10.0, 20.0, 30.0, 40.0], [50.0, 60.0, 70.0, 80.0]])

    datas = [data1, data2]
    dataset = gdal_multiband_dataset_with_data((0.0, 0.0), 0.02, datas)
    layer1 = RasterLayer(dataset, band=1)
    layer2 = RasterLayer(dataset, band=2)

    layers = [layer1, layer2]
    for i in range(2):
        data = datas[i]
        layer = layers[i]
        actual = layer.read_array(0, 0, 4, 2)
        assert (data == actual).all()

def test_read_array_is_numpy():
    # This test will fail if we use say the MLX backend without casting it back to numpy
    data = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    dataset = gdal_dataset_with_data((0.0, 0.0), 0.02, data)
    with RasterLayer(dataset) as layer1:
        actual = layer1.read_array(0, 0, 4, 2).astype(int)
        expected = data.astype(int)
        assert (actual == expected).all
