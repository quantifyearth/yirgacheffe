import numpy
import pytest
from osgeo import gdal

from helpers import gdal_dataset_with_data
from yirgacheffe.layers import Layer

def test_add_byte_layers() -> None:
    data1 = numpy.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    data2 = numpy.array([[10, 20, 30, 40], [50, 60, 70, 80]])

    layer1 = Layer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))
    layer2 = Layer(gdal_dataset_with_data((0.0, 0.0), 0.02, data2))

    comp = layer1 + layer2
    print(comp)

    result_data = gdal.GetDriverByName('mem').Create(
        'mem',
        4,
        2,
        1,
        gdal.GDT_Byte,
        []
    )
    band = result_data.GetRasterBand(1)
    comp.save(band=band)

    expected = data1 + data2
    actual = band.ReadAsArray(0, 0, 4, 2)

    assert (expected == actual).all()

def test_sub_byte_layers() -> None:
    data1 = numpy.array([[10, 20, 30, 40], [50, 60, 70, 80]])
    data2 = numpy.array([[1, 2, 3, 4], [5, 6, 7, 8]])

    layer1 = Layer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))
    layer2 = Layer(gdal_dataset_with_data((0.0, 0.0), 0.02, data2))

    comp = layer1 - layer2

    result_data = gdal.GetDriverByName('mem').Create(
        'mem',
        4,
        2,
        1,
        gdal.GDT_Byte,
        []
    )
    band = result_data.GetRasterBand(1)
    comp.save(band=band)

    expected = data1 - data2
    actual = band.ReadAsArray(0, 0, 4, 2)

    assert (expected == actual).all()

def test_add_float_layers() -> None:
    data1 = numpy.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    data2 = numpy.array([[10.0, 20.0, 30.0, 40.0], [50.0, 60.0, 70.0, 80.0]])

    layer1 = Layer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))
    layer2 = Layer(gdal_dataset_with_data((0.0, 0.0), 0.02, data2))

    comp = layer1 + layer2

    result_data = gdal.GetDriverByName('mem').Create(
        'mem',
        4,
        2,
        1,
        gdal.GDT_Float64,
        []
    )
    band = result_data.GetRasterBand(1)
    comp.save(band=band)

    expected = data1 + data2
    actual = band.ReadAsArray(0, 0, 4, 2)

    assert (expected == actual).all()

def test_sub_float_layers() -> None:
    data1 = numpy.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    data2 = numpy.array([[10.0, 20.0, 30.0, 40.0], [50.0, 60.0, 70.0, 80.0]])

    layer1 = Layer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))
    layer2 = Layer(gdal_dataset_with_data((0.0, 0.0), 0.02, data2))

    comp = layer1 - layer2

    result_data = gdal.GetDriverByName('mem').Create(
        'mem',
        4,
        2,
        1,
        gdal.GDT_Float64,
        []
    )
    band = result_data.GetRasterBand(1)
    comp.save(band=band)

    expected = data1 - data2
    actual = band.ReadAsArray(0, 0, 4, 2)

    assert (expected == actual).all()

def test_mult_float_layers() -> None:
    data1 = numpy.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    data2 = numpy.array([[10.0, 20.0, 30.0, 40.0], [50.0, 60.0, 70.0, 80.0]])

    layer1 = Layer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))
    layer2 = Layer(gdal_dataset_with_data((0.0, 0.0), 0.02, data2))

    comp = layer1 * layer2

    result_data = gdal.GetDriverByName('mem').Create(
        'mem',
        4,
        2,
        1,
        gdal.GDT_Float64,
        []
    )
    band = result_data.GetRasterBand(1)
    comp.save(band=band)

    expected = data1 * data2
    actual = band.ReadAsArray(0, 0, 4, 2)

    assert (expected == actual).all()

def test_mult_float_layer_by_const() -> None:
    data1 = numpy.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    layer1 = Layer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))

    comp = layer1 * 2.5

    result_data = gdal.GetDriverByName('mem').Create(
        'mem',
        4,
        2,
        1,
        gdal.GDT_Float64,
        []
    )
    band = result_data.GetRasterBand(1)
    comp.save(band=band)

    expected = data1 * 2.5
    actual = band.ReadAsArray(0, 0, 4, 2)

    assert (expected == actual).all()

def test_div_float_layer_by_const() -> None:
    data1 = numpy.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    layer1 = Layer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))

    comp = layer1 / 2.5

    result_data = gdal.GetDriverByName('mem').Create(
        'mem',
        4,
        2,
        1,
        gdal.GDT_Float64,
        []
    )
    band = result_data.GetRasterBand(1)
    comp.save(band=band)

    expected = data1 / 2.5
    actual = band.ReadAsArray(0, 0, 4, 2)

    assert (expected == actual).all()

def test_power_float_layer_by_const() -> None:
    data1 = numpy.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    layer1 = Layer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))

    comp = layer1 ** 2.5

    result_data = gdal.GetDriverByName('mem').Create(
        'mem',
        4,
        2,
        1,
        gdal.GDT_Float64,
        []
    )
    band = result_data.GetRasterBand(1)
    comp.save(band=band)

    expected = data1 ** 2.5
    actual = band.ReadAsArray(0, 0, 4, 2)

    assert (expected == actual).all()

def test_unary_apply() -> None:
    data1 = numpy.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    layer1 = Layer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))

    def simple_add(value):
        return value + 1.0

    comp = layer1.apply(simple_add)

    result_data = gdal.GetDriverByName('mem').Create(
        'mem',
        4,
        2,
        1,
        gdal.GDT_Float64,
        []
    )
    band = result_data.GetRasterBand(1)
    comp.save(band=band)

    expected = data1 + 1.0
    actual = band.ReadAsArray(0, 0, 4, 2)

    assert (expected == actual).all()

def test_binary_apply() -> None:
    data1 = numpy.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    data2 = numpy.array([[10.0, 20.0, 30.0, 40.0], [50.0, 60.0, 70.0, 80.0]])

    layer1 = Layer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))
    layer2 = Layer(gdal_dataset_with_data((0.0, 0.0), 0.02, data2))

    def simple_add(value1, value2):
        return value1 + value2

    comp = layer1.apply(simple_add, layer2)

    result_data = gdal.GetDriverByName('mem').Create(
        'mem',
        4,
        2,
        1,
        gdal.GDT_Float64,
        []
    )
    band = result_data.GetRasterBand(1)
    comp.save(band=band)

    expected = data1 + data2
    actual = band.ReadAsArray(0, 0, 4, 2)

    assert (expected == actual).all()


@pytest.mark.parametrize("operator",
    [
        lambda a, b: a == b,
        lambda a, b: a != b,
        lambda a, b: a > b,
        lambda a, b: a >= b,
        lambda a, b: a < b,
        lambda a, b: a <= b,
    ]
)
def test_comparison_float_layer_by_const(operator) -> None:
    data1 = numpy.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    layer1 = Layer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))

    comp = operator(layer1, 3.0)

    result_data = gdal.GetDriverByName('mem').Create(
        'mem',
        4,
        2,
        1,
        gdal.GDT_Byte,
        []
    )
    band = result_data.GetRasterBand(1)
    comp.save(band=band)

    expected = operator(data1, 3.0)
    actual = band.ReadAsArray(0, 0, 4, 2)

    assert (expected == actual).all()
