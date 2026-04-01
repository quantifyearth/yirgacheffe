from typing import Any

import numpy as np
import pytest
import yirgacheffe as yg

def test_simple_clip() -> None:
    data = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
    projection = yg.MapProjection("esri:54009", 1.0, -1.0)

    target_area = yg.Area(1.0, -1.0, 3.0, -3.0, projection)
    expected_data = data[1:3,1:3]

    with yg.from_array(data, (0, 0), projection) as layer:
        clipped = layer.as_area(target_area)

        assert clipped.dimensions == (2, 2)
        assert clipped.area == target_area

        read_data = clipped.read_array(0, 0, 2, 2)
        assert (read_data == expected_data).all()


def test_simple_pad() -> None:
    data = np.array([[1, 2], [3, 4]])
    projection = yg.MapProjection("esri:54009", 1.0, -1.0)

    target_area = yg.Area(0.0, 0.0, 4.0, -4.0, projection)
    expected_data = np.array([[0, 0, 0, 0], [0, 1, 2, 0], [0, 3, 4, 0], [0, 0, 0, 0]])

    with yg.from_array(data, (1, -1), projection) as layer:
        clipped = layer.as_area(target_area)

        assert clipped.dimensions == (4, 4)
        assert clipped.area == target_area

        read_data = clipped.read_array(0, 0, 4, 4)
        assert (read_data == expected_data).all()


def test_simple_clip_from_layer() -> None:
    data = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
    projection = yg.MapProjection("esri:54009", 1.0, -1.0)

    target_area = yg.Area(1.0, -1.0, 3.0, -3.0, projection)
    expected_data = data[1:3,1:3]

    with (
        yg.from_array(data, (0, 0), projection) as layer,
        yg.from_array(np.zeros((2, 2)), (1, -1), projection) as other,
    ):
        assert other.area == target_area
        clipped = layer.as_area(other)

        assert clipped.dimensions == (2, 2)
        assert clipped.area == target_area

        read_data = clipped.read_array(0, 0, 2, 2)
        assert (read_data == expected_data).all()


def test_simple_pad_from_layer() -> None:
    data = np.array([[1, 2], [3, 4]])
    projection = yg.MapProjection("esri:54009", 1.0, -1.0)

    target_area = yg.Area(0.0, 0.0, 4.0, -4.0, projection)
    expected_data = np.array([[0, 0, 0, 0], [0, 1, 2, 0], [0, 3, 4, 0], [0, 0, 0, 0]])

    with (
        yg.from_array(data, (1, -1), projection) as layer,
        yg.from_array(np.zeros((4, 4)), (0, 0), projection) as other,
    ):
        assert other.area == target_area
        clipped = layer.as_area(other)

        assert clipped.dimensions == (4, 4)
        assert clipped.area == target_area

        read_data = clipped.read_array(0, 0, 4, 4)
        assert (read_data == expected_data).all()


@pytest.mark.parametrize("area", [42, None, "boo"])
def test_invalid_areas(area: Any) -> None:
    data = np.array([[1, 2], [3, 4]])
    projection = yg.MapProjection("esri:54009", 1.0, -1.0)
    with yg.from_array(data, (1, -1), projection) as layer:
        with pytest.raises(TypeError):
            _ = layer.as_area(area)


def test_add_byte_layers_with_union() -> None:
    data1 = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
    data2 = np.array([[10, 20], [50, 60]])
    projection = yg.MapProjection("epsg:4326", 1.0, -1.0)

    with (
        yg.from_array(data1, (0, 0), projection) as layer1,
        yg.from_array(data2, (1, -1), projection) as layer2,
    ):
        layers = [layer1, layer2]
        union = yg.find_union(layers)
        comp = layer1.as_area(union) + layer2.as_area(union)
        assert comp.dimensions == (4, 4)
        assert comp.area == union

        expected = np.array([[1, 2, 3, 4], [5, 16, 27, 8], [9, 60, 71, 12], [13, 14, 15, 16]])
        actual = comp.read_array(0, 0, 4, 4)
        assert (expected == actual).all()


def test_add_byte_layers_with_intersection_with_max_save_raster() -> None:
    data1 = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
    data2 = np.array([[10, 20], [50, 60]])
    projection = yg.MapProjection("epsg:4326", 1.0, -1.0)

    with (
        yg.from_array(data1, (0, 0), projection) as layer1,
        yg.from_array(data2, (1, -1), projection) as layer2,
    ):
        layers = [layer1, layer2]
        intersection = yg.find_intersection(layers)

        comp = layer1.as_area(intersection) + layer2.as_area(intersection)
        assert comp.dimensions == (2, 2)
        assert comp.area == intersection

        expected = np.array([[16, 27], [60, 71]])
        actual = comp.read_array(0, 0, 2, 2)

        assert (expected == actual).all()


@pytest.mark.parametrize("other_projection", [
    yg.MapProjection("epsg:4326", 10.0, -10.0), # same projection, different pixel scale
    yg.MapProjection("esri:54009", 1.0, -1.0), # different projection, same pixel_scale
])
def test_projection_mismatch(other_projection: yg.MapProjection) -> None:
    data1 = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
    projection = yg.MapProjection("epsg:4326", 1.0, -1.0)

    other_area = yg.Area(0, 0, 20, -20, other_projection)

    with yg.from_array(data1, (0, 0), projection) as layer:
        with pytest.raises(ValueError):
            _ = layer.as_area(other_area)


def test_infer_projection() -> None:
    data1 = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
    projection = yg.MapProjection("epsg:4326", 1.0, -1.0)

    other_area = yg.Area(0, 0, 20, -20) # no explicit projection

    with yg.from_array(data1, (0, 0), projection) as layer:
        adjusted_layer = layer.as_area(other_area)
        assert adjusted_layer.area.projection == projection


def test_clip_and_pad() -> None:
    # This tests shows that setting an area doesn't apply a mask or anything, just that the
    # most recent as_area wins.
    data = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
    projection = yg.MapProjection("esri:54009", 1.0, -1.0)

    target_area = yg.Area(1.0, -1.0, 3.0, -3.0, projection)

    with yg.from_array(data, (0, 0), projection) as layer:
        clipped = layer.as_area(target_area).as_area(layer.area)

        assert clipped.dimensions == (4, 4)
        assert clipped.area == layer.area

        read_data = clipped.read_array(0, 0, 4, 4)
        assert (read_data == data).all()


def test_pad_and_clip() -> None:
    data = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
    projection = yg.MapProjection("esri:54009", 1.0, -1.0)

    target_area = yg.Area(-1.0, 1.0, 5.0, -5.0, projection)

    with yg.from_array(data, (0, 0), projection) as layer:
        clipped = layer.as_area(target_area).as_area(layer.area)

        assert clipped.dimensions == (4, 4)
        assert clipped.area == layer.area

        read_data = clipped.read_array(0, 0, 4, 4)
        assert (read_data == data).all()
