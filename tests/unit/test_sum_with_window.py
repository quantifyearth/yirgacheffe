import operator

import numpy as np
import pytest

import yirgacheffe as yg


def test_sum_sans_window_update() -> None:
    data1 = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
    projection = yg.MapProjection("epsg:4326", 1.0, -1.0)

    with yg.from_array(data1, (0, 0), projection) as layer1:
        assert layer1.sum() == np.sum(data1)


# In this and subsequent tests we use the add and multiply operators as
# add normally will trigger an automatic union and multiply will trigger an
# automatic intersection, so in all cases we check that we both override the
# default correctly and telling it to do what it would do anyway doesn't break it
@pytest.mark.parametrize("op", [operator.add, operator.mul])
def test_sum_with_union_inputs(op) -> None:
    data1 = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
    data2 = np.array([[10, 20], [50, 60]])
    projection = yg.MapProjection("epsg:4326", 1.0, -1.0)

    with (
        yg.from_array(data1, (0, 0), projection) as layer1,
        yg.from_array(data2, (1, -1), projection) as layer2,
    ):
        layers = [layer1, layer2]
        union = yg.find_union(layers)
        comp = op(layer1.as_area(union), layer2.as_area(union))

        padded_data2 = np.pad(data2, (1, 1))
        expected = op(data1, padded_data2)
        assert comp.sum() == np.sum(expected)


@pytest.mark.parametrize("op", [operator.add, operator.mul])
def test_sum_with_union_calc(op) -> None:
    data1 = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
    data2 = np.array([[10, 20], [50, 60]])
    projection = yg.MapProjection("epsg:4326", 1.0, -1.0)

    with (
        yg.from_array(data1, (0, 0), projection) as layer1,
        yg.from_array(data2, (1, -1), projection) as layer2,
    ):
        layers = [layer1, layer2]
        union = yg.find_union(layers)
        comp = op(layer1, layer2)

        padded_data2 = np.pad(data2, (1, 1))
        expected = op(data1, padded_data2)
        assert comp.as_area(union).sum() == np.sum(expected)


@pytest.mark.parametrize("op", [operator.add, operator.mul])
def test_sum_with_intersection_inputs(op) -> None:
    data1 = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
    data2 = np.array([[10, 20], [50, 60]])
    projection = yg.MapProjection("epsg:4326", 1.0, -1.0)

    with (
        yg.from_array(data1, (0, 0), projection) as layer1,
        yg.from_array(data2, (1, -1), projection) as layer2,
    ):
        layers = [layer1, layer2]
        intersection = yg.find_intersection(layers)
        comp = op(layer1.as_area(intersection), layer2.as_area(intersection))

        clipped_data1 = data1[1:3,1:3]
        expected = op(clipped_data1, data2)
        assert comp.sum() == np.sum(expected)


@pytest.mark.parametrize("op", [operator.add, operator.mul])
def test_sum_with_intersection_calc(op) -> None:
    data1 = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
    data2 = np.array([[10, 20], [50, 60]])
    projection = yg.MapProjection("epsg:4326", 1.0, -1.0)

    with (
        yg.from_array(data1, (0, 0), projection) as layer1,
        yg.from_array(data2, (1, -1), projection) as layer2,
    ):
        layers = [layer1, layer2]
        intersection = yg.find_intersection(layers)
        comp = op(layer1, layer2)

        clipped_data1 = data1[1:3,1:3]
        expected = op(clipped_data1, data2)
        assert comp.as_area(intersection).sum() == np.sum(expected)
