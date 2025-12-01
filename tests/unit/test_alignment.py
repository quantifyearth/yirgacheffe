import tempfile
from pathlib import Path

import numpy as np
import pytest
import yirgacheffe as yg

from .helpers import make_vectors_with_id

def test_nearest_neighbour_raster_alignment_large_step() -> None:
    data1 = np.array([[1]])
    data2 = np.array([[3, 2, 1], [6, 5, 4], [9, 8, 7]])

    projection = yg.MapProjection("esri:54009", 10.0, -10.0)

    with yg.from_array(data1, (30.0, 10.0), projection) as focus_layer:
        for y_offset in range(40):
            for x_offset in range(40):

                x_index = (x_offset - 6) // 10
                y_index = (y_offset - 6) // 10

                expect_error = not 0 <= x_index < 3 or not 0 <= y_index < 3

                with yg.from_array(
                    data2, (0.0 + x_offset, 0.0 + y_offset), projection
                ) as sliding_layer:
                    sliding_layer.name = "sliding"

                    # Just test the area intersection, as this can
                    # be one source of error, and the read_array
                    # is another potential source, and so we want
                    # to fail with it being obvious which one is
                    # causing trouble
                    if expect_error:
                        with pytest.raises(ValueError):
                            _ = sliding_layer.area & focus_layer.area
                    else:
                        _ = sliding_layer.area & focus_layer.area

                    if expect_error:
                        with pytest.raises(ValueError):
                            selected_pixel = sliding_layer * focus_layer
                            _ = selected_pixel.sum()
                    else:
                        selected_pixel = sliding_layer * focus_layer
                        val = selected_pixel.read_array(0, 0, 1, 1)[0][0]
                        expected = ((y_index * 3) + x_index) + 1
                        assert val == expected


def test_nearest_neighbour_raster_alignment_small_step() -> None:
    data1 = np.array([[1]])
    data2 = np.array([[3, 2, 1], [6, 5, 4], [9, 8, 7]])

    projection = yg.MapProjection("epsg:4326", 0.01, -0.01)

    with yg.from_array(data1, (0.03, 0.01), projection) as focus_layer:
        for y_offset in range(40):
            for x_offset in range(40):
                x_index = (x_offset - 6) // 10
                y_index = (y_offset - 6) // 10

                expect_error = not 0 <= x_index < 3 or not 0 <= y_index < 3

                with yg.from_array(
                    data2,
                    (0.0 + (x_offset * 0.001), 0.0 + (y_offset * 0.001)),
                    projection,
                ) as sliding_layer:

                    # Just test the area intersection, as this can
                    # be one source of error, and the read_array
                    # is another potential source, and so we want
                    # to fail with it being obvious which one is
                    # causing trouble
                    if expect_error:
                        with pytest.raises(ValueError):
                            _ = sliding_layer.area & focus_layer.area
                    else:
                        _ = sliding_layer.area & focus_layer.area

                    if expect_error:
                        with pytest.raises(ValueError):
                            selected_pixel = sliding_layer * focus_layer
                            _ = selected_pixel.sum()
                    else:
                        selected_pixel = sliding_layer * focus_layer
                        val = selected_pixel.read_array(0, 0, 1, 1)[0][0]
                        expected_value = ((y_index * 3) + x_index) + 1
                        assert val == expected_value


def test_nearest_neighbour_vector_alignment_small_step() -> None:
    data2 = np.array([[3, 2, 1], [6, 5, 4], [9, 8, 7]])
    projection = yg.MapProjection("epsg:4326", 0.01, -0.01)

    with tempfile.TemporaryDirectory() as tmpdir:
        filename = Path(tmpdir) / "focus.gpkg"
        area = yg.Area(0.03, 0.01, 0.04, 0.00)
        make_vectors_with_id(42, {area}, filename)
        with yg.read_shape(filename, projection) as focus_layer:

            for y_offset in range(40):
                for x_offset in range(40):
                    x_index = (x_offset - 6) // 10
                    y_index = (y_offset - 6) // 10

                    expect_error = not 0 <= x_index < 3 or not 0 <= y_index < 3

                    with yg.from_array(
                        data2,
                        (0.0 + (x_offset * 0.001), 0.0 + (y_offset * 0.001)),
                        projection,
                    ) as sliding_layer:

                        # Just test the area intersection, as this can
                        # be one source of error, and the read_array
                        # is another potential source, and so we want
                        # to fail with it being obvious which one is
                        # causing trouble
                        if expect_error:
                            with pytest.raises(ValueError):
                                _ = sliding_layer.area & focus_layer.area
                        else:
                            _ = sliding_layer.area & focus_layer.area

                        if expect_error:
                            with pytest.raises(ValueError):
                                selected_pixel = sliding_layer * focus_layer
                                _ = selected_pixel.sum()
                        else:
                            selected_pixel = sliding_layer * focus_layer
                            val = selected_pixel.read_array(0, 0, 1, 1)[0][0]
                            expected_value = ((y_index * 3) + x_index) + 1
                            assert val == expected_value


def test_vector_layer_consistency() -> None:
    # This test is based on an issue I observed with an AOH where the habitat and elevation
    # layers were offset enough that if you anchored the vector to a different one it
    # rendered as either 4 or 2 pixels. This is unfortunate, but okay if you are at least consistent, but
    # the rasterization of the vector was happening in the space of which layer you were comparing it
    # with which meant you got different answers on revert to range on elevation and on habitat :/

    projection = yg.MapProjection('ESRI:54009', 1000.0, -1000.0)
    species_area = yg.Area(1.0, 1999.0, 1999.0, 1.0)

    data = np.array([[1, 2, 3, 4], [5, 4, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]])
    expected = np.array([[4, 6], [9, 10]])

    with tempfile.TemporaryDirectory() as tmpdir:
        filename = Path(tmpdir) / "focus.gpkg"
        make_vectors_with_id(42, {species_area}, filename, 'ESRI:54009')

        with yg.from_array(data, (-1000.0, 3000.0), projection) as raster1:

            # Unprojected shape first
            with yg.read_shape(filename) as range_layer:
                calc = raster1 * range_layer
                assert calc.window.xsize == 2
                assert calc.window.ysize == 2
                result = calc.read_array(0, 0, 2, 2)
                assert (result == expected).all()

            # Check with a projected shape
            with yg.read_shape_like(filename, raster1) as range_layer:
                assert range_layer.sum() == 4
                calc = raster1 * range_layer
                assert calc.window.xsize == 2
                assert calc.window.ysize == 2
                result = calc.read_array(0, 0, 2, 2)
                assert (result == expected).all()

        # Where we expect the size of the range_layer to change size as the polygon aligns or doesn't with
        # the grid of the raster it is matched with. However, if we then multiple that with a large grid of
        # ones that is zero aligned we shouldn't see it suddenly to back to a 2x2 grid, which was the bug
        # that made me write this test case.
        with yg.from_array(np.ones((16, 16)), (-8000.0, 8000.0), projection) as raster0:
            for offset in range(-500, 500):
                with yg.from_array(data, (-1000.0 + offset, 3000.0), projection) as raster1:
                    with yg.read_shape_like(filename, raster1) as range_layer:
                        calc = raster0 * range_layer
                        assert range_layer.sum() == calc.sum()

                with yg.from_array(data, (-1000.0, 3000.0 + offset), projection) as raster1:
                    with yg.read_shape_like(filename, raster1) as range_layer:
                        calc = raster0 * range_layer
                        assert range_layer.sum() == calc.sum()
