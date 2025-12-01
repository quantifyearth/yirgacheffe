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
