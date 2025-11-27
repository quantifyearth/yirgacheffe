
import numpy as np
import yirgacheffe as yg

import pytest

def test_nearest_neighbour_raster_alignment() -> None:
    data1 = np.array([[1]])
    data2 = np.array([[3, 2, 1], [6, 5, 4], [9, 8, 7]])

    projection = yg.MapProjection("esri:54009", 10.0, -10.0)

    with yg.from_array(data1, (30.0, 10.0), projection) as focus_layer:
        focus_layer.name = "focus"
        for y_offset in range(40):
            for x_offset in range(40):

                x_index = (x_offset - 6) // 10
                y_index = (y_offset - 6) // 10

                expect_error = not (0 <= x_index < 3) or not (0 <= y_index < 3)

                with yg.from_array(data2, (0.0 + x_offset, 0.0 + y_offset), projection) as sliding_layer:

                    if expect_error:
                        with pytest.raises(ValueError):
                            selected_pixel = sliding_layer * focus_layer
                            _ = selected_pixel.sum()
                    else:
                        selected_pixel = sliding_layer * focus_layer
                        val = selected_pixel.read_array(0, 0, 1, 1)[0][0]
                        expected = ((y_index * 3) + x_index) + 1
                        assert val == expected
