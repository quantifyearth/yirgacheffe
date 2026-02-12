
import numpy as np
import yirgacheffe as yg

def test_simple_wgs84_layer() -> None:
    projection = yg.MapProjection("epsg:4326", 10.0, -10.0)
    with yg.area_raster(projection) as area_raster:
        assert area_raster.map_projection == projection
        assert area_raster.window == yg.Window(0, 0, 36, 18)
        assert area_raster.area == yg.Area(-180, 90, 180, -90, projection)

        # I don't want to test specific values, just trends
        area_values = area_raster.read_array(0, 0, 36, 18)
        # 1: All values on each row are the same
        assert np.allclose(area_values, area_values[:, 0:1]), "All columns should match first column"
        # 2: Top half mirrors bottom half (symmetric around equator)
        assert np.allclose(area_values, np.flip(area_values, axis=0))
        # 3: Top half values increase (from pole toward equator)
        top_half = area_values[:9, 0]
        assert np.all(np.diff(top_half) > 0)

def test_windowed_wgs84_layer() -> None:
    projection = yg.MapProjection("epsg:4326", 10.0, -10.0)
    with yg.area_raster(projection) as area_raster:
        area_values = area_raster.read_array(0, 0, 36, 18)
        for x in range(36):
            for y in range(18):
                xpos = -18 + x
                ypos = 9 - y
                with yg.from_array(np.array([[1]]), (xpos * 10, ypos * 10), projection) as mask:
                    res = mask * area_raster
                    assert res.window.xsize == 1
                    assert res.window.ysize == 1
                    val = res.read_array(0, 0, 1, 1)
                    assert val[0][0] == area_values[y][x]

def test_simple_mollweide_layer() -> None:
    projection = yg.MapProjection("esri:54009", 1000.0, -1000.0)
    with yg.area_raster(projection) as area_raster:
        assert area_raster.map_projection == projection
        assert area_raster.window == yg.Window(0, 0, 36082, 18041)
        assert area_raster.area == yg.Area(
            left=-18041000.0,
            top=9020000.0,
            right=18041000.0,
            bottom=-9021000.0,
            projection=projection
        )

        # For Mollweide and friends all values should be the same
        area_values = area_raster.read_array(0, 0, 36, 18)
        assert (area_values == 1000000.0).all()

def test_windowed_mollweide_layer() -> None:
    projection = yg.MapProjection("esri:54009", 10000.0, -10000.0)
    with yg.area_raster(projection) as area_raster:
        # This is subset, as doing everything is very slow, and it should always be the same value
        for pos in range(10):
            with yg.from_array(np.array([[1]]), ((5 - pos) * 10000, (5 - pos) * 10000), projection) as mask:
                res = mask * area_raster
                assert res.window.xsize == 1
                assert res.window.ysize == 1
                val = res.read_array(0, 0, 1, 1)
                assert val[0][0] == 100000000.0
