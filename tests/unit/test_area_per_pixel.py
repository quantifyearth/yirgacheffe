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


def test_approximate_wgs84_values_1() -> None:
    projection = yg.MapProjection("epsg:4326", 1.0, -1.0)
    with yg.area_raster(projection) as area_raster:

        # Read single pixels at known latitudes
        equator_area = area_raster.read_array(0, 90, 1, 1)[0, 0]  # lat=0°
        lat_60_area = area_raster.read_array(0, 30, 1, 1)[0, 0]   # lat=60°

        # Single degree at equator: ~12,365,000,000 m²
        # And at 60°: ~6,182,000,000 m² (roughly half)
        assert 12.3e9 < equator_area < 12.4e9
        assert 6.1e9 < lat_60_area < 6.4e9
        ratio = lat_60_area / equator_area
        assert 0.48 < ratio < 0.52


def test_approximate_wgs84_values_2() -> None:
    # https://web.archive.org/web/20110424104419/http://home.online.no/~sigurdhu/WGS84_Eng.html
    CIRCUMFERENCE_OF_EARTH = 40_075_017 # metres
    DEGREES_PER_KILOMETER = 360 / (CIRCUMFERENCE_OF_EARTH / 1000)
    projection = yg.MapProjection("epsg:4326", DEGREES_PER_KILOMETER, DEGREES_PER_KILOMETER * -1.0)
    with yg.area_raster(projection) as area_raster:
        equator_y_value = round(90 / DEGREES_PER_KILOMETER)
        equator_area = area_raster.read_array(0, equator_y_value, 1, 1)[0, 0]
        expected = 1_000_000  # m²
        assert (expected * 0.99) < equator_area < (expected * 1.01)


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
        assert (area_values == 1_000_000).all()


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
                assert val[0][0] == 100_000_000


def test_non_global_area():
    # EPSG:32633 - UTM Zone 33N (covers Sweden, Norway)
    # Valid: 12°E to 18°E, 0°N to 84°N
    projection = yg.MapProjection("epsg:32633", 1000.0, -1000.0)
    west, south, east, north = projection.crs.area_of_use.bounds
    assert west == 12.0 and east == 18.0
    assert south == 0.0 and north == 84.0

    with yg.area_raster(projection) as area_raster:
        # Check projected bounds are reasonable
        # Should be roughly 166km to 834km E-W, 0 to 9329km N-S
        area = area_raster.area
        assert 100_000 < area.left < 200_000
        assert 800_000 < area.right < 900_000
        assert -10_000 < area.bottom < 10_000
        assert 9_200_000 < area.top < 9_400_000

        # whilst we're here, check the areas are okay
        result = area_raster.read_array(0, 0, 100, 100)
        assert (result == 1_000_000).all()
