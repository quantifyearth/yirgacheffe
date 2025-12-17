import tempfile
from pathlib import Path

import numpy as np
import pyproj
import pytest
from osgeo import gdal

import yirgacheffe as yg
from yirgacheffe import MapProjection
from yirgacheffe._datatypes.mapprojection import MINIMAL_DEGREE_OF_INTEREST


@pytest.mark.parametrize(
    "name",
    [
        "epsg:4326",
        "esri:54009",
        "EPSG:5845",
        "EPSG:32633",
        "EPSG:27700",
        "EPSG:2154",
        pyproj.CRS.from_string("epsg:4326").to_wkt(),
        pyproj.CRS.from_string("esri:54009").to_wkt(),
    ],
)
def test_scale_from_projection(name) -> None:
    projection = MapProjection(name, 0.1, -0.1)
    assert projection == MapProjection(name, 0.1, -0.1)
    assert projection.name == pyproj.CRS.from_string(name).to_wkt()
    assert projection.xstep == 0.1
    assert projection.ystep == -0.1
    scale = projection.scale
    assert scale.xstep == 0.1
    assert scale.ystep == -0.1


@pytest.mark.parametrize("crsname", [
    "epsg:4326",
    "EPSG:5845", # This is the only known real world fail
    "EPSG:32633",
    "EPSG:27700",
    "EPSG:2154",
    "ESRI:54009",
])
def test_projection_stability(crsname: str) -> None:
    # This test case is for the issue described by
    # https://github.com/quantifyearth/yirgacheffe/issues/113
    # whereby if we read the projection from GDAL and then save
    # a new GeoTIFF with that same description then
    # for certain projections it changes :/
    data = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])

    # Create a GeoTIFF using the first name
    with tempfile.TemporaryDirectory() as tmpdirstr:
        tmpdir = Path(tmpdirstr)

        # We use GDAL to write the first file as this test was
        # written whilst trying to improve Yirgacheffe's projection
        # name management.
        driver = gdal.GetDriverByName('GTiff')
        ds = driver.Create(
            tmpdir / "first.tif",
            4,
            2,
            1,
            gdal.GDT_Byte,
            []
        )
        ds.SetProjection(crsname)
        ds.SetGeoTransform((0.0, 0.1, 0.0, 0.0, 0.0, -0.1))
        ds.GetRasterBand(1).WriteArray(data, 0, 0)
        ds.Close()

        with yg.read_raster(tmpdir / "first.tif") as gen1:
            first_generation_projection = gen1.map_projection
            gen1.to_geotiff(tmpdir / "second.tif")

        with yg.read_raster(tmpdir / "second.tif") as gen2:
            second_generation_projection = gen2.map_projection

        assert first_generation_projection == second_generation_projection


PROJ_A = "epsg:4326"
PROJ_B = "esri:54009"


@pytest.mark.parametrize(
    "lhs,rhs,is_equal",
    [
        (MapProjection(PROJ_A, 0.1, -0.1), MapProjection(PROJ_A, 0.1, -0.1), True),
        (MapProjection(PROJ_A, 0.1, -0.1), MapProjection(PROJ_B, 0.1, -0.1), False),
        (MapProjection(PROJ_A, 0.1, -0.1), MapProjection(PROJ_A, 0.1, 0.1), False),
        (MapProjection(PROJ_A, 0.1, -0.1), MapProjection(PROJ_A, -0.1, 0.1), False),
        (
            MapProjection(PROJ_A, 0.1, -0.1),
            MapProjection(PROJ_A, 0.1 + (MINIMAL_DEGREE_OF_INTEREST / 2), -0.1),
            True,
        ),
        (
            MapProjection(PROJ_A, 0.1, -0.1),
            MapProjection(PROJ_A, 0.1 - (MINIMAL_DEGREE_OF_INTEREST / 2), -0.1),
            True,
        ),
        (
            MapProjection(PROJ_A, 0.1, -0.1),
            MapProjection(PROJ_A, 0.1, -0.1 + (MINIMAL_DEGREE_OF_INTEREST / 2)),
            True,
        ),
        (
            MapProjection(PROJ_A, 0.1, -0.1),
            MapProjection(PROJ_A, 0.1, -0.1 - (MINIMAL_DEGREE_OF_INTEREST / 2)),
            True,
        ),
    ],
)
def test_projection_equality(
    lhs: MapProjection, rhs: MapProjection, is_equal: bool
) -> None:
    assert MINIMAL_DEGREE_OF_INTEREST > 0.0
    assert (lhs == rhs) == is_equal
    assert (lhs != rhs) == (not is_equal)


def test_invalid_projection_name() -> None:
    with pytest.raises(ValueError):
        _ = MapProjection("random name", 1.0, -1.0)
