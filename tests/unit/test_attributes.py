import datetime
import tempfile
from pathlib import Path

import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon

import yirgacheffe as yg

def test_constant_attributes() -> None:
    with yg.constant(42) as c:
        assert c.attributes is None

def test_raster_attributes() -> None:
    projection = yg.MapProjection("epsg:4326", 0.02, -0.02)
    data = np.array([[1, 2], [3, 4]])
    with yg.from_array(data, (0, 0), projection) as raster:
        assert raster.attributes is None

def box(midx, midy, size=1.0):
    return Polygon([
        (midx - size / 2, midy - size / 2),
        (midx + size / 2, midy - size / 2),
        (midx + size / 2, midy + size / 2),
        (midx - size / 2, midy + size / 2),
    ])

def gpkg_data() -> dict:
    data = {
        "geometry": [
            box(0, 0),
            box(2, 0),
            box(4, 0),
            box(0, 2),
            box(2, 2),
        ],
        "strings": [
            "foo",
            "bar",
            "wibble",
            "baz",
            "flib",
        ],
        "integers": [
            250, 1800, 60, 120, 15,
        ],
        "floats": [
            1523.4, 8890.2, 210.75, 640.0, 95.3,
        ],
        "bools": [
            True, True, False, True, False,
        ],
        "dates": [
            datetime.date(2023, 6, 1),
            datetime.date(2024, 1, 15),
            datetime.date(2022, 11, 30),
            datetime.date(2024, 3, 22),
            datetime.date(2021, 9, 10),
        ],
        "optional_strings": [
            "here",
            None,
            "also",
            "not none",
            None,
        ],
    }
    # If we won't force the type here the dates end up as strings when
    # written to disk. We also need to specify which subset of datetime64 to use
    # as pandas is sensitive to the input type and seems to pick different accuracies
    data["dates"] = pd.to_datetime(data["dates"]).astype("datetime64[us]") # type: ignore
    return data

def test_gpkg_simple_attributes() -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        filename = Path(tempdir) / "test.gpkg"

        data = gpkg_data()
        pdf = gpd.GeoDataFrame(data, crs="epsg:4326")
        pdf.to_file(filename, driver="GPKG")

        with yg.read_shape(filename) as layer:
            attrs = layer.attributes
            assert attrs is not None
            assert len(attrs) == 5

            # our attributes table will not have geometry in it
            expected_keys = list(data.keys())
            expected_keys.remove("geometry")
            assert (attrs.columns == expected_keys).all()
            for key in attrs.columns:
                assert list(data[key]) == [None if pd.isna(x) else x for x in attrs[key]]

def test_geojson_simple_attributes() -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        filename = Path(tempdir) / "test.geojson"

        data = gpkg_data()
        pdf = gpd.GeoDataFrame(data, crs="epsg:4326")
        pdf.to_file(filename, driver="GeoJSON")

        with yg.read_shape(filename) as layer:
            attrs = layer.attributes
            assert attrs is not None
            assert len(attrs) == 5

            # our attributes table will not have geometry in it
            expected_keys = list(data.keys())
            expected_keys.remove("geometry")
            assert (attrs.columns == expected_keys).all()
            for key in attrs.columns:
                assert list(data[key]) == [None if pd.isna(x) else x for x in attrs[key]]

def test_gpkg_filtered_attributes() -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        filename = Path(tempdir) / "test.gpkg"

        data = gpkg_data()
        pdf = gpd.GeoDataFrame(data, crs="epsg:4326")
        pdf.to_file(filename, driver="GPKG")

        with yg.read_shape(filename, where_filter="bools=true") as layer:
            attrs = layer.attributes
            assert attrs is not None
            assert len(attrs) == 3

        del data["geometry"]
        df = pd.DataFrame(data)
        df = df[df.bools==True].reset_index(drop=True) # pylint: disable=C0121

        assert df.equals(attrs)
