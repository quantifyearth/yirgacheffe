import tempfile
from pathlib import Path

import numpy as np
import pytest
import tifffile
import yirgacheffe as yg

def test_sprase_needs_nodata() -> None:
    data = np.zeros((5000, 5000))
    with tempfile.TemporaryDirectory() as tempdir:
        filename = Path(tempdir) / "test.tif"
        with yg.from_array(data, (0, 0), yg.MapProjection("EPSG:4326", 10.0, -10.0)) as layer:
            with pytest.raises(ValueError):
                layer.to_geotiff(filename, sparse=True)


@pytest.mark.parametrize("dtype,zero", [
    (np.int8, 0),
    (np.float32, 0.0),
])
def test_all_sparse(dtype: type, zero: int | float) -> None:
    data = np.zeros((5000, 5000)).astype(dtype) # type: ignore
    with tempfile.TemporaryDirectory() as tempdir:
        filename = Path(tempdir) / "test.tif"
        with yg.from_array(data, (0, 0), yg.MapProjection("EPSG:4326", 10.0, -10.0)) as layer:
            layer.to_geotiff(filename, nodata=zero, sparse=True)
        with yg.read_raster(filename) as layer:
            # no data is nan, so convert that to 0, as otherwise
            # this test will miss things
            assert layer.nan_to_num().sum() == 0

        # now look into the tiff structure to see that all blocks are tiff
        with tifffile.TiffFile(filename) as tif:
            page = tif.pages[0] # type: ignore
            offsets = page.tags.get('StripOffsets').value # type: ignore
            byte_counts = page.tags.get('StripByteCounts').value # type: ignore

            for offset, count in zip(offsets, byte_counts):
                assert offset == 0
                assert count == 0


@pytest.mark.parametrize("dtype,zero", [
    (np.int8, 0),
    (np.float32, 0.0),
])
def test_none_sparse(dtype: type, zero: int | float) -> None:
    data = np.ones((5000, 5000)).astype(dtype)  # type: ignore
    with tempfile.TemporaryDirectory() as tempdir:
        filename = Path(tempdir) / "test.tif"
        with yg.from_array(data, (0, 0), yg.MapProjection("EPSG:4326", 10.0, -10.0)) as layer:
            layer.to_geotiff(filename, nodata=zero, sparse=True)
        with yg.read_raster(filename) as layer:
            assert layer.sum() == 5000 * 5000

        # now look into the tiff structure to see that all blocks are tiff
        with tifffile.TiffFile(filename) as tif:
            page = tif.pages[0] # type: ignore
            offsets = page.tags.get('StripOffsets').value # type: ignore
            byte_counts = page.tags.get('StripByteCounts').value # type: ignore

            for offset, count in zip(offsets, byte_counts):
                assert offset != 0
                assert count != 0


@pytest.mark.parametrize("dtype,zero", [
    (np.int8, 0),
    (np.float32, 0.0),
])
def test_mixed_sparse(dtype: type, zero: int | float) -> None:
    data = np.array([[i % 2] * 5000 for i in range(5000)]).astype(dtype)  # type: ignore
    with tempfile.TemporaryDirectory() as tempdir:
        filename = Path(tempdir) / "test.tif"
        with yg.from_array(data, (0, 0), yg.MapProjection("EPSG:4326", 10.0, -10.0)) as layer:
            layer.to_geotiff(filename, nodata=zero, sparse=True)
        with yg.read_raster(filename) as layer:
            # no data is nan, so convert that to 0, as otherwise
            # this test will miss things
            assert layer.nan_to_num().sum() == 5000 * 5000 / 2

        # now look into the tiff structure to see that all blocks are tiff
        with tifffile.TiffFile(filename) as tif:
            page = tif.pages[0] # type: ignore
            offsets = page.tags.get('StripOffsets').value # type: ignore
            byte_counts = page.tags.get('StripByteCounts').value # type: ignore

            for i, (offset, count) in enumerate(zip(offsets, byte_counts)):
                if i % 2:
                    assert offset != 0
                    assert count != 0
                else:
                    assert offset == 0
                    assert count == 0
