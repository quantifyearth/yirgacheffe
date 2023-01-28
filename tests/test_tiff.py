import pickle

import numpy
import pytest

from yirgacheffe.tiff import GDALGeoTiff, TiffFileGeoTiff

@pytest.mark.parametrize(
	"cls", [GDALGeoTiff, TiffFileGeoTiff],
)
def test_basic_load(cls):
	tiff = cls("tests/testdata/small_made_with_gdal.tif")
	assert tiff.dimensions == (73, 69)
	assert tiff.geo_transform == (-59.9939862499542, 0.00089831528412, 0.0, -27.134511472128718, 0.0, -0.00089831528412)
	assert 'WGS 84' in tiff.projection
	data = tiff.read_array(0, 0, 73, 69)
	assert numpy.sum(data) == 3345.0

@pytest.mark.parametrize(
	"cls", [GDALGeoTiff, TiffFileGeoTiff],
)
def test_file_missing(cls):
	with pytest.raises(FileNotFoundError):
		_ = cls("this_file_does_not_exist.tif")

@pytest.mark.parametrize(
	"cls", [GDALGeoTiff, TiffFileGeoTiff],
)
def test_pickle(cls):
	tiff = cls("tests/testdata/small_made_with_gdal.tif")
	p = pickle.dumps(tiff)
	new_tiff = pickle.loads(p)

	assert tiff.dimensions == new_tiff.dimensions
	assert tiff.geo_transform == new_tiff.geo_transform
	assert tiff.projection == new_tiff.projection
