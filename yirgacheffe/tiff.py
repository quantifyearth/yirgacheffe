import os
from typing import Any, Optional, Tuple

from osgeo import gdal
import tifffile

# In testing, using POSIX file I/O on Linux/macOS was faster with tifffile than using Python's standard I/O,
# but that API didn't work so well on Windows, so we try to detect it here based on whether we have the
# right flags available to seek
USE_POSIX = hasattr(os, 'SEEK_DATA')

class FDWrapper:
	"""A class that wraps a posix file descriptor for use with TiffFile"""
	def __init__(self, fn):
		self.fd = os.open(fn, os.O_RDONLY)

	def seek(self, offset, rel=os.SEEK_DATA):
		os.lseek(self.fd, offset, rel)

	def read(self, length):
		return os.read(self.fd, length)

	def tell(self):
		return os.lseek(self.fd, 0, os.SEEK_CUR)


class _GeoTiff:
	def __init__(self, filename: Optional[str]):
		self.filename = filename
		# force a file open, as user of class will expect it
		if self.filename is not None:
			_ = self._file

	def __getstate__(self):
		if self.filename is None:
			raise ValueError("Cannot pickle memory backed TIFF")
		return {'filename': self.filename}

	@property
	def dimensions(self) -> Tuple[int, int]:
		raise NotImplementedError("Please instantiate GDALGeoTiff or TIffFileGeoTiff")

	@property
	def geo_transform(self) -> Tuple[float, float, float, float, float, float]:
		raise NotImplementedError("Please instantiate GDALGeoTiff or TIffFileGeoTiff")

	@property
	def projection(self) -> str:
		raise NotImplementedError("Please instantiate GDALGeoTiff or TIffFileGeoTiff")

	@property
	def datatype(self) -> int:
		raise NotImplementedError("Please instantiate GDALGeoTiff or TIffFileGeoTiff")


class GDALGeoTiff(_GeoTiff):

	@classmethod
	def from_dataset(cls, dataset: gdal.Dataset):
		tiff = cls(None)
		tiff._dataset_cache = dataset
		return tiff

	@property
	def _file(self):
		if not hasattr(self, '_dataset_cache'):
			self._dataset_cache = gdal.Open(self.filename)
			if self._dataset_cache is None:
				raise FileNotFoundError(self.filename)
		return self._dataset_cache

	@property
	def dimensions(self) -> Tuple[int, int]:
		dataset = self._file
		return dataset.RasterXSize, dataset.RasterYSize

	@property
	def geo_transform(self) -> Tuple[float, float, float, float, float, float]:
		return self._file.GetGeoTransform()

	@property
	def projection(self) -> str:
		return self._file.GetProjection()

	@property
	def datatype(self) -> int:
		return self._file.GetRasterBand(1).DataType

	def read_array(self, xoffset: int, yoffset: int, xsize: int, ysize: int) -> Any:
		return self._file.GetRasterBand(1).ReadAsArray(xoffset, yoffset, xsize, ysize)


class TiffFileGeoTiff(_GeoTiff):
	@property
	def _file(self):
		if not hasattr(self, '_tiff_cache'):
			if USE_POSIX:
				self._tiff_cache = tifffile.TiffFile(FDWrapper(self.filename))
			else:
				self._tiff_cache = tifffile.TiffFile(self.filename)
		return self._tiff_cache

	@property
	def dimensions(self) -> Tuple[int, int]:
		dim = self._file.series[0].sizes
		return dim["width"], dim["height"]

	@property
	def geo_transform(self) -> Tuple[float, float, float, float, float, float]:
		geotiff_metadata = self._file.geotiff_metadata
		scale = geotiff_metadata['ModelPixelScale']
		offsets = geotiff_metadata['ModelTiepoint']
		return (offsets[3], scale[0], 0.0, offsets[4], 0.0, scale[1] * -1)

	@property
	def projection(self) -> str:
		geotiff_metadata = self._file.geotiff_metadata
		return geotiff_metadata['GeogCitationGeoKey'] # this is weaker than GDAL

	def read_array(self, xoffset: int, yoffset: int, xsize: int, ysize: int) -> Any:
		page = self._file.pages[0]




