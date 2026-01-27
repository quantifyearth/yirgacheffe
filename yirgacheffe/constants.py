import pyproj

YSTEP = 512
MINIMUM_CHUNKS_PER_THREAD = 1

# Both GDAL and MLX assume that there is one instance running and it has the right to use
# all the memory it can. In general Yirgacheffe's chunking and it's own caching is what we shou
# be relying on, so we set some limits here. These are applied before each calculation, and so
# in theory they can be tweaked if necessary on demand.
GDAL_CACHE_LIMIT = 1 * 1024 * 1024 * 1024
MLX_CACHE_LIMIT = 1 * 1024 * 1024 * 1024

# I don't really want this here, but it's just too useful having it exposed
# This used to be a fixed string, but now it is at least programmatically generated
WGS_84_PROJECTION = pyproj.CRS.from_epsg(4326).to_wkt(version='WKT1_GDAL')
