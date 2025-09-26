import pyproj

YSTEP = 512
MINIMUM_CHUNKS_PER_THREAD = 1

# I don't really want this here, but it's just too useful having it exposed
# This used to be a fixed string, but now it is at least programmatically generated
WGS_84_PROJECTION = pyproj.CRS.from_epsg(4326).to_wkt(version='WKT1_GDAL')
