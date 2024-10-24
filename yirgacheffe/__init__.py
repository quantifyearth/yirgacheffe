from osgeo import gdal
try:
    from importlib import metadata
    __version__ = metadata.version(__name__)
except ModuleNotFoundError:
    __version__ = "unknown"

gdal.UseExceptions()

# I don't really want this here, but it's just too useful having it exposed
WGS_84_PROJECTION = 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,'\
    'AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0],'\
    'UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AXIS["Latitude",NORTH],'\
    'AXIS["Longitude",EAST],AUTHORITY["EPSG","4326"]]'

# For legacy reasons [facepalm]
WSG_84_PROJECTION = WGS_84_PROJECTION
