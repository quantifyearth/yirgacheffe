YSTEP = 512
MINIMUM_CHUNKS_PER_THREAD = 1

# Both GDAL and MLX assume that there is one instance running and it has the right to use
# all the memory it can. In general Yirgacheffe's chunking and it's own caching is what we shou
# be relying on, so we set some limits here. These are applied before each calculation, and so
# in theory they can be tweaked if necessary on demand.
GDAL_CACHE_LIMIT = 1 * 1024 * 1024 * 1024
MLX_CACHE_LIMIT = 1 * 1024 * 1024 * 1024
