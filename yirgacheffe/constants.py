YSTEP = 512
MINIMUM_CHUNKS_PER_THREAD = 1

# For the MLX backend we have to be careful with how much of the AST we evaluate at once, as if
# we do too much in a single MLX kernel we fail due to watchdog timeouts. As such, each operations
# has a cost, and as the tree is evaluated, whenever we hit this cost value we'll force the
# current sub-expression to be evaluated.
EVALUATION_THRESHOLD = 16 * 1024

# Both GDAL and MLX assume that there is one instance running and it has the right to use
# all the memory it can. In general Yirgacheffe's chunking and it's own caching is what we shou
# be relying on, so we set some limits here. These are applied before each calculation, and so
# in theory they can be tweaked if necessary on demand.
GDAL_CACHE_LIMIT = 1 * 1024 * 1024 * 1024
MLX_CACHE_LIMIT = 1 * 1024 * 1024 * 1024
