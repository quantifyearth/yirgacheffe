
import numpy as np
import mlx.core as mx # pylint: disable=E0001,E0611

array_t = mx.array
float_t = mx.float32

promote = mx.array
demote_array = np.asarray
demote_scalar = np.float64

eval_op = mx.eval

add_op = mx.array.__add__
sub_op = mx.array.__sub__
truediv_op = mx.array.__truediv__
pow_op = mx.array.__pow__
eq_op = mx.array.__eq__
ne_op = mx.array.__ne__
lt_op = mx.array.__lt__
le_op = mx.array.__le__
gt_op = mx.array.__gt__
ge_op = mx.array.__ge__
and_op = mx.array.__and__
or_op = mx.array.__or__
log = mx.log
log2 = mx.log2
log10 = mx.log10
exp = mx.exp
clip = mx.clip
where = mx.where
min_op = mx.min
max_op = mx.max
maximum = mx.maximum
minimum = mx.minimum
zeros = mx.zeros
pad = mx.pad
isscalar = np.isscalar
full = mx.full
allclose = mx.allclose

def sum_op(a):
    # There are weird issues around how MLX overflows int8, so just promote the data ahead of summing
    match a.dtype:
        case mx.int8:
            return mx.sum(a.astype(mx.int32))
        case mx.uint8:
            return mx.sum(a.astype(mx.uint32))
        case _:
            return mx.sum(a)

def _is_float(x):
    if isinstance(x, float):
        return True
    try:
        np_floats = [np.dtype('float16'), np.dtype('float32'), np.dtype('float64')]
        if x.dtype in np_floats:
            return True
        match x.dtype:
            case mx.float32 | mx.float64:
                return True
            case _:
                return False
    except AttributeError:
        return False

def mul_op(a, b):
    # numpy will promote an operation between float and int to float, whereas it looks like mlx does the inverse
    # and so for consistency with the numpy path, we do some fiddling here if necessary
    if _is_float(b):
        print("okay")
        match a.dtype:
            case mx.int8 | mx.int32 | mx.uint8 | mx.uint32:
                a = a.astype(mx.float32)
            case mx.int64 | mx.uint64:
                a = a.astype(mx.float64)
            case _:
                pass
    return mx.multiply(a, b, stream=mx.gpu)

def exp2(a):
    mx.eval(a)
    return promote(np.exp2(a))

def nan_to_num(a, nan, posinf, neginf, copy): # pylint: disable=W0613
    return mx.nan_to_num(a, float(nan), posinf, neginf)

def isin(a, test_elements):
    # There is no `isin` on MLX currently, so we need to fallback to CPU behaviour here
    # https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html#using-shape-strides
    mx.eval(a)
    return promote(np.isin(a, test_elements))
