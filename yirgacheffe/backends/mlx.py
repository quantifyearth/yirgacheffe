
import numpy as np
import mlx.core as mx # pylint: disable=E0001,E0611

array_t = mx.array
float_t = mx.float32

promote = mx.array
demote_array = lambda a: np.array(a, copy=False)
demote_scalar = np.float64

eval_op = mx.eval

add_op = mx.array.__add__
sub_op = mx.array.__sub__
mul_op = lambda a, b: mx.multiply(a, b, stream=mx.gpu)
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
nan_to_num = mx.nan_to_num
isin = mx.isin
log = mx.log
log2 = mx.log2
log10 = mx.log10
exp = mx.exp
exp2 = mx.exp2
clip = mx.clip
where = mx.where
min_op = mx.min
max_op = mx.max
maximum = mx.maximum
minimum = mx.minimum
zeros = mx.zeros
pad = mx.pad
sum_op = mx.sum
isscalar = mx.isscalar
