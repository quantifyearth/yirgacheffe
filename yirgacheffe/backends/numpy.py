
import numpy as np

array_t = np.ndarray
float_t = np.float64

promote = lambda a: a
demote_array = lambda a: a
demote_scalar = lambda a: a
eval_op = lambda a: a

add_op = np.ndarray.__add__
sub_op = np.ndarray.__sub__
mul_op = np.ndarray.__mul__
truediv_op = np.ndarray.__truediv__
pow_op = np.ndarray.__pow__
eq_op = np.ndarray.__eq__
ne_op = np.ndarray.__ne__
lt_op = np.ndarray.__lt__
le_op = np.ndarray.__le__
gt_op = np.ndarray.__gt__
ge_op = np.ndarray.__ge__
and_op = np.ndarray.__and__
or_op = np.ndarray.__or__
nan_to_num = np.nan_to_num
isin = np.isin
log = np.log
log2 = np.log2
log10 = np.log10
exp = np.exp
exp2 = np.exp2
clip = np.clip
where = np.where
min_op = np.min
max_op = np.max
maximum = np.maximum
minimum = np.minimum
zeros = np.zeros
pad = np.pad
sum_op = lambda a: np.sum(a.astype(np.float64))
isscalar = np.isscalar
full = np.full
allclose = np.allclose
