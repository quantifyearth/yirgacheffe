
import numpy as np

from .enumeration import operators as op

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

operator_map = {
	op.ADD: np.ndarray.__add__,
	op.SUB: np.ndarray.__sub__,
	op.MUL: np.ndarray.__mul__,
	op.TRUEDIV: np.ndarray.__truediv__,
	op.POW: np.ndarray.__pow__,
	op.EQ: np.ndarray.__eq__,
	op.NE: np.ndarray.__ne__,
	op.LT: np.ndarray.__lt__,
	op.LE: np.ndarray.__le__,
	op.GT: np.ndarray.__gt__,
	op.GE: np.ndarray.__ge__,
	op.AND: np.ndarray.__and__,
	op.OR: np.ndarray.__or__,
	op.LOG: np.log,
	op.LOG2: np.log2,
	op.LOG10: np.log10,
	op.EXP: np.exp,
	op.EXP2: np.exp2,
	op.CLIP: np.clip,
	op.WHERE: np.where,
	op.MIN: np.min,
	op.MAX: np.max,
	op.MINIMUM: np.minimum,
	op.MAXIMUM: np.maximum,
	op.NAN_TO_NUM: np.nan_to_num,
	op.ISIN: np.isin,
}
