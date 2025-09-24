from __future__ import annotations

from typing import Callable

import numpy as np
import torch

from .enumeration import operators as op
from .enumeration import dtype

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
isnan = np.isnan
isscalar = np.isscalar
full = np.full
allclose = np.allclose
remainder_op = np.ndarray.__mod__
floordiv_op = np.ndarray.__floordiv__
abs_op = np.abs
floor_op = np.floor
round_op = np.round
ceil_op = np.ceil

def conv2d_op(data, weights):
    # torch wants to process dimensions of channels of width of height
    # Which is why both the data and weights get nested into two arrays here,
    # and then we have to unpack it from that nesting.

    preped_weights = np.array([[weights]])
    conv = torch.nn.Conv2d(1, 1, weights.shape, bias=False)
    conv.weight = torch.nn.Parameter(torch.from_numpy(preped_weights))
    preped_data = torch.from_numpy(np.array([[data]]))

    res = conv(preped_data)
    return res.detach().numpy()[0][0]

def dtype_to_backend(dt):
    match dt:
        case dtype.Float32:
            return np.float32
        case dtype.Float64:
            return np.float64
        case dtype.Byte:
            return np.uint8
        case dtype.Int8:
            return np.int8
        case dtype.Int16:
            return np.int16
        case dtype.Int32:
            return np.int32
        case dtype.Int64:
            return np.int64
        case dtype.UInt8:
            return np.uint8
        case dtype.UInt16:
            return np.uint16
        case dtype.UInt32:
            return np.uint32
        case dtype.UInt64:
            return np.uint64
        case _:
            raise ValueError

def backend_to_dtype(val):
    match val:
        case np.float32:
            return dtype.Float32
        case np.float64:
            return dtype.Float64
        case np.int8:
            return dtype.Int8
        case np.int16:
            return dtype.Int16
        case np.int32:
            return dtype.Int32
        case np.int64:
            return dtype.Int64
        case np.uint8:
            return dtype.Byte
        case np.uint16:
            return dtype.UInt16
        case np.uint32:
            return dtype.UInt32
        case np.uint64:
            return dtype.UInt64
        case _:
            raise ValueError

def astype_op(data, datatype):
    return data.astype(dtype_to_backend(datatype))

operator_map: dict[op, Callable] = {
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
    op.REMAINDER: np.ndarray.__mod__,
    op.FLOORDIV: np.ndarray.__floordiv__,
    op.CONV2D: conv2d_op,
    op.ABS: np.abs,
    op.ASTYPE: astype_op,
    op.FLOOR: np.floor,
    op.ROUND: np.round,
    op.CEIL: np.ceil,
    op.ISNAN: np.isnan,
}
