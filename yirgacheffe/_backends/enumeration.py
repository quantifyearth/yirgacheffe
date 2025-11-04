from __future__ import annotations

from enum import Enum

import numpy as np
from osgeo import gdal

class operators(Enum):
    ADD = 1
    SUB = 2
    MUL = 3
    TRUEDIV = 4
    POW = 5
    EQ = 6
    NE = 7
    LT = 8
    LE = 9
    GT = 10
    GE = 11
    AND = 12
    OR = 13
    LOG = 14
    LOG2 = 15
    LOG10 = 16
    EXP = 17
    EXP2 = 18
    CLIP = 19
    WHERE = 20
    MIN = 21
    MAX = 22
    SUM = 23
    MINIMUM = 24
    MAXIMUM = 25
    NAN_TO_NUM = 26
    ISIN = 27
    REMAINDER = 28
    FLOORDIV = 29
    CONV2D = 30
    ABS = 31
    ASTYPE = 32
    FLOOR = 33
    ROUND = 34
    CEIL = 35
    ISNAN = 36
    RADD = 37
    RSUB = 38
    RMUL = 39
    RTRUEDIV = 40
    RFLOORDIV = 41
    RREMAINDER = 42
    RPOW = 43

class dtype(Enum):
    """Represents the type of data returned by a layer.

    This enumeration defines the valid data types supported by Yirgacheffe, and is
    what is returned by  calling `datatype` on a layer or expression, and can be
    passed to `astype` to convert values between types.

    Attributes:
        Float32: 32 bit floating point value
        Float64: 64 bit floating point value
        Byte: Unsigned 8 bit integer value
        Int8: Signed 8 bit integer value
        Int16: Signed 16 bit integer value
        Int32: Signed 32 bit integer value
        Int64: Signed 64 bit integer value
        UInt8: Unsigned 8 bit integer value
        UInt16: Unsigned 16 bit integer value
        UInt32: Unsigned 32 bit integer value
        UInt64: Unsigned 64 bit integer value
    """

    Float32 = gdal.GDT_Float32
    Float64 = gdal.GDT_Float64
    Byte = gdal.GDT_Byte
    Int8 = gdal.GDT_Int8
    Int16 = gdal.GDT_Int16
    Int32 = gdal.GDT_Int32
    Int64 = gdal.GDT_Int64
    UInt8 = gdal.GDT_Byte
    UInt16 = gdal.GDT_UInt16
    UInt32 = gdal.GDT_UInt32
    UInt64 = gdal.GDT_UInt64

    def to_gdal(self) -> int:
        """Coverts the Yirgacheffe data type to the corresponding GDAL constant.

        Returns:
            An integer with the corresponding GDAL type constant.
        """
        return self.value

    @classmethod
    def of_gdal(cls, val: int) -> dtype:
        """Generates the Yirgacheffe data type value from the correspondiong GDAL value.

        Returns:
            A Yirgacheffe data type value.
        """
        return cls(val)

    @classmethod
    def of_array(cls, val: np.ndarray) -> dtype:
        """Generates the Yirgacheffe data type value from a numpy array.

        Returns:
            A Yirgacheffe data type value.
        """
        match val.dtype:
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
                return dtype.UInt8
            case np.uint16:
                return dtype.UInt16
            case np.uint32:
                return dtype.UInt32
            case np.uint64:
                return dtype.UInt64
            case _:
                raise ValueError

    def sizeof(self) -> int:
        """Returns the number of bytes used to store the data type.

        Returns:
            The number of bytes used to store the data type.
        """
        match self:
            case dtype.Byte | dtype.Int8 | dtype.UInt8:
                return 1
            case dtype.Int16 | dtype.UInt16:
                return 2
            case dtype.Int32 | dtype.UInt32 | dtype.Float32:
                return 4
            case dtype.Int64 | dtype.UInt64 | dtype.Float64:
                return 8
            case _:
                raise ValueError
