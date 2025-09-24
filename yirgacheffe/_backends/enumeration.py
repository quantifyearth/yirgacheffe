from enum import Enum

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

    def to_gdal(self):
        return self.value

    @classmethod
    def of_gdal(cls, val):
        return cls(val)
