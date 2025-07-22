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

class dtype(Enum):
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
