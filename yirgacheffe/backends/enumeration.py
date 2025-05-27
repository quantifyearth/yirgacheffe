from enum import Enum

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
