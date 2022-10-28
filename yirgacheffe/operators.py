import numpy

from .window import Window

YSIZE = 1

class LayerConstant:
    def __init__(self, val):
        self.val = val

    def __str__(self):
        return str(self.val)

    def _eval(self, _index):
        return self.val


class LayerMathMixin:

    def __add__(self, other):
        return LayerOperation(self, "__add__", other)

    def __sub__(self, other):
        return LayerOperation(self, "__sub__", other)

    def __mul__(self, other):
        return LayerOperation(self, "__mul__", other)

    def __truediv__(self, other):
        return LayerOperation(self, "__truediv__", other)

    def __pow__(self, other):
        return LayerOperation(self, "__pow__", other)

    def __eq__(self, other):
        return LayerOperation(self, "__eq__", other)

    def __ne__(self, other):
        return LayerOperation(self, "__ne__", other)

    def __lt__(self, other):
        return LayerOperation(self, "__lt__", other)

    def __le__(self, other):
        return LayerOperation(self, "__le__", other)

    def __gt__(self, other):
        return LayerOperation(self, "__gt__", other)

    def __ge__(self, other):
        return LayerOperation(self, "__ge__", other)

    def _eval(self, index):
        try:
            window = self.window
            return self.read_array(0, index, window.xsize, YSIZE)
        except AttributeError:
            return self.read_array(0, index, 1, YSIZE)

    def numpy_apply(self, func, other=None):
        return LayerOperation(self, func, other)

    def shader_apply(self, func, other=None):
        return ShaderStyleOperation(self, func, other)


class LayerOperation(LayerMathMixin):

    def __init__(self, lhs, operator=None, rhs=None):
        self.lhs = lhs
        if operator:
            self.operator = operator
        if rhs is not None:
            if isinstance(rhs, (float, int)):
                self.rhs = LayerConstant(rhs)
            else:
                self.rhs = rhs

    def __str__(self):
        try:
            return f"({self.lhs} {self.operator} {self.rhs})"
        except AttributeError:
            return str(self.lhs)

    def __len__(self):
        return len(self.lhs)

    @property
    def window(self) -> Window:
        try:
            return self.lhs.window
        except AttributeError:
            # If neither side had a window attribute then
            # the operation doesn't have anything useful to
            # say, so let the exception propegate up
            return self.rhs.window

    def _eval(self, index):
        try:
            lhs = self.lhs._eval(index)
            # we want operator to fail first, before the rhs check, as we
            # support unary operations
            operator = getattr(lhs, self.operator)
            rhs = self.rhs._eval(index)
            result = operator(rhs)

            # This is currently a hurried work around for the fact that
            #   0.0 + numpy array
            # is valid, but
            #   getattr(0.0, '__add__')(numpy array)
            # returns NotImplemented
            if result.__class__ == NotImplemented.__class__:
                if self.operator in ['__add__', '__mul__']:
                    result = getattr(rhs, self.operator)(lhs)

            return result
        except TypeError: # operator not a string
            try:
                return self.operator(self.lhs._eval(index), self.rhs._eval(index))
            except AttributeError: # no rhs
                return self.operator(self.lhs._eval(index))
        except AttributeError: # no operator attr
            window = self.window
            val = self.lhs.read_array(0, index, window.xsize, YSIZE)
            return val

    def sum(self):
        total = 0.0
        window = self.window
        for yoffset in range(window.ysize):
            line = self._eval(yoffset)
            total += numpy.sum(line)
        return total

    def save(self, band):
        window = self.window
        for yoffset in range(window.ysize):
            line = self._eval(yoffset)
            try:
                band.WriteArray(line, 0, yoffset)
            except AttributeError:
                # Likely that line is a constant value
                if isinstance(line, (float, int)):
                    constline = numpy.array([[line] * window.xsize])
                    band.WriteArray(constline, 0, yoffset)
                else:
                    raise


class ShaderStyleOperation(LayerOperation):

    def _eval(self, index):
        lhs = self.lhs._eval(index)
        window = self.lhs.window
        result = numpy.empty_like(lhs)
        try:
            rhs = self.rhs._eval(index)
        except AttributeError: # no rhs
            rhs = None

        for yoffset in range(YSIZE):
            for xoffset in range(window.xsize):
                if rhs is not None:
                    result[yoffset][xoffset] = self.operator(
                        lhs[yoffset][xoffset],
                        rhs[yoffset][xoffset]
                    )
                else:
                    result[yoffset][xoffset] = self.operator(
                        lhs[yoffset][xoffset]
                    )

        return result
