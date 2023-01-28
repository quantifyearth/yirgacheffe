import numpy

from .window import Window

YSTEP = 512

class LayerConstant:
    def __init__(self, val):
        self.val = val

    def __str__(self):
        return str(self.val)

    def _eval(self, _index, _step):
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

    def _eval(self, index, step):
        try:
            window = self.window
            return self.read_array(0, index, window.xsize, step)
        except AttributeError:
            return self.read_array(0, index, 1, step)

    def numpy_apply(self, func, other=None):
        return LayerOperation(self, func, other)

    def shader_apply(self, func, other=None):
        return ShaderStyleOperation(self, func, other)

    def save(self, destination_layer, and_sum=False):
        return LayerOperation(self).save(destination_layer, and_sum)

    def sum(self):
        return LayerOperation(self).sum()


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
            # say, so let the exception propagate up
            return self.rhs.window

    def _eval(self, index, step):
        try:
            lhs = self.lhs._eval(index, step)
            # we want operator to fail first, before the rhs check, as we
            # support unary operations
            operator = getattr(lhs, self.operator)
            rhs = self.rhs._eval(index, step)
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
                return self.operator(lhs, self.rhs._eval(index,step))
            except AttributeError: # no rhs
                return self.operator(lhs)
        except AttributeError: # no operator attr
            return lhs

    def sum(self):
        total = 0.0
        computation_window = self.window
        for yoffset in range(0, computation_window.ysize, YSTEP):
            step=YSTEP
            if yoffset+step > computation_window.ysize:
                step = computation_window.ysize - yoffset
            chunk = self._eval(yoffset, step)
            total += numpy.sum(chunk)
        return total

    def save(self, destination_layer, and_sum=False):
        """
        Calling save will write the output of the operation to the provied layer.
        If you provide sum as true it will additionall compute the sum and return that.
        """

        if destination_layer is None:
            raise ValueError("Layer is required")
        try:
            band = destination_layer._tiff._file.GetRasterBand(1)
        except AttributeError as exc:
            raise ValueError("Layer must be a raster backed layer") from exc

        computation_window = self.window
        destination_window = destination_layer.window

        total = 0.0

        for yoffset in range(0, computation_window.ysize, YSTEP):
            step=YSTEP
            if yoffset+step > computation_window.ysize:
                step = computation_window.ysize - yoffset
            chunk = self._eval(yoffset, step)
            if isinstance(chunk, (float, int)):
                chunk = numpy.full((step, destination_window.xsize), chunk)
            band.WriteArray(
                chunk,
                destination_window.xoff,
                yoffset + destination_window.yoff,
            )
            if and_sum:
                total += numpy.sum(chunk)

        return total if and_sum else None


class ShaderStyleOperation(LayerOperation):

    def _eval(self, index, step):
        lhs = self.lhs._eval(index, step)
        try:
            rhs = self.rhs._eval(index, step)
        except AttributeError: # no rhs
            rhs = None

        # Costant results make this a bit messier. Might in future
        # be nicer to promote them to arrays sooner?
        if isinstance(lhs, (int, float)):
            if rhs is None:
                return self.operator(lhs)
            if isinstance(rhs, (int, float)):
                return self.operator(lhs, rhs)
            else:
                result = numpy.empty_like(rhs)
        else:
            result = numpy.empty_like(lhs)

        window = self.window
        for yoffset in range(step):
            for xoffset in range(window.xsize):
                try:
                    lhs_val = lhs[yoffset][xoffset]
                except TypeError:
                    lhs_val = lhs
                if rhs is not None:
                    try:
                        rhs_val = rhs[yoffset][xoffset]
                    except TypeError:
                        rhs_val = rhs
                    result[yoffset][xoffset] = self.operator(lhs_val, rhs_val)
                else:
                    result[yoffset][xoffset] = self.operator(lhs_val)

        return result
