import numpy as np

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
        return LayerOperation(self, np.ndarray.__add__, other)

    def __sub__(self, other):
        return LayerOperation(self, np.ndarray.__sub__, other)

    def __mul__(self, other):
        return LayerOperation(self, np.ndarray.__mul__, other)

    def __truediv__(self, other):
        return LayerOperation(self, np.ndarray.__truediv__, other)

    def __pow__(self, other):
        return LayerOperation(self, np.ndarray.__pow__, other)

    def __eq__(self, other):
        return LayerOperation(self, np.ndarray.__eq__, other)

    def __ne__(self, other):
        return LayerOperation(self, np.ndarray.__ne__, other)

    def __lt__(self, other):
        return LayerOperation(self, np.ndarray.__lt__, other)

    def __le__(self, other):
        return LayerOperation(self, np.ndarray.__le__, other)

    def __gt__(self, other):
        return LayerOperation(self, np.ndarray.__gt__, other)

    def __ge__(self, other):
        return LayerOperation(self, np.ndarray.__ge__, other)

    def _eval(self, index, step, target_window=None):
        try:
            window = self.window
            return self.read_array(0, index, window.xsize, step)
        except AttributeError:
            return self.read_array(0, index, target_window.xsize if target_window else 1, step)

    def numpy_apply(self, func, other=None):
        return LayerOperation(self, func, other)

    def shader_apply(self, func, other=None):
        return ShaderStyleOperation(self, func, other)

    def save(self, destination_layer, and_sum=False):
        return LayerOperation(self).save(destination_layer, and_sum)

    def sum(self):
        return LayerOperation(self).sum()

    def min(self):
        return LayerOperation(self).min()

    def max(self):
        return LayerOperation(self).max()


class LayerOperation(LayerMathMixin):

    def __init__(self, lhs, operator=None, rhs=None):
        self.ystep = YSTEP

        if lhs is None:
            raise ValueError("LHS on operation should not be none")
        self.lhs = lhs

        self.operator = operator

        if rhs is not None:
            if isinstance(rhs, (float, int)):
                self.rhs = LayerConstant(rhs)
            elif isinstance(rhs, (np.ndarray)):
                if rhs.shape == ():
                    self.rhs = LayerConstant(rhs.item())
                else:
                    raise ValueError("Numpy arrays are no allowed")
            else:
                self.rhs = rhs
        else:
            self.rhs = None

    def __str__(self):
        try:
            return f"({self.lhs} {self.operator} {self.rhs})"
        except AttributeError:
            try:
                return f"({self.operator} {self.lhs})"
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

    def _eval(self, index, step): # pylint: disable=W0221
        lhs_data = self.lhs._eval(index, step)

        if self.operator is None:
            return lhs_data

        if self.rhs is not None:
            rhs_data = self.rhs._eval(index, step)
            return self.operator(lhs_data, rhs_data)
        else:
            return self.operator(lhs_data)

    def sum(self):
        res = 0.0
        computation_window = self.window
        for yoffset in range(0, computation_window.ysize, self.ystep):
            step=self.ystep
            if yoffset+step > computation_window.ysize:
                step = computation_window.ysize - yoffset
            chunk = self._eval(yoffset, step)
            res += np.sum(chunk)
        return res

    def min(self):
        res = None
        computation_window = self.window
        for yoffset in range(0, computation_window.ysize, self.ystep):
            step=self.ystep
            if yoffset+step > computation_window.ysize:
                step = computation_window.ysize - yoffset
            chunk = self._eval(yoffset, step)
            chunk_min = np.min(chunk)
            if (res is None) or (res > chunk_min):
                res = chunk_min
        return res

    def max(self):
        res = None
        computation_window = self.window
        for yoffset in range(0, computation_window.ysize, self.ystep):
            step=self.ystep
            if yoffset+step > computation_window.ysize:
                step = computation_window.ysize - yoffset
            chunk = self._eval(yoffset, step)
            chunk_max = np.max(chunk)
            if (res is None) or (chunk_max > res):
                res = chunk_max
        return res

    def save(self, destination_layer, and_sum, callback=None):
        """
        Calling save will write the output of the operation to the provied layer.
        If you provide sum as true it will additionall compute the sum and return that.
        """

        if destination_layer is None:
            raise ValueError("Layer is required")
        try:
            band = destination_layer._dataset.GetRasterBand(1)
        except AttributeError as exc:
            raise ValueError("Layer must be a raster backed layer") from exc

        computation_window = self.window
        destination_window = destination_layer.window

        if (computation_window.xsize != destination_window.xsize) \
                or (computation_window.ysize != destination_window.ysize):
            raise ValueError("Destination raster window size does not match input raster window size.")

        total = 0.0

        for yoffset in range(0, computation_window.ysize, self.ystep):
            if callback:
                callback(yoffset / computation_window.ysize)
            step=self.ystep
            if yoffset+step > computation_window.ysize:
                step = computation_window.ysize - yoffset
            chunk = self._eval(yoffset, step)
            if isinstance(chunk, (float, int)):
                chunk = np.full((step, destination_window.xsize), chunk)
            band.WriteArray(
                chunk,
                destination_window.xoff,
                yoffset + destination_window.yoff,
            )
            if and_sum:
                total += np.sum(chunk)
        if callback:
            callback(1.0)

        return total if and_sum else None


class ShaderStyleOperation(LayerOperation):

    def _eval(self, index, step):
        lhs_data = self.lhs._eval(index, step, self.window)
        if self.rhs is not None:
            rhs_data = self.rhs._eval(index, step, self.window)
        else:
            rhs_data = None

        # Constant results make this a bit messier. Might in future
        # be nicer to promote them to arrays sooner?
        if isinstance(lhs_data, (int, float)):
            if rhs_data is None:
                return self.operator(lhs_data)
            if isinstance(rhs_data, (int, float)):
                return self.operator(lhs_data, rhs_data)
            else:
                result = np.empty_like(rhs_data)
        else:
            result = np.empty_like(lhs_data)

        print(self.window)
        print(self.lhs._window)
        if self.rhs is not None:
            print(self.rhs._window)
        print(lhs_data)
        if self.rhs is not None:
            print(self.rhs)
        print(rhs_data)
        print(result)

        window = self.window
        for yoffset in range(step):
            for xoffset in range(window.xsize):
                try:
                    lhs_val = lhs_data[yoffset][xoffset]
                except TypeError:
                    lhs_val = lhs_data
                if rhs_data is not None:
                    try:
                        rhs_val = rhs_data[yoffset][xoffset]
                    except TypeError:
                        rhs_val = rhs_data
                    result[yoffset][xoffset] = self.operator(lhs_val, rhs_val)
                else:
                    result[yoffset][xoffset] = self.operator(lhs_val)

        return result
