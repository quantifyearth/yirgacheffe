import logging
import math
import multiprocessing
import sys
import time
import types
from enum import Enum
from multiprocessing import Semaphore, Process
from multiprocessing.managers import SharedMemoryManager
from typing import Optional

import numpy as np
from osgeo import gdal
from dill import dumps, loads # type: ignore

from . import constants
from .rounding import are_pixel_scales_equal_enough, round_up_pixels, round_down_pixels
from .window import Area, PixelScale, Window
from ._backends import backend
from ._backends.enumeration import operators as op
from ._backends.enumeration import dtype as DataType

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

class WindowOperation(Enum):
    NONE = 1
    UNION = 2
    INTERSECTION = 3
    LEFT = 4
    RIGHT = 5

class LayerConstant:
    def __init__(self, val):
        self.val = val

    def __str__(self):
        return str(self.val)

    def _eval(self, _area, _index, _step, _target_window):
        return self.val


class LayerMathMixin:

    def __add__(self, other):
        return LayerOperation(self, op.ADD, other, window_op=WindowOperation.UNION)

    def __sub__(self, other):
        return LayerOperation(self, op.SUB, other, window_op=WindowOperation.UNION)

    def __mul__(self, other):
        return LayerOperation(self, op.MUL, other, window_op=WindowOperation.INTERSECTION)

    def __truediv__(self, other):
        return LayerOperation(self, op.TRUEDIV, other, window_op=WindowOperation.INTERSECTION)

    def __floordiv__(self, other):
        return LayerOperation(self, op.FLOORDIV, other, window_op=WindowOperation.INTERSECTION)

    def __mod__(self, other):
        return LayerOperation(self, op.REMAINDER, other, window_op=WindowOperation.INTERSECTION)

    def __pow__(self, other):
        return LayerOperation(self, op.POW, other, window_op=WindowOperation.UNION)

    def __eq__(self, other):
        return LayerOperation(self, op.EQ, other, window_op=WindowOperation.INTERSECTION)

    def __ne__(self, other):
        return LayerOperation(self, op.NE, other, window_op=WindowOperation.UNION)

    def __lt__(self, other):
        return LayerOperation(self, op.LT, other, window_op=WindowOperation.UNION)

    def __le__(self, other):
        return LayerOperation(self, op.LE, other, window_op=WindowOperation.UNION)

    def __gt__(self, other):
        return LayerOperation(self, op.GT, other, window_op=WindowOperation.UNION)

    def __ge__(self, other):
        return LayerOperation(self, op.GE, other, window_op=WindowOperation.UNION)

    def __and__(self, other):
        return LayerOperation(self, op.AND, other, window_op=WindowOperation.INTERSECTION)

    def __or__(self, other):
        return LayerOperation(self, op.OR, other, window_op=WindowOperation.UNION)

    def _eval(self, area, index, step, target_window=None):
        try:
            window = self.window if target_window is None else target_window
            return self.read_array_for_area(area, 0, index, window.xsize, step)
        except AttributeError:
            return self.read_array_for_area(area, 0, index, target_window.xsize if target_window else 1, step)

    def nan_to_num(self, nan=0, posinf=None, neginf=None):
        return LayerOperation(
            self,
            op.NAN_TO_NUM,
            window_op=WindowOperation.NONE,
            copy=False,
            nan=nan,
            posinf=posinf,
            neginf=neginf,
        )

    def isin(self, test_elements):
        return LayerOperation(
            self,
            op.ISIN,
            window_op=WindowOperation.NONE,
            test_elements=test_elements,
        )

    def abs(self):
        return LayerOperation(
            self,
            op.ABS,
            window_op=WindowOperation.NONE,
        )

    def floor(self):
        return LayerOperation(
            self,
            op.FLOOR,
            window_op=WindowOperation.NONE,
        )

    def round(self):
        return LayerOperation(
            self,
            op.ROUND,
            window_op=WindowOperation.NONE,
        )

    def ceil(self):
        return LayerOperation(
            self,
            op.CEIL,
            window_op=WindowOperation.NONE,
        )

    def log(self):
        return LayerOperation(
            self,
            op.LOG,
            window_op=WindowOperation.NONE,
        )

    def log2(self):
        return LayerOperation(
            self,
            op.LOG2,
            window_op=WindowOperation.NONE,
        )

    def log10(self):
        return LayerOperation(
            self,
            op.LOG10,
            window_op=WindowOperation.NONE,
        )

    def exp(self):
        return LayerOperation(
            self,
            op.EXP,
            window_op=WindowOperation.NONE,
        )

    def exp2(self):
        return LayerOperation(
            self,
            op.EXP2,
            window_op=WindowOperation.NONE,
        )

    def clip(self, min=None, max=None): # pylint: disable=W0622
        # In the numpy 1 API np.clip(array) used a_max, a_min arguments and array.clip() used max and min as arguments
        # In numpy 2 they moved so that max and min worked on both, but still support a_max, and a_min on np.clip.
        # For now I'm only going to support the newer max/min everywhere notion, but I have to internally call
        # a_max, a_min so that yirgacheffe can work on older numpy installs.
        return LayerOperation(
            self,
            op.CLIP,
            window_op=WindowOperation.NONE,
            a_min=min,
            a_max=max,
        )

    def conv2d(self, weights):
        # A set of limitations that are just down to implementation time restrictions
        weights_width, weights_height = weights.shape
        if weights_width != weights_height:
            raise ValueError("Currently only square matrixes are supported for weights")
        padding = (weights_width - 1) / 2
        if padding != int(padding):
            raise ValueError("Currently weights dimensions must be odd")

        return LayerOperation(
            self,
            op.CONV2D,
            window_op=WindowOperation.NONE,
            buffer_padding=padding,
            weights=weights.astype(np.float32),
        )

    def numpy_apply(self, func, other=None):
        return LayerOperation(self, func, other)

    def shader_apply(self, func, other=None):
        return ShaderStyleOperation(self, func, other)

    def save(self, destination_layer, and_sum=False, callback=None, band=1):
        return LayerOperation(self).save(destination_layer, and_sum, callback, band)

    def parallel_save(self, destination_layer, and_sum=False, callback=None, parallelism=None, band=1):
        return LayerOperation(self).parallel_save(destination_layer, and_sum, callback, parallelism, band)

    def parallel_sum(self, callback=None, parallelism=None, band=1):
        return LayerOperation(self).parallel_sum(callback, parallelism, band)

    def sum(self):
        return LayerOperation(self).sum()

    def min(self):
        return LayerOperation(self).min()

    def max(self):
        return LayerOperation(self).max()

    def astype(self, datatype):
        return LayerOperation(
            self,
            op.ASTYPE,
            window_op=WindowOperation.NONE,
            datatype=datatype
        )


class LayerOperation(LayerMathMixin):

    @staticmethod
    def where(cond, a, b):
        return LayerOperation(
            cond,
            op.WHERE,
            rhs=a,
            other=b
        )

    @staticmethod
    def maximum(a, b):
        return LayerOperation(
            a,
            op.MAXIMUM,
            b,
            window_op=WindowOperation.UNION,
        )

    @staticmethod
    def minimum(a, b):
        return LayerOperation(
            a,
            op.MINIMUM,
            rhs=b,
            window_op=WindowOperation.UNION,
        )

    def __init__(
        self,
        lhs,
        operator=None,
        rhs=None,
        other=None,
        window_op=WindowOperation.NONE,
        buffer_padding=0,
        **kwargs
    ):
        self.ystep = constants.YSTEP
        self.kwargs = kwargs
        self.window_op = window_op
        self.buffer_padding = buffer_padding

        if lhs is None:
            raise ValueError("LHS on operation should not be none")
        self.lhs = lhs

        self.operator = operator

        if rhs is not None:
            if backend.isscalar(rhs):
                self.rhs = LayerConstant(rhs)
            elif isinstance(rhs, (backend.array_t)):
                if rhs.shape == ():
                    self.rhs = LayerConstant(rhs.item())
                else:
                    raise ValueError("Numpy arrays are no allowed")
            else:
                if not are_pixel_scales_equal_enough([lhs.pixel_scale, rhs.pixel_scale]):
                    raise ValueError("Not all layers are at the same pixel scale")
                self.rhs = rhs
        else:
            self.rhs = None

        if other is not None:
            if backend.isscalar(other):
                self.other = LayerConstant(other)
            elif isinstance(other, (backend.array_t)):
                if other.shape == ():
                    self.rhs = LayerConstant(other.item())
                else:
                    raise ValueError("Numpy arrays are no allowed")
            else:
                if not are_pixel_scales_equal_enough([lhs.pixel_scale, other.pixel_scale]):
                    raise ValueError("Not all layers are at the same pixel scale")
                self.other = other
        else:
            self.other = None

    def __str__(self) -> str:
        try:
            return f"({self.lhs} {self.operator} {self.rhs})"
        except AttributeError:
            try:
                return f"({self.operator} {self.lhs})"
            except AttributeError:
                return str(self.lhs)

    def __len__(self) -> int:
        return len(self.lhs)

    def __getstate__(self) -> object:
        odict = self.__dict__.copy()
        if isinstance(self.operator, types.LambdaType):
            odict['operator_dill'] = dumps(self.operator)
            del odict['operator']
        return odict

    def __setstate__(self, state) -> None:
        if 'operator_dill' in state:
            state['operator'] = loads(state['operator_dill'])
            del state['operator_dill']
        self.__dict__.update(state)

    @property
    def area(self) -> Optional[Area]:
        # The type().__name__ here is to avoid a circular import dependancy
        lhs_area = self.lhs.area if not type(self.lhs).__name__ == "ConstantLayer" else None
        try:
            rhs_area = self.rhs.area if not type(self.rhs).__name__ == "ConstantLayer" else None # type: ignore[return-value] # pylint: disable=C0301
        except AttributeError:
            rhs_area = None
        try:
            other_area = self.other.area if not type(self.other).__name__ == "ConstantLayer" else None # type: ignore[return-value] # pylint: disable=C0301
        except AttributeError:
            other_area = None

        all_areas = []
        if lhs_area is not None:
            all_areas.append(lhs_area)
        if rhs_area is not None:
            all_areas.append(rhs_area)
        if other_area is not None:
            all_areas.append(other_area)

        match self.window_op:
            case WindowOperation.NONE:
                return all_areas[0]
            case WindowOperation.LEFT:
                return lhs_area
            case WindowOperation.RIGHT:
                assert rhs_area is not None
                return rhs_area
            case WindowOperation.INTERSECTION:
                intersection = Area(
                    left=max(x.left for x in all_areas),
                    top=min(x.top for x in all_areas),
                    right=min(x.right for x in all_areas),
                    bottom=max(x.bottom for x in all_areas)
                )
                if (intersection.left >= intersection.right) or (intersection.bottom >= intersection.top):
                    raise ValueError('No intersection possible')
                return intersection
            case WindowOperation.UNION:
                return Area(
                    left=min(x.left for x in all_areas),
                    top=max(x.top for x in all_areas),
                    right=max(x.right for x in all_areas),
                    bottom=min(x.bottom for x in all_areas)
                )
            case _:
                assert False, "Should not be reached"

    @property
    def pixel_scale(self) -> PixelScale:
        # Because we test at construction that pixel scales for RHS/other are roughly equal,
        # I believe this should be sufficient...
        try:
            pixel_scale = self.lhs.pixel_scale
        except AttributeError:
            pixel_scale = None

        if pixel_scale is None:
            return self.rhs.pixel_scale
        return pixel_scale

    @property
    def window(self) -> Window:
        pixel_scale = self.pixel_scale
        area = self.area
        assert area is not None

        return Window(
            xoff=round_down_pixels(area.left / pixel_scale.xstep, pixel_scale.xstep),
            yoff=round_down_pixels(area.top / (pixel_scale.ystep * -1.0), pixel_scale.ystep * -1.0),
            xsize=round_up_pixels(
                (area.right - area.left) / pixel_scale.xstep, pixel_scale.xstep
            ),
            ysize=round_up_pixels(
                (area.top - area.bottom) / (pixel_scale.ystep * -1.0),
                (pixel_scale.ystep * -1.0)
            ),
        )

    @property
    def datatype(self) -> DataType:
        # TODO: Work out how to indicate type promotion via numpy
        return self.lhs.datatype

    @property
    def projection(self):
        try:
            return self.lhs.projection
        except AttributeError:
            return self.rhs.projection

    def _eval(self, area: Area, index: int, step: int, target_window:Optional[Window]=None):

        if self.buffer_padding:
            if target_window:
                target_window = target_window.grow(self.buffer_padding)
            pixel_scale = self.pixel_scale
            area = area.grow(self.buffer_padding * pixel_scale.xstep)
            # The index doesn't need updating because we updated area/window
            step += (2 * self.buffer_padding)

        lhs_data = self.lhs._eval(area, index, step, target_window)

        if self.operator is None:
            return lhs_data

        try:
            operator = backend.operator_map[self.operator]
        except KeyError:
            # Handles things like `numpy_apply` where a custom operator is provided
            operator = self.operator

        if self.other is not None:
            assert self.rhs is not None
            rhs_data = self.rhs._eval(area, index, step, target_window)
            other_data = self.other._eval(area, index, step, target_window)
            return operator(lhs_data, rhs_data, other_data, **self.kwargs)

        if self.rhs is not None:
            rhs_data = self.rhs._eval(area, index, step, target_window)
            return operator(lhs_data, rhs_data, **self.kwargs)

        return operator(lhs_data, **self.kwargs)

    def sum(self):
        # The result accumulator is float64, and for precision reasons
        # we force the sum to be done in float64 also. Otherwise we
        # see variable results depending on chunk size, as different parts
        # of the sum are done in different types.
        res = 0.0
        computation_window = self.window
        for yoffset in range(0, computation_window.ysize, self.ystep):
            step=self.ystep
            if yoffset+step > computation_window.ysize:
                step = computation_window.ysize - yoffset
            chunk = self._eval(self.area, yoffset, step, computation_window)
            res += backend.sum_op(chunk)
        return res

    def min(self):
        res = None
        computation_window = self.window
        for yoffset in range(0, computation_window.ysize, self.ystep):
            step=self.ystep
            if yoffset+step > computation_window.ysize:
                step = computation_window.ysize - yoffset
            chunk = self._eval(self.area, yoffset, step, computation_window)
            chunk_min = backend.min_op(chunk)
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
            chunk = self._eval(self.area, yoffset, step, computation_window)
            chunk_max = backend.max_op(chunk)
            if (res is None) or (chunk_max > res):
                res = chunk_max
        return res

    def save(self, destination_layer, and_sum=False, callback=None, band=1):
        """
        Calling save will write the output of the operation to the provied layer.
        If you provide sum as true it will additionall compute the sum and return that.
        """

        if destination_layer is None:
            raise ValueError("Layer is required")
        try:
            band = destination_layer._dataset.GetRasterBand(band)
        except AttributeError as exc:
            raise ValueError("Layer must be a raster backed layer") from exc

        destination_window = destination_layer.window

        # If we're calculating purely from a constant layer, then we don't have a window or area
        # so we should use the destination raster details.
        try:
            computation_window = self.window
            computation_area = self.area
        except AttributeError:
            computation_window = destination_window
            computation_area = destination_layer.area

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
            chunk = self._eval(computation_area, yoffset, step, computation_window)
            if isinstance(chunk, (float, int)):
                chunk = backend.full((step, destination_window.xsize), chunk)
            band.WriteArray(
                backend.demote_array(chunk),
                destination_window.xoff,
                yoffset + destination_window.yoff,
            )
            if and_sum:
                total += backend.sum_op(chunk)
        if callback:
            callback(1.0)

        return total if and_sum else None

    def _parallel_worker(self, index, shared_mem, sem, np_dtype, width, input_queue, output_queue, computation_window):
        arr = np.ndarray((self.ystep, width), dtype=np_dtype, buffer=shared_mem.buf)

        try:
            while True:
                # We acquire the lock so we know we have somewhere to put the
                # result before we take work. This is because in practice
                # it seems the writing to GeoTIFF is the bottleneck, and
                # we had workers taking a task, then waiting for somewhere to
                # write to for ages when other workers were exiting because there
                # was nothing to do.
                sem.acquire()

                task = input_queue.get()
                if task is None:
                    sem.release()
                    output_queue.put(None)
                    break
                yoffset, step = task

                result = self._eval(self.area, yoffset, step, computation_window)
                backend.eval_op(result)

                arr[:step] = backend.demote_array(result)

                output_queue.put((index, yoffset, step))

        except Exception as e: # pylint: disable=W0718
            logger.exception(e)
            sem.release()
            output_queue.put(None)

    def _park(self):
        try:
            self.lhs._park()
        except AttributeError:
            pass
        try:
            self.rhs._park()
        except AttributeError:
            pass
        try:
            self.other._park()
        except AttributeError:
            pass

    def _parallel_save(self, destination_layer, and_sum=False, callback=None, parallelism=None, band=1):
        assert (destination_layer is not None) or and_sum
        try:
            computation_window = self.window
        except AttributeError:
            # This is most likely because the calculation is on a constant layer (or combination of only constant
            # layers) and there's no real benefit to parallel saving then, so to keep this code from getting yet
            # more complicated just fall back to the single threaded path
            if destination_layer:
                return self.save(destination_layer, and_sum, callback, band)
            elif and_sum:
                return self.sum()
            else:
                assert False

        worker_count = parallelism or multiprocessing.cpu_count()
        work_blocks = len(range(0, computation_window.ysize, self.ystep))
        adjusted_blocks = math.ceil(work_blocks / constants.MINIMUM_CHUNKS_PER_THREAD)
        worker_count = min(adjusted_blocks, worker_count)

        if worker_count == 1:
            if destination_layer:
                return self.save(destination_layer, and_sum, callback, band)
            elif and_sum:
                return self.sum()
            else:
                assert False

        if destination_layer is not None:
            try:
                band = destination_layer._dataset.GetRasterBand(band)
            except AttributeError as exc:
                raise ValueError("Layer must be a raster backed layer") from exc

            destination_window = destination_layer.window

            if (computation_window.xsize != destination_window.xsize) \
                    or (computation_window.ysize != destination_window.ysize):
                raise ValueError("Destination raster window size does not match input raster window size.")

            np_dtype = {
                gdal.GDT_Byte:    np.dtype('byte'),
                gdal.GDT_Float32: np.dtype('float32'),
                gdal.GDT_Float64: np.dtype('float64'),
                gdal.GDT_Int8:    np.dtype('int8'),
                gdal.GDT_Int16:   np.dtype('int16'),
                gdal.GDT_Int32:   np.dtype('int32'),
                gdal.GDT_Int64:   np.dtype('int64'),
                gdal.GDT_UInt16:  np.dtype('uint16'),
                gdal.GDT_UInt32:  np.dtype('uint32'),
                gdal.GDT_UInt64:  np.dtype('uint64'),
            }[band.DataType]
        else:
            band = None
            np_dtype = np.dtype('float64')

        # The parallel save will cause a fork on linux, so we need to
        # remove all SWIG references
        self._park()

        total = 0.0

        with multiprocessing.Manager() as manager:
            with SharedMemoryManager() as smm:

                mem_sem_cast = []
                for i in range(worker_count):
                    shared_buf = smm.SharedMemory(size=np_dtype.itemsize * self.ystep * computation_window.xsize)
                    cast_buf = np.ndarray((self.ystep, computation_window.xsize), dtype=np_dtype, buffer=shared_buf.buf)
                    cast_buf[:] = np.zeros((self.ystep, computation_window.xsize), np_dtype)
                    mem_sem_cast.append((shared_buf, Semaphore(), cast_buf))

                source_queue = manager.Queue()
                result_queue = manager.Queue()

                for yoffset in range(0, computation_window.ysize, self.ystep):
                    step = ((computation_window.ysize - yoffset)
                        if yoffset+self.ystep > computation_window.ysize
                        else self.ystep)
                    source_queue.put((
                        yoffset,
                        step
                    ))
                for _ in range(worker_count):
                    source_queue.put(None)

                if callback:
                    callback(0.0)

                workers = [Process(target=self._parallel_worker, args=(
                    i,
                    mem_sem_cast[i][0],
                    mem_sem_cast[i][1],
                    np_dtype,
                    computation_window.xsize,
                    source_queue,
                    result_queue,
                    computation_window
                )) for i in range(worker_count)]
                for worker in workers:
                    worker.start()

                sentinal_count = len(workers)
                retired_blocks = 0
                while sentinal_count > 0:
                    res = result_queue.get()
                    if res is None:
                        sentinal_count -= 1
                        continue
                    index, yoffset, step = res
                    _, sem, arr = mem_sem_cast[index]
                    if band:
                        band.WriteArray(
                            arr[0:step],
                            destination_window.xoff,
                            yoffset + destination_window.yoff,
                        )
                    if and_sum:
                        total += np.sum(np.array(arr[0:step]).astype(np.float64))
                    sem.release()
                    retired_blocks += 1
                    if callback:
                        callback(retired_blocks / work_blocks)

                processes = workers
                while processes:
                    candidates = [x for x in processes if not x.is_alive()]
                    for candidate in candidates:
                        candidate.join()
                        if candidate.exitcode:
                            for victim in processes:
                                victim.kill()
                            sys.exit(candidate.exitcode)
                        processes.remove(candidate)
                    time.sleep(0.01)

        return total if and_sum else None

    def parallel_save(self, destination_layer, and_sum=False, callback=None, parallelism=None, band=1):
        if destination_layer is None:
            raise ValueError("Layer is required")
        return self._parallel_save(destination_layer, and_sum, callback, parallelism, band)

    def parallel_sum(self, callback=None, parallelism=None, band=1):
        return self._parallel_save(None, True, callback, parallelism, band)

class ShaderStyleOperation(LayerOperation):

    def _eval(self, area, index, step, target_window=None):
        if target_window is None:
            target_window = self.window
        lhs_data = self.lhs._eval(area, index, step, target_window)
        if self.rhs is not None:
            rhs_data = self.rhs._eval(area, index, step, target_window)
        else:
            rhs_data = None

        # Constant results make this a bit messier. Might in future
        # be nicer to promote them to arrays sooner?
        if isinstance(lhs_data, (int, float)):
            if rhs_data is None:
                return self.operator(lhs_data, **self.kwargs)
            if isinstance(rhs_data, (int, float)):
                return self.operator(lhs_data, rhs_data, **self.kwargs)
            else:
                result = np.empty_like(rhs_data)
        else:
            result = np.empty_like(lhs_data)

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
                    result[yoffset][xoffset] = self.operator(lhs_val, rhs_val, **self.kwargs)
                else:
                    result[yoffset][xoffset] = self.operator(lhs_val, **self.kwargs)

        return result

# We provide these module level accessors as it's often nicer to write `log(x/y)` rather than `(x/y).log()`
where = LayerOperation.where
minumum = LayerOperation.minimum
maximum = LayerOperation.maximum
clip = LayerOperation.clip
log = LayerOperation.log
log2 = LayerOperation.log2
log10 = LayerOperation.log10
exp = LayerOperation.exp
exp2 = LayerOperation.exp2
nan_to_num = LayerOperation.nan_to_num
isin = LayerOperation.isin
abs = LayerOperation.abs # pylint: disable=W0622
floor = LayerOperation.floor
round = LayerOperation.round # pylint: disable=W0622
ceil = LayerOperation.ceil
