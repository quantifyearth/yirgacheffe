import sys
import time
import multiprocessing
from multiprocessing import Semaphore, Process
from multiprocessing.managers import SharedMemoryManager

import numpy as np
from osgeo import gdal


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

    def save(self, destination_layer, and_sum=False, callback=None):
        return LayerOperation(self).save(destination_layer, and_sum, callback)

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

    def save(self, destination_layer, and_sum=False, callback=None):
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

    def _parallel_worker(self, index, shared_mem, sem, np_dtype, width, input_queue, output_queue):
        arr = np.ndarray((self.ystep, width), dtype=np_dtype, buffer=shared_mem.buf)

        while True:
            task = input_queue.get()
            if task is None:
                output_queue.put(None)
                break
            yoffset, step = task

            sem.acquire()
            arr[:step] = self._eval(yoffset, step)

            output_queue.put((index, yoffset, step))

    def _park(self):
        try:
            self.lhs._park()
        except AttributeError:
            pass
        try:
            self.rhs._park()
        except AttributeError:
            pass

    def parallel_save(self, destination_layer, and_sum=False, callback=None, parallelism=None):
        if destination_layer is None:
            raise ValueError("Layer is required")
        try:
            band = destination_layer._dataset.GetRasterBand(1)
        except AttributeError as exc:
            raise ValueError("Layer must be a raster backed layer") from exc

        computation_window = self.window
        destination_window = destination_layer.window

        # The parallel save will cause a fork on linux, so we need to
        # remove all SWIG references
        self._park()

        if (computation_window.xsize != destination_window.xsize) \
                or (computation_window.ysize != destination_window.ysize):
            raise ValueError("Destination raster window size does not match input raster window size.")

        total = 0.0

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


        with multiprocessing.Manager() as manager:
            with SharedMemoryManager() as smm:

                worker_count = parallelism or multiprocessing.cpu_count()
                worker_count = min(len(range(0, computation_window.ysize, self.ystep)), worker_count)

                mem_and_locks = [
                    (
                        smm.SharedMemory(size=np_dtype.itemsize * self.ystep * destination_window.xsize),
                        Semaphore(),
                    ) for _ in range(worker_count)
                ]

                cast_mem = [
                    np.ndarray((self.ystep, destination_window.xsize), dtype=np_dtype, buffer=shared_mem.buf)
                for shared_mem, _ in mem_and_locks]

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

                workers = [Process(target=self._parallel_worker, args=(
                    i,
                    mem_and_locks[i][0],
                    mem_and_locks[i][1],
                    np_dtype,
                    destination_window.xsize,
                    source_queue,
                    result_queue
                )) for i in range(worker_count)]
                for worker in workers:
                    worker.start()

                sentinal_count = len(workers)
                while sentinal_count > 0:
                    res = result_queue.get()
                    if res is None:
                        sentinal_count -= 1
                        continue
                    index, yoffset, step = res
                    _, sem = mem_and_locks[index]
                    arr = cast_mem[index]
                    band.WriteArray(
                        arr[0:step],
                        destination_window.xoff,
                        yoffset + destination_window.yoff,
                    )
                    if and_sum:
                        total += np.sum(arr[0:step])
                    sem.release()

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
