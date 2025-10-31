
from .._backends import backend
from ..window import Window

class CSECacheTable:

    def __init__(self, expression, window: Window) -> None:
        self._table: dict[tuple[int, Window], tuple[int, backend.array_t | None]] = {}
        self._populate(expression, window)

    def __len__(self) -> int:
        return len(self._table)

    def _add(self, cse_hash: int | None, window: Window) -> int:
        if cse_hash is None:
            return 0

        try:
            count, data = self._table[(cse_hash, window)]
            cache_line = (count + 1, data)
        except KeyError:
            cache_line = (1, None)
        self._table[(cse_hash, window)] = cache_line
        count, _ = cache_line
        return count

    def _populate(self, expression, window: Window) -> None:
        used_window = window.grow(expression.buffer_padding)
        count = self._add(expression._cse_hash, used_window)
        if count == 1:
            # We only add children the first time we see an expression, otherwise
            # we will cache data covered by this node's cache line potentially
            for child in expression._children:
                try:
                    self._populate(child, used_window)
                except AttributeError:
                    try:
                        self._add(child._cse_hash, used_window)
                    except TypeError:
                        pass

    def set_data(self, cse_hash: int | None, window: Window, data: backend.array_t) -> None:
        if cse_hash is not None:
            try:
                count, old_data = self._table[(cse_hash, window)]
                if count < 2:
                    return
                if old_data is not None:
                    raise RuntimeWarning("Failure in CSE logic, setting data that is already set")
                self._table[(cse_hash, window)] = (count, data)
            except KeyError:
                raise RuntimeWarning("Failure in CSE logic, setting data for unknown term") # pylint: disable=W0707

    def get_data(self, cse_hash: int | None, window: Window) -> backend.array_t | None:
        # TODO: In theory we could release data from the table if we decremented the count each read
        # but we'd also need to fixed copy of the count for reset
        if cse_hash is None:
            return None
        try:
            _count, data = self._table[(cse_hash, window)]
            return data
        except KeyError:
            return None

    def reset_cache(self):
        self._table = {k: (count, None) for k, (count, _data) in self._table.items()}
