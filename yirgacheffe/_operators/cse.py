
from .._backends import backend
from ..window import Window

class CSECacheTable:

    def __init__(self) -> None:
        self._table: dict[tuple[int, Window], tuple[int, backend.array_t | None]] = {}

    def __len__(self) -> int:
        return len(self._table)

    def add(self, cse_hash: int | None, window: Window) -> int:
        # TODO: Ideally we'd have a constructor or factory that does this and _populate_hash_table
        # in one so we know that once that phase is done, the table only changes with data
        # being cached, not the hash/window/count values.
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

    def set_data(self, cse_hash: int | None, window: Window, data: backend.array_t):
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
