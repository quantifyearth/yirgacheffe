import json

import numpy as np
import pytest

import yirgacheffe as yg
from yirgacheffe._backends import backend, BACKEND

def test_sum_result_is_scalar() -> None:
    rng = np.random.default_rng(seed=42)
    data = rng.integers(0, 128, size=(1000, 1000))
    with yg.from_array(data, (0, 0), ("epsg:4326", (0.01, -0.01))) as layer:
        total = layer.sum()
        json.dumps(total)

@pytest.mark.parametrize("c,dtype,maxval", [
    (int(2), np.int8, 120),
    (int(2), np.uint8, 250),
    (int(2), np.int16, 32000),
    (int(2), np.uint16, 64000),
    (int(2), np.int32, 66000),
    (int(2), np.uint32, 66000),
])
@pytest.mark.parametrize("step", [1, 2, 4, 8])
def test_sums_of_calc_int(monkeypatch, step, c, dtype: type, maxval: int) -> None:
    with monkeypatch.context() as m:
        m.setattr(yg.constants, "YSTEP", step)

        rng = np.random.default_rng(seed=42)

        data = rng.integers(0, maxval, size=(1000, 1000), dtype=dtype)
        typed_data = backend.promote(data)

        assert np.sum(data) == backend.sum_op(typed_data)

        with yg.from_array(data, (0, 0), ("epsg:4326", (0.01, -0.01))) as layer:
            assert layer.sum() == backend.sum_op(typed_data)
            calc = layer * c
            actual = calc.sum()
            expected = backend.sum_op(typed_data * c)
            assert actual == expected

@pytest.mark.parametrize("c,dtype,maxval", [
    (float(2.5), np.int8, 120),
    (float(2.5), np.uint8, 250),
    (float(2.5), np.int16, 32000),
    (float(2.5), np.uint16, 640),
    (float(2.5), np.int32, 660),
    (float(2.5), np.uint32, 660),
])
@pytest.mark.parametrize("step", [1, 2, 4, 8])
def test_sums_of_calc_float_mlx(monkeypatch, step, c, dtype: type, maxval: int) -> None:
    with monkeypatch.context() as m:
        m.setattr(yg.constants, "YSTEP", step)

        rng = np.random.default_rng(seed=42)

        data = rng.integers(0, maxval, size=(100, 100), dtype=dtype)
        typed_data = backend.promote(data)

        with yg.from_array(data, (0, 0), ("epsg:4326", (0.01, -0.01))) as layer:
            assert layer.sum() == backend.sum_op(typed_data)
            calc = layer * c
            actual = calc.sum()
            expected = backend.sum_op(typed_data * c)
            # MLX has a maximum float size of 32 bit for GPU and NUMPY has 64 bit on CPU
            rel = 1e-6 if BACKEND == "MLX" else 1e-10
            assert float(actual) == pytest.approx(float(expected), rel=rel)
