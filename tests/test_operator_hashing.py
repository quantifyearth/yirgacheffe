import numpy as np
import pytest

import yirgacheffe as yg

def test_simple_constant_expression() -> None:
    with (
        yg.constant(1) as lhs,
        yg.constant(2) as rhs,
    ):
        calc0 = lhs + rhs
        calc1 = lhs + rhs
        calc2 = rhs + lhs

        # One day this test should fail when we take commutative operators into account
        assert calc0 is not calc1
        assert calc0._cse_hash == calc1._cse_hash
        assert calc1._cse_hash != calc2._cse_hash

def test_simple_raster_expression() -> None:
    data1 = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    data2 = np.array([[10, 20, 30, 40], [50, 60, 70, 80]])

    with(
        yg.from_array(data1, (0, 0), ("epsg:4326", (1.0, -1.0))) as layer1,
        yg.from_array(data1, (0, 0), ("epsg:4326", (1.0, -1.0))) as layer2,
    ):
        calc0 = layer1 + layer2
        calc1 = layer1 + layer2
        calc2 = layer1 + layer1

        assert calc0 is not calc1
        assert calc0._cse_hash == calc1._cse_hash
        assert calc1._cse_hash != calc2._cse_hash

def test_mixed_raster_constant() -> None:
    data1 = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])

    with(
        yg.from_array(data1, (0, 0), ("epsg:4326", (1.0, -1.0))) as layer1,
        yg.constant(2) as layer2,
        yg.constant(2.0) as layer3,
    ):
        calc0 = layer1 + layer2
        calc1 = layer1 + layer2
        calc2 = layer1 * layer2
        calc3 = layer1 + layer3

        assert calc0 is not calc1
        assert layer2 is not layer3
        assert calc0._cse_hash == calc1._cse_hash
        assert calc1._cse_hash != calc2._cse_hash
        assert calc1._cse_hash == calc3._cse_hash

def test_cse_simple(mocker, monkeypatch) -> None:
    with monkeypatch.context() as m:
        m.setattr(yg.constants, "YSTEP", 1)

        data1 = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        with (
            yg.from_array(data1, (0, 0), ("epsg:4326", (1.0, -1.0))) as lhs,
            yg.constant(3) as rhs,
        ):
            calc = (lhs + rhs) * (lhs + rhs)

            # this is an API violation, but let's check the table used for CSE
            hash_table = {}
            calc._populate_hash_table(hash_table, calc.window)

            assert len(hash_table) == 4

            top_level_hash = calc._cse_hash
            common_term_hash = (lhs + rhs)._cse_hash
            lhs_hash = lhs._cse_hash
            rhs_hash = rhs._cse_hash

            assert hash_table[(top_level_hash, calc.window)] == (1, None)
            assert hash_table[(common_term_hash, calc.window)] == (2, None)
            assert hash_table[(lhs_hash, calc.window)] == (1, None)
            assert hash_table[(rhs_hash, calc.window)] == (1, None)

            lhs_spy = mocker.spy(lhs, '_read_array_with_window')
            rhs_spy = mocker.spy(rhs, '_read_array_for_area')

            expected = (data1 + 3) * (data1 + 3)
            actual = calc.read_array(0, 0, 4, 2)
            assert (expected == actual).all()

            assert lhs_spy.call_count == 2
            assert rhs_spy.call_count == 2

def test_simple_aoh_style_range_check(mocker, monkeypatch) -> None:
    with monkeypatch.context() as m:
        m.setattr(yg.constants, "YSTEP", 1)

        data1 = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        with yg.from_array(data1, (0, 0), ("epsg:4326", (1.0, -1.0))) as lhs:
            calc = (lhs > 2) & (lhs < 7)

            # this is an API violation, but let's check the table used for CSE
            hash_table = {}
            calc._populate_hash_table(hash_table, calc.window)

            assert len(hash_table) == 6
            assert hash_table[(lhs._cse_hash, calc.window)] == (2, None)
            for k in hash_table.keys():
                if k == (lhs._cse_hash, calc.window):
                    continue
                assert hash_table[k] == (1, None)

            lhs_spy = mocker.spy(lhs, '_read_array_with_window')

            expected = (data1 > 2) & (data1 < 7)

            actual = calc.read_array(0, 0, 4, 2)
            assert (expected == actual).all()

            assert lhs_spy.call_count == 2

def test_caching_versus_boundary_expansion(mocker, monkeypatch) -> None:
    # If you have a layer that is the source for a convolution, then the window read from it is
    # expanded, so if we have the same layer in and out a convolution, it'll be read at different
    # window sizes, and cause a mess with caching
    with monkeypatch.context() as m:
        m.setattr(yg.constants, "YSTEP", 1)

        matrix = np.array([[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]])
        data1 = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [12, 13, 14, 15]]).astype(np.float32)
        with yg.from_array(data1, (0, 0), ("epsg:4326", (1.0, -1.0))) as lhs:
            calc = lhs.conv2d(matrix) * lhs
            calc.pretty_print()

            # this is an API violation, but let's check the table used for CSE
            hash_table = {}
            calc._populate_hash_table(hash_table, calc.window)

            # In current CSE implementation, we can't hash np arrays, so the only
            # thing we can actually cache in theory is the raster array
            assert len(hash_table) == 2
            assert hash_table[(lhs._cse_hash, calc.window)] == (1, None)
            assert hash_table[(lhs._cse_hash, calc.window.grow(1))] == (1, None)

@pytest.mark.parametrize("sequence", [
    [1, 2, 3],
    (1, 2, 3),
    {1, 2, 3},
])
def test_isin_hashable(sequence) -> None:
    data1 = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    with yg.from_array(data1, (0, 0), ("epsg:4326", (1.0, -1.0))) as layer:
        calc = layer.isin(sequence)

        assert calc._cse_hash is not None

        expected = np.isin(data1, list(sequence))
        actual = calc.read_array(0, 0, 4, 2)
        assert (expected == actual).all()
