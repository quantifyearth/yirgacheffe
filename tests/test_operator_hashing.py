import numpy as np

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
        assert calc0._cse_hash() == calc1._cse_hash()
        assert calc1._cse_hash() != calc2._cse_hash()

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
        assert calc0._cse_hash() == calc1._cse_hash()
        assert calc1._cse_hash() != calc2._cse_hash()

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
            assert calc0._cse_hash() == calc1._cse_hash()
            assert calc1._cse_hash() != calc2._cse_hash()
            assert calc1._cse_hash() == calc3._cse_hash()
