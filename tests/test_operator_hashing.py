
import yirgacheffe as yg

def test_simple_constant_expression() -> None:
    lhs = yg.constant(1)
    rhs = yg.constant(2)
    calc0 = lhs + rhs
    calc1 = lhs + rhs
    calc2 = rhs + lhs

    # One day this test should fail when we take commutative operators into account
    assert calc0 == calc1
    assert calc1 == calc0
    assert calc1 != calc2
    assert calc2 != calc1

    assert hash(calc0) == hash(calc1)
    assert hash(calc1) != hash(calc2)
