import pytest

# I've no idea why pylint dislikes this particular import but accepts
# other entries in the module.
from yirgacheffe.window import Area # pylint: disable=E0401, E0611

@pytest.mark.parametrize(
    "lhs,rhs,is_equal,overlaps",
    [
        # Obvious equality
        (Area(-10.0, 10.0, 10.0, -10.0), Area(-10.0, 10.0, 10.0, -10.0), True,  True),
        (Area(-9.0, 9.0, 9.0, -9.0),     Area(-10.0, 10.0, 10.0, -10.0), False, True), # subset
        (Area(-9.0, 9.0, -1.0, 1.0),     Area(1.0, -1.0, 9.0, -9.0),     False, False),
        (Area(-10.0, 10.0, 1.0, -10.0),  Area(-1.0, 10.0, 10.0, -10.0),  False, True),
    ]
)
def test_area_operators(lhs: Area, rhs: Area, is_equal: bool, overlaps: bool) -> None:
    assert (lhs == rhs) == is_equal
    assert (lhs != rhs) == (not is_equal)
    assert (lhs.overlaps(rhs)) == overlaps
    assert (rhs.overlaps(lhs)) == overlaps
