import pytest

# I've no idea why pylint dislikes this particular import but accepts
# other entries in the module.
from yirgacheffe.window import Window # pylint: disable=E0401, E0611

@pytest.mark.parametrize(
    "lhs,rhs,is_greater,is_equal",
    [
        # Obvious equality
        (Window(0, 0, 100, 100),     Window(0, 0, 100, 100),     False, True),
        (Window(-10, 0, 100, 100),   Window(-10, 0, 100, 100),   False, True),
        (Window(0, -10, 100, 100),   Window(0, -10, 100, 100),   False, True),
        (Window(-10, -10, 100, 100), Window(-10, -10, 100, 100), False, True),
        (Window(10, 0, 100, 100),    Window(10, 0, 100, 100),    False, True),
        (Window(0, 10, 100, 100),    Window(0, 10, 100, 100),    False, True),
        (Window(10, 10, 100, 100),   Window(10, 10, 100, 100),   False, True),

        # Obvious inequality
        (Window(0, 0, 100, 200), Window(0, 0, 100, 100), False, False),
        (Window(0, 0, 200, 100), Window(0, 0, 100, 100), False, False),
        (Window(0, 2, 100, 100), Window(0, 0, 100, 100), False, False),
        (Window(2, 0, 100, 100), Window(0, 0, 100, 100), False, False),

        # is greater is always lhs the subset
        (Window(10, 10, 10, 10), Window(0, 0, 100, 100), True, False),
        (Window(0, 0, 10, 10),   Window(0, 0, 100, 100), True, False),
        (Window(11, 11, 8, 8),   Window(10, 10, 10, 10), True, False),
        (Window(0, 0, 10, 10), Window(-1, -1, 12, 12), True, False),

        # Here the LHS is smaller, but isn't a subset
        (Window(-5, -5, 10, 10), Window(0, 0, 100, 100), False, False),
        (Window(95, 95, 10, 10), Window(0, 0, 100, 100), False, False),
        (Window(9, 9, 5, 5),     Window(10, 10, 10, 10), False, False),
        (Window(19, 19, 5, 5),   Window(10, 10, 10, 10), False, False),

        # Smaller but with bounaries touching
        (Window(0, 0, 10, 10), Window(-2, -2, 12, 12), True, False),
        (Window(0, 0, 10, 10), Window(0, 0, 11, 11), True, False),
    ]
)
def test_window_operators(lhs: Window, rhs: Window, is_greater: bool, is_equal: bool) -> None:
    assert (lhs == rhs) == is_equal
    assert (lhs != rhs) == (not is_equal)
    assert (lhs < rhs) == is_greater
    assert (rhs > lhs) == is_greater
    assert (lhs <= rhs) == (is_equal or is_greater)
    assert (rhs >= lhs) == (is_equal or is_greater)

def test_find_intersection_empty_list() -> None:
    with pytest.raises(ValueError):
        Window.find_intersection([])

def test_find_intersection_single_item() -> None:
    window = Window(10, 10, 10, 10)
    intersection = Window.find_intersection([window])
    assert intersection == window

def test_find_intersection_same() -> None:
    windows = [
        Window(-10, 10, 10, 10),
        Window(-10, 10, 10, 10),
    ]
    intersection = Window.find_intersection(windows)
    assert intersection == windows[0]

def test_find_intersection_subset() -> None:
    windows = [
        Window(0, 0, 10, 10),
        Window(1, 1, 1, 1),
    ]
    intersection = Window.find_intersection(windows)
    assert intersection == windows[1]

def test_find_intersection_overlap() -> None:
    windows = [
        Window(0, 0, 10, 10),
        Window(5, 5, 10, 10),
    ]
    intersection = Window.find_intersection(windows)
    assert intersection == Window(5, 5, 5, 5)

def test_find_intersection_distinct() -> None:
    windows = [
        Window(0, 0, 10, 10),
        Window(20, 0, 10, 10),
    ]
    with pytest.raises(ValueError):
        _ = Window.find_intersection(windows)
