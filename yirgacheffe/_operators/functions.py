
import operator as pyoperator

from . import LayerOperation, WindowOperation
from .._backends.enumeration import operators as op

def where(cond, a, b):
    """Return elements chosen from `a` or `b` depending on `cond`.

    Behaves like numpy.where(condition, x, y), returning a layer operation
    where elements from `a` are selected where `cond` is True, and elements
    from `b` are selected where `cond` is False.

    Args:
        cond: Layer or constant used as condition. Where True, yield `a`, otherwise yield `b`.
        a: Layer or constant with values from which to choose where `cond` is True.
        b: Layer or constant with values from which to choose where `cond` is False.

    Returns:
        New layer representing the conditional selection.
    """
    return LayerOperation(
        cond,
        op.WHERE,
        rhs=a,
        other=b
    )

def maximum(a, b):
    """Element-wise maximum of layer elements.

    Behaves like numpy.maximum(x1, x2), comparing two layers element-by-element
    and returning a new layer with the maximum values.

    Args:
        a: First layer or constant to compare.
        b: Second layer or constant to compare.

    Returns:
        New layer representing the element-wise maximum of the inputs.
    """
    return LayerOperation(
        a,
        op.MAXIMUM,
        b,
        window_op=WindowOperation.UNION,
    )

def minimum(a, b):
    """Element-wise minimum of layer elements.

    Behaves like numpy.minimum(x1, x2), comparing two layers element-by-element
    and returning a new layer with the minimum values.

    Args:
        a: First layer or constant to compare.
        b: Second layer or constant to compare.

    Returns:
        New layer representing the element-wise minimum of the inputs.
    """
    return LayerOperation(
        a,
        op.MINIMUM,
        rhs=b,
        window_op=WindowOperation.UNION,
    )

def _balanced_reduce(layers, operator):
    if len(layers) == 0:
        raise RuntimeError("Internal precondition violation")
    if len(layers) == 1:
        return layers[0]
    mid = len(layers) // 2
    left = _balanced_reduce(layers[:mid], operator)
    right = _balanced_reduce(layers[mid:], operator)
    return operator(left, right)

def sum(layers: list): # pylint: disable=W0622
    """Combine multiple layers by summing spatially corresponding pixels.

    Creates a new raster where each pixel is the sum of that pixel's
    values across all input rasters for the same location.

    Args:
        layers: List/sequence of layers to sum

    Returns:
        A new raster layer with pixel-wise sums

    Examples:
        # Combine 100 species habitat rasters into richness map
        richness = yg.sum(habitat_layers)

    Note:
        To sum all pixels within a single raster to get a scalar,
        use the .sum() method instead: `layer.sum()`
    """
    if len(layers) == 0:
        raise ValueError("List of layers is empty")
    return _balanced_reduce(layers, pyoperator.__add__)

def any(layers: list): # pylint: disable=W0622
    """Combine layers with OR operation.

    Returns a raster where each pixel is 1 if ANY input layers
    has a non-zero value at that geospatial location.

    Similar to Python's built-in any() but for Yirgacheffe layers.

    Args:
        layers: List/sequence of layers to OR

    Returns:
        A new raster layer with pixel-wise sums
    """
    if len(layers) == 0:
        raise ValueError("List of layers is empty")
    return _balanced_reduce([layer != 0 for layer in layers], pyoperator.or_)

def all(layers: list): # pylint: disable=W0622
    """Combine layers with AND operation.

    Returns a raster where each pixel is 1 only if ALL input layers
    have non-zero values at that geospatial location.

    Similar to Python's built-in all() but for Yirgacheffe layers.

    Args:
        layers: List/sequence of layers to AND

    Returns:
        A new raster layer with pixel-wise sums
    """
    if len(layers) == 0:
        raise ValueError("List of layers is empty")
    return _balanced_reduce([layer != 0 for layer in layers], pyoperator.and_)

# We provide these module level accessors as it's often nicer to write `log(x/y)` rather than `(x/y).log()`
# But they need to be functions rather than references so that mkdoc does the right thing (originally they
# were just aliases).
def clip(layer, min=None, max=None): # pylint: disable=W0622
    """Clip a layer to either an miniumum or maximum value, or both.

    Args:
        layer: The layer on which to clip.
        min: If specified, the lower bound of value that will be in the result layer.
        max: If specified, the upper bound of value that will be in the result layer.

    Returns:
        A new layer with the values clipped as specified.
    """
    return layer.clip(min, max)

def log(layer):
    """Returns the natural logarithm of the layer.

    Args:
        layer: The layer of which to take the natural logarithm.

    Returns:
        A layer where each value is the natural logarithm of the input.

    Note:
        Can also be called as `layer.log()`.
    """
    return layer.log()

def log2(layer):
    """Returns the base-2 logarithm of the layer.

    Args:
        layer: The layer of which to take the base-2 logarithm.

    Returns:
        A layer where each value is the base-2 logarithm of the input.

    Note:
        Can also be called as `layer.log2()`.
    """
    return layer.log2()

def log10(layer):
    """Returns the base-10 logarithm of the layer.

    Args:
        layer: The layer of which to take the base-10 logarithm.

    Returns:
        A layer where each value is the base-10 logarithm of the input.

    Note:
        Can also be called as `layer.log10()`.
    """
    return layer.log10()

def exp(layer):
    """Returns the exponent of the layer.

    Args:
        layer: The layer of which to take the exponent.

    Returns:
        A layer where each value is the exponent of the input.

    Note:
        Can also be called as `layer.exp()`.
    """
    return layer.exp()

def exp2(layer):
    """Returns the base-2 exponent of the layer.

    Args:
        layer: The layer of which to take the base-2 exponent.

    Returns:
        A layer where each value is the base-2 exponent of the input.

    Note:
        Can also be called as `layer.exp2()`.
    """
    return layer.exp2()

def nan_to_num(layer, nan=0, posinf=None, neginf=None):
    """Replace nan and infinity values with zero and large finate values respecitively, or
    with the values provided.

    Args:
        layer: The layer to evaluate.
        nan: The value used to replace nan values. Defaults to 0.
        posinf: The value used to replace positive infinity. Defaults to a large finite value dependant on type.
        posinf: The value used to replace negative infinity. Defaults to a negative large finite value dependant
            on type.

    Returns:
        A new layer with the values appropriately substituted.

    Note:
        Can also be called as `layer.nan_to_num(...)`.
    """
    return layer.nan_to_num(nan, posinf, neginf)

def isin(layer, test_elements):
    """Returns a layer of boolean values that indicate if the corresponding value of the source was
    fouind in the provided test_elements.

    Args:
        layer: The layer to be evaluted.
        test_elements: A sequence (list, tuple, set) of values to be tested against.

    Returns:
        A new layer where the values are True if the source layer value was in the test_elements,
        otherwise False.

    Note:
        Can also be called as `layer.isin(test_elements)`.
    """
    return layer.isin(test_elements)

def abs(layer): # pylint: disable=W0622
    """Returns a layer where the values are the absolute value of the input values.

    Args:
        layer: The layer of which to take the absolute values.

    Returns:
        A new layer where the values are the absolute value of the input layer's values.

    Note:
        Can also be called as `layer.abs()`.
    """
    return layer.abs()

def floor(layer):
    """Returns a layer where the values are the rounded down (floor) of the input values.

    Args:
        layer: The layer of which to floor.

    Returns:
        A new layer where the values are the floor of the input layer's values.

    Note:
        Can also be called as `layer.floor()`.
    """
    return layer.floor()

def round(layer): # pylint: disable=W0622
    """Returns a layer where the values are rounded from the input values.

    Args:
        layer: The layer of which to round.

    Returns:
        A new layer where the values are rounded from the input layer's values.

    Note:
        Can also be called as `layer.round()`.
    """
    return layer.round()

def ceil(layer):
    """Returns a layer where the values are the rounded up (ceiling) of the input values.

    Args:
        layer: The layer of which to ceiling.

    Returns:
        A new layer where the values are the ceiling of the input layer's values.

    Note:
        Can also be called as `layer.ceil()`.
    """
    return layer.ceil()

def logical_and(layer1, layer2):
    """Returns a boolean layer that is the logical and of the values in the two input layers.

    Note that this is not the same as using the `&` operator which will do a bitwise and.

    Args:
        layer1: First input layer.
        layer2: Second input layer.

    Returns:
        A new layer that is the logical and of the two input layers.
    """
    return LayerOperation.logical_and(layer1, layer2)

def logical_or(layer1, layer2):
    """Returns a boolean layer that is the logical or of the values in the two input layers.

    Note that this is not the same as using the `|` operator which will do a bitwise or.

    Args:
        layer1: First input layer.
        layer2: Second input layer.

    Returns:
        A new layer that is the logical or of the two input layers.
    """
    return LayerOperation.logical_or(layer1, layer2)

def logical_xor(layer1, layer2):
    """Returns a boolean layer that is the logical xor of the values in the two input layers.

    Args:
        layer1: First input layer.
        layer2: Second input layer.

    Returns:
        A new layer that is the logical xor of the two input layers.
    """
    return LayerOperation.logical_xor(layer1, layer2)

def logical_not(layer):
    """Returns a boolean layer that is the logical inverse of the input layer.


    Args:
        layer: The input layer.

    Returns:
        A new layer that is the logical inverse of the input.
    """
    return LayerOperation.logical_not(layer)
