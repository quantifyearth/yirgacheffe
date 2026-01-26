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


# We provide these module level accessors as it's often nicer to write `log(x/y)` rather than `(x/y).log()`
clip = LayerOperation.clip
log = LayerOperation.log
log2 = LayerOperation.log2
log10 = LayerOperation.log10
exp = LayerOperation.exp
exp2 = LayerOperation.exp2
nan_to_num = LayerOperation.nan_to_num
isin = LayerOperation.isin
abs = LayerOperation.abs # pylint: disable=W0622
floor = LayerOperation.floor
round = LayerOperation.round # pylint: disable=W0622
ceil = LayerOperation.ceil
