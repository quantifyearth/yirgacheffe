from __future__ import annotations
import operator
from functools import reduce
from typing import Sequence

from .base import YirgacheffeLayer
from .._datatypes import Area

def find_intersection(layers: Sequence[YirgacheffeLayer]) -> Area:
    """Given a collection of layers return the intersecting area.

    All layers must have the same map projection and pixel scale. If
    no intersection is possible then a ValueError is raised.

    Args:
        layers: The collection of layers to compare.

    Returns:
        An area representing the pixel aligned intersection.
    """

    if not layers:
        raise ValueError("Expected list of layers")

    # This only makes sense (currently) if all layers
    # have the same pixel pitch (modulo desired accuracy)
    projections = [x.projection for x in layers if x.projection is not None]
    if not projections:
        raise ValueError("No layers have a projection")
    if not all(projections[0] == x for x in projections[1:]):
        raise ValueError("Not all layers are at the same projection or pixel scale")

    layer_areas = [x._get_operation_area(projections[0]) for x in layers]
    return reduce(operator.and_, layer_areas)

def find_union(layers: Sequence[YirgacheffeLayer]) -> Area:
    """Given a collection of layers return the union area.

    All layers must have the same map projection and pixel scale.

    Args:
        layers: The collection of layers to compare.

    Returns:
        An area representing the pixel aligned union.
    """
    if not layers:
        raise ValueError("Expected list of layers")

    # This only makes sense (currently) if all layers
    # have the same pixel pitch (modulo desired accuracy)
    projections = [x.projection for x in layers if x.projection is not None]
    if not projections:
        raise ValueError("No layers have a projection")
    if not all(projections[0] == x for x in projections[1:]):
        raise ValueError("Not all layers are at the same projectin or pixel scale")

    layer_areas = [x._get_operation_area(projections[0]) for x in layers]
    # This removal of global layers is to stop constant layers forcing everything to be global
    return reduce(operator.or_, [x for x in layer_areas if not x.is_world])
