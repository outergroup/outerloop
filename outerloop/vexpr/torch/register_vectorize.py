import vexpr.vectorization as v
from vexpr import Vexpr

from .primitives import matern_p

def unary_elementwise_vectorize(shapes, expr):
    return Vexpr(
        expr.op,
        (v._vectorize(shapes, expr.args[0]),),
        **expr.kwargs)

v.vectorize_impls[matern_p] = unary_elementwise_vectorize
