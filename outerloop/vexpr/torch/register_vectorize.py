import vexpr.core

from .primitives import matern_p

def unary_elementwise_vectorize(shapes, expr):
    return vexpr.core.Vexpr(
        expr.op,
        (vexpr.core_vectorize(shapes, expr.args[0]),),
        **expr.kwargs)

vexpr.core.vectorize_impls[matern_p] = unary_elementwise_vectorize
