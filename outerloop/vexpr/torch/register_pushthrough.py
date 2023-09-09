import vexpr.core as core
import vexpr.torch as vtorch
import vexpr.torch.primitives as t_p

from .primitives import matern_p, matern

def push_stack_through_matern(shapes, expr, allow_partial=True):
    assert expr.op is t_p.stack_p

    exprs_to_stack = expr.args[0]
    assert all(isinstance(child_expr, core.Vexpr)
               and child_expr.op is matern_p
               for child_expr in exprs_to_stack)

    nu = exprs_to_stack[0].kwargs.get("nu", 2.5)
    grandchildren = []
    for child_expr in exprs_to_stack:
        assert child_expr.kwargs.get("nu", 2.5) == nu
        grandchildren.append(child_expr.args[0])

    grandchildren = core._vectorize(shapes,
                                    vtorch.stack(grandchildren, **expr.kwargs))
    return matern(grandchildren, nu=nu)

core.pushthrough_impls[(t_p.stack_p, matern_p)] = push_stack_through_matern
