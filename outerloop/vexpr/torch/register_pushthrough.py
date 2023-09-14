import vexpr.core as core
import vexpr.torch as vtorch
import vexpr.torch.primitives as t_p

from .primitives import matern_p, matern

def push_stack_through_matern(shapes, expr, allow_partial=True):
    assert expr.op == t_p.stack_p

    exprs_to_stack = expr.args[0]
    assert all(isinstance(child_expr, core.Vexpr)
               and child_expr.op == matern_p
               for child_expr in exprs_to_stack)

    nu = exprs_to_stack[0].kwargs.get("nu", 2.5)
    grandchildren = []
    for child_expr in exprs_to_stack:
        assert child_expr.kwargs.get("nu", 2.5) == nu
        grandchildren.append(child_expr.args[0])

    grandchildren = core._vectorize(shapes,
                                    vtorch.stack(grandchildren, **expr.kwargs))
    ret = matern(grandchildren, nu=nu)

    grandchildren_shapes = [shapes[id(child_expr.args[0])]
                            for child_expr in exprs_to_stack]
    assert all(grandchildren_shape == grandchildren_shapes[0]
                for grandchildren_shape in grandchildren_shapes)
    dim = expr.kwargs.get("dim", 0)
    # use dim to determine result shape after stack
    result_shape = grandchildren_shapes[0]
    if dim < 0:
        dim += len(result_shape) + 1
    result_shape = (result_shape[:dim]
                    + (len(exprs_to_stack),)
                    + result_shape[dim:])
    shapes[id(ret)] = result_shape

    return ret

def push_concat_through_matern(shapes, expr, allow_partial=True):
    assert expr.op == t_p.concat_p

    exprs_to_concat = expr.args[0]
    assert all(isinstance(child_expr, core.Vexpr)
               and child_expr.op == matern_p
               for child_expr in exprs_to_concat)

    nu = exprs_to_concat[0].kwargs.get("nu", 2.5)
    grandchildren = []
    for child_expr in exprs_to_concat:
        assert child_expr.kwargs.get("nu", 2.5) == nu
        grandchildren.append(child_expr.args[0])

    grandchildren = core._vectorize(shapes,
                                    vtorch.concat(grandchildren, **expr.kwargs))
    ret = matern(grandchildren, nu=nu)

    grandchildren_shapes = [shapes[id(child_expr.args[0])]
                            for child_expr in exprs_to_concat]
    dim = expr.kwargs.get("dim", 0)
    if dim < 0:
        dim += len(result_shape)

    # use dim to determine result shape after concat
    result_shape = []
    for i in range(len(grandchildren_shapes[0])):
        if i == dim:
            result_shape.append(sum(grandchildren_shape[i]
                                   for grandchildren_shape in grandchildren_shapes))
        else:
            assert all(grandchildren_shape[i] == grandchildren_shapes[0][i]
                        for grandchildren_shape in grandchildren_shapes)
            result_shape.append(grandchildren_shapes[0][i])

    shapes[id(ret)] = tuple(result_shape)

    return ret

core.pushthrough_impls[(t_p.stack_p, matern_p)] = push_stack_through_matern
core.pushthrough_impls[(t_p.concat_p, matern_p)] = push_concat_through_matern