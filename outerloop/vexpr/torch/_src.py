import math
from functools import partial

import torch
import vexpr.core
import vexpr.torch.primitives as t_p
import vexpr.custom.torch.primitives as tc_p
import vexpr.vectorization as v
from vexpr import Vexpr
from vexpr.torch.register_pushthrough import (
    push_cat_through_unary_elementwise,
    push_stack_through_unary_elementwise,
)
from vexpr.custom.torch.register_pushthrough import (
    identity_pushthrough,
    push_shuffle_through_unary_elementwise,
)
from vexpr.custom.torch.register_lift import (
    lift_shuffle_from_unary_elementwise,
)


matern_p, matern = vexpr.core._p_and_constructor("matern")

def matern_impl(d, nu=2.5):
    with torch.profiler.record_function("matern"):
        assert nu == 2.5
        neg_sqrt5_times_d = (-math.sqrt(5)) * d
        exp_component = torch.exp(neg_sqrt5_times_d)
        constant_component = 1. - neg_sqrt5_times_d + (5. / 3.) * d**2
        return constant_component * exp_component


vexpr.core.eval_impls[matern_p] = matern_impl
v.vectorize_impls[matern_p] = v.unary_elementwise_vectorize
v.pushthrough_impls[(t_p.stack_p, matern_p)] = partial(
    push_stack_through_unary_elementwise, matern_p)
v.pushthrough_impls[(t_p.cat_p, matern_p)] = partial(
    push_cat_through_unary_elementwise, matern_p)
v.pushthrough_impls[(tc_p.shuffle_p, matern_p)] = partial(
    push_shuffle_through_unary_elementwise, matern_p
)
v.pushthrough_impls[(tc_p.mul_along_dim_p, matern_p)] = identity_pushthrough
v.lift_impls[(matern_p, tc_p.shuffle_p)] = partial(
    lift_shuffle_from_unary_elementwise, matern_p
)
