import math

import torch
import vexpr.core
import vexpr.vectorization as v

# TODO gross hack, this is needed to make register_unary_elementwise_op do all
# the right things
import vexpr.torch

matern_p, matern = vexpr.core._p_and_constructor("matern")

def matern_impl(d, nu=2.5):
    with torch.profiler.record_function("matern"):
        assert nu == 2.5
        neg_sqrt5_times_d = (-math.sqrt(5)) * d
        exp_component = torch.exp(neg_sqrt5_times_d)
        constant_component = 1. - neg_sqrt5_times_d + (5. / 3.) * d**2
        return constant_component * exp_component


vexpr.core.eval_impls[matern_p] = matern_impl

v.register_unary_elementwise_op(matern_p)
