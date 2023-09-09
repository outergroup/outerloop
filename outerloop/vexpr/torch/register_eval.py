import math

import torch

import vexpr.core
from .primitives import matern_p

def matern_impl(d, nu=2.5):
    assert nu == 2.5
    exp_component = torch.exp(-math.sqrt(5) * d)
    constant_component = 1. + (math.sqrt(5) * d) + (5. / 3.) * d**2
    return constant_component * exp_component

vexpr.core.eval_impls[matern_p] = matern_impl
