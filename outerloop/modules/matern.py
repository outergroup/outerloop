import math

import torch
import torch.profiler


class Matern(torch.nn.Module):
    """
    """
    def __init__(self, nu=2.5):
        super().__init__()
        self.nu = nu

    def forward(self, d):
        with torch.profiler.record_function("Matern.forward"):
            # TODO implement for other nu values
            assert self.nu == 2.5
            exp_component = torch.exp(-math.sqrt(5) * d)
            constant_component = 1. + (math.sqrt(5) * d) + (5. / 3.) * d**2
            return constant_component * exp_component


__all__ = [
    "Matern",
]
