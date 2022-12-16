import gpytorch
import torch


class BetaPrior(gpytorch.priors.Prior, torch.distributions.Beta):
    def __init__(self, concentration1, concentration0, validate_args=False,
                 transform=None):
        super().__init__(concentration1, concentration0, validate_args)
        self._transform = transform

    def expand(self, batch_shape):
        batch_shape = torch.Size(batch_shape)
        return BetaPrior(self.concentration1.expand(batch_shape),
                         self.concentration0.expand(batch_shape))

    def _apply(self, fn):
        self._dirichlet.concentration = fn(self._dirichlet.concentration)
