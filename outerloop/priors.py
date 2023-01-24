import gpytorch
import torch


class BetaPrior(gpytorch.priors.Prior, torch.distributions.Beta):
    def __init__(self, concentration1, concentration0, validate_args=False,
                 transform=None):
        super().__init__(concentration1, concentration0,
                         validate_args=validate_args)
        self._transform = transform

    def expand(self, batch_shape):
        batch_shape = torch.Size(batch_shape)
        return BetaPrior(self.concentration1.expand(batch_shape),
                         self.concentration0.expand(batch_shape))

    def _apply(self, fn):
        self._dirichlet.concentration = fn(self._dirichlet.concentration)


class DirichletPrior(gpytorch.priors.Prior, torch.distributions.Dirichlet):
    def __init__(self, concentration, validate_args=False,
                 transform=None):
        super().__init__(concentration, validate_args=validate_args)
        self._transform = transform

    def _apply(self, fn):
        self.concentration = fn(self.concentration)

    def expand(self, batch_shape):
        batch_shape = torch.Size(batch_shape)
        return DirichletPrior(self.concentration.expand(batch_shape))


class DirichletAtPrior(gpytorch.priors.Prior):
    def __init__(self, concentrations, validate_args=False, transform=None):
        super().__init__(validate_args=validate_args)
        self._transform = transform
        self.lengths = [len(c) for c in concentrations]
        self.priors = torch.nn.ModuleList(
            DirichletPrior(c)
            for c in concentrations
        )

    def log_prob(self, x):
        return torch.stack(
            [prior.log_prob(x_)
             for x_, prior in zip(x.split(self.lengths, -1), self.priors)]
        ).sum()
