import gpytorch
import torch
import torch.profiler


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

    def log_prob(self, x):
        with torch.profiler.record_function("BetaPrior.log_prob"):
            return super().log_prob(x)


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
        self.concentrations = [torch.as_tensor(c) for c in concentrations]
        self.lengths = [len(c) for c in concentrations]

        concentration = torch.cat(self.concentrations)
        self.concentration_minus_1 = concentration - 1.0

        reduced_size = len(self.lengths)
        reduce_indices = torch.arange(
            reduced_size
        ).repeat_interleave(torch.tensor(self.lengths))
        component1 = torch.lgamma(
            torch.zeros(reduced_size).index_add_(
                -1, reduce_indices, concentration
            )
        )
        component2 = torch.zeros(reduced_size).index_add_(
            -1, reduce_indices, torch.lgamma(concentration)
        )
        self.additive_constant = (component1 - component2).sum()

    def _apply(self, fn):
        self.concentrations = [fn(c) for c in self.concentrations]
        self.concentration_minus_1 = fn(self.concentration_minus_1)
        self.additive_constant = fn(self.additive_constant)

    def log_prob(self, x):
        with torch.profiler.record_function("DirichletAtPrior.log_prob"):
            return ((x.log() * self.concentration_minus_1).sum(-1)
                    + self.additive_constant)

    def rsample(self, sample_shape=()):
        return torch.cat(
            [DirichletPrior(c).rsample(sample_shape)
             for c in self.concentrations],
            -1
        )

