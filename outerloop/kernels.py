import math

import gpytorch
import torch


class SuppressDebugChecksKernel(gpytorch.kernels.Kernel):
    def __call__(self, *args, **kwargs):
        with gpytorch.settings.debug(False):
            return super().__call__(*args, **kwargs)


class InnerOuterAlphaWeights(gpytorch.Module):
    """
    This provides a 2-weight parameterization of a set of 3 weights.

    Stores the weights for a weighted sum
        w1*(w2*k1 + (1-w2)*k2) + (1-w1)*k1*k2

    and uses them to generate three weights in the expanded form of this
    expression, i.e
        w1*w2
        w1*(1-w2)
        (1-s1)
    """
    def __init__(self, alpha_inner_prior, alpha_outer_prior,
                 batch_shape=torch.Size([])):
        super().__init__()

        self.register_parameter(
            name="raw_alpha_outer",
            parameter=torch.nn.Parameter(torch.zeros(*batch_shape, 1)))
        if alpha_outer_prior is not None:
            self.register_prior("alpha_outer_prior", alpha_outer_prior,
                                self.alpha_outer_getter, self.alpha_outer_setter)
        self.register_constraint("raw_alpha_outer", gpytorch.constraints.Interval(0.0, 1.0))

        self.register_parameter(
            name="raw_alpha_inner",
            parameter=torch.nn.Parameter(torch.zeros(*batch_shape, 1)))
        if alpha_inner_prior is not None:
            self.register_prior("alpha_inner_prior", alpha_inner_prior,
                                self.alpha_inner_getter, self.alpha_inner_setter)
        self.register_constraint("raw_alpha_inner", gpytorch.constraints.Interval(0.0, 1.0))

    @property
    def alpha_outer(self):
        return self.raw_alpha_outer_constraint.transform(self.raw_alpha_outer)

    @classmethod
    def alpha_outer_getter(cls, k):
        return k.alpha_outer

    @classmethod
    def alpha_outer_setter(cls, k, alpha_outer):
        if not torch.is_tensor(alpha_outer):
            alpha_outer = torch.as_tensor(alpha_outer).to(k.raw_alpha_outer)

        k.initialize(
            raw_alpha_outer=
            k.raw_alpha_outer_constraint.inverse_transform(alpha_outer))

    @property
    def alpha_inner(self):
        return self.raw_alpha_inner_constraint.transform(self.raw_alpha_inner)

    @classmethod
    def alpha_inner_getter(cls, k):
        return k.alpha_inner

    @classmethod
    def alpha_inner_setter(cls, k, alpha_inner):
        if not torch.is_tensor(alpha_inner):
            alpha_inner = torch.as_tensor(alpha_inner).to(k.raw_alpha_inner)

        k.initialize(
            raw_alpha_inner=
            k.raw_alpha_inner_constraint.inverse_transform(alpha_inner))

    def get_expanded_weights(self):
        outer = self.alpha_outer
        inner = self.alpha_inner
        return torch.cat([outer * inner, outer * (1 - inner), (1 - outer)],
                         dim=-1)


class WeightedSPSMaternKernel(SuppressDebugChecksKernel):
    """
    Weighted sum of products of sums of 1D Matern kernels (with nu=2.5)
    """

    has_lengthscale = True

    def __init__(self, weight_getter, sum_prod_sum_indices, **kwargs):
        sum_prod_sum_lengths = [len(sum_indices)
                                for prod_sum_indices in sum_prod_sum_indices
                                for sum_indices in prod_sum_indices]
        super().__init__(ard_num_dims=sum(sum_prod_sum_lengths),

                         lengthscale_prior=gpytorch.priors.GammaPrior(3.0, 6.0),
                         **kwargs)

        self.weight_getter = weight_getter

        self.operand_indices = torch.tensor(
            [idx
             for prod_sum_indices in sum_prod_sum_indices
             for sum_indices in prod_sum_indices
             for idx in sum_indices])

        self.sum_prod_size = len(sum_prod_sum_lengths)
        self.sum_prod_sum_parent_indices = torch.arange(
            self.sum_prod_size
        ).repeat_interleave(torch.tensor(sum_prod_sum_lengths))

        sum_prod_lengths = [len(prod_sum_indices)
                            for prod_sum_indices in sum_prod_sum_indices]
        self.sum_size = len(sum_prod_lengths)

        self.sum_prod_parent_indices = torch.arange(
            self.sum_size
        ).repeat_interleave(torch.tensor(sum_prod_lengths))

    def forward(self, x1, x2, diag: bool = False, last_dim_is_batch: bool = False):
        assert not diag
        assert not last_dim_is_batch

        x1_ = x1[..., self.operand_indices]
        x2_ = x2[..., self.operand_indices]

        # For numerical stability (similar to other gpytorch kernels)
        mean = x1_.reshape(-1, x1_.size(-1)).mean(0)[(None,) * (x1_.dim() - 1)]
        x1_ = x1_.subtract(mean)
        x2_ = x2_.subtract(mean)

        # More efficient to do this on these vectors than on the big distance
        # matrix
        lengthscale = self.lengthscale
        x1_ = x1_.div_(lengthscale)
        x2_ = x2_.div_(lengthscale)
        del lengthscale

        d = x2_.unsqueeze(-3) - x1_.unsqueeze(-2)

        # Measuring a bunch of 1D distances. abs is suffiecient.
        # d = d.square_()
        # d = d.clamp_min_(1e-30).sqrt_()
        d = d.abs_()

        exp_component = torch.exp(-math.sqrt(5) * d)
        constant_component = 1. + (math.sqrt(5) * d) + (5. / 3.) * d**2
        d = constant_component * exp_component
        del exp_component
        del constant_component

        summed = torch.zeros((*d.shape[:-1], self.sum_prod_size),
                             device=d.device)
        summed.index_add_(-1, self.sum_prod_sum_parent_indices, d)
        d = summed
        del summed

        # Do a prod via a sum of logs. Both index_reduce_ and prod are slow, in
        # part because their backward pass handles the scenario where the
        # product is 0. The code below is much faster in the backward pass
        # because it is asynchronous, unlike prod's backward pass.
        d = d.log()
        summed = torch.zeros((*d.shape[:-1], self.sum_size),
                             device=d.device)
        summed.index_add_(-1, self.sum_prod_parent_indices, d)
        d = summed
        del summed
        d = d.exp()

        w = self.weight_getter()
        w = w.view(*w.shape[:-1], 1, 1, w.shape[-1])
        d = (d * w).sum(dim=-1)

        return d

    def _apply(self, fn):
        self.operand_indices = fn(self.operand_indices)
        self.sum_prod_sum_parent_indices = fn(self.sum_prod_sum_parent_indices)
        self.sum_prod_parent_indices = fn(self.sum_prod_parent_indices)
        return super()._apply(fn)


def weighted_sps_matern_kernel(weight_getter, sum_prod_sum_names,
                               gp_feature_space, batch_shape=torch.Size([])):
    sum_prod_sum_indices = []
    names_ = []
    for prod_sum_names in sum_prod_sum_names:
        prod_sum_indices = []
        for sum_names in prod_sum_names:
            sum_indices = []
            for name in sum_names:
                found = False
                for i, p in enumerate(gp_feature_space):
                    if p.name == name:
                        found = True
                        sum_indices.append(i)
                        names_.append(name)
                if not found:
                    raise KeyError(name)
            prod_sum_indices.append(sum_indices)
        sum_prod_sum_indices.append(prod_sum_indices)
    k = WeightedSPSMaternKernel(weight_getter, sum_prod_sum_indices,
                                batch_shape=batch_shape)
    k.raw_lengthscale._ol_names = names_

    return k
