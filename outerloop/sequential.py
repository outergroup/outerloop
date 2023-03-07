import gpytorch
import torch

import outerloop as ol


def passthrough(f):
    def passthrough_function(*args, **kwargs):
        def passthrough_constructor(levels):
            return f(*args, **kwargs), levels
        return passthrough_constructor
    return passthrough_function


clamp = passthrough(ol.modules.Clamp)
cdist1d = passthrough(ol.modules.CDist1d)
cdist1d_hamming = passthrough(ol.modules.CDist1dHamming)
matern = passthrough(ol.modules.Matern)
gibbs = passthrough(ol.modules.Gibbs)


def sum_at():
    def sum_at_constructor(levels):
        *levels, final = levels
        return (ol.modules.SumAt(final.grouplengths),
                levels)
    return sum_at_constructor


def prod_at(all_positive=False):
    def prod_at_constructor(levels):
        *levels, final = levels
        return (ol.modules.ProdAt(final.grouplengths, all_positive),
                levels)
    return prod_at_constructor


def cdist_at():
    def cdist_at_constructor(levels):
        *levels, final = levels
        return (ol.modules.CDistAt(final.grouplengths),
                levels)
    return cdist_at_constructor


def reweight(construct_weights, *transform_weights, ndims_per_model=None):
    def reweight_constructor(levels):
        construct_instance, levels, w_grouplengths = construct_weights(levels)

        transform_instances = []
        if len(transform_weights) > 0:
            for op_constructor in transform_weights:
                (op_instance,
                 levels,
                 w_grouplengths) = op_constructor(levels, w_grouplengths)
                if op_instance is not None:
                    transform_instances.append(op_instance)

        if w_grouplengths != levels[-1].grouplengths:
            raise ValueError(
                f"Must output weights of for groups {levels[-1].grouplengths}"
                f"actual: {w_grouplengths}"
            )

        return (ol.modules.Reweight(construct_instance,
                                    *transform_instances,
                                    ndims_per_model=ndims_per_model),
                levels)

    return reweight_constructor


def w_beta_pair(beta_prior_key=None, batch_shape=torch.Size([])):
    def w_beta_pair_constructor(levels):
        group_indices = [i
                         for i, length in enumerate(levels[-1].grouplengths)
                         if length > 1]
        num_pairs = len(group_indices)
        w_grouplengths = [2] * num_pairs

        if beta_prior_key is not None:
            prior_args = levels[-2].values[beta_prior_key]
            prior_args = [prior_args[i]
                          for i in group_indices]
            concentration1, concentration0 = (torch.tensor(x)
                                              for x in zip(*prior_args))
            prior = ol.priors.BetaPrior(concentration1, concentration0)
        else:
            prior = None

        return ol.modules.WBetaPair(
            num_pairs=num_pairs,
            prior=prior,
            batch_shape=batch_shape), levels, w_grouplengths

    return w_beta_pair_constructor


def w_dirichlet(dirichlet_prior_key=None,
                dirichlet_prior_concentration=None,
                batch_shape=torch.Size([])):
    assert dirichlet_prior_key is None or dirichlet_prior_concentration is None

    def w_dirichlet_constructor(levels):
        w_grouplengths = [length
                          for length in levels[-1].grouplengths
                          if length > 1]

        if dirichlet_prior_key is not None:
            concentration = torch.as_tensor(
                levels[-1].values[dirichlet_prior_key]
            )
            concentrations = [
                c
                for c in concentration.split(levels[-1].grouplengths)
                if c.numel() > 1
            ]
            prior = ol.priors.DirichletAtPrior(concentrations)

        if dirichlet_prior_concentration is not None:
            concentrations = [torch.tensor([dirichlet_prior_concentration]
                                           * length)
                              for length in w_grouplengths]
            prior = ol.priors.DirichletAtPrior(concentrations)

        return ol.modules.WDirichlet(
            lengths=w_grouplengths,
            prior=prior,
            batch_shape=batch_shape), levels, w_grouplengths

    return w_dirichlet_constructor


def w_lengthscale(gamma_prior_key=None, batch_shape=torch.Size([])):
    def lengthscale_constructor(levels):
        if gamma_prior_key is not None:
            prior_args = levels[-1].values[gamma_prior_key]
            concentration, rate = (torch.tensor(x)
                                   for x in zip(*prior_args))
            prior = gpytorch.priors.GammaPrior(concentration, rate)
        else:
            prior = None

        w_grouplengths = levels[-1].grouplengths

        return ol.modules.WLengthscale(
            sum(levels[-1].grouplengths),
            prior=prior,
            batch_shape=batch_shape,
        ), levels, w_grouplengths
    return lengthscale_constructor


def w_beta_pair_compose(beta_prior_key=None, batch_shape=torch.Size([])):
    def w_beta_pair_compose_constructor(levels, w_grouplengths):
        if beta_prior_key is not None:
            prior_args = levels[-2].values[beta_prior_key]
            concentration1, concentration0 = (torch.tensor(x)
                                              for x in zip(*prior_args))
            prior = ol.priors.BetaPrior(concentration1, concentration0)
        else:
            prior = None

        if len(w_grouplengths) == 1:
            instance = ol.modules.WSingleBetaPairCompose(w_grouplengths, prior,
                                                         batch_shape)
        else:
            instance = ol.modules.WBetaPairCompose(w_grouplengths, prior,
                                                   batch_shape)

        return (instance,
                levels,
                [length + 1 for length in w_grouplengths])

    return w_beta_pair_compose_constructor


def w_scale_compose(gamma_prior_key=None, batch_shape=torch.Size([])):
    def w_scale_compose_constructor(levels, w_grouplengths):
        if gamma_prior_key is not None:
            prior_args = levels[-2].values[gamma_prior_key]
            concentration, rate = (torch.tensor(x)
                                   for x in zip(*prior_args))
            prior = gpytorch.priors.GammaPrior(concentration, rate)
        else:
            prior = None

        if len(w_grouplengths) == 1:
            instance = ol.modules.WSingleScaleCompose(w_grouplengths,
                                                      prior, batch_shape)
        else:
            instance = ol.modules.WScaleCompose(w_grouplengths, prior,
                                                batch_shape)

        return (instance,
                levels,
                w_grouplengths)
    return w_scale_compose_constructor


def w_identity_compose():
    def w_identity_compose_constructor(levels, grouplengths_w):
        n = 0
        other_indices = []
        for i, length in enumerate(levels[-1].grouplengths):
            if length == 1:
                grouplengths_w.insert(i, 1)
            elif length > 1:
                other_indices += list(range(n, n + length))
            n += length

        return (ol.modules.WIdentityCompose(n, other_indices),
                levels, grouplengths_w)
    return w_identity_compose_constructor


def select(space=None):
    def select_constructor(levels):
        if space is not None:
            keys = levels[-1].values["key"]
            indices = []
            for key in keys:
                try:
                    indices.append(
                        next(i
                             for i, p in enumerate(space)
                             if p.name == key))
                except StopIteration:
                    raise ValueError(key)
        else:
            indices = levels[-1].values["index"]
        return ol.modules.Select(indices), levels
    return select_constructor


zero_center = passthrough(ol.modules.ZeroCenter)


def build_list(operations, levels):
    module_instances = []
    for constructor in operations:
        instance, levels = constructor(levels)
        module_instances.append(instance)

    return module_instances


def build(operations, levels):
    return torch.nn.Sequential(*build_list(operations, levels))
