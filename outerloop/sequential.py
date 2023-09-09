import collections

import gpytorch
import torch

import outerloop as ol


def passthrough(f):
    def passthrough_function(*args, **kwargs):
        def passthrough_constructor(levels):
            return f(*args, **kwargs), levels
        return passthrough_constructor
    return passthrough_function


def passthrough2(f, w_grouplengths):
    def passthrough2_function(*args, **kwargs):
        def passthrough2_constructor(levels):
            return f(*args, **kwargs), levels, w_grouplengths
        return passthrough2_constructor
    return passthrough2_function


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


def mean_at():
    def mean_at_constructor(levels):
        *levels, final = levels
        return (ol.modules.MeanAt(final.grouplengths),
                levels)
    return mean_at_constructor


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


def w_beta_pair(prior=None, beta_prior_key=None, constraint=None,
                batch_shape=torch.Size([])):
    assert prior is None or beta_prior_key is None
    def w_beta_pair_constructor(levels):
        group_indices = [i
                         for i, length in enumerate(levels[-1].grouplengths)
                         if length > 1]
        num_pairs = len(group_indices)
        w_grouplengths = [2] * num_pairs

        if prior is not None:
            prior_ = prior
        elif beta_prior_key is not None:
            prior_args = levels[-2].values[beta_prior_key]
            prior_args = [prior_args[i]
                          for i in group_indices]
            concentration1, concentration0 = (torch.tensor(x)
                                              for x in zip(*prior_args))
            prior_ = ol.priors.BetaPrior(concentration1, concentration0)
        else:
            prior_ = None

        return ol.modules.WBetaPair(
            num_pairs=num_pairs,
            prior=prior_,
            constraint=constraint,
            batch_shape=batch_shape), levels, w_grouplengths

    return w_beta_pair_constructor


def w_dirichlet(dirichlet_prior_key=None,
                dirichlet_prior_concentration=None,
                num_ignore=0,
                batch_shape=torch.Size([])):
    assert dirichlet_prior_key is None or dirichlet_prior_concentration is None

    def w_dirichlet_constructor(levels):
        w_grouplengths = [length - num_ignore
                          for length in levels[-1].grouplengths
                          if length > 1]

        if dirichlet_prior_key is not None:
            assert False
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


def w_scale(prior=None, constraint=None, # TODO keep?
            gamma_prior_key=None,
            batch_shape=torch.Size([])):
    assert prior is None or gamma_prior_key is None
    def scale_constructor(levels):
        if prior is not None:
            prior_ = prior
        elif gamma_prior_key is not None:
            prior_args = levels[-1].values[gamma_prior_key]
            concentration, rate = (torch.tensor(x)
                                   for x in zip(*prior_args))
            prior_ = gpytorch.priors.GammaPrior(concentration, rate)
        else:
            prior_ = None

        w_grouplengths = levels[-1].grouplengths

        return ol.modules.WScale(
            sum(levels[-1].grouplengths),
            prior=prior_,
            constraint=constraint,
            batch_shape=batch_shape,
        ), levels, w_grouplengths

    return scale_constructor


def w_lengthscale(prior=None, constraint=None, # TODO keep?
                  gamma_prior_key=None,
                  initialize=False,
                  batch_shape=torch.Size([])):
    assert prior is None or gamma_prior_key is None
    def lengthscale_constructor(levels):
        if prior is not None:
            prior_ = prior
        elif gamma_prior_key is not None:
            prior_args = levels[-1].values[gamma_prior_key]
            concentration, rate = (torch.tensor(x)
                                   for x in zip(*prior_args))
            # TODO this is a quick hack, decide whether to actually support this.
            if rate.ndim == 2 or concentration.ndim == 2:
                # reverse to support batches
                concentration = concentration.t()
                rate = rate.t()
            prior_ = gpytorch.priors.GammaPrior(concentration, rate)
        else:
            prior_ = None

        w_grouplengths = levels[-1].grouplengths

        return ol.modules.WLengthscale(
            sum(levels[-1].grouplengths),
            prior=prior_,
            constraint=constraint,
            initialize=initialize,
            batch_shape=batch_shape,
        ), levels, w_grouplengths
    return lengthscale_constructor


def w_beta_pair_compose(prior=None, beta_prior_key=None, constraint=None,
                        batch_shape=torch.Size([])):
    assert prior is None or beta_prior_key is None

    def w_beta_pair_compose_constructor(levels, w_grouplengths):
        if prior is not None:
            prior_ = prior
        elif beta_prior_key is not None:
            prior_args = levels[-2].values[beta_prior_key]
            concentration1, concentration0 = (torch.tensor(x)
                                              for x in zip(*prior_args))
            prior_ = ol.priors.BetaPrior(concentration1, concentration0)
        else:
            prior_ = None

        if len(w_grouplengths) == 1:
            instance = ol.modules.WSingleBetaPairCompose(w_grouplengths, prior_,
                                                         constraint,
                                                         batch_shape)
        else:
            instance = ol.modules.WBetaPairCompose(w_grouplengths, prior_,
                                                   constraint,
                                                   batch_shape)

        return (instance,
                levels,
                [length + 1 for length in w_grouplengths])

    return w_beta_pair_compose_constructor


def w_scale_compose(prior=None, gamma_prior_key=None, batch_shape=torch.Size([])):
    assert prior is None or gamma_prior_key is None

    def w_scale_compose_constructor(levels, w_grouplengths):
        if prior is not None:
            prior_ = prior
        elif gamma_prior_key is not None:
            prior_args = levels[-2].values[gamma_prior_key]
            concentration, rate = (torch.tensor(x)
                                   for x in zip(*prior_args))
            prior_ = gpytorch.priors.GammaPrior(concentration, rate)
        else:
            prior_ = None

        if len(w_grouplengths) == 1:
            instance = ol.modules.WSingleScaleCompose(w_grouplengths,
                                                      prior_, batch_shape)
        else:
            instance = ol.modules.WScaleCompose(w_grouplengths, prior_,
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


def select(k, space=None):
    def select_constructor(levels):
        if space is not None:
            keys = levels[-1].values[k]
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
            indices = levels[-1].values[k]
        return ol.modules.Select(indices), levels
    return select_constructor


zero_center = passthrough(ol.modules.ZeroCenter)


def switch(switch_key, value_key, cases):
    def switch_constructor(levels):
        nodes_by_key = collections.defaultdict(list)
        indices_by_key = collections.defaultdict(list)
        for i, (key, unparsed_tree) in enumerate(
                zip(levels[-1].values[switch_key],
                    levels[-1].values[value_key])):
            nodes_by_key[key].append(unparsed_tree)
            indices_by_key[key].append(i)

        modules = []
        result_index_groups = []
        for key, nodes in nodes_by_key.items():
            headers, operations = cases[key]
            subtree_levels_before = ol.treelevels.parse(headers, nodes)
            (instances,
             subtree_levels_after) = _build_list(operations,
                                                 subtree_levels_before)
            modules.append(torch.nn.Sequential(*instances))
            result_index_groups.append(indices_by_key[key])

        result_length = sum(levels[-1].grouplengths)

        return (ol.modules.CallAndMerge(modules, result_index_groups,
                                        result_length),
                levels)
    return switch_constructor


def _build_list(operations, levels):
    module_instances = []
    for constructor in operations:
        instance, levels = constructor(levels)
        module_instances.append(instance)

    return module_instances, levels


def build_list(operations, levels):
    module_instances, levels = _build_list(operations, levels)
    return module_instances


def build(operations, levels):
    return torch.nn.Sequential(*build_list(operations, levels))


def check(module, space, ignored_parameters=[]):
    # iterate over all modules, submodules, sub-submodules, etc.
    all_indices = set()

    for m in module.modules():
        if isinstance(m, ol.modules.Select):
            all_indices.update(m.indices.tolist())

    for i, p in enumerate(space):
        expected = p.name not in ignored_parameters

        if i in all_indices:
            if not expected:
                raise ValueError("Parameter {} is not expected to be used in "
                                 "the model.".format(p.name))
        else:
            if expected:
                raise ValueError("Parameter {} is expected to be used in the "
                                 "model.".format(p.name))

    return module