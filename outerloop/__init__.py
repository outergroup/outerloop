import itertools

import torch

from . import kernels
from . import priors
from . import transforms


class Scalar:
    def __init__(self, name, lower=None, upper=None, condition=None,
                 inactive_value=None):
        self.name = name
        self.bounds = [lower, upper]
        self.condition = condition

        if condition is not None and inactive_value is None:
            if lower is not None:
                if upper is not None:
                    self.inactive_value = lower + (upper - lower) / 2
                else:
                    self.inactive_value = lower
            elif upper is not None:
                self.inactive_value = upper
            else:
                self.inactive_value = 0.0

    def to_user_space(self, x):
        return float(x)

    def to_search_space(self, v):
        return v


class Int:
    def __init__(self, name, lower, upper, condition=None, inactive_value=None):
        self.name = name

        self.bounds = [lower, upper]
        self.condition = condition

        if condition is not None and inactive_value is None:
            if lower is not None:
                if upper is not None:
                    self.inactive_value = lower + (upper - lower) // 2
                else:
                    self.inactive_value = lower
            elif upper is not None:
                self.inactive_value = upper
            else:
                self.inactive_value = 0

    def to_user_space(self, x):
        return int(x)

    def to_search_space(self, v):
        return float(v)


class Choice(Int):
    inactive_value = -1

    def __init__(self, name, choices, condition=None):
        self.name = name
        self.choices = choices
        self.bounds = [0, len(choices) - 1]
        self.condition = condition

    def to_user_space(self, x):
        return self.choices[int(x)]

    def to_search_space(self, v):
        return self.choices.index(v)



class LinearConstraint:
    """
    Encodes w * x <= bound, using an indices-and-coefficients description of w
    """
    def __init__(self, indices, coefficients, bound):
        self.indices = indices
        self.coefficients = coefficients
        self.bound = bound

    def check(self, x):
        return (sum(c * v
                    for c, v in zip(self.coefficients, x[self.indices]))
                <= self.bound)

    def as_botorch_inequality_constraint(self):
        return (torch.tensor(self.indices),
                -torch.tensor(self.coefficients),
                -self.bound)


def botorch_inequality_constraints(linear_constraints):
    if len(linear_constraints) == 0:
        return None
    else:
        return [c.as_botorch_inequality_constraint()
                for c in linear_constraints]


def botorch_bounds(parameters):
    return torch.tensor([p.bounds for p in parameters]).T


def X_to_configs(parameters, X):
    return [x_to_config(parameters, x)
            for x in X]


def x_to_config(parameters, x):
    config = {}
    for i, p in enumerate(parameters):
        if isinstance(p, Choice):
            if x[i] != p.inactive_value:
                config[p.name] = p.to_user_space(x[i])

    for i, p in enumerate(parameters):
        if not isinstance(p, Choice):
            active = p.condition is None or p.condition(config)
            if active:
                config[p.name] = p.to_user_space(x[i])

    return config



def all_choice_combinations(space):
    choice_space = [p for p in space
                    if isinstance(p, Choice)]

    results = []
    for choices in itertools.product(*[p.choices
                                       for p in choice_space]):
        choices = dict(zip((p.name for p in choice_space),
                           choices))

        result = []
        for p in choice_space:
            if p.condition is not None and p.name in choices:
                active = p.condition(choices)
                if not active:
                    del choices[p.name]
                    continue

            result.append((p.name, choices[p.name]))
        results.append(tuple(result))

    return list(set(results))


def all_base_configurations(space):
    """
    All choice variable settings, along with any fixed scalar values for those
    settings (for scalar parameters that are inactive for those choice
    parameter settings)
    """
    all_fixed_features = []
    for choices in all_choice_combinations(space):
        choices = dict(choices)

        fixed_features = {}
        for i, p in enumerate(space):
            if isinstance(p, Choice):
                if p.name in choices:
                    x = p.to_search_space(choices[p.name])
                else:
                    x = p.inactive_value

                fixed_features[i] = x
            else:
                if p.condition is not None:
                    active = p.condition(choices)
                    if not active:
                        fixed_features[i] = p.inactive_value

        all_fixed_features.append(fixed_features)

    return all_fixed_features


__all__ = [
    "kernels",
    "transforms",
    "Scalar",
    "Int",
    "Choice",
    "LinearConstraint",
    "botorch_inequality_constraints",
    "botorch_bounds",
    "X_to_configs",
    "x_to_config",
    "all_base_configurations",
]
