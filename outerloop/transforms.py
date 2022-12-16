import abc
import math
from collections import OrderedDict
from functools import partial

import botorch
import numpy as np
import scipy
import torch

import outerloop as ol


class UntransformThenTransform(torch.nn.Module):
    def __init__(self, space, xform):
        super().__init__()
        self.xform = xform
        self.space1 = space
        self.space2 = space

    def transform(self, X):
        # Preserve the gradients, don't track gradients for the roundtrip.
        X_new = X.clone()
        with torch.no_grad():
            X_rounded = self.xform.transform(self.xform.untransform(X))
            X_new = X_new.copy_(X_rounded)
        return X_new


class EnforceBounds:
    def __init__(self, space):
        self.space1 = space
        self.space2 = space

    def transform(self, X):
        return X

    def untransform(self, X):
        X = X.clone()
        for i, p in enumerate(self.space2):
            if isinstance(p, ol.Choice):
                X[..., i].clamp_(min=min(p.bounds[0], p.inactive_value),
                                 max=p.bounds[1])
            else:
                if p.bounds[0] is not None or p.bounds[1] is not None:
                    X[..., i].clamp_(min=p.bounds[0], max=p.bounds[1])
        return X


class IntToScalar(torch.nn.Module):
    def __init__(self, space):
        super().__init__()
        self.int_indices = torch.tensor(
            [i
             for i, p in enumerate(space)
             if isinstance(p, ol.Int) or isinstance(p, ol.Choice)]
        )
        self.space1 = space

        space2 = list(space)
        for i, p in enumerate(space):
            if isinstance(p, ol.Int):
                space2[i] = ol.Scalar(p.name, p.bounds[0], p.bounds[1] + (1 - 1e-9),
                                      p.condition)
        for i, p in enumerate(space):
            if isinstance(p, ol.Choice):
                space2[i] = ol.Choice(p.name, p.choices, p.condition)
                space2[i].bounds = [p.bounds[0], p.bounds[1] + (1 - 1e-9)]
        self.space2 = space2

    def transform(self, X):
        return X

    def untransform(self, X):
        X = X.clone()
        X[..., self.int_indices] = X[..., self.int_indices].floor_()
        return X

    def _apply(self, fn):
        self.int_indices = fn(self.int_indices)
        return super()._apply(fn)


class Chain(torch.nn.ModuleDict):
    def __init__(self, space, *transform_classes):

        self.space1 = space
        transforms = []
        for t_class in transform_classes:
            t = t_class(space)
            space = t.space2
            transforms.append(t)
        self.transforms = transforms

        # TODO this is getting gross, maybe make all transforms modules
        modules = [t for t in transforms if isinstance(t, torch.nn.Module)]
        super().__init__(OrderedDict((str(i), m) for i, m in enumerate(modules)))

        self.space2 = space

    def transform(self, X):
        for t in self.transforms:
            X = t.transform(X)
        return X

    def untransform(self, X):
        for t in reversed(self.transforms):
            X = t.untransform(X)
        return X


class ToScalarSpace(Chain):
    def __init__(self, space, *transform_classes):
        super().__init__(space, IntToScalar, *transform_classes)


class Add(torch.nn.Module):
    def __init__(self, name_to_new_name, operand_name, space):
        super().__init__()
        self.space1 = space

        try:
            operand_i, operand_p = next((i, p) for i, p in enumerate(space)
                                        if p.name == operand_name)
        except StopIteration:
            raise KeyError(operand_name)
        self.operand_index = operand_i

        space2 = list(space)
        parameter_indices = []
        for name1, name2 in name_to_new_name.items():
            found = False
            for i, p in enumerate(space):
                if p.name == name1:
                    found = True
                    space2[i] = ol.Scalar(name2,
                                          space[i].bounds[0] + operand_p.bounds[0],
                                          space[i].bounds[1] + operand_p.bounds[1],
                                          p.condition)
                    if hasattr(p, "inactive_value"):
                        space2[i].inactive_value = (space[i].inactive_value
                                                    + operand_p.inactive_value)
                    parameter_indices.append(i)
            if not found:
                raise KeyError(name1)
        self.space2 = space2
        self.parameter_indices = torch.tensor(parameter_indices)

    def transform(self, X):
        X = X.clone()
        X[..., self.parameter_indices] += X[..., self.operand_index].unsqueeze(-1)
        return X

    def untransform(self, X):
        X = X.clone()
        X[..., self.parameter_indices] -= X[..., self.operand_index].unsqueeze(-1)
        return X

    def _apply(self, fn):
        self.parameter_indices = fn(self.parameter_indices)
        return super()._apply(fn)


def add(name_to_new_name, operand_name):
    return partial(Add, name_to_new_name, operand_name)


class Subtract(torch.nn.Module):
    def __init__(self, name_to_new_name, operand_name, space):
        super().__init__()
        self.space1 = space

        try:
            operand_i, operand_p = next((i, p) for i, p in enumerate(space)
                                        if p.name == operand_name)
        except StopIteration:
            raise KeyError(operand_name)
        self.operand_index = operand_i

        space2 = list(space)
        parameter_indices = []
        for name1, name2 in name_to_new_name.items():
            found = False
            for i, p in enumerate(space):
                if p.name == name1:
                    found = True
                    space2[i] = ol.Scalar(name2,
                                          space[i].bounds[0] - operand_p.bounds[0],
                                          space[i].bounds[1] - operand_p.bounds[1],
                                          p.condition)
                    if hasattr(p, "inactive_value"):
                        space2[i].inactive_value = (space[i].inactive_value
                                                    - operand_p.inactive_value)
                    parameter_indices.append(i)
            if not found:
                raise KeyError(name1)
        self.space2 = space2
        self.parameter_indices = torch.tensor(parameter_indices)

    def transform(self, X):
        X = X.clone()
        X[..., self.parameter_indices] -= X[..., self.operand_index].unsqueeze(-1)
        return X

    def untransform(self, X):
        X = X.clone()
        X[..., self.parameter_indices] -= X[..., self.operand_index].unsqueeze(-1)
        return X

    def _apply(self, fn):
        self.parameter_indices = fn(self.parameter_indices)
        return super()._apply(fn)


def subtract(name_to_new_name, operand_name):
    return partial(Subtract, name_to_new_name, operand_name)


class Multiply(torch.nn.Module):
    def __init__(self, name_to_new_name, operand_name, space):
        super().__init__()
        self.space1 = space

        try:
            operand_i, operand_p = next((i, p) for i, p in enumerate(space)
                                        if p.name == operand_name)
        except StopIteration:
            raise KeyError(operand_name)
        self.operand_index = operand_i

        space2 = list(space)
        parameter_indices = []
        for name1, name2 in name_to_new_name.items():
            found = False
            for i, p in enumerate(space):
                if p.name == name1:
                    found = True
                    space2[i] = ol.Scalar(name2,
                                          space[i].bounds[0] * operand_p.bounds[0],
                                          space[i].bounds[1] * operand_p.bounds[1],
                                          p.condition)
                    if hasattr(p, "inactive_value"):
                        space2[i].inactive_value = (space[i].inactive_value
                                                    * operand_p.inactive_value)
                    parameter_indices.append(i)
            if not found:
                raise KeyError(name1)
        self.space2 = space2
        self.parameter_indices = torch.tensor(parameter_indices)

    def transform(self, X):
        # An in-place update to X will interfere with backprop to the operands.
        operands = X[..., self.operand_index].unsqueeze(-1)
        X = X.clone()
        X[..., self.parameter_indices] *= operands
        return X

    def untransform(self, X):
        operands = X[..., self.operand_index].unsqueeze(-1)
        X = X.clone()
        X[..., self.parameter_indices] /= operands
        return X

    def _apply(self, fn):
        self.parameter_indices = fn(self.parameter_indices)
        return super()._apply(fn)


def multiply(name_to_new_name, operand_name):
    return partial(Multiply, name_to_new_name, operand_name)



class Log(torch.nn.Module):
    def __init__(self, name_to_new_name, space):
        super().__init__()
        self.space1 = space

        space2 = list(space)
        log_parameter_indices = []
        for name1, name2 in name_to_new_name.items():
            found = False
            for i, p in enumerate(space):
                if p.name == name1:
                    found = True
                    lower = (None if p.bounds[0] is None else math.log(p.bounds[0]))
                    upper = (None if p.bounds[1] is None else math.log(p.bounds[1]))
                    space2[i] = ol.Scalar(name2, lower, upper, p.condition)
                    if hasattr(p, "inactive_value"):
                        space2[i].inactive_value = math.log(p.inactive_value)
                    log_parameter_indices.append(i)
            if not found:
                raise KeyError(name1)
        self.space2 = space2
        self.log_parameter_indices = torch.tensor(log_parameter_indices)

    def transform(self, X):
        X = X.clone()
        X[..., self.log_parameter_indices] = X[..., self.log_parameter_indices].log()
        return X

    def untransform(self, X):
        X = X.clone()
        X[..., self.log_parameter_indices] = X[..., self.log_parameter_indices].exp()
        return X

    def _apply(self, fn):
        self.log_parameter_indices = fn(self.log_parameter_indices)
        return super()._apply(fn)


def log(name_to_new_name):
    return partial(Log, name_to_new_name)


class AppendMean(torch.nn.Module):
    def __init__(self, operands, mean_name, space):
        super().__init__()
        self.space1 = space

        operand_indices = []
        for name in operands:
            found = False
            for i, p in enumerate(space):
                if p.name == name:
                    found = True
                    operand_indices.append(i)
            if not found:
                raise KeyError(name)
        self.operand_indices = torch.tensor(operand_indices)

        lower = (sum(space[i].bounds[0] for i in operand_indices)
                 / len(operand_indices))
        upper = (sum(space[i].bounds[1] for i in operand_indices)
                 / len(operand_indices))
        self.space2 = [*space, ol.Scalar(mean_name, lower, upper)]

    def transform(self, X):
        new_feature = X[..., self.operand_indices].mean(dim=-1)
        X = torch.cat((X, new_feature.unsqueeze(-1)), dim=-1)
        return X

    def _apply(self, fn):
        self.operand_indices = fn(self.operand_indices)
        return super()._apply(fn)


def append_mean(operands, mean_name):
    return partial(AppendMean, operands, mean_name)


class BotorchInputTransform(botorch.models.transforms.input.InputTransform,
                            torch.nn.Module):
    def __init__(
            self,
            transform,
            transform_on_train: bool = True,
            transform_on_eval: bool = True,
            transform_on_fantasize: bool = True,
    ):
        super().__init__()
        self.transform_on_train = transform_on_train
        self.transform_on_eval = transform_on_eval
        self.transform_on_fantasize = transform_on_fantasize

        # Save it here so that if it's a nn.Module, it is included in the
        # state_dict logic.
        self.wrapped_t = transform

    def transform(self, X):
        return self.wrapped_t.transform(X)

    def extra_repr(self):
        return f"transform={self.wrapped_t}"


class ChoiceParameterLearnedProjection(torch.nn.Module):
    def __init__(self, space, names, out_name, scalar_offsets=[],
                 batch_shape=torch.Size([])):
        super().__init__()

        self.space1 = space

        embedded_indices = []
        for name in names:
            found = False
            for i, p in enumerate(space):
                if p.name == name:
                    found = True
                    assert isinstance(p, ol.Choice)
                    embedded_indices.append(i)
            if not found:
                raise KeyError(name)
        self.embedded_indices = torch.tensor(embedded_indices)

        self.other_indices = torch.tensor([i for i in range(len(space))
                                           if i not in embedded_indices])

        # Precompute offsets for converting between list of choice integers and
        # indices in an n-hot vector.
        offsets = []
        tot = 0
        for i in self.embedded_indices:
            offsets.append(tot)
            tot += len(space[i].choices)
        choice_offsets = torch.tensor(offsets).unsqueeze(0)
        self.choice_offsets = choice_offsets
        self.D_choice = tot
        self.D_choice_compressed = tot - len(self.embedded_indices)
        # Remove all choice parameters and put an embedding at the end.
        self.space2 = ([space[i] for i in self.other_indices]
                       # TODO I inserted a hack to append indices to the name
                       + [ol.Scalar(out_name + str(i))
                          for i in range(self.D_choice_compressed)])

        offset_scalar_indices = []
        for name in scalar_offsets:
            found = False
            for i, p in enumerate(self.space2):
                if p.name == name:
                    found = True
                    offset_scalar_indices.append(i)
            if not found:
                raise KeyError(name)
        self.offset_scalar_indices = torch.tensor(offset_scalar_indices,
                                                  dtype=torch.long)

        self.D_scalar = len(self.offset_scalar_indices)
        self.D = self.D_choice_compressed + self.D_scalar

        # Project into a space where initially each choice is equidistant from
        # each other choice (the points on a regular n-simplex)
        proj = torch.zeros((self.D, self.D_choice))
        d_offset = self.D_scalar
        c_offset = 0
        for i in self.embedded_indices:
            p = space[i]
            n = len(space[i].choices) - 1

            # Select n orthogonal vectors in n + 1 dimensional space that are
            # orthogonal to the [1, 1, ..., 1] vector. The transpose of this
            # list of vectors is a list of n+1 n-dimensional points that form a
            # regular n-simplex (i.e. they are equally spaced). (Why this works:
            # we could easily form this simplex in n+1 dimensional space by just
            # using the n+1 dimensional identity matrix as a list of
            # equally-spaced points. We then project these points into the
            # n-dimensional hyperplane normal to [1,1,...,1] to get a set of
            # equally-spaced points in n dimensions. But performing this
            # projection is silly, as it's just multiplying with an identity
            # matrix. It's sufficient to just calculate a set of orthonormal
            # vectors in this hyperplane and don't bother projecting the points
            # into it.)

            # Start with a non-orthogonal basis for the hyperplane.
            # https://math.stackexchange.com/questions/3598477/given-u-1-find-orthogonal-matrix-whose-first-column-is-u-1
            basis = np.zeros((n, n + 1))
            basis[:, 0] = 1.0
            if n > 1:
                basis[list(range(n)), list(range(1, n + 1))] = -1.0
            # Now make it orthogonal.
            simplex = scipy.linalg.orth(basis.T).T

            proj[d_offset:(d_offset + n),
                 c_offset:(c_offset + len(p.choices))] = torch.tensor(simplex)
            d_offset += n
            c_offset += len(p.choices)

        proj = proj.repeat(*batch_shape, *[1] * len(proj.shape)).view(
            *batch_shape, *proj.shape)
        proj.requires_grad_()

        self.proj = torch.nn.Parameter(proj)

    def transform(self, X):
        choice_X = X[..., self.embedded_indices]

        # Convert choice ints to n-hot vector
        on_indices = choice_X.long() + self.choice_offsets
        # Temporarily use an extra bit at the end to capture inactive parameters.
        on_indices = torch.where(choice_X == -1, self.D_choice, on_indices)
        # on_indices[torch.where(choice_X == -1)] = self.D_choice
        choice_X_hot = torch.zeros((*choice_X.shape[:-1], self.D_choice + 1),
                                   device=choice_X.device)
        choice_X_hot.scatter_(-1, on_indices, 1.0)
        # Strip the final "inactive" bit.
        choice_X_hot = choice_X_hot[..., :self.D_choice]

        d = torch.matmul(choice_X_hot, self.proj.transpose(-2, -1))

        other_X = X[..., self.other_indices]
        other_X[..., self.offset_scalar_indices] += d[..., :self.D_scalar]
        X = torch.cat((other_X, d[..., self.D_scalar:]), dim=-1)
        return X

    def _apply(self, fn):
        self.embedded_indices = fn(self.embedded_indices)
        self.other_indices = fn(self.other_indices)
        self.choice_offsets = fn(self.choice_offsets)
        self.offset_scalar_indices = fn(self.offset_scalar_indices)
        return super()._apply(fn)
