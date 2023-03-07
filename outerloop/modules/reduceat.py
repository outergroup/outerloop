import abc

import gpytorch.kernels.kernel
import torch
import torch.profiler


class ReduceAtBase(torch.nn.Module, abc.ABC):
    def __init__(self, lengths):
        super().__init__()
        self.reduced_size = len(lengths)
        self.reduce_indices = torch.arange(
            self.reduced_size
        ).repeat_interleave(torch.tensor(lengths))

    def _apply(self, fn):
        self.reduce_indices = fn(self.reduce_indices)
        return super()._apply(fn)

    @abc.abstractmethod
    def forward(self, x):
        pass


class SumAt(ReduceAtBase):
    def forward(self, x):
        with torch.profiler.record_function("SumAt.forward"):
            return torch.zeros(
                (*x.shape[:-1], self.reduced_size),
                device=x.device
            ).index_add_(-1, self.reduce_indices, x)


class MeanAt(SumAt):
    def __init__(self, lengths):
        super().__init__(lengths)
        self.lengths = torch.as_tensor(lengths)

    def _apply(self, fn):
        self.lengths = fn(self.lengths)
        return super()._apply(fn)

    def forward(self, x):
        with torch.profiler.record_function("MeanAt.forward"):
            return super().forward(x) / self.lengths


class ProdAt(ReduceAtBase):
    def __init__(self, lengths, all_positive=False):
        super().__init__(lengths)
        self.all_positive = all_positive

    def forward(self, x):
        with torch.profiler.record_function("ProdAt.forward"):
            if self.all_positive:
                # Do a prod via a sum of logs. Both index_reduce_ and prod are
                # slow, in part because their backward pass performs a
                # synchronize to detect whether the product is 0. This code is
                # much faster in the backward pass because it is asynchronous.
                x = x.log()
                x = torch.zeros(
                    (*x.shape[:-1], self.reduced_size),
                    device=x.device
                ).index_add_(-1, self.reduce_indices, x)
                return x.exp()
            else:
                return torch.ones(
                    (*x.shape[:-1], self.reduced_size),
                    device=x.device
                ).index_reduce_(-1, self.reduce_indices, x, "prod")

            return x


class CDistAt(torch.nn.Module):
    def __init__(self, lengths):
        super().__init__()
        self.lengths = lengths
        self.reduced_size = len(lengths)

        self.max_length = max(lengths)
        padded_matrix_rows = []
        padded_matrix_cols = []
        for i, length in enumerate(lengths):
            padded_matrix_rows += [i] * length
            padded_matrix_cols += range(length)
        self.padded_matrix_rows = torch.tensor(padded_matrix_rows)
        self.padded_matrix_cols = torch.tensor(padded_matrix_cols)

    def _apply(self, fn):
        self.padded_matrix_rows = fn(self.padded_matrix_rows)
        self.padded_matrix_cols = fn(self.padded_matrix_cols)
        return super()._apply(fn)

    def forward(self, x):
        with torch.profiler.record_function("CDistAt.forward"):
            x1, x2 = x
            x1_mat = torch.zeros((*x1.shape[:-1], self.reduced_size, self.max_length),
                                 device=x1.device)
            x1_mat[..., self.padded_matrix_rows, self.padded_matrix_cols] = x1
            x1_mat = x1_mat.transpose(-3, -2)
            x2_mat = torch.zeros((*x2.shape[:-1], self.reduced_size, self.max_length),
                                 device=x2.device)
            x2_mat[..., self.padded_matrix_rows, self.padded_matrix_cols] = x2
            x2_mat = x2_mat.transpose(-3, -2)

            # TODO: support setting something like
            # with ol.x1_eq_x2():
            #    ...
            # so that the outside caller can specify this (they know already, we
            # shouldn't infer it here)

            result = gpytorch.kernels.kernel.dist(x1_mat, x2_mat, x1_eq_x2=False)
            return result.permute(*range(result.dim() - 3), -2, -1, -3)


__all__ = [
    "SumAt",
    "MeanAt",
    "ProdAt",
    "CDistAt",
]
