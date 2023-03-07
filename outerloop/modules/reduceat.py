import abc

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


__all__ = [
    "SumAt",
    "ProdAt",
]
