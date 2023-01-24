import torch


class Select(torch.nn.Module):
    def __init__(self, indices):
        super().__init__()
        self.indices = torch.tensor(indices)

    def _apply(self, fn):
        self.indices = fn(self.indices)
        return super()._apply(fn)

    def forward(self, x):
        if isinstance(x, tuple):
            return tuple(x_[..., self.indices] for x_ in x)
        else:
            return x[..., self.indices]


__all__ = [
    "Select"
]
