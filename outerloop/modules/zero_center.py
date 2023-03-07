import torch
import torch.profiler


class ZeroCenter(torch.nn.Module):
    """
    Mimics the zero-centering of gpytorch's kernels.
    """

    def forward(self, x):
        with torch.profiler.record_function("ZeroCenter.forward"):
            if isinstance(x, tuple):
                # Use mean of x1
                xm = x[0]
            else:
                xm = x

            mean = xm.reshape(-1, xm.size(-1)).mean(0)[(None,) * (xm.dim() - 1)]

            if isinstance(x, tuple):
                return tuple(x_ - mean for x_ in x)
            else:
                return x - mean


__all__ = [
    "ZeroCenter",
]
