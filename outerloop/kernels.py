import gpytorch


class KernelFromSequential(gpytorch.kernels.Kernel):
    def __init__(self, module, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.module = module

    def forward(self, x1, x2, diag: bool = False, last_dim_is_batch: bool = False):
        assert not diag
        assert not last_dim_is_batch

        d = self.module((x1, x2))
        return d.squeeze(-1)
