import torch


class SoftmaxConstraint(torch.nn.Module):
    def __init__(self, initial_value=None, eps=1e-8):
        super().__init__()
        self.canary = torch.tensor(0)  # track the current device
        self._initial_value = initial_value
        self.eps = eps

    def _apply(self, fn):
        self.canary = fn(self.canary)
        return super()._apply(fn)

    def transform(self, tensor):
        return torch.special.softmax(tensor, dim=-1)

    def inverse_transform(self, transformed_tensor):
        if not torch.is_tensor(transformed_tensor):
            transformed_tensor = torch.as_tensor(transformed_tensor,
                                                 device=self.canary.device)
        return torch.special.logit(transformed_tensor, self.eps)

    @property
    def initial_value(self):
        return self._initial_value

    def __iter__(self):
        # Support botorch's get_parameters_and_bounds, which assumes it can get
        # bounds by iterating over a constraint.
        yield self.eps
        yield 1.0 - self.eps
