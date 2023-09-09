import collections

import torch


# gpytorch requires all modules with constraints to have string paths, so can't
# use ModuleList.
class CallAndMerge(torch.nn.ModuleDict):
    def __init__(self, modules, result_index_groups, result_length):
        self.result_index_groups = [torch.as_tensor(group)
                                    for group in result_index_groups]
        self.result_length = result_length

        super().__init__(
            collections.OrderedDict((f"m{i}", m)
                                    for i, m in enumerate(modules)))

    def _apply(self, fn):
        self.result_index_groups = [fn(group)
                                    for group in self.result_index_groups]
        return super()._apply(fn)

    def forward(self, x):
        ret = None

        for module, result_indices in zip(self.values(),
                                          self.result_index_groups):
            v = module(x)
            if ret is None:
                # Use first return value to infer shape.
                ret = torch.zeros((*v.shape[:-1], self.result_length),
                                  device=v.device)
            ret[..., result_indices] = v
        return ret


__all__ = [
    "CallAndMerge",
]
