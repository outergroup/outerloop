import gpytorch
import torch
import torch.profiler


class Reweight(torch.nn.Module):
    def __init__(self, w_module, *transform_modules, ndims_per_model=None):
        super().__init__()
        self.ndims_per_model = ndims_per_model
        self.w_module = w_module
        if len(transform_modules) > 0:
            if len(transform_modules) == 1:
                self.w_transform = transform_modules[0]
            else:
                self.w_transform = torch.nn.Sequential(*transform_modules)

        self.cached_w = None

    def train(self, mode=True):
        if mode:
            self.cached_w = None
        return super().train(mode=mode)

    def _apply(self, fn):
        if self.cached_w is not None:
            self.cached_w = fn(self.cached_w)
        return super()._apply(fn)

    def compute_weights(self):
        w = self.w_module()
        if hasattr(self, "w_transform"):
            w = self.w_transform(w)
        if self.ndims_per_model is not None:
            # if w is just one row, it will broadcast properly to any batch
            # shape.
            if w.ndim > 1:
                needed = self.ndims_per_model - 1
                if needed > 0:
                    w = w.view(*w.shape[:-1], *([1] * needed), w.shape[-1])
        return w

    def forward(self, x):
        with torch.profiler.record_function("Reweight.forward"):
            if self.training:
                w = self.compute_weights()
            else:
                w = self.cached_w
                if w is None:
                    with torch.no_grad():
                        w = self.compute_weights()
                    self.cached_w = w

            if isinstance(x, tuple):
                return tuple(x_ * w for x_ in x)
            else:
                return x * w


class WBetaPairBase(gpytorch.Module):
    def __init__(self,
                 num_pairs,
                 prior=None,
                 batch_shape=torch.Size([])):
        super().__init__()

        name = "raw_alpha"
        self.register_parameter(
            name=name,
            parameter=torch.nn.Parameter(
                torch.zeros(*batch_shape, num_pairs))
        )

        self.register_constraint(name, gpytorch.constraints.Interval(0, 1))

        if prior is not None:
            self.register_prior(f"alpha_prior", prior,
                                WBetaPair.get_alpha,
                                WBetaPair.set_alpha)

    @property
    def alpha(self):
        with torch.profiler.record_function("WBetaPairBase.alpha"):
            if self.training:
                return self.raw_alpha_constraint.transform(self.raw_alpha)
            else:
                with torch.no_grad():
                    return self.raw_alpha_constraint.transform(self.raw_alpha)

    @staticmethod
    def get_alpha(instance):
        return instance.alpha

    @staticmethod
    def set_alpha(instance, v):
        if not torch.is_tensor(v):
            v = torch.as_tensor(v).to(instance.raw_alpha)

        instance.initialize(
            **{"raw_alpha":
               instance.raw_alpha_constraint.inverse_transform(v)}
        )


class WBetaPair(WBetaPairBase):
    def __call__(self):
        with torch.profiler.record_function("WBetaPair.__call__"):
            alpha = self.alpha
            # Build [alpha1, 1-alpha1, alpha2, 1-alpha2, ...]
            w = torch.stack((alpha, 1 - alpha), dim=-1).view(*alpha.shape[:-1], -1)
            return w


class WBetaPairCompose(WBetaPairBase):
    """
    Composes a pair of weights with the given weights.

    Given weights
      [w1, w2, ..., wn]
    the pair of weights
      [alpha, 1 - alpha]
    are composed with these weights to create:
      [w1*alpha, w2*alpha, ..., wn*alpha, 1 - alpha].

    This is useful, for example, when weighting additive vs multiplicative
    kernels
      alpha*(w1*k1 + w2*k2 + ... + wn*kn) + (1-alpha)*(k1*k2*...*kn)
    and expanding the summation so that it's a single weighted sum.
    """

    def __init__(self, lengths, *args, **kwargs):
        super().__init__(len(lengths), *args, **kwargs)

        self.initial_lengths = torch.tensor(lengths)
        adjusted_lengths = self.initial_lengths + 1
        self.total_size = adjusted_lengths.sum()

        indices1 = []
        indices2 = []
        base = 0
        for length in adjusted_lengths:
            indices1 += list(range(base, base + length - 1))
            indices2.append(base + length - 1)
            base += length

        self.indices1 = torch.tensor(indices1)
        self.indices2 = torch.tensor(indices2)

    def _apply(self, fn):
        self.initial_lengths = fn(self.initial_lengths)
        self.indices1 = fn(self.indices1)
        self.indices2 = fn(self.indices2)
        return super()._apply(fn)

    def forward(self, w):
        with torch.profiler.record_function("WBetaPairCompose.forward"):
            alpha = self.alpha

            w2 = torch.zeros(
                (*w.shape[:-1], self.total_size), device=alpha.device
            )
            w2[..., self.indices1] = w * alpha.repeat_interleave(self.initial_lengths, dim=-1)
            w2[..., self.indices2] = 1 - alpha
            return w2


class WSingleBetaPairCompose(WBetaPairBase):
    def __init__(self, lengths, *args, **kwargs):
        super().__init__(len(lengths), *args, **kwargs)
        assert len(lengths) == 1

    def forward(self, w):
        with torch.profiler.record_function("WSingleBetaPairCompose.forward"):
            alpha = self.alpha
            w = alpha * w
            return torch.cat([w, 1 - alpha], dim=-1)


class SoftmaxAtConstraint(torch.nn.Module):
    def __init__(self, lengths, initial_value=None, eps=1e-8):
        super().__init__()
        self.reduced_size = len(lengths)
        self.reduce_indices = torch.arange(
            self.reduced_size
        ).repeat_interleave(torch.tensor(lengths))

        self._initial_value = initial_value
        self.eps = eps

    def _apply(self, fn):
        self.reduce_indices = fn(self.reduce_indices)
        return super()._apply(fn)

    def transform(self, tensor):
        # Perform multiple variable-sized softmaxes in parallel.
        tensor = tensor.exp()
        sums = torch.zeros(
            (*tensor.shape[:-1], self.reduced_size),
            device=tensor.device
        ).index_add_(-1, self.reduce_indices, tensor)
        return tensor / sums[..., self.reduce_indices]

    def inverse_transform(self, transformed_tensor):
        if not torch.is_tensor(transformed_tensor):
            transformed_tensor = torch.as_tensor(
                transformed_tensor,
                device=self.reduce_indices.device)
        return torch.special.logit(transformed_tensor, self.eps)

    @property
    def initial_value(self):
        return self._initial_value

    def __iter__(self):
        # Support botorch's get_parameters_and_bounds, which assumes it can get
        # bounds by iterating over a constraint.
        yield 0.0
        yield 1.0


class WDirichlet(gpytorch.Module):
    def __init__(self, lengths, prior=None, batch_shape=torch.Size([])):
        super().__init__()

        name = "weight_logits"
        self.register_parameter(
            name=name,
            parameter=torch.nn.Parameter(
                torch.zeros(*batch_shape, sum(lengths)))
        )

        self.register_constraint(name, SoftmaxAtConstraint(lengths))

        if prior is not None:
            self.register_prior("weight_prior",
                                prior,
                                WDirichlet.get_weight,
                                WDirichlet.set_weight)

    def __call__(self):
        return self.weight

    @property
    def weight(self):
        with torch.profiler.record_function("WDirichlet.weight"):
            if self.training:
                return self.weight_logits_constraint.transform(self.weight_logits)
            else:
                with torch.no_grad():
                    return self.weight_logits_constraint.transform(
                        self.weight_logits)

    @staticmethod
    def get_weight(instance):
        return instance.weight

    @staticmethod
    def set_weight(instance, v):
        if not torch.is_tensor(v):
            v = torch.as_tensor(v).to(instance.weight_logits)

        instance.initialize(
            **{"weight_logits":
               instance.weight_logits_constraint.inverse_transform(v)}
        )



class WLengthscale(gpytorch.Module):
    def __init__(self,
                 n,
                 prior=None,
                 batch_shape=torch.Size([])):
        super().__init__()

        name = "raw_lengthscale"
        self.register_parameter(
            name=name,
            parameter=torch.nn.Parameter(
                torch.zeros(*batch_shape, n))
        )

        self.register_constraint(name, gpytorch.constraints.Positive())

        if prior is not None:
            self.register_prior(f"lengthscale_prior", prior,
                                WLengthscale.get_lengthscale,
                                WLengthscale.set_lengthscale)

    @property
    def lengthscale(self):
        with torch.profiler.record_function("WLengthscale.lengthscale"):
            if self.training:
                return self.raw_lengthscale_constraint.transform(self.raw_lengthscale)
            else:
                with torch.no_grad():
                    return self.raw_lengthscale_constraint.transform(
                        self.raw_lengthscale)

    @staticmethod
    def get_lengthscale(instance):
        return instance.lengthscale

    @staticmethod
    def set_lengthscale(instance, v):
        if not torch.is_tensor(v):
            v = torch.as_tensor(v).to(instance.raw_lengthscale)

        instance.initialize(
            **{"raw_lengthscale":
               instance.raw_lengthscale_constraint.inverse_transform(v)}
        )

    def __call__(self):
        with torch.profiler.record_function("WLengthscale.__call__"):
            return 1 / self.lengthscale


class WScaleComposeBase(gpytorch.Module):
    """
    Scale a set of weights.
    """
    def __init__(self, lengths, prior=None, batch_shape=torch.Size([])):
        super().__init__()
        name = "raw_scale"
        num_groups = len(lengths)
        self.register_parameter(
            name=name,
            parameter=torch.nn.Parameter(
                torch.zeros(*batch_shape, num_groups))
        )

        self.register_constraint(name, gpytorch.constraints.Positive())

        if prior is not None:
            self.register_prior(f"scale_prior", prior,
                                WScaleComposeBase.get_scale,
                                WScaleComposeBase.set_scale)

    @property
    def scale(self):
        with torch.profiler.record_function("WScaleComposeBase.scale"):
            if self.training:
                return self.raw_scale_constraint.transform(self.raw_scale)
            else:
                with torch.no_grad():
                    return self.raw_scale_constraint.transform(self.raw_scale)

    @staticmethod
    def get_scale(instance):
        return instance.scale

    @staticmethod
    def set_scale(instance, v):
        if not torch.is_tensor(v):
            v = torch.as_tensor(v).to(instance.raw_scale)

        instance.initialize(
            **{"raw_scale":
               instance.raw_scale_constraint.inverse_transform(v)}
        )


class WScaleCompose(WScaleComposeBase):
    def __init__(self, lengths, *args, **kwargs):
        super().__init__(lengths, *args, **kwargs)
        self.scalar_indices = torch.arange(
            sum(lengths)
        ).repeat_interleave(torch.tensor(lengths))

    def _apply(self, fn):
        self.scalar_indices = fn(self.scalar_indices)
        return super()._apply(fn)

    def forward(self, w):
        with torch.profiler.record_function("WScaleCompose.forward"):
            scale = self.scale[..., self.scalar_indices]
            return scale * w


class WSingleScaleCompose(WScaleComposeBase):
    def forward(self, w):
        with torch.profiler.record_function("WSingleScaleCompose.forward"):
            scale = self.scale
            return scale * w


class WIdentityCompose(torch.nn.Module):
    def __init__(self, total_size, indices):
        super().__init__()
        self.total_size = total_size
        self.indices = torch.tensor(indices)

    def _apply(self, fn):
        self.indices = fn(self.indices)
        return super()._apply(fn)

    def forward(self, w):
        with torch.profiler.record_function("WIdentityCompose.forward"):
            w2 = torch.ones(
                (*w.shape[:-1], self.total_size),
                device=w.device
            )
            w2[..., self.indices] = w
            return w2



__all__ = [
    "Reweight",
    "WBetaPair",
    "WBetaPairCompose",
    "WSingleBetaPairCompose",
    "WDirichlet",
    "WLengthscale",
    "WScaleCompose",
    "WSingleScaleCompose",
    "WIdentityCompose",
]
