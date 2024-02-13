# Copyright (c) 2023, Tri Dao.
# Implement residual + layer_norm / rms_norm.

import math
import torch
import torch.nn.functional as F


def layer_norm_fn(x, weight, bias, residual=None, eps=1e-6, prenorm=False, upcast=False):
    dtype = x.dtype
    if upcast:
        weight = weight.float()
        bias = bias.float() if bias is not None else None
    if upcast:
        x = x.float()
        residual = residual.float() if residual is not None else residual
    if residual is not None:
        x = (x + residual).to(x.dtype)
    out = F.layer_norm(x.to(weight.dtype), x.shape[-1:], weight=weight, bias=bias, eps=eps).to(
        dtype
    )
    return out if not prenorm else (out, x)


def rms_norm_fn(x, weight, bias, residual=None, eps=1e-6, prenorm=False, upcast=False):
    dtype = x.dtype
    if upcast:
        weight = weight.float()
        bias = bias.float() if bias is not None else None
    if upcast:
        x = x.float()
        residual = residual.float() if residual is not None else residual
    if residual is not None:
        x = (x + residual).to(x.dtype)
    rstd = 1 / torch.sqrt((x.square()).mean(dim=-1, keepdim=True) + eps)
    out = (x * rstd * weight) + bias if bias is not None else (x * rstd * weight)
    out = out.to(dtype)
    return out if not prenorm else (out, x)


# class RMSNorm(torch.nn.Module):
#     def __init__(self, hidden_size, eps=1e-5, device=None, dtype=None):
#         factory_kwargs = {"device": device, "dtype": dtype}
#         super().__init__()
#         self.eps = eps
#         self.weight = torch.nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
#         self.register_parameter("bias", None)
#         self.reset_parameters()

#     def reset_parameters(self):
#         torch.nn.init.ones_(self.weight)

#     def forward(self, x, residual=None, prenorm=False, residual_in_fp32=False):
#         return rms_norm_fn(
#             x,
#             self.weight,
#             self.bias,
#             residual=residual,
#             eps=self.eps,
#             prenorm=prenorm,
#             #residual_in_fp32=residual_in_fp32,
#         )

class RMSNorm(torch.nn.Module):
    #https://github.com/bzhangGo/rmsnorm/blob/master/rmsnorm_torch.py
    def __init__(self, d, p=-1., eps=1e-5, bias=False, device=None, dtype=None):
        """
            Root Mean Square Layer Normalization
        :param d: model size
        :param p: partial RMSNorm, valid value [0, 1], default -1.0 (disabled)
        :param eps:  epsilon value, default 1e-8
        :param bias: whether use bias term for RMSNorm, disabled by
            default because RMSNorm doesn't enforce re-centering invariance.
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.eps = eps
        self.d = d
        self.p = p
        self.bias = bias

        self.weight = torch.nn.Parameter(torch.ones(d,**factory_kwargs))
        self.register_parameter("weight", self.weight)

        if self.bias:
            self.offset = torch.nn.Parameter(torch.zeros(d,**factory_kwargs))
            self.register_parameter("offset", self.offset)

    def forward(self, x):
        if self.p < 0. or self.p > 1.:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)

            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size

        rms_x = norm_x * d_x ** (-1. / 2)
        x_normed = x / (rms_x + self.eps)

        if self.bias:
            return self.weight * x_normed + self.offset

        return self.weight * x_normed