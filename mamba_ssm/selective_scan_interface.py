# Copyright (c) 2023, Tri Dao, Albert Gu.

import torch
import torch.nn.functional as F
from einops import rearrange, repeat


def selective_scan(u, delta, A, B, C, D=None, z=None):
    """
    u: r(B D L)
    delta: r(B D L)
    A: c(D N) or r(D N)
    B: c(D N) or r(B N L) or r(B N 2L) or r(B G N L) or (B G N L)
    C: c(D N) or r(B N L) or r(B N 2L) or r(B G N L) or (B G N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32

    out: r(B D L)
    last_state (optional): r(B D dstate) or c(B D dstate)
    """
    dtype_in = u.dtype
    u = u.float()
    B = B.float()
    C = C.float()
    delta = delta.float()

    batch, seqlen, dim, dstate = u.shape[0], u.shape[2], A.shape[0], A.shape[1]

    x = A.new_zeros((batch, dim, dstate))

    deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
    deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)

    ys = []
    for i in range(seqlen):
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
        #if y.is_complex():
        #    y = y.real * 2
        ys.append(y)
    y = torch.stack(ys, dim=2) # (batch dim L)

    out = y + u * rearrange(D, "d -> d 1")
    if z is not None:
        out = out * F.silu(z)
    out = out.to(dtype=dtype_in)
    return (out, x)
