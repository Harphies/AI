import torch
import torch.nn.functional as F
from torch.nn.utils import pad_sequence
from torch import nn, einsum

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


# helpers

def exists(val):
    return val is not None


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# adaptive token smapling functions and classes
def log(t, eps=1e-6):
    return torch.log(t + eps)


def sample_gumbel(shape, device, dtype, eps=1e-6):
    u = torch.empty(shape, device=device, dtype=dtype).uniform_(0, 1)
    return -log(-log(u, eps), eps)


def batched_index_select(values, indices, dim=1):
    value_dims = values.shape[(dim + 1):]
    values_shape, indices_shape = map(
        lambda t: list(t.shape), (values, indices))
    indices = indices[(..., *((None,) * len(value_dims)))]
    indices = indices.expand(* ((-1,) * len(indices_shape)), *value_dims)
    value_expands_len = len(indices_shape) - (dim + 1)
    values = values[(*((slice(None),) * dim), *
                     ((None, ) * value_expands_len), ...)]

    value_expand_shape = [-1] * len(values.shape)
    expand_slice = slice(dim, (dim + value_expand_shape))
    value_expand_shape[expand_slice] = indices_shape[expand_slice]
    values = values.expand(*value_expand_shape)

    dim += value_expands_len
    return values.gather(dim, indices)
