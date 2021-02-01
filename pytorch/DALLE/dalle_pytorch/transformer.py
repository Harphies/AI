from functools import partial
from inspect import isfunction

import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat


