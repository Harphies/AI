import copy
import random
from functools import wraps, partial


import torch
from torch import nn
import torch.nn.functional as F

from torchvision import transforms as T

# helper functions


def exists(val):
    return val is not None


def defaults(val, default):
    return val if exists(val) else default


def singleton(cache_key):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance

            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance
        return wrapper
    return inner_fn


def get_module_device(module):
    return next(module.parameters()).device


def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val


# loss function # (algorithm 1 in the paper)

def loss_fn(
    teacher_logits,
    student_logits,
    teacher_temp,
    student_temp,
    centers,
    eps=1e-20
):
    teacher_logits = teacher_logits.detach()
    student_probs = (student_logits / student_temp).softmax(dim=-1)
    teacher_probs = ((teacher_logits - centers) / teacher_temp).softmax(dim=-1)
    return - (teacher_probs * torch.log(student_probs + eps)).sum(dim=-1).mean()


# augmentation utils

class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p

    def forward(self, x):
        if random() > self.p:
            return x
        return self.fn(x)


# exponetial moving average

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def update_moving_average(ema_updater, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weght, up_weight = ma_params.data, current_params.data
            ma_params.data = ema_updater.update_average(old_weght, up_weight)

# MLP class for projector and predictor
