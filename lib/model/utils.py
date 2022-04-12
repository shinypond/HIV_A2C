import torch.nn as nn


def get_norm(name):
    if name.lower() == 'layernorm':
        norm = nn.LayerNorm
    elif name.lower() == 'batchnorm':
        norm = nn.BatchNorm1d
    else:
        raise ValueError(f'Normalization {name} is not yet supported.')
    return norm


def get_act(name):
    if name.lower() == 'silu':
        act = nn.SiLU()
    elif name.lower() == 'gelu':
        act = nn.GELU()
    elif name.lower() == 'relu':
        act = nn.ReLU()
    elif name.lower() == 'leakyrelu':
        act = nn.LeakyReLU()
    else:
        raise ValueError(f'Activation {name} is not yet supported.')
    return act

