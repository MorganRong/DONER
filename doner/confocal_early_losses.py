import torch
from torch.nn import functional as F

def reconstruction(pred, clean):
    error = pred - clean
    absolute = (error.square() + 1e-9).sqrt().mean()
    gradient = sum(torch.diff(error, dim=dim).abs().mean() for dim in (-1, -2))
    return absolute + .2 * gradient

def response_loss(pred, plus, signal_value):


    error = plus - pred - signal_value
    energy = signal_value.square().mean((1, 2, 3)).detach().clamp_min(1e-10)
    value = (error.square().mean((1, 2, 3)) / energy).mean()
    for dim in (-1, -2):
        de = torch.diff(error, dim=dim)
        ds = torch.diff(signal_value, dim=dim)
        energy = ds.square().mean((1, 2, 3)).detach().clamp_min(1e-10)
        value = value + .1 * (de.square().mean((1, 2, 3)) / energy).mean()
    return value
