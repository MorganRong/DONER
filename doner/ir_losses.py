import torch
from torch.nn import functional as F

def reconstruction(pred, clean):
    error = pred - clean
    absolute = (error.square() + 1e-9).sqrt().mean()
    gradient = sum(torch.diff(error, dim=dim).abs().mean() for dim in (-1, -2))
    return absolute + .2 * gradient


def response_loss(pred, plus, signal_value):


    error = plus - pred - signal_value

    def valid_ratio(squared_error, energy):
        energy = energy.detach()
        valid = energy > 1e-10
        if not bool(valid.any()):
            return squared_error.sum()*0.
        return (squared_error[valid]/energy[valid]).mean()

    value = valid_ratio(error.square().mean((1,2,3)),
                        signal_value.square().mean((1,2,3)))
    gradient_error = error.new_zeros(error.shape[0])
    gradient_energy = error.new_zeros(error.shape[0])
    for dim in (-1, -2):
        de = torch.diff(error, dim=dim)
        ds = torch.diff(signal_value, dim=dim)
        gradient_error = gradient_error + de.square().mean((1,2,3))
        gradient_energy = gradient_energy + ds.square().mean((1,2,3))


    value = value + .2*valid_ratio(gradient_error, gradient_energy)
    return value


def analytic_identity_loss(pred, clean, signal_value):


    error = pred - clean
    with torch.no_grad():
        support = F.max_pool2d((signal_value.abs() > 1e-6).to(error.dtype),
                              kernel_size=9, stride=1, padding=4)

    def relative_term(numerator, known, mask):
        energy = (known.square()*mask).sum((1,2,3)).detach()
        valid = energy > 1e-12
        if not bool(valid.any()):
            return numerator.sum()*0.
        residual = (numerator.square()*mask).sum((1,2,3))
        return (residual[valid]/energy[valid]).mean()

    value = relative_term(error, signal_value, support)
    gradient_error = error.new_zeros(error.shape[0])
    gradient_energy = error.new_zeros(error.shape[0])
    for dim in (-1, -2):
        derivative_error = torch.diff(error, dim=dim)
        derivative_signal = torch.diff(signal_value, dim=dim)
        if dim == -1:
            adjacent_support = torch.maximum(support[..., 1:], support[..., :-1])
        else:
            adjacent_support = torch.maximum(support[..., 1:, :], support[..., :-1, :])
        gradient_error = gradient_error + (derivative_error.square()*adjacent_support).sum((1,2,3))
        gradient_energy = gradient_energy + (derivative_signal.square()*adjacent_support).sum((1,2,3)).detach()
    valid_gradient = gradient_energy > 1e-10
    if bool(valid_gradient.any()):
        value = value + .2*(gradient_error[valid_gradient]/gradient_energy[valid_gradient]).mean()
    far_background = 1.-support
    pixels = far_background.sum((1,2,3)).detach()
    valid_background = pixels > 0
    if bool(valid_background.any()):
        squared_error = (error.square()*far_background).sum((1,2,3))
        background_mse = squared_error[valid_background]/pixels[valid_background]
        value = value + .1*(background_mse/(.01**2)).mean()
    return value
