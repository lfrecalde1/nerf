import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import torch.nn.functional as F


def compute_accumulated_transmittance(betas):
    accumulated_transmittance = torch.cumprod(betas, dim=1)
    ones_column = torch.ones(
        accumulated_transmittance.shape[0], 1, device=accumulated_transmittance.device
    )
    result = torch.cat((ones_column, accumulated_transmittance[:, :-1]), dim=1)
    return result


def rendering_manual(model, rays_o, rays_d, tn, tf, nb_bins=200, device="gpu"):

    t = torch.linspace(tn, tf, nb_bins + 1).to(device)  # [nb_bins]

    aux = torch.zeros((rays_o.shape[0], nb_bins)).to(device)
    C = torch.zeros((rays_o.shape[0], 3, nb_bins)).to(device)

    for k in range(0, nb_bins):
        x = rays_o + t[k] * rays_d  # [nb_rays, nb_bins, 3]
        colors, density = model.intersect(x)
        delta = t[k + 1] - t[k]
        alpha = 1 - torch.exp(-density * delta)  # [nb_rays, nb_bins, 1]
        aux_multiplication = density * delta
        aux[:, k] = aux_multiplication.reshape((rays_o.shape[0],))
        T = torch.exp(-aux[:, 0:k].sum(1))  # [nb_rays, nb_bins, 1]
        T = T.unsqueeze(-1)
        C[:, :, k] = T * alpha * colors

    C = C.sum(2)
    return C


def rendering(
    model, rays_o, rays_d, tn, tf, nb_bins=200, device="gpu", white_background=True
):
    # Create linspace on the GPU
    t = torch.linspace(tn, tf, nb_bins, device=device)  # [nb_bins]
    # Make sure the constant tensor is on the same device
    delta = torch.cat((torch.tensor([1e10], device=device), t[1:] - t[:-1]))

    # Compute the sample positions along the rays
    x = rays_o.unsqueeze(1) + t.unsqueeze(0).unsqueeze(-1) * rays_d.unsqueeze(
        1
    )  # [nb_rays, nb_bins, 3]

    colors, density = model.intersect(x.reshape(-1, 3))
    colors = colors.reshape((x.shape[0], nb_bins, 3))  # [nb_rays, nb_bins, 3]
    density = density.reshape((x.shape[0], nb_bins))

    alpha = 1 - torch.exp(-density * delta.unsqueeze(0))  # [nb_rays, nb_bins]
    # T = compute_accumulated_transmittance(1 - alpha)
    weights = compute_accumulated_transmittance(1 - alpha) * alpha  # [nb_rays, nb_bins]
    if white_background:
        c_aux = (weights.unsqueeze(-1) * colors).sum(1)  # [nb_rays, 3]
        weight_sum = weights.sum(-1)
        c = c_aux + 1 - weight_sum.unsqueeze(-1)
    else:
        c = (weights.unsqueeze(-1) * colors).sum(1)  # [nb_rays, 3]
    return c
