import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch


def stratified_samples(t_n, t_f, N, seed=None):
    """
    Stratified sampling of the interval [t_n, t_f] into N bins,
    returning one random sample in each bin.
    """
    if seed is not None:
        np.random.seed(seed)  # for reproducibility if desired

    Delta = (t_f - t_n) / N
    samples = []
    samples = torch.zeros((N,))
    for i in range(N):
        t_min = t_n + i * Delta
        t_max = t_n + (i + 1) * Delta
        t_i = np.random.uniform(t_min, t_max)
        samples[i] = t_i
    return samples


def compute_accumulated_transmittance(betas):
    accumulated_transmittance = torch.cumprod(betas, 1)
    accumulated_transmittance[:, 0] = 1.0
    return accumulated_transmittance


def rendering(model, rays_o, rays_d, tn, tf, nb_bins=200, device="cpu"):

    t = torch.linspace(tn, tf, nb_bins).to(device)  # [nb_bins]
    delta = torch.cat((torch.tensor([1e10]), t[1:] - t[:-1]))

    # t = stratified_samples(tn, tf, nb_bins)
    # delta = torch.cat((torch.tensor([1e10]), t[1:] - t[:-1]))

    x = rays_o.unsqueeze(1) + t.unsqueeze(0).unsqueeze(-1) * rays_d.unsqueeze(
        1
    )  # [nb_rays, nb_bins, 3]

    colors, density = model.intersect(x.reshape(-1, 3))

    colors = colors.reshape((x.shape[0], nb_bins, 3))  # [nb_rays, nb_bins, 3]
    density = density.reshape((x.shape[0], nb_bins))

    alpha = 1 - torch.exp(-density * delta.unsqueeze(0))  # [nb_rays, nb_bins, 1]
    T = compute_accumulated_transmittance(1 - alpha)  # [nb_rays, nb_bins, 1]
    c = (T.unsqueeze(-1) * alpha.unsqueeze(-1) * colors).sum(1)  # [nb_rays, 3]
    return c


def rendering_manual(model, rays_o, rays_d, tn, tf, nb_bins=200, device="cpu"):

    t = torch.linspace(tn, tf, nb_bins + 1).to(device)  # [nb_bins]
    delta = torch.cat((torch.tensor([1e10]), t[1:] - t[:-1]))

    # t = stratified_samples(tn, tf, nb_bins)
    # delta = torch.cat((torch.tensor([1e10]), t[1:] - t[:-1]))
    aux = torch.zeros((rays_o.shape[0], nb_bins))
    C = torch.zeros((rays_o.shape[0], 3, nb_bins))

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


class Sphere:
    def __init__(self, p1, r1, c1, p2, r2, c2):

        self.p1 = p1
        self.r1 = r1
        self.c1 = c1

        self.p2 = p2
        self.r2 = r2
        self.c2 = c2

    def intersect(self, x):
        cond = (x[:, 0] - self.p1[0]) ** 2 + (x[:, 1] - self.p1[1]) ** 2 + (
            x[:, 2] - self.p1[2]
        ) ** 2 <= self.r1**2
        num_rays = x.shape[0]
        colors = torch.zeros((num_rays, 3))
        density = torch.zeros((num_rays, 1))

        colors[cond] = self.c1
        density[cond] = 20
        return colors, density


def plot_rays(o, d, t):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    pt1 = o
    pt2 = o + t * d
    for k in range(0, pt1.shape[0], 50):
        plt.plot([pt1[k, 0], pt2[k, 0]], [pt1[k, 1], pt2[k, 1]], [pt1[k, 2], pt2[k, 2]])
    plt.savefig("simple.png")
    return None


def main():
    # define parameters of the camera
    W = 400
    H = 200
    f = 1200

    # creating rays
    rays_o = np.zeros((H * W, 3))
    rays_d = np.zeros((H * W, 3))

    v = np.arange(H)
    u = np.arange(W)

    # create a grid all possible values
    u, v = np.meshgrid(u, v)

    # We can compute the direction vector from a point in the world to the image
    dirs = np.stack((u - W / 2, -(v - H / 2), -np.ones_like(u) * f), axis=-1)
    rays_d = dirs / np.linalg.norm(dirs, axis=2, keepdims=True)
    rays_d = rays_d.reshape(-1, 3)

    model = Sphere(
        torch.tensor([0, 0, -1.2]),
        0.05,
        torch.tensor([0.0, 0.8, 0.7]),
        torch.tensor([0, 0, -1.2]),
        0.05,
        torch.tensor([0.0, 0.8, 0.7]),
    )
    px_colors = rendering(
        model, torch.from_numpy(rays_o), torch.from_numpy(rays_d), 0.0, 1.2
    )
    px_manual = rendering_manual(
        model, torch.from_numpy(rays_o), torch.from_numpy(rays_d), 0.0, 1.2
    )
    img = px_colors.reshape(H, W, 3).cpu().numpy()
    img_manual = px_manual.reshape(H, W, 3).cpu().numpy()
    plt.figure(dpi=200)
    plt.imshow(img)
    plt.savefig("nerf_projection.png")

    plt.figure(dpi=200)
    plt.imshow(img_manual)
    plt.savefig("nerf_projection_manual.png")


if __name__ == "__main__":
    main()
