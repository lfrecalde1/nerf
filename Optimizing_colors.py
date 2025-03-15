import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import torch.nn.functional as F


def stratified_samples(t_n, t_f, N, seed=None):
    """
    Stratified sampling of the interval [t_n, t_f] into N bins,
    returning one random sample in each bin.
    """
    if seed is not None:
        np.random.seed(seed)  # for reproducibility if desired

    Delta = (t_f - t_n) / N
    samples = torch.zeros((N,))
    for i in range(N):
        t_min = t_n + i * Delta
        t_max = t_n + (i + 1) * Delta
        t_i = np.random.uniform(t_min, t_max)
        samples[i] = t_i
    return samples


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


def rendering(model, rays_o, rays_d, tn, tf, nb_bins=200, device="gpu"):
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
    T = compute_accumulated_transmittance(1 - alpha)  # [nb_rays, nb_bins]
    c = (T.unsqueeze(-1) * alpha.unsqueeze(-1) * colors).sum(1)  # [nb_rays, 3]
    return c


class Sphere:
    def __init__(self, p1, r1, c1, p2, r2, c2):
        self.p1 = p1
        self.r1 = r1
        self.c1 = c1

        self.p2 = p2
        self.r2 = r2
        self.c2 = c2

    def intersect(self, x):
        # Compute a simple sphere intersection condition
        cond = (
            (x[:, 0] - self.p1[0]) ** 2
            + (x[:, 1] - self.p1[1]) ** 2
            + (x[:, 2] - self.p1[2]) ** 2
        ) <= self.r1**2
        num_rays = x.shape[0]
        colors = torch.zeros((num_rays, 3), device=x.device)
        density = torch.zeros((num_rays, 1), device=x.device)

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
    # Select device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Define parameters of the camera
    W = 400
    H = 200
    f = 1200

    # Creating rays on the CPU first using numpy
    rays_o_np = np.zeros((H * W, 3))
    rays_d_np = np.zeros((H * W, 3))

    v = np.arange(H)
    u = np.arange(W)
    u, v = np.meshgrid(u, v)

    # Compute direction vectors for each pixel
    dirs = np.stack((u - W / 2, -(v - H / 2), -np.ones_like(u) * f), axis=-1)
    rays_d_np = dirs / np.linalg.norm(dirs, axis=2, keepdims=True)
    rays_d_np = rays_d_np.reshape(-1, 3)

    # Convert numpy arrays to torch tensors and move to GPU
    rays_o = torch.from_numpy(rays_o_np).float().to(device)
    rays_d = torch.from_numpy(rays_d_np).float().to(device)

    # Create a model (Sphere) on the GPU
    model = Sphere(
        torch.tensor([0, 0, -1.2], device=device, dtype=torch.float32),
        0.05,
        torch.tensor([0.0, 0.8, 0.7], device=device, dtype=torch.float32),
        torch.tensor([0, 0, -1.2], device=device, dtype=torch.float32),
        0.05,
        torch.tensor([0.0, 0.8, 0.7], device=device, dtype=torch.float32),
    )
    # Render the image to obtain target colors
    b = rendering(model, rays_o, rays_d, 0.0, 1.2, device=device)
    b = b.reshape(H, W, 3)

    # Set up parameter optimization: optimize sphere color
    c_to_optimize = torch.tensor(
        [0.0, 0.0, 0.0], device=device, dtype=torch.float32, requires_grad=True
    )
    model = Sphere(
        torch.tensor([0, 0, -1.2], device=device, dtype=torch.float32),
        0.05,
        c_to_optimize,
        torch.tensor([0, 0, -1.2], device=device, dtype=torch.float32),
        0.05,
        c_to_optimize,
    )
    optimizer = torch.optim.SGD([c_to_optimize], lr=1e-1)

    # Optimize over several epochs
    for epoch in range(1000):
        model = Sphere(
            torch.tensor([0, 0, -1.2], device=device, dtype=torch.float32),
            0.05,
            c_to_optimize,
            torch.tensor([0, 0, -1.2], device=device, dtype=torch.float32),
            0.05,
            c_to_optimize,
        )
        Ax = rendering(model, rays_o, rays_d, 0.0, 1.2, device=device)
        px_colors_model = Ax.reshape(H, W, 3)

        loss = F.l1_loss(b, px_colors_model)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            img = (
                px_colors_model.cpu().data.numpy()
            )  # Move image back to CPU for visualization
            print(f"Epoch {epoch}, Loss: {loss.item()}")
            print("Current optimized color:", c_to_optimize)

    # Visualize the result (back on CPU for matplotlib)
    plt.figure(dpi=200)
    plt.imshow(img)
    plt.savefig("Optimization_projection.png")


if __name__ == "__main__":
    main()
