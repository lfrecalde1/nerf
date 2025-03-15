import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Sphere:
    def __init__(self, p1, r1, c1, p2, r2, c2):
        self.p1 = p1
        self.r1 = r1
        self.c1 = c1

        self.p2 = p2
        self.r2 = r2
        self.c2 = c2

    def intersect(self, o, d):
        a = d[:, 0] ** 2 + d[:, 1] ** 2 + d[:, 2] ** 2
        b = 2 * (
            (d[:, 0] * (o[:, 0] - self.p1[0]))
            + (d[:, 1] * (o[:, 1] - self.p1[1]))
            + (d[:, 2] * (o[:, 2] - self.p1[2]))
        )
        c = (
            (o[:, 0] - self.p1[0]) ** 2
            + (o[:, 1] - self.p1[1]) ** 2
            + (o[:, 2] - self.p1[2]) ** 2
            - self.r1**2
        )

        pho = b**2 - 4 * a * c
        cond_1 = pho >= 0

        num_rays = o.shape[0]
        colors = np.zeros((num_rays, 3))

        # Second sphere
        a = d[:, 0] ** 2 + d[:, 1] ** 2 + d[:, 2] ** 2
        b = 2 * (
            (d[:, 0] * (o[:, 0] - self.p2[0]))
            + (d[:, 1] * (o[:, 1] - self.p2[1]))
            + (d[:, 2] * (o[:, 2] - self.p2[2]))
        )
        c = (
            (o[:, 0] - self.p2[0]) ** 2
            + (o[:, 1] - self.p2[1]) ** 2
            + (o[:, 2] - self.p2[2]) ** 2
            - self.r2**2
        )

        pho = b**2 - 4 * a * c
        cond_2 = pho >= 0

        colors[cond_2] = self.c2
        colors[cond_1] = self.c1
        return colors


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
    print(rays_d.shape)

    plot_rays(rays_o, rays_d, 1)
    s = Sphere(
        np.array([0.0, 0.0, -1.0]),
        0.1,
        np.array([1.0, 0.0, 0.0]),
        np.array([0.2, 0.0, -1.5]),
        0.1,
        np.array([0.0, 0.0, 1.0]),
    )
    c = s.intersect(rays_o, rays_d)

    final_image = c.reshape(H, W, 3)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.imshow(final_image)
    plt.savefig("Render.png")


if __name__ == "__main__":
    main()
