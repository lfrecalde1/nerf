import numpy as np
import matplotlib.pyplot as plt


def stratified_samples(t_n, t_f, N, seed=None):
    """
    Stratified sampling of the interval [t_n, t_f] into N bins,
    returning one random sample in each bin.
    """
    if seed is not None:
        np.random.seed(seed)  # for reproducibility if desired

    Delta = (t_f - t_n) / N
    samples = []
    for i in range(N):
        t_min = t_n + i * Delta
        t_max = t_n + (i + 1) * Delta
        t_i = np.random.uniform(t_min, t_max)
        samples.append(t_i)
    return np.array(samples)


def classic_samples(t_n, t_f, N, seed=None):
    """
    Stratified sampling of the interval [t_n, t_f] into N bins,
    returning one random sample in each bin.
    """
    t = np.linspace(t_n, t_f, N + 1)
    delta = t[1:] - t[:-1]
    return delta


# Parameters
t_n, t_f = 0.0, 1.0
N = 64

# Generate stratified samples
ts = stratified_samples(t_n, t_f, N, seed=42)
ts_classic = classic_samples(t_n, t_f, N, seed=42)
print(ts.shape)
print(ts_classic.shape)

# Optionally, you can also make a simple scatter plot
plt.figure(figsize=(8, 2))
plt.scatter(range(N), ts, alpha=0.7)
plt.xlabel("Sample index (i)")
plt.ylabel("Sample value (t)")
plt.title("Stratified Samples vs. Index")

plt.figure(figsize=(8, 2))
plt.scatter(range(N), ts_classic, alpha=0.7)
plt.xlabel("Sample index (i)")
plt.ylabel("Sample value (t)")
plt.title("Classic Samples vs. Index")

plt.show()
