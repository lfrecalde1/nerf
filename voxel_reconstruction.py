import torch
import numpy as np
import os
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt

from dataset import get_rays
from rendering import rendering
from tqdm import tqdm

from voxels_nn import Voxels
import torch.nn.functional as F


def training(
    model,
    optimizer,
    schedules,
    tn,
    tf,
    nb_bins,
    nb_epoch,
    data_loader,
    H,
    W,
    device="gpu",
):
    training_loss = []
    for epoch in tqdm(range(nb_epoch)):
        for batch in tqdm(data_loader):
            o = batch[:, :3].to(device)
            d = batch[:, 3:6].to(device)
            target = batch[:, 6:].to(device)

            prediction = rendering(
                model,
                o,
                d,
                tn,
                tf,
                nb_bins=nb_bins,
                device=device,
            )

            loss = ((prediction - target) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # training_loss.append(loss.item())
        schedules.step()

    return np.array(training_loss)


batch_size = 1024
o, d, target_px_values, H, W = get_rays("fox", mode="train")
dataloader_warmup = DataLoader(
    torch.cat(
        (
            torch.from_numpy(o)
            .reshape(90, 400, 400, 3)[:, 100:300, 100:300, :]
            .reshape(-1, 3),
            torch.from_numpy(d)
            .reshape(90, 400, 400, 3)[:, 100:300, 100:300, :]
            .reshape(-1, 3),
            torch.from_numpy(target_px_values)
            .reshape(90, 400, 400, 3)[:, 100:300, 100:300, :]
            .reshape(-1, 3),
        ),
        dim=1,
    ),
    batch_size=batch_size,
    shuffle=True,
)

dataloader_train = DataLoader(
    torch.cat(
        (
            torch.from_numpy(o).reshape(-1, 3),
            torch.from_numpy(d).reshape(-1, 3),
            torch.from_numpy(target_px_values).reshape(-1, 3),
        ),
        dim=1,
    ),
    batch_size=batch_size,
    shuffle=True,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
model = Voxels(scale=3.0, device=device)
epochs = 15
lr = 1e-3
gamma = 0.5

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
schedule = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[5, 10], gamma=gamma
)
tn = 8
tf = 12
bins = 100

loss_init = training(
    model,
    optimizer,
    schedule,
    tn,
    tf,
    bins,
    1,
    dataloader_warmup,
    H,
    W,
    device=device,
)

loss = training(
    model,
    optimizer,
    schedule,
    tn,
    tf,
    bins,
    epochs,
    dataloader_train,
    H,
    W,
    device=device,
)

img = rendering(
    model,
    torch.from_numpy(o[0]).to(device),
    torch.from_numpy(d[0]).to(device),
    tn,
    tf,
    nb_bins=bins,
    device=device,
)
img = img.cpu().data.numpy()
img = img.reshape(H, W, 3)
plt.figure(dpi=200)
plt.imshow(img)
plt.savefig("optimization_fox_1.png")

img = rendering(
    model,
    torch.from_numpy(o[1]).to(device),
    torch.from_numpy(d[1]).to(device),
    tn,
    tf,
    nb_bins=bins,
    device=device,
)

img = img.cpu().data.numpy()
img = img.reshape(H, W, 3)
plt.figure(dpi=200)
plt.imshow(img)
plt.savefig("optimization_fox_2.png")

img = rendering(
    model,
    torch.from_numpy(o[2]).to(device),
    torch.from_numpy(d[2]).to(device),
    tn,
    tf,
    nb_bins=bins,
    device=device,
)

img = img.cpu().data.numpy()
img = img.reshape(H, W, 3)
plt.figure(dpi=200)
plt.imshow(img)
plt.savefig("optimization_fox_3.png")
