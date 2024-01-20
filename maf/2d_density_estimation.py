import argparse
import os

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

import wandb

from mades import MADE, MADE_MOG
from mafs import MAF, MAF_MOG


# Parse arguments, check validity of arguments, and init wandb

parser = argparse.ArgumentParser()
parser.add_argument("dataset", help="Name of potential function")
parser.add_argument("model", type=str)
parser.add_argument("seed", type=int)
parser.add_argument("-num_ar_layers", help="Number of autoregressive layers", type=int)
parser.add_argument("-alternate_input_order", type=int)

args = parser.parse_args()
config = vars(args)

dataset = config["dataset"]
model = config["model"]
seed = config["seed"]
num_ar_layers = config["num_ar_layers"]
alternate = bool(config["alternate_input_order"])

print(alternate)

assert model in ["made", "made-mog", "maf", "maf-mog"]

if model.startswith("maf"):
    assert num_ar_layers is not None
    assert alternate is not None
else:
    assert num_ar_layers is None
    assert alternate is None

run = wandb.init(
    project="masked-autoregressive-flow",
    group=f"{dataset} {model}" if model.startswith("made") else f"{dataset} {model} {num_ar_layers} {alternate}",
    name=f"seed={seed}",
    reinit=True
)

# Select the correct dataset

data = np.load(f"./2d_data/{dataset}_gmm_samples.npy")
train_data = torch.from_numpy(data[:10000])
test_data = torch.from_numpy(data[10000:])

train_ds = TensorDataset(train_data)
train_dl = DataLoader(train_ds, batch_size=100)

# Instantiate the correct neural density model

np.random.seed(seed)
torch.manual_seed(seed)

print(f"Selected neural density model: {model}")

if model == "made":
    dist = MADE(data_dim=2, hidden_dims=[100, 100])
elif model == "made-mog":
    dist = MADE_MOG(data_dim=2, hidden_dims=[100, 100], num_components=10)
elif model == "maf":
    dist = MAF(
        data_dim=2, hidden_dims=[100, 100], num_ar_layers=num_ar_layers,
        alternate_input_order=alternate
    )
elif model == "maf-mog":
    dist = MAF_MOG(
        data_dim=2, hidden_dims=[100, 100], num_components=10, num_ar_layers=num_ar_layers,
        alternate_input_order=alternate
    )

# Training

opt = optim.Adam(dist.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[100, 200], gamma=1 / 3)

for i in range(300):

    losses_batch = []

    for (xb,) in train_dl:

        loss = - dist.log_prob(xb).mean()

        losses_batch.append(float(loss))

        opt.zero_grad()
        loss.backward()
        opt.step()

    train_loss = np.mean(losses_batch)

    with torch.no_grad():
        test_loss = float(- dist.log_prob(test_data).mean())

    scheduler.step()

    wandb.log({
        "Loss (Train)": train_loss,
        "Loss (Test)": test_loss,
    }, step=i+1)

    print(f"Epoch {i + 1:3.0f} | Train Loss {train_loss:6.3f} | Test Loss {test_loss:6.3f}")

# Plotting and saving

xs = torch.linspace(-6, 6, 100)
ys = torch.linspace(-6, 6, 100)
xxs, yys = torch.meshgrid(xs, ys)
xxs_flat, yys_flat = xxs.reshape(-1, 1), yys.reshape(-1, 1)
grid = torch.hstack([xxs_flat, yys_flat])

with torch.no_grad():
    if model in ["made", "made-mog"]:
        probs = dist.log_prob(grid).exp()
    elif model in ["maf", "maf-mog"]:
        ms, vs = dist.get_ms_and_vs(train_data)  # batch norm parameters
        probs = dist.log_prob(grid, ms=ms, vs=vs).exp()

plt.figure(figsize=(5, 5))

plt.contourf(
    xxs.numpy(), yys.numpy(), probs.numpy().reshape(100, 100),
    levels=20, cmap="turbo"
)

plt.xticks([])
plt.yticks([])

# TODO: generate some data (especially make sure this works for MAFs)
# TODO: add the half moon datasets

if model in ["made", "made-mog"]:
    png_name = f"{dataset} {model.upper()} Density"
elif model in ["maf", "maf-mog"]:
    if alternate:
        png_name = f"{dataset} {model.upper()} ({num_ar_layers}) Density"
    else:
        png_name = f"{dataset} {model.upper()} ({num_ar_layers} fixed) Density"

plt.title(png_name, fontsize=20, fontweight="bold")

plt.savefig(os.path.join(wandb.run.dir, png_name + ".png"), dpi=300, bbox_inches="tight")

torch.save(dist.state_dict(), os.path.join(wandb.run.dir, "dist.pth"))

# Rest now

run.finish()
