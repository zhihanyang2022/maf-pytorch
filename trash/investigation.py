import torch
import numpy as np
import matplotlib.pyplot as plt

from mafs import MAF

def main():

    xs = torch.linspace(-6, 6, 300)
    ys = torch.linspace(-6, 6, 300)
    xxs, yys = torch.meshgrid(xs, ys)
    xxs_flat, yys_flat = xxs.reshape(-1, 1), yys.reshape(-1, 1)
    grid = torch.hstack([xxs_flat, yys_flat])

    data = torch.from_numpy(np.load("../maf/2d_data/U3_gmm_samples.npy")).double()

    maf = MAF(data_dim=2, hidden_dims=[100, 100], num_ar_layers=20).double()
    maf.load_state_dict(torch.load("dist_u3.pth"))

    print(data[6919])
    ms10000, vs10000 = maf.get_ms_and_vs(data[:10000])

    # 6919
    # log_prob = maf.log_prob(x=data[6919].reshape(1, -1).double(), ms=ms10000, vs=vs10000)
    log_prob = maf.log_prob(x=data[:10000].double(), ms=ms10000, vs=vs10000)

    print(log_prob)


if __name__ == "__main__":
    main()
