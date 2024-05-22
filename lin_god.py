import sys
sys.path.append('../stylegan_xl')
import numpy as np
import torch
from torch.utils.data import DataLoader
import dnnlib
import legacy
from classes import Data
from datasets import get_god
from decoder import train_linear, reconstruct
from imports import *


def main():
    device = torch.device("cuda")
    seed = 6
    n_batch = 32
    n_epochs = 100

    # StyleGAN-XL
    sys.path.append('../stylegan_xl')
    with dnnlib.util.open_url("../imagenet512.pkl") as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)

    w_te, x_te, w_tr, x_tr, _ = get_god()
    x_te = torch.from_numpy(x_te).to(torch.float32).to(device)
    training_data = Data(x_tr, w_tr)
    training_loader = DataLoader(training_data, batch_size=n_batch, shuffle=True)
    model = train_linear(device, n_epochs, x_te.shape[1], seed, training_loader, "lin_god")
    model.eval()
    with torch.no_grad():
        y = model(x_te)
    ys = y.unsqueeze(1).repeat(1, 37, 1)
    for i in range(len(ys)):
        reconstruct(G, i, ys, "god/recon_lin")
    np.save("/home/tdado/pam/god/y_god_lin.npy", y.cpu().detach().numpy())


if __name__ == "__main__":
    main()