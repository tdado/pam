import sys
sys.path.append('../stylegan_xl')
import numpy as np
import torch
from torch.utils.data import DataLoader
import dnnlib
import legacy
from classes import Data
from datasets import get_styxl
from decoder import train_linear, reconstruct
from imports import *


def main():
    device = torch.device("cuda")
    seed = 9
    n_batch = 32
    n_epochs = 150

    # StyleGAN-XL
    sys.path.append('../stylegan_xl')
    with dnnlib.util.open_url("../imagenet512.pkl") as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)

    w_te, x_te, w_tr, x_tr, _ = get_styxl()
    x_te = torch.from_numpy(x_te).to(torch.float32).to(device)
    training_data = Data(x_tr, w_tr)
    training_loader = DataLoader(training_data, batch_size=n_batch, shuffle=True)
    model = train_linear(device, n_epochs, x_te.shape[1], seed, training_loader, "lin_sty")
    model.eval()
    with torch.no_grad():
        y = model(x_te)
    ys = y.unsqueeze(1).repeat(1, 37, 1)
    for i in range(len(ys)):
        reconstruct(G, i, ys, "sty/recon_lin")
    np.save("/home/pam/sty/y_sty_lin.npy", y.cpu().detach().numpy())


if __name__ == "__main__":
    main()