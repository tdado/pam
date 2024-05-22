import sys
sys.path.append('../stylegan_xl')
import numpy as np
import torch
from torch.utils.data import DataLoader
import dnnlib
import legacy
from classes import Data
from datasets import get_styxl
from decoder import train_pam, reconstruct
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

    w_te, x_te, w_tr, x_tr, h = get_styxl()
    x_te = torch.from_numpy(x_te).to(torch.float32).to(device)
    x_te = [x_te[:, h[i]:h[i+1]] for i in range(len(h)-1)]
    training_data = Data(x_tr, w_tr)
    training_loader = DataLoader(training_data, batch_size=n_batch, shuffle=True)
    model = train_pam(device, n_epochs, seed, training_loader, h, "sty")
    model.eval()
    with torch.no_grad():
        y, k, v, q = model(x_te)

    ys = y.unsqueeze(1).repeat(1, 37, 1)
    for i in range(len(ys)):
        reconstruct(G, i, ys, "sty/recon")

    np.save("/home/tdado/pam/sty/y_sty.npy", y.cpu().detach().numpy())
    np.save("/home/tdado/pam/sty/k_sty.npy", k.cpu().detach().numpy())
    np.save("/home/tdado/pam/sty/q_sty.npy", q.cpu().detach().numpy())
    np.save("/home/tdado/pam/sty/v_sty.npy", v.cpu().detach().numpy())
    np.save("/home/tdado/pam/sty/w_sty.npy", model._w.cpu().detach().numpy())


if __name__ == "__main__":
    main()