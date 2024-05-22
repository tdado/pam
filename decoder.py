from classes import LinearDecoder, PredictiveAttentionMechanism
from imports import *


def train_linear(device, n_epochs, n_input, seed, training_loader, filename):
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = LinearDecoder(n_input).train().to(device)
    model.apply(lambda x: torch.nn.init.xavier_uniform_(x.weight) if type(x) == nn.Linear else None)
    mse = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())
    loss_tr = []
    for epoch in range(1, n_epochs + 1):
        model.train()
        l_tr = 0
        for x, t in training_loader: # brain, latent;
            x = x.to(torch.float32).to(device)
            t = t.to(torch.float32).to(device)
            y = model(x)
            _l_tr = mse(y, t)
            optimizer.zero_grad()
            _l_tr.backward()
            optimizer.step()
            l_tr += _l_tr.item()
        loss_tr.append(l_tr / len(training_loader))
    plt.figure()
    plt.plot(loss_tr)
    plt.savefig("/home/tdado/pam/fig/loss_%s_%i.png" % (filename, epoch))
    plt.close()
    return model

def train_pam(device, n_epochs, seed, training_loader, h, filename):
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = PredictiveAttentionMechanism(len(h)-1, 512).train().to(device)
    model.apply(lambda x: torch.nn.init.xavier_uniform_(x.weight) if type(x) == nn.Linear else None)
    mse = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())
    loss_tr = []
    for epoch in range(1, n_epochs + 1):
        model.train()
        l_tr = 0
        for x, t in training_loader: # brain, latent;
            x = x.to(torch.float32).to(device)
            t = t.to(torch.float32).to(device)
            y, _, _, _ = model([x[:, h[i]:h[i+1]] for i in range(len(h)-1)])
            _l_tr = mse(y, t)
            optimizer.zero_grad()
            _l_tr.backward()
            optimizer.step()
            l_tr += _l_tr.item()
        loss_tr.append(l_tr / len(training_loader))
    plt.figure()
    plt.plot(loss_tr)
    plt.savefig("/home/tdado/pam/fig/loss_%s_%i.png" % (filename, epoch))
    plt.close()
    return model


def pearson_correlation_coefficient(x: np.ndarray, y: np.ndarray, axis: int) -> np.ndarray:
    r = (np.nan_to_num(stats.zscore(x)) * np.nan_to_num(stats.zscore(y))).mean(axis)
    p = 2 * t.sf(np.abs(r / np.sqrt((1 - r ** 2) / (x.shape[0] - 2))), x.shape[0] - 2)
    return r, p


def reconstruct(G, i, y, folder):
    _img = G.synthesis(y[i, None], noise_mode="none")
    _img = (_img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    _img = Image.fromarray(_img[0].cpu().numpy(), 'RGB')
    _img.save("/home/tdado/pam/%s/%s.png" % (folder, str(i+1).zfill(4)))