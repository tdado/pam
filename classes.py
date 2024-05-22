from imports import *


class CustomRidgeCV:
    def __init__(self, X, target, n_alphas):
        self.X = X
        self.target = target
        self.n_alphas = n_alphas
        self._alphas = None

    @property
    def alphas(self):
        if self._alphas is not None:
            return self._alphas

        # SVD for alpha generation
        U, s, Vt = svd(self.X)
        s = s[s > 0]  # Filter non-zero singular values

        # Generate alphas based on the distribution of singular values (simplified approach)
        max_alpha = np.max(s) ** 2
        min_alpha = np.min(s[s > 0]) ** 2
        self._alphas = np.logspace(np.log10(min_alpha), np.log10(max_alpha), self.n_alphas)

        return self._alphas

    def train(self):
        model = RidgeCV(alphas=self.alphas, store_cv_values=True)
        model.fit(self.X, self.target)
        best_alpha = model.alpha_
        cv_values = model.cv_values_
        best_error = np.min(cv_values.mean(axis=0))
        return model


class Data(Dataset):
    def __init__(self, x, w):
        self.x = x
        self.w = w

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.w[idx]
    

class LinearDecoder(nn.Module):
    def __init__(self, input_size):
        super(LinearDecoder, self).__init__()
        self.linear = nn.Sequential(nn.Linear(input_size, 512, False), nn.BatchNorm1d(512), nn.ReLU())

    def forward(self, x):
        return self.linear(x)


class PredictiveAttentionMechanism(nn.Module):
    """
    Predictive attention mechanism (PAM) module.

    Attributes:
        e (nn.ModuleList): Embedding of input heads.
        a (nn.Sequential): Attention of output channels.

    Args:
        h (int): Number of input heads.
        c (int): Number of output channels.
    """
    def __init__(self, h: int, c: int) -> None:
        super().__init__()
        self.e = nn.ModuleList(
            nn.Sequential(
                nn.LazyLinear(c),
                nn.LazyBatchNorm1d(),
                nn.ReLU(True),
                nn.Linear(c, 2 * c)
            ) for _ in range(h)
        )
        self.a = nn.Sequential(
            nn.Linear(c, c),
            _ScalarMultiplication(c ** -0.5),
            nn.Softmax(1)
        )
        self._w: Optional[torch.Tensor] = None

    def forward(self, x: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (Tuple[torch.Tensor, ...]): Input heads (h, (n, *)).

        Returns:
            torch.Tensor: Output channels (n, c).
        """
        k, v = torch.stack(tuple(e(x_) for e, x_ in zip(self.e, x)), 1).tensor_split(2, 2)
        w = self.a(k)
        self._w = w.detach()
        return (w * v).sum(1), k, v, self.a[0].weight.detach()  # out, k, q


class _ScalarMultiplication(nn.Module):
    def __init__(self, c: float) -> None:
        super().__init__()
        self._c = c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._c * x