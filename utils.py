import numpy as np
import torch
from torch import nn


class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.in_features) / self.omega_0,
                    np.sqrt(6 / self.in_features) / self.omega_0,
                )

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class SoftThreshold(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, lam):
        x_abs = x.abs() - lam
        zeros = x_abs - x_abs
        n_sub = torch.max(x_abs, zeros)
        return torch.mul(torch.sign(x), n_sub)


def make_coord_input(size, device):
    return (
        torch.from_numpy(np.arange(1, size + 1))
        .reshape(size, 1)
        .float()
        .to(device)
    )
