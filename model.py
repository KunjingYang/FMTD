import math
import torch
from torch import nn
from utils import SineLayer


def triple_product(A, B, C):
    return torch.einsum('iqs,pjs,pqt->ijt', A, B, C)


def quadruple_product(A, B, C, D):
    return torch.einsum('iqst,pjst,pqmt,pqsk->ijmk', A, B, C, D)


def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    return torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]


class FMTDNetwork3D(nn.Module):
    """3D FMTD network for image tasks (triple product)."""

    def __init__(self, r1, r2, r3, mid_channel, omega):
        super().__init__()
        self.r1 = r1
        self.r2 = r2
        self.r3 = r3

        self.U_net = nn.Sequential(
            SineLayer(1, mid_channel, is_first=True, omega_0=omega),
            SineLayer(mid_channel, mid_channel, is_first=True, omega_0=omega),
            nn.Linear(mid_channel, r2 * r3),
        )
        self.V_net = nn.Sequential(
            SineLayer(1, mid_channel, is_first=True, omega_0=omega),
            SineLayer(mid_channel, mid_channel, is_first=True, omega_0=omega),
            nn.Linear(mid_channel, r3 * r1),
        )
        self.W_net = nn.Sequential(
            SineLayer(1, mid_channel, is_first=True, omega_0=omega),
            SineLayer(mid_channel, mid_channel, is_first=True, omega_0=omega),
            nn.Linear(mid_channel, r1 * r2),
        )

    def forward(self, U_input, V_input, W_input, n1, n2, n3):
        U_tube = self.U_net(U_input).reshape(n1, self.r2, self.r3)
        V_tube = self.V_net(V_input).reshape(self.r1, n2, self.r3)
        W_tube = self.W_net(W_input).reshape(self.r1, self.r2, n3)
        return triple_product(U_tube, V_tube, W_tube)


class FMTDNetwork4D(nn.Module):
    """4D FMTD network for video/simulation tasks (quadruple product)."""

    def __init__(self, r_1, r_2, r_3, r_4, mid_channel, omega):
        super().__init__()
        self.r_1 = r_1
        self.r_2 = r_2
        self.r_3 = r_3
        self.r_4 = r_4

        self.U_net = nn.Sequential(
            SineLayer(1, mid_channel, is_first=True, omega_0=omega),
            SineLayer(mid_channel, mid_channel, is_first=True, omega_0=omega),
            nn.Linear(mid_channel, r_2 * r_3 * r_4),
        )
        self.V_net = nn.Sequential(
            SineLayer(1, mid_channel, is_first=True, omega_0=omega),
            SineLayer(mid_channel, mid_channel, is_first=True, omega_0=omega),
            nn.Linear(mid_channel, r_1 * r_3 * r_4),
        )
        self.W_net = nn.Sequential(
            SineLayer(1, mid_channel, is_first=True, omega_0=omega),
            SineLayer(mid_channel, mid_channel, is_first=True, omega_0=omega),
            nn.Linear(mid_channel, r_1 * r_2 * r_4),
        )
        self.R_net = nn.Sequential(
            SineLayer(1, mid_channel, is_first=True, omega_0=omega),
            SineLayer(mid_channel, mid_channel, is_first=True, omega_0=omega),
            nn.Linear(mid_channel, r_1 * r_2 * r_3),
        )

    def forward(self, U_input, V_input, W_input, R_input):
        U = self.U_net(U_input).reshape(U_input.size(0), self.r_2, self.r_3, self.r_4)
        V = self.V_net(V_input).reshape(self.r_1, V_input.size(0), self.r_3, self.r_4)
        W = self.W_net(W_input).reshape(self.r_1, self.r_2, W_input.size(0), self.r_4)
        R = self.R_net(R_input).reshape(self.r_1, self.r_2, self.r_3, R_input.size(0))
        X = quadruple_product(U, V, W, R)
        return X, U, V, W, R


class FMTDPointCloudNetwork(nn.Module):
    """FMTD network for point cloud upsampling tasks."""

    def __init__(self, r_1, r_2, r_3, mid_channel, omega):
        super().__init__()

        self.U_net = nn.Sequential(
            SineLayer(1, mid_channel, is_first=True, omega_0=omega),
            SineLayer(mid_channel, mid_channel, is_first=True, omega_0=omega),
            SineLayer(mid_channel, mid_channel, is_first=True, omega_0=omega),
            nn.Linear(mid_channel, r_1),
        )
        self.V_net = nn.Sequential(
            SineLayer(1, mid_channel, is_first=True, omega_0=omega),
            SineLayer(mid_channel, mid_channel, is_first=True, omega_0=omega),
            SineLayer(mid_channel, mid_channel, is_first=True, omega_0=omega),
            nn.Linear(mid_channel, r_2),
        )
        self.W_net = nn.Sequential(
            SineLayer(1, mid_channel, is_first=True, omega_0=omega),
            SineLayer(mid_channel, mid_channel, is_first=True, omega_0=omega),
            SineLayer(mid_channel, mid_channel, is_first=True, omega_0=omega),
            nn.Linear(mid_channel, r_3),
        )

    def forward(self, U_tube, V_tube, W_tube, x, flag):
        centre = torch.einsum('iqs,pjs,pqt->ijt', U_tube, V_tube, W_tube)
        U = self.U_net(x[:, 0].unsqueeze(-1))
        V = self.V_net(x[:, 1].unsqueeze(-1))
        W = self.W_net(x[:, 2].unsqueeze(-1))
        if flag == 1:
            centre = centre.permute(1, 2, 0)
            centre = centre @ U.t()
            centre = centre.permute(2, 1, 0)
            centre = torch.matmul(centre, V.unsqueeze(-1))
            centre = centre.permute(0, 2, 1)
            centre = torch.matmul(centre, W.unsqueeze(-1))
        elif flag == 2:
            centre = centre.permute(1, 2, 0)
            centre = centre @ U.t()
            centre = centre.permute(2, 1, 0)
            centre = centre @ V.t()
            centre = centre.permute(0, 2, 1)
            centre = centre @ W.t()
        return centre
