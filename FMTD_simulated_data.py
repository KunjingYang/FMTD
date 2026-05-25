import os
import sys
os.chdir(sys.path[0])
import numpy as np
import torch
import argparse
from torch import optim

from utils import make_coord_input
from model import FMTDNetwork4D, quadruple_product

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Current computing device is: {device}")


###################
parser = argparse.ArgumentParser(description='robust tensor completion for color video')
parser.add_argument('--ranks', type=int, nargs=4, default=[4, 4, 4, 4], help='Multiple ranks r1, r2, r3, r4 (default: [8, 8, 8, 8])')
args = parser.parse_args()

r_1, r_2, r_3, r_4 = args.ranks

'''------ network parameters ------'''
outer_iter    = 2001     # total number of iterations
omega         = 3        # a parameter of the sine activation function
mid_channel   = 200      # hidden layer parameters
learning_rate = 0.0002   # step size


if __name__ == "__main__":
    for k in range(1, 2):

        '''-------  Data preparation  --------'''
        I1, I2, I3, I4 = 7, 8, 9, 10    # The dimensions of a tensor
        r1, r2, r3, r4 = 4, 5, 4, 6     # selected multiple rank

        A = torch.tensor(np.random.rand(I1, r2, r3, r4), dtype=torch.float32).to(device)
        B = torch.tensor(np.random.rand(r1, I2, r3, r4), dtype=torch.float32).to(device)
        C = torch.tensor(np.random.rand(r1, r2, I3, r4), dtype=torch.float32).to(device)
        D = torch.tensor(np.random.rand(r1, r2, r3, I4), dtype=torch.float32).to(device)

        # Randomly generate a fourth-order tensor with multiple rank (r1, r2, r3, r4)
        X = quadruple_product(A, B, C, D)

        '''-------  Main program starts  -------'''
        n_1, n_2, n_3, n_4 = X.shape

        U_input = make_coord_input(n_1, device)
        V_input = make_coord_input(n_2, device)
        W_input = make_coord_input(n_3, device)
        R_input = make_coord_input(n_4, device)

        # (r_1, r_2, r_3, r_4) may not be equal to (r1, r2, r3, r4)
        model = FMTDNetwork4D(r_1, r_2, r_3, r_4, mid_channel, omega).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        for step in range(outer_iter):
            X_Out, U_tube, V_tube, W_tube, R_tube = model(U_input, V_input, W_input, R_input)
            loss = torch.norm(X_Out - X, 2)

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            if step % 500 == 0:
                rel_error = torch.norm(X_Out - X) / torch.norm(X)
                print('iteration:', step, 'rel_error', rel_error)
