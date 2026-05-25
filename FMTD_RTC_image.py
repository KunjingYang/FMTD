import os
import sys
os.chdir(sys.path[0])
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import argparse
from PIL import Image
import torch
from scipy.io import savemat
from skimage.metrics import peak_signal_noise_ratio
from torch import optim
from torchvision.utils import save_image

from utils import SoftThreshold, make_coord_input
from model import FMTDNetwork3D


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Current computing device is: {device}")

parser = argparse.ArgumentParser(description='robust tensor completion for color image')
parser.add_argument('--data_path', type=str, default='data/pepper.jpg', help='Path to the input image (default: data/pepper.jpg)')
parser.add_argument('--sr', type=float, default=0.4, help='Sampling rate (default: 0.4)')
parser.add_argument('--noise_level', type=float, default=0.2, help='Noise level (default: 0.2)')
parser.add_argument('--ranks', type=int, nargs=3, default=[12, 12, 16], help='Multiple ranks r1, r2, r3 (default: [12, 12, 16])')
parser.add_argument('--is_imshow', type=int, default=1, help='0 or 1: Whether to display the image using imshow')
parser.add_argument('--is_save_mat', type=int, default=0, help='0 or 1: Whether to save the result as a .mat file')
args = parser.parse_args()

print("\n" + "="*30 + "  Configuration  " + "="*30)
print(f" {'Input Data Path:':<25} {args.data_path}")
print(f" {'Sampling Rate (sr):':<25} {args.sr}")
print(f" {'Noise Level:':<25} {args.noise_level}")
print(f" {'Ranks (r1, r2, r3):':<25} {args.ranks}")
print(f" {'Show Image (is_imshow):':<25} {args.is_imshow}")
print(f" {'Save MAT (is_save_mat):':<25} {args.is_save_mat}")
print("="*77 + "\n")

'''------ experimental setup ------'''
data_path     = args.data_path
sr            = args.sr
noise_level   = args.noise_level
is_imshow     = args.is_imshow
is_save_mat   = args.is_save_mat
r1, r2, r3    = args.ranks  # multiple rank

'''------- hyperparameters --------'''
gamma = 0.0025           # weight of sparse term (l1 norm)
phi = 5 * 10e-6          # weight of smooth term (TVl1)
eta = 0.0001             # proximal parameter
mu = 1                   # penalty parameter
'''------ network parameters ------'''
outer_iter    = 4001     # total number of iterations
mid_channel   = 800     # hidden layer parameters
learning_rate = 0.0001   # step size
omega         = 3        # a parameter of the sine activation function



""" ----------------------------------------------- """
""" -------------        Main      ---------------- """
""" ----------------------------------------------- """

if __name__ == "__main__":
    for k in range(1, 2):
        grad_of_subproblem_X = []
        steps_history = []
        soft_thres = SoftThreshold()

        '''-------  Data preparation  --------'''
        img = Image.open(data_path)  
        gt = np.array(img)/255
        gt = torch.from_numpy(gt).to(device)

        # Add salt-and-pepper noise
        noise_mask = torch.rand(gt.shape, device=gt.device)
        noisy = gt.clone()
        noisy[noise_mask < noise_level * 0.5] = 1.0
        noisy[(noise_mask >= noise_level * 0.5) & (noise_mask < noise_level)] = 0.0

        # Random sampling
        total_elements = noisy.numel()
        N_sample = int(sr * total_elements)
        indices = torch.randperm(total_elements, device=gt.device)[:N_sample]
        mask_flat = torch.zeros(total_elements, dtype=torch.bool, device=gt.device)
        mask_flat[indices] = True
        mask = mask_flat.view(gt.shape)

        '''-------  Main program starts  -------'''
        X = noisy * mask.float()
        n_1, n_2, n_3 = X.shape
        mask = torch.ones(X.shape).to(device)
        mask[X == 0] = 0
        X[mask == 0] = 0

        U_input = make_coord_input(n_1, device)
        V_input = make_coord_input(n_2, device)
        W_input = make_coord_input(n_3, device)

        model = FMTDNetwork3D(r1, r2, r3, mid_channel, omega).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        X_Out_old = torch.zeros_like(X)
        for step in range(outer_iter):

            X_Out = model(U_input, V_input, W_input, n_1, n_2, n_3)

            if step == 0:
                S = (X - X_Out.clone().detach()).to(device)
                V = S.clone().detach().to(device)

            S_old = S
            V = soft_thres(S, gamma)
            S = (2 * X - 2 * X_Out.clone().detach() + mu * V + eta * S_old) / (2 + mu + eta)

            loss = torch.norm(X * mask - X_Out * mask - V * mask, 2) + 0.5 * eta * torch.norm(X_Out - X_Out_old)
            loss = loss + phi * torch.norm(X_Out[1:, :, :] - X_Out[:-1, :, :], 1)
            loss = loss + phi * torch.norm(X_Out[:, 1:, :] - X_Out[:, :-1, :], 1)

            
            loss.backward(retain_graph=True)
            optimizer.step()
            optimizer.zero_grad()

            if step % 100 == 0:
                save_image(X_Out.permute(2, 0, 1).cpu(), "./output/color_image.jpg")
                grad_norm = torch.norm(X * mask - X_Out * mask - V * mask) / torch.norm(X_Out)   # since eta and phi are very small,                            
                grad_of_subproblem_X.append(grad_norm)                                           # the gradient is dominated by the data fidelity term.
                steps_history.append(step)
                #print( 'relative_grad_norm_of_X', grad_norm)
                if grad_norm < 0.01:
                    psnr = peak_signal_noise_ratio(
                        np.clip(gt.cpu().detach().numpy(), 0, 1),
                        X_Out.cpu().detach().numpy(),
                    )
                    print('iteration:', step, 'PSNR', psnr)
                    print(f"Termination criterion reached at iter {step}")
                    break
            if step % 200 == 0:
                psnr = peak_signal_noise_ratio(
                    np.clip(gt.cpu().detach().numpy(), 0, 1),
                    X_Out.cpu().detach().numpy(),
                )
                print('iteration:', step, 'PSNR', psnr)
            X_Out_old = X_Out.detach().clone()

    """ ----------------------------------------------- """
    """ -------------      imshow      ---------------- """
    """ ----------------------------------------------- """

    if is_imshow == 1:
        plt.figure(figsize=(15, 45))
        plt.subplot(131)
        plt.imshow(X.cpu().detach().numpy())
        plt.title('Observed Image')

        plt.subplot(132)
        plt.imshow(X_Out.cpu().detach().numpy())
        plt.title('Reconstructed Image')

        plt.subplot(133)
        plt.imshow(gt.cpu().detach().numpy())
        plt.title('Ground Truth')

        plt.savefig('images/color_image.png', dpi=300, bbox_inches='tight')
        print('The visualization has been saved to: images/color_image.png')
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.plot(steps_history, torch.stack(grad_of_subproblem_X).detach().cpu().numpy(), label='Gradient Norm of X_out')
        plt.title('Gradient Norm of X_out during Training')
        plt.xlabel('Training Step')
        plt.ylabel('Gradient Norm (L2)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.savefig("output/grad_norm_history.png")
        print("The gradient of subproblem X_Out is grad_norm_history.png")

    if is_save_mat == 1:
        savemat("output/FMTD_RTC_image.mat", {'X': X_Out.cpu().detach().numpy()})
        print('The reconstructed image has been saved to: output/FMTD_RTC_image.mat')


