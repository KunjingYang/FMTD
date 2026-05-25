import math
import matplotlib.pyplot as plt
import numpy as np
import argparse
import open3d as o3d
import torch
from torch import optim
from model import FMTDPointCloudNetwork, gradient

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Current computing device is: {device}")

parser = argparse.ArgumentParser(description='robust tensor completion for color video')
parser.add_argument('--data_path', type=str, default='data\heart.pcd', help='Path to the input image (default: data\heart.pcd)')
parser.add_argument('--sr', type=float, default=0.05, help='Sampling rate (default: 0.05)')
parser.add_argument('--ranks', type=int, nargs=1, default=4, help='High-order triple rank r (default: 4)')
parser.add_argument('--is_imshow', type=int, default=1, help='0 or 1: Whether to display the image using imshow')
args = parser.parse_args()

print("\n" + "="*30 + "  Configuration  " + "="*30)
print(f" {'Input Data Path:':<25} {args.data_path}")
print(f" {'Sampling Rate (sr):':<25} {args.sr}")
print(f" {'Ranks (r1, r2, r3):':<25} {args.ranks}")
print(f" {'Show Image (is_imshow):':<25} {args.is_imshow}")
print("="*77 + "\n")


'''------ experimental setup ------'''
data_path       = args.data_path
sr              = args.sr
is_imshow       = args.is_imshow
r               = args.ranks

'''------- hyperparameters --------'''
r             = 4       # High order triple rank
gamma_1       = 0.4     # the weight of the second loss
gamma_2       = 0.4     # the weight of the third loss
down          = 5       # Tucker rank
'''------ network parameters ------'''
mid_channel   = 200     # hidden layer parameters
outer_iter    = 801     # total number of iterations
omega         = 5       # a parameter of the sine activation function
learning_rate = 0.00001 # step size
thres         = 0.01    # Threshold of the distance function




""" ----------------------------------------------- """
""" -------------        Main      ---------------- """
""" ----------------------------------------------- """

if __name__ == "__main__":
    for k in range(1, 2):

        # input dataset
        pcd = o3d.io.read_point_cloud(data_path)
        point_cloud = np.array(pcd.points)

        # Random sampling
        num_sample = int(len(point_cloud) * sr)
        indices = np.random.choice(len(point_cloud), num_sample, replace=False)
        sampled_points = point_cloud[indices]
        sampled_pcd = o3d.geometry.PointCloud()
        sampled_pcd.points = o3d.utility.Vector3dVector(sampled_points)

        X_np = np.array(sampled_pcd.points)
        n = X_np.shape[0]
        print('Number of sparse sampling points:', n)

        # tucker rank
        r_1 = int(n / down)
        r_2 = int(n / down)
        r_3 = int(n / down)

        X_gt = torch.zeros(n, 1).to(device)
        U_input = torch.from_numpy(X_np[:, 0]).reshape(n, 1).float().to(device)
        U_input.requires_grad = True
        V_input = torch.from_numpy(X_np[:, 1]).reshape(n, 1).float().to(device)
        V_input.requires_grad = True
        W_input = torch.from_numpy(X_np[:, 2]).reshape(n, 1).float().to(device)
        W_input.requires_grad = True

        U_tube = torch.Tensor(r_1, r, r).to(device)
        V_tube = torch.Tensor(r, r_2, r).to(device)
        W_tube = torch.Tensor(r, r, r_3).to(device)
        stdv = 1 / math.sqrt(n)
        U_tube.data.uniform_(-stdv, stdv)
        V_tube.data.uniform_(-stdv, stdv)
        W_tube.data.uniform_(-stdv, stdv)
        x_input = torch.cat((U_input, V_input, W_input), dim=1)

        # network setting
        model = FMTDPointCloudNetwork(r_1, r_2, r_3, mid_channel, omega).to(device)
        params = list(model.parameters())
        U_tube.requires_grad = True
        params += [U_tube]
        V_tube.requires_grad = True
        params += [V_tube]
        W_tube.requires_grad = True
        params += [W_tube]
        optimizer = optim.Adam(params, lr=learning_rate)

        rand_num = 30
        add_border = 0.1
        for step in range(outer_iter):

            U_random = (torch.min(U_input) - add_border +
                        (torch.max(U_input) - torch.min(U_input) + 2 * add_border) * torch.rand(rand_num, 1).to(device))
            V_random = (torch.min(V_input) - add_border +
                        (torch.max(V_input) - torch.min(V_input) + 2 * add_border) * torch.rand(rand_num, 1).to(device))
            W_random = (torch.min(W_input) - add_border +
                        (torch.max(W_input) - torch.min(W_input) + 2 * add_border) * torch.rand(rand_num, 1).to(device))
            x_random = torch.cat((U_random, V_random, W_random), dim=1)

            X_Out = model(U_tube, V_tube, W_tube, x_input, flag=1)
            loss_1 = torch.norm((X_Out) - X_gt, 1)
            X_Out_off = model(U_tube, V_tube, W_tube, x_random, flag=2)
            grad_ = gradient(X_Out_off, x_random)
            loss_2 = gamma_1 * torch.norm(grad_.norm(dim=-1) - rand_num ** 2, 1)
            loss_3 = gamma_2 * torch.norm(torch.exp(-torch.abs(X_Out_off)), 1)
            loss = loss_1 + loss_2 + loss_3

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            if step % 200 == 0:
                print('iteration:', step)
                number = 60
                range_ = torch.from_numpy(np.array(range(number))).to(device)
                u = (torch.min(U_input) - add_border + (
                    torch.max(U_input) - torch.min(U_input) + 2 * add_border) * (range_ / number)).reshape(number, 1)
                v = (torch.min(V_input) - add_border + (
                    torch.max(V_input) - torch.min(V_input) + 2 * add_border) * (range_ / number)).reshape(number, 1)
                w = (torch.min(W_input) - add_border + (
                    torch.max(W_input) - torch.min(W_input) + 2 * add_border) * (range_ / number)).reshape(number, 1)
                x_in = torch.cat((u, v, w), dim=1)
                out = model(U_tube, V_tube, W_tube, x_in, flag=2).detach().cpu().clone()
                idx = torch.where(torch.abs(out) < thres)
                Pts = torch.cat((u[idx[0]], v[idx[1]]), dim=1)
                Pts = torch.cat((Pts, w[idx[2]]), dim=1).detach().cpu().clone().numpy()

                fig = plt.figure(figsize=(10, 10), dpi=300)
                ax = fig.add_axes([0, 0, 1, 1], projection='3d')
                xs, ys, zs = Pts[:, 0], Pts[:, 1], Pts[:, 2]
                sc = ax.scatter(xs, ys, zs, c=zs, cmap='viridis', s=6)
                ax.set_axis_off()
                ax.grid(False)
                ax.view_init(elev=30, azim=90)
                plt.savefig("./output/point_cloud.png", bbox_inches='tight', pad_inches=-1.9, transparent=False)

    """ ----------------------------------------------- """
    """ -------------      imshow      ---------------- """
    """ ----------------------------------------------- """

    if is_imshow == 1:
        plt.close()
        size_pc = 4
        fig1 = plt.figure(figsize=(15, 30))
        ax = fig1.add_subplot(131, projection='3d')
        ax.scatter(X_np[:, 0], X_np[:, 1], X_np[:, 2], s=size_pc)
        ax.set_title('Input Sparse Point Cloud', fontsize=20, pad=20)
        ax.view_init(elev=30, azim=90)

        ax = fig1.add_subplot(132, projection='3d')
        ax.scatter(Pts[:, 0], Pts[:, 1], Pts[:, 2], s=size_pc)
        ax.set_title('FMTD Reconstructed', fontsize=20, pad=20)
        ax.view_init(elev=30, azim=90)

        ax = fig1.add_subplot(133, projection='3d')
        ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], s=size_pc)
        ax.view_init(elev=30, azim=90)
        ax.set_title('Ground Truth', fontsize=20, pad=20)

        plt.savefig('images/point_cloud.png', dpi=300, bbox_inches='tight')
        print('The visualization has been saved to: images/point_cloud.png')
        #plt.show()
