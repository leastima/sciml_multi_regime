import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import random
from torch.optim import LBFGS
from tqdm import tqdm
import argparse
import math
from util import *
from model_dict import get_model

# seed = 0
# np.random.seed(seed)
# random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)

parser = argparse.ArgumentParser('Training Region Optimization')
parser.add_argument('--model', type=str, default='PINN')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--initial_region', type=float, default=1e-5)
parser.add_argument('--sample_num', type=int, default=1)
parser.add_argument('--past_iterations', type=int, default=5)
parser.add_argument('--beta', type=int, default=50)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--n_colloc', type=int, default=1024)
args = parser.parse_args()

# Move seed initialization after parsing arguments
seed = args.seed
print(f"Setting random seed to: {seed}")
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

device = args.device
beta = args.beta
n_colloc = args.n_colloc

# Calculate grid dimensions based on n_colloc
# Use a rounded integer value and ensure it's at least 5 for stability
# n_points = max(5, int(round(math.sqrt(n_colloc))))
n_points = int(round(math.sqrt(n_colloc)))
print(f"Using grid size: {n_points}×{n_points} (total points: {n_points*n_points})")

res, b_left, b_right, b_upper, b_lower = get_data([0, 2 * np.pi], [0, 1], n_points, n_points)
res_test, _, _, _, _ = get_data([0, 2 * np.pi], [0, 1], n_points, n_points)

if args.model == 'PINNsFormer' or args.model == 'PINNsFormer_Enc_Only':
    res = make_time_sequence(res, num_step=5, step=1e-4)
    b_left = make_time_sequence(b_left, num_step=5, step=1e-4)
    b_right = make_time_sequence(b_right, num_step=5, step=1e-4)
    b_upper = make_time_sequence(b_upper, num_step=5, step=1e-4)
    b_lower = make_time_sequence(b_lower, num_step=5, step=1e-4)

res = torch.tensor(res, dtype=torch.float32, requires_grad=True).to(device)
b_left = torch.tensor(b_left, dtype=torch.float32, requires_grad=True).to(device)
b_right = torch.tensor(b_right, dtype=torch.float32, requires_grad=True).to(device)
b_upper = torch.tensor(b_upper, dtype=torch.float32, requires_grad=True).to(device)
b_lower = torch.tensor(b_lower, dtype=torch.float32, requires_grad=True).to(device)

x_res, t_res = res[:, ..., 0:1], res[:, ..., 1:2]
x_left, t_left = b_left[:, ..., 0:1], b_left[:, ..., 1:2]
x_right, t_right = b_right[:, ..., 0:1], b_right[:, ..., 1:2]
x_upper, t_upper = b_upper[:, ..., 0:1], b_upper[:, ..., 1:2]
x_lower, t_lower = b_lower[:, ..., 0:1], b_lower[:, ..., 1:2]


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.0)


if args.model == 'KAN':
    model = get_model(args).Model(width=[2, 5, 5, 1], grid=5, k=3, grid_eps=1.0, \
                                  noise_scale_base=0.25, device=device).to(device)
elif args.model == 'QRes':
    model = get_model(args).Model(in_dim=2, hidden_dim=256, out_dim=1, num_layer=2).to(device)
    model.apply(init_weights)
elif args.model == 'PINNsFormer' or args.model == 'PINNsFormer_Enc_Only':
    model = get_model(args).Model(in_dim=2, hidden_dim=32, out_dim=1, num_layer=1).to(device)
    model.apply(init_weights)
else:
    model = get_model(args).Model(in_dim=2, hidden_dim=512, out_dim=1, num_layer=4).to(device)
    model.apply(init_weights)

optim = LBFGS(model.parameters(), line_search_fn='strong_wolfe')

print(model)
print(get_n_params(model))
loss_track = []

# for region optimization
initial_region = args.initial_region
sample_num = args.sample_num
past_iterations = args.past_iterations
gradient_list_overall = []
gradient_list_temp = []
gradient_variance = 1

for i in tqdm(range(1000)):

    ###### Region Optimization with Monte Carlo Approximation ######
    def closure():
        x_res_region_sample_list = []
        t_res_region_sample_list = []
        for i in range(sample_num):
            x_region_sample = (torch.rand(x_res.shape).to(x_res.device)) * np.clip(initial_region / gradient_variance,
                                                                                   a_min=0,
                                                                                   a_max=0.01)
            t_region_sample = (torch.rand(x_res.shape).to(t_res.device)) * np.clip(initial_region / gradient_variance,
                                                                                   a_min=0,
                                                                                   a_max=0.01)
            x_res_region_sample_list.append(x_res + x_region_sample)
            t_res_region_sample_list.append(t_res + t_region_sample)
        x_res_region_sample = torch.cat(x_res_region_sample_list, dim=0)
        t_res_region_sample = torch.cat(t_res_region_sample_list, dim=0)
        pred_res = model(x_res_region_sample, t_res_region_sample)
        pred_left = model(x_left, t_left)
        pred_right = model(x_right, t_right)
        pred_upper = model(x_upper, t_upper)
        pred_lower = model(x_lower, t_lower)

        u_x = \
            torch.autograd.grad(pred_res, x_res_region_sample, grad_outputs=torch.ones_like(pred_res),
                                retain_graph=True,
                                create_graph=True)[0]
        u_t = \
            torch.autograd.grad(pred_res, t_res_region_sample, grad_outputs=torch.ones_like(pred_res),
                                retain_graph=True,
                                create_graph=True)[0]

        loss_res = torch.mean((u_t + beta * u_x) ** 2) # Beta value here
        loss_bc = torch.mean((pred_upper - pred_lower) ** 2)
        loss_ic = torch.mean((pred_left[:, 0] - torch.sin(x_left[:, 0])) ** 2)

        loss_track.append([loss_res.item(), loss_bc.item(), loss_ic.item()])

        loss = loss_res + loss_bc + loss_ic
        optim.zero_grad()
        loss.backward(retain_graph=True)
        gradient_list_temp.append(torch.cat([(p.grad.view(-1)) if p.grad is not None else torch.zeros(1).cuda() for p in
                                             model.parameters()]).cpu().numpy())  # hook gradients from computation graph
        return loss


    optim.step(closure)

    ###### Trust Region Calibration ######
    gradient_list_overall.append(np.mean(np.array(gradient_list_temp), axis=0))
    gradient_list_overall = gradient_list_overall[-past_iterations:]
    gradient_list = np.array(gradient_list_overall)
    gradient_variance = (np.std(gradient_list, axis=0) / (
            np.mean(np.abs(gradient_list), axis=0) + 1e-6)).mean()  # normalized variance
    gradient_list_temp = []
    if gradient_variance == 0:
        gradient_variance = 1  # for numerical stability

print('Loss Res: {:4f}, Loss_BC: {:4f}, Loss_IC: {:4f}'.format(loss_track[-1][0], loss_track[-1][1], loss_track[-1][2]))
print('Train Loss: {:4f}'.format(np.sum(loss_track[-1])))

if not os.path.exists('./results/'):
    os.makedirs('./results/')
torch.save(model.state_dict(), f'./results/convection_{args.model}_region.pt')

# Visualize
if args.model == 'PINNsFormer' or args.model == 'PINNsFormer_Enc_Only':
    res_test = make_time_sequence(res_test, num_step=5, step=1e-4)

res_test = torch.tensor(res_test, dtype=torch.float32, requires_grad=True).to(device)
x_test, t_test = res_test[:, ..., 0:1], res_test[:, ..., 1:2]

with torch.no_grad():
    pred = model(x_test, t_test)[:, 0:1]
    pred = pred.cpu().detach().numpy()

pred = pred.reshape(n_points, n_points)


def u_res(x, t):
    print(x.shape)
    print(t.shape)
    return np.sin(x - beta * t)   # Beta here


res_test, _, _, _, _ = get_data([0, 2 * np.pi], [0, 1], n_points, n_points)   # n_colloc here
u = u_res(res_test[:, 0], res_test[:, 1]).reshape(n_points, n_points)

rl1 = np.sum(np.abs(u - pred)) / np.sum(np.abs(u))
rl2 = np.sqrt(np.sum((u - pred) ** 2) / np.sum(u ** 2))

print('relative L1 error: {:4f}'.format(rl1))
print('relative L2 error: {:4f}'.format(rl2))

plt.figure(figsize=(4, 3))
plt.imshow(pred, aspect='equal')
plt.xlabel('x')
plt.ylabel('t')
plt.title('Predicted u(x,t)')
plt.colorbar()
plt.tight_layout()
plt.axis('off')
plt.savefig(f'./results/convection_{args.model}_region_optimization_pred.pdf', bbox_inches='tight')

plt.figure(figsize=(4, 3))
plt.imshow(u, aspect='equal')
plt.xlabel('x')
plt.ylabel('t')
plt.title('Exact u(x,t)')
plt.colorbar()
plt.tight_layout()
plt.axis('off')
plt.savefig('./results/convection_exact.pdf', bbox_inches='tight')

plt.figure(figsize=(4, 3))
plt.imshow(pred - u, aspect='equal', cmap='coolwarm', vmin=-1, vmax=1)
plt.xlabel('x')
plt.ylabel('t')
plt.title('Absolute Error')
plt.colorbar()
plt.tight_layout()
plt.axis('off')
plt.savefig(f'./results/convection_{args.model}_region_optimization_error.pdf', bbox_inches='tight')
