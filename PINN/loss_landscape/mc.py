import wandb
import argparse
import sys
import traceback
import torch
import os
import numpy as np
import math
import copy
import json
import time

from src.train_utils import *
from src.models import *
from src.curves import *

import matplotlib.pyplot as plt
from collections import OrderedDict


def plot_loss_curve(loss_list, alphas, output_dir, model_i, model_j, loss_name):
    """
    绘制并保存损失曲线图
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.figure()
    plt.plot(alphas, loss_list)
    plt.xlabel('t (Bezier curve parameter)')
    plt.ylabel(f'{loss_name} Loss')
    plt.title(f'Loss along the Bezier Curve: Model {model_i} -> Model {model_j}')
    plt.grid(True)

    file_path = os.path.join(output_dir, f'loss_curve_{loss_name}_model{model_i}_model{model_j}.png')
    plt.savefig(file_path)
    plt.close()
    print(f"Saved loss curve plot to {file_path}")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clean_state_dict(state_dict):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        # name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict


def loss_total(loss_res, loss_ic, loss_bc):
    return loss_res + loss_ic + loss_bc


def loss_f(loss_res, loss_ic, loss_bc):
    return loss_res


def loss_u(loss_ref, loss_ic, loss_bc):
    return loss_ic


def loss_b(loss_ref, loss_ic, loss_bc):
    return loss_bc


def learning_rate_schedule(base_lr, epoch, total_epochs):
    """The learning rate schedule for testing the mode connectivity
    """

    alpha = epoch / total_epochs
    if alpha <= 0.5:
        factor = 1.0
    elif alpha <= 0.9:
        factor = 1.0 - (alpha - 0.5) / 0.4 * 0.99
    else:
        factor = 0.01
    return factor * base_lr


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


class SimpleCurveWrapper(torch.nn.Module):
    def __init__(self, dataset_path, model1, model2, device, args):
        super(SimpleCurveWrapper, self).__init__()

        self.device = device
        self.args = args
        self.arch = PINN
        self.curve = Bezier
        self.curve_arch = PINNCurve
        self.arch_kwargs = {}
        self.num_bends = args.num_bends

        # for debug
        self.test_model_list = []

        curve_model = CurveNet(
            in_dim=2,
            hidden_dim=args.num_neurons,
            out_dim=1,
            num_layer=args.num_layers,
            curve=self.curve,
            architecture=self.curve_arch,
            num_bends=self.num_bends,
            fix_start=True,
            fix_end=True,
            architecture_kwargs=self.arch_kwargs,
        )

        base_model = PINN(
            in_dim=2, hidden_dim=args.num_neurons, out_dim=1, num_layer=args.num_layers
        ).to(device)
        for k in range(self.num_bends):
            if k == 0:
                checkpoint = clean_state_dict(model1.state_dict())
            elif k == self.num_bends - 1:
                checkpoint = clean_state_dict(model2.state_dict())
            else:
                checkpoint = OrderedDict()
                state_dict_1 = clean_state_dict(model1.state_dict())
                state_dict_2 = clean_state_dict(model2.state_dict())
                alpha = k / (self.num_bends - 1)
                for key in state_dict_1.keys():
                    checkpoint[key] = (1 - alpha) * state_dict_1[key] + alpha * state_dict_2[key]
            base_model.load_state_dict(checkpoint)
            curve_model.import_base_parameters(base_model, k)
            self.test_model_list.append(base_model)

        self.curve_model = curve_model.to(device)

        # 初始化 Adam（用于预训练）
        self.adam_optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.curve_model.parameters()),
            lr=1e-3
        )

        # 初始化 LBFGS（用于精调）
        self.lbfgs_optimizer = torch.optim.LBFGS(
            filter(lambda p: p.requires_grad, self.curve_model.parameters()),
            lr=1.0,
            max_iter=2000,
            history_size=100,
            tolerance_grad=1e-7,
            tolerance_change=1e-9,
            line_search_fn="strong_wolfe"
        )

        # 数据加载逻辑不变
        data = torch.load(dataset_path, map_location=device, weights_only=False)
        x_range, t_range, self.loss_func, pde_coefs, _ = get_pde(args.pde, args.pde_params, args.loss)
        self.x = (data['x_res'], data['x_left'], data['x_upper'], data['x_lower'])
        self.t_data = (data['t_res'], data['t_left'], data['t_upper'], data['t_lower'])

    def train_curve(self, stage):
        set_random_seed(0)

        """ 先使用 Adam 进行随机采样训练，再用 L-BFGS 在固定采样上精调 """
        device = self.device
        curve_model = self.curve_model
        num_epochs = self.args.num_epochs
        num_t_per_epoch = self.args.num_t_per_epoch
        debug = self.args.debug
        pretrain_epoch = 0
        record = {}

        if num_epochs == 0:
            pretrain_epoch = 0

        beg_tmp = time.time() if debug else 0

        # ======== 第一阶段：Adam 随机采样优化 ========
        if "1" in stage:
            print("Stage 1: Training with Adam (random sampling)...")
            for epoch in range(pretrain_epoch):
                # t_vals = torch.rand(num_t_per_epoch, device=device)
                lr = learning_rate_schedule(1e-3, epoch, pretrain_epoch)
                adjust_learning_rate(self.adam_optimizer, lr)
                self.adam_optimizer.zero_grad()
                total_loss = 0.0

                y_pred = curve_model.predict(self.x, self.t_data)
                loss_res, loss_bc, loss_ic = self.loss_func(self.x, self.t_data, y_pred)
                loss = loss_res + loss_bc + loss_ic
                total_loss += loss

                total_loss.backward()
                self.adam_optimizer.step()

                if debug and epoch % 1000 == 0:
                    print(f"[Adam] epoch {epoch}/{pretrain_epoch}, loss = {total_loss.item():.6f}")

            if debug:
                num_steps = self.args.num_steps
                test_loss_list = []
                alphas = np.linspace(0, 1, num_steps)

                for t_val in alphas:
                    y_pred = curve_model.predict(self.x, self.t_data, t_val)
                    loss_res, loss_bc, loss_ic = self.loss_func(self.x, self.t_data, y_pred)
                    loss_final = loss_res + loss_bc + loss_ic
                    test_loss_list.append(loss_final.item())

                print('[Adam] end,', 'test loss :', test_loss_list)
                record['adam_max_loss'] = max(test_loss_list)
                record['adam_mean_loss'] = np.mean(test_loss_list)

        # ======== 第二阶段：L-BFGS 精调（fully） ========
        if "2" in stage:
            print("Stage 2: Fine-tuning with L-BFGS (fixed sampling)...")
            num_t_per_epoch = self.args.num_t_per_epoch
            num_t_lbfgs = num_t_per_epoch
            post_train_epoch = self.args.num_epochs
            segments = torch.linspace(0.0, 1.0, num_t_per_epoch + 1, device=device)

            for epoch in range(1):
                # 初始化 LBFGS（用于精调）
                self.lbfgs_optimizer = torch.optim.LBFGS(
                    filter(lambda p: p.requires_grad, self.curve_model.parameters()),
                    lr=1.0,
                    max_iter=post_train_epoch,
                    history_size=100,
                    tolerance_grad=1e-7,
                    tolerance_change=1e-9,
                    line_search_fn="strong_wolfe"
                )
                t_vals = segments

                # t_vals = torch.rand(num_t_per_epoch, device=device)
                # t_vals = torch.cat([
                #     torch.rand(1, device=device) * (segments[i + 1] - segments[i]) + segments[i]
                #     for i in range(num_t_lbfgs)
                # ])
                def closure():
                    self.lbfgs_optimizer.zero_grad()
                    total_loss = 0.0
                    for t_val in t_vals:
                        y_pred = curve_model.predict(self.x, self.t_data, t_val)
                        loss_res, loss_bc, loss_ic = self.loss_func(self.x, self.t_data, y_pred)
                        total_loss += (loss_res + loss_bc + loss_ic)
                    total_loss = total_loss / num_t_lbfgs
                    total_loss.backward()
                    return total_loss

                self.lbfgs_optimizer.step(closure)

                if debug and epoch % debug == 0:
                    num_steps = self.args.num_steps
                    train_loss_list = []
                    test_loss_list = []
                    alphas = np.linspace(0, 1, num_steps)

                    for t_val in t_vals:
                        y_pred = curve_model.predict(self.x, self.t_data, t_val)
                        loss_res, loss_bc, loss_ic = self.loss_func(self.x, self.t_data, y_pred)
                        loss_final = loss_res + loss_bc + loss_ic
                        train_loss_list.append(loss_final.item())

                    for t_val in alphas:
                        y_pred = curve_model.predict(self.x, self.t_data, t_val)
                        loss_res, loss_bc, loss_ic = self.loss_func(self.x, self.t_data, y_pred)
                        loss_final = loss_res + loss_bc + loss_ic
                        test_loss_list.append(loss_final.item())

                    print('epoch :', epoch, 'train loss :', train_loss_list, 'test loss :', test_loss_list)

        if debug:
            end_tmp = time.time()
            print(f"train cost for one curve: {end_tmp - beg_tmp:.4f}s")

        if debug:
            num_steps = self.args.num_steps
            test_loss_list = []
            alphas = np.linspace(0, 1, num_steps)

            for t_val in alphas:
                y_pred = curve_model.predict(self.x, self.t_data, t_val)
                loss_res, loss_bc, loss_ic = self.loss_func(self.x, self.t_data, y_pred)
                loss_final = loss_res + loss_bc + loss_ic
                test_loss_list.append(loss_final.item())

            print('[L-BFGS] end,', 'test loss :', test_loss_list)
            # record['lbfgs_loss_list'] = test_loss_list

        return record

    def get_loss_at_t(self, t):
        y_pred = self.curve_model.predict(self.x, self.t_data, t)
        loss_res, loss_bc, loss_ic = self.loss_func(self.x, self.t_data, y_pred)
        return loss_res + loss_bc + loss_ic


def cal_model(param1, param2, alpha, cal_type):
    if cal_type == 'linear':
        return alpha * param2 + (1 - alpha) * param1
    elif cal_type == 'bezier':
        return param1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pde', type=str,
                        default='convection', help='PDE type')
    parser.add_argument('--pde_params', type=str,
                        default=None, help='PDE coefficients')
    parser.add_argument('--num_layers', type=int, default=4,
                        help='number of layers of the neural net')
    parser.add_argument('--num_neurons', type=int, default=50,
                        help='number of neurons per layer')
    parser.add_argument('--loss', type=str, default='mse',
                        help='type of loss function')
    parser.add_argument('--num_x', type=int, default=257,
                        help='number of spatial sample points (power of 2 + 1)')
    parser.add_argument('--num_t', type=int, default=101,
                        help='number of temporal sample points')
    parser.add_argument('--num_res', type=int, default=1000,
                        help='number of sampled residual points')
    parser.add_argument('--wandb_project', type=str,
                        default='pinns', help='W&B project name')
    parser.add_argument('--set_idx', type=int, default=0, help='the index of dataset')
    parser.add_argument('--device', type=str, default=0, help='GPU to use')
    parser.add_argument('--save_path', type=str, help='path to save the results of experiments')

    parser.add_argument('--hc', type=str, default='alm', help='hc method')

    parser.add_argument('--max_model_num', type=int, default=5, help='max model num for mc')
    parser.add_argument('--num_bends', type=int, default=3, help='bends for bezier')
    parser.add_argument('--optimizer_type', type=str, default='lbfgs', help='optimizer for train bezier curve')
    parser.add_argument('--num_epochs', type=int, default=1000, help='sample epochs for train bezier curve')
    parser.add_argument('--num_t_per_epoch', type=int, default=50, help='number of sample points per epoch')
    parser.add_argument('--num_steps', type=int, default=101, help='number for test')
    parser.add_argument('--debug', type=int, default=1, help='debug log')
    parser.add_argument('--stage', type=str, default="2", help='train stage')

    # Extract arguments from parser
    args = parser.parse_args()

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    set_idx = args.set_idx

    model_list = []
    pde_param = json.loads(args.pde_params)
    if args.pde == "convection":
        model_folder = os.path.join("output", "system_convection", f"N_f_{args.num_res}",
                                    f"beta_{float(pde_param['beta'])}", f"set_{set_idx}", args.hc)
        dataset_path = os.path.join("dataset", f'system_{args.pde}',
                                    f'N_f_{args.num_res}', f"beta_{float(pde_param['beta'])}",
                                    'train_' + str(set_idx) + '.pt')
    elif args.pde == "reaction":
        model_folder = os.path.join("output", "system_reaction", f"N_f_{args.num_res}",
                                    f"rho_{float(pde_param['rho'])}", f"set_{set_idx}", args.hc)
        dataset_path = os.path.join("dataset", "system_reaction", f"N_f_{args.num_res}",
                                    f"rho_{float(pde_param['rho'])}", 'train_' + str(set_idx) + '.pt')
    elif args.pde == "wave":
        model_folder = os.path.join("output", "system_wave", f"N_f_{args.num_res}",
                                    f'beta_{float(pde_param["beta"])}_c_{float(pde_param["c"])}', f"set_{set_idx}",
                                    args.hc)
        dataset_path = os.path.join("dataset", "system_wave", f"N_f_{args.num_res}",
                                    f'beta_{float(pde_param["beta"])}_c_{float(pde_param["c"])}',
                                    'train_' + str(set_idx) + '.pt')
    elif args.pde == "reaction_diffusion":
        model_folder = os.path.join("output", "system_reaction_diffusion", f"N_f_{args.num_res}",
                                    f'nu_{float(pde_param["nu"])}_rho_{float(pde_param["rho"])}', f"set_{set_idx}",
                                    args.hc)
        dataset_path = os.path.join("dataset", "system_reaction_diffusion", f"N_f_{args.num_res}",
                                    f'nu_{float(pde_param["nu"])}_rho_{float(pde_param["rho"])}',
                                    'train_' + str(set_idx) + '.pt')

    model_files = [f for f in os.listdir(model_folder) if f.endswith(".pt")]
    if len(model_files) == 0:
        print(f"No model files found in {model_folder} with pattern '*{hc}*'")
        return

    max_model_num = args.max_model_num
    for filename in sorted(model_files):
        path = os.path.join(model_folder, filename)
        try:
            model = PINN(in_dim=2, hidden_dim=args.num_neurons, out_dim=1, num_layer=args.num_layers).to(device)
            model.load_state_dict(torch.load(path))
            model.eval()
            model_list.append(model)
            print(f"Loaded model: {filename}")

            if len(model_list) == max_model_num:
                break
        except Exception as e:
            print(f"Error loading model {filename}: {e}")

    if len(model_list) < 2:
        print("Need at least 2 models to compute mode connectivity")
        return

    print(f"Loaded {len(model_list)} models for mode connectivity analysis")

    loss_list_func = [loss_total]
    result = {}

    # calculate the LMC
    lmc_matrix = {}
    for i in range(len(model_list)):
        lmc_matrix[i] = {}
        for j in range(len(model_list)):
            lmc_matrix[i][j] = {}

            model1 = model_list[i].state_dict()
            model2 = model_list[j].state_dict()
            new_state_dict = OrderedDict()

            num_steps = args.num_steps
            cal_type = 'linear'
            error_list = []
            loss_list = []
            alphas = np.linspace(0, 1, num_steps)

            for alpha in alphas:
                for key in model1.keys():
                    if key in model2:
                        new_state_dict[key] = alpha * model1[key] + (1 - alpha) * model2[key]
                    else:
                        raise ValueError(f"Key '{key}' not found in model2")

                model = PINN(in_dim=2, hidden_dim=args.num_neurons, out_dim=1, num_layer=args.num_layers).to(device)
                model.load_state_dict(new_state_dict)
                model.eval()

                x_range, t_range, loss_func, pde_coefs, _ = get_pde(args.pde, args.pde_params, args.loss)
                data = torch.load(dataset_path, map_location=device)
                x = (data['x_res'], data['x_left'], data['x_upper'], data['x_lower'])
                t = (data['t_res'], data['t_left'], data['t_upper'], data['t_lower'])
                data_params = data['data_params']

                loss_res, loss_bc, loss_ic = loss_func(x, t, predict(x, t, model))
                loss = loss_res + loss_bc + loss_ic
                loss_list.append(loss.item())

                n_x_test = int((args.num_x - 1) / 2) + 1
                n_t_test = args.num_t
                x_test, t_test, data_params_test = get_data(x_range, t_range, n_x_test,
                                                            n_t_test, random=False, device=device)

                with torch.no_grad():
                    predictions = torch.vstack(predict(x_test, t_test, model)).cpu().detach().numpy()
                targets = get_ref_solutions(args.pde, pde_coefs, x_test, t_test, data_params_test)
                test_l2re = l2_relative_error(predictions, targets)
                error_list.append(test_l2re)

            lmc_matrix[i][j]['loss'] = loss_list
            lmc_matrix[i][j]['error'] = error_list

    lmc = np.zeros((len(model_list), len(model_list)))

    for i in range(len(model_list)):
        for j in range(len(model_list)):
            if j <= i:
                lmc[i][j] = 0

            else:
                loss_list = lmc_matrix[i][j]['loss']  # loss list alpha from 0 to 1

                L_theta = loss_list[-1]  # alpha=1.0
                L_theta_prime = loss_list[0]  # alpha=0.0

                mid_losses = np.array(loss_list)
                midpoint = 0.5 * (L_theta + L_theta_prime)

                deviations = np.abs(midpoint - mid_losses)
                t_star_idx = np.argmax(deviations)
                L_gamma_t_star = mid_losses[t_star_idx]

                mc_val = midpoint - L_gamma_t_star
                lmc[i][j] = mc_val

    result["lmc"] = lmc

    for cal_loss in loss_list_func:
        # print(f"Computing Bezier mode connectivity with loss {cal_loss.__name__}")
        mc_matrix = {}

        total_pairs = len(model_list) * (len(model_list) - 1) // 2
        current_pair = 0

        for i in range(len(model_list)):
            for j in range(len(model_list)):
                if j <= i:
                    continue

                if j != i + 1:
                    continue

                current_pair += 1
                print(f"Training Bezier curve between model {i} and model {j} ({current_pair}/{total_pairs})")
                mc_matrix[i] = mc_matrix.get(i, {})
                mc_matrix[i][j] = {}

                try:
                    # 创建贝塞尔曲线包装器并训练
                    curve_wrapper = SimpleCurveWrapper(dataset_path, model_list[i], model_list[j], device, args)
                    record = curve_wrapper.train_curve(args.stage)

                    # !!!!!!!!
                    # 对应 mc_taxonomy 265行往后
                    num_steps = args.num_steps
                    error_list_bezier = []
                    loss_list_bezier = []
                    alphas = np.linspace(0, 1, num_steps)

                    for alpha in alphas:
                        loss = curve_wrapper.get_loss_at_t(alpha)
                        loss_list_bezier.append(loss.item())

                    mc_matrix[i][j]['loss'] = loss_list_bezier

                except Exception as e:
                    print(f"Error training Bezier curve between model {i} and {j}: {e}")
                    traceback.print_exc()
                    mc_matrix[i][j]['loss'] = [np.nan] * num_steps
                    continue

        mc_bezier = np.zeros((len(model_list), len(model_list)))

        for i in range(len(model_list)):
            for j in range(len(model_list)):
                if j <= i:
                    mc_bezier[i][j] = 0
                elif j != i + 1:
                    mc_bezier[i][j] = 0
                else:
                    if i in mc_matrix and j in mc_matrix[i]:
                        loss_list_vals = mc_matrix[i][j]['loss']
                        # 检查是否有有效的损失值
                        if all(not np.isnan(x) for x in loss_list_vals):
                            L_theta = loss_list_vals[-1]
                            L_theta_prime = loss_list_vals[0]
                            mid_losses = np.array(loss_list_vals)
                            midpoint = 0.5 * (L_theta + L_theta_prime)
                            deviations = np.abs(midpoint - mid_losses)
                            t_star_idx = np.argmax(deviations)
                            L_gamma_t_star = mid_losses[t_star_idx]
                            mc_val = midpoint - L_gamma_t_star
                            mc_bezier[i][j] = mc_val
                        else:
                            mc_bezier[i][j] = 0
                    else:
                        mc_bezier[i][j] = 0

        result["mc_" + cal_loss.__name__] = mc_bezier  # 贝塞尔曲线使用mc前缀

    print('The lmc matrix is:')
    print(result['lmc'])
    print('The mc matrix is:')
    print(result['mc_loss_total'])

    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
        np.save(model_folder + '/mode_connectivity.npy', result)
    else:
        np.save(model_folder + '/mode_connectivity.npy', result)


if __name__ == "__main__":
    main()