# external libraries and packages
import wandb
import argparse
import sys
import traceback
import torch
import os
import numpy as np

os.environ["WANDB_MODE"] = "offline"

from src.train_utils import set_random_seed, train, parse_params_list
from src.models import PINN
sys.path.append('./multiadam')  # Add multiadam to path
from multiadam.benchmark import run_multiadam_experiment

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--initial_seed', type=int, default=1234, help='initial seed')
    parser.add_argument('--pde', type=str,
                        default='convection', help='PDE type')
    parser.add_argument('--pde_params', type=str,
                        default='{"beta":30}', help='PDE coefficients')
    parser.add_argument('--opt', type=str, default='lbfgs',
                        help='optimizer to use')
    parser.add_argument('--opt_params', nargs='+', type=str,
                        default=None, help='optimizer parameters')
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
    parser.add_argument('--num_res', type=int, default=10000,
                        help='number of sampled residual points')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of epochs to run')
    parser.add_argument('--wandb_project', type=str,
                        default='pinns', help='W&B project name')
    parser.add_argument('--new_data', action="store_true", help='whether to create a new training set')
    parser.add_argument('--set_idx', type=int, default=0, help='the index of dataset')
    parser.add_argument('--device', type=str, default=0, help='GPU to use')
    parser.add_argument('--save_path', type=str, default=None, help='path to save the results of experiments')
    parser.add_argument('--save_model', action="store_true", help='Save the model for analysis later.')

    parser.add_argument('--hc', type=str, default='none', help='hc method')
    parser.add_argument('--L', type=float, default=1, help='pde loss weight')
    parser.add_argument('--alm_mu', type=float, default=2, help='hc method')
    parser.add_argument('--alm_L', type=float, default=1, help='hc method')
    parser.add_argument('--alm_beta', type=float, default=2, help='hc method')
    parser.add_argument('--alm_iter', type=float, default=10, help='hc method')
    parser.add_argument('--alm_hc', type=int, default=0b11110, help='hc method')
    parser.add_argument('--alm_weight_decay', type=float, default=0, help='hc method')

    parser.add_argument('--cl', action="store_true", help='whether to create a new training set')

    # Extract arguments from parser
    args = parser.parse_args()
    # set initial seed
    initial_seed = args.initial_seed
    set_random_seed(initial_seed)

    # organize arguments for the experiment into a dictionary for logging purpose
    experiment_args = {
        "initial_seed": args.initial_seed,
        "pde": args.pde,
        "pde_params": args.pde_params,
        "opt": args.opt,
        "opt_params": args.opt_params,
        "num_layers": args.num_layers,
        "num_neurons": args.num_neurons,
        "loss": args.loss,
        "num_x": args.num_x,
        "num_t": args.num_t,
        "num_res": args.num_res, 
        "epochs": args.epochs,
        "wandb_project": args.wandb_project,
        "new_data": bool(args.new_data),
        "device": f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu',
        "save_path": args.save_path,
        "save_model": args.save_model
    }

    # print out arguments
    print("Seed set to: {}".format(initial_seed))
    print("Selected PDE type: {}".format(experiment_args["pde"]))
    print("Specified PDE coefficients: {}".format(
        experiment_args["pde_params"]))
    print("Optimizer to use: {}".format(experiment_args["opt"]))
    print("Specified optimizer parameters: {}".format(
        experiment_args["opt_params"]))
    print("Number of layers: {}".format(experiment_args["num_layers"]))
    print("Number of neurons per layer: {}".format(experiment_args["num_neurons"]))
    print("Number of spatial points (x): {}".format(experiment_args["num_x"]))
    print("Number of temporal points (t): {}".format(experiment_args["num_t"]))
    print("Number of random residual points to sample: {}".format(experiment_args["num_res"]))
    print("Number of epochs: {}".format(experiment_args["epochs"]))
    print("Weights and Biases project: {}".format(
        experiment_args["wandb_project"]))
    print("GPU to use: {}".format(experiment_args["device"]))

    pde_params = parse_params_list(args.pde_params)

    if experiment_args["pde"] == 'reaction_diffusion':
        folder = os.path.join(
            experiment_args["save_path"],
            f'system_{experiment_args["pde"]}',
            f'N_f_{experiment_args["num_res"]}',
            f'nu_{float(pde_params["nu"])}_rho_{float(pde_params["rho"])}'
        )
        dataset_path = os.path.join(
            "./dataset",
            f'system_{experiment_args["pde"]}',
            f'N_f_{experiment_args["num_res"]}',
            f'nu_{float(pde_params["nu"])}_rho_{float(pde_params["rho"])}'
        )
    elif args.pde == 'convection':
        folder = os.path.join(
            experiment_args["save_path"],
            f'system_{experiment_args["pde"]}',
            f'N_f_{experiment_args["num_res"]}',
            f'beta_{float(pde_params["beta"])}'
        )
        dataset_path = os.path.join(
            "./dataset",
            f'system_{experiment_args["pde"]}',
            f'N_f_{experiment_args["num_res"]}',
            f'beta_{float(pde_params["beta"])}'
        )
    elif args.pde == 'reaction':
        folder = os.path.join(
            experiment_args["save_path"],
            f'system_{experiment_args["pde"]}',
            f'N_f_{experiment_args["num_res"]}',
            f'rho_{float(pde_params["rho"])}'
        )
        dataset_path = os.path.join(
            "./dataset",
            f'system_{experiment_args["pde"]}',
            f'N_f_{experiment_args["num_res"]}',
            f'rho_{float(pde_params["rho"])}'
        )
    elif args.pde == 'wave':
        folder = os.path.join(
            experiment_args["save_path"],
            f'system_{experiment_args["pde"]}',
            f'N_f_{experiment_args["num_res"]}',
            f'beta_{float(pde_params["beta"])}_c_{float(pde_params["c"])}'
        )
        dataset_path = os.path.join(
            "./dataset",
            f'system_{experiment_args["pde"]}',
            f'N_f_{experiment_args["num_res"]}',
            f'beta_{float(pde_params["beta"])}_c_{float(pde_params["c"])}'
        )

    # Check if we should use MultiAdam benchmark - ONLY for multiadam optimizer
    if experiment_args["opt"].lower() == 'multiadam':
        print(f"\n{'='*60}")
        print(f"Using MultiAdam benchmark implementation")
        print(f"{'='*60}\n")
        
        # Prepare PDE parameters based on PDE type
        # pde_params_dict = {}
        # if experiment_args["pde"] == 'convection':
        #     if experiment_args["pde_params"] and len(experiment_args["pde_params"]) > 1:
        #         pde_params_dict['beta'] = float(experiment_args["pde_params"][1])
        #     else:
        #         pde_params_dict['beta'] = 2.0
        # elif experiment_args["pde"] == 'reaction':
        #     if experiment_args["pde_params"] and len(experiment_args["pde_params"]) > 1:
        #         pde_params_dict['alpha'] = float(experiment_args["pde_params"][1])
        #     else:
        #         pde_params_dict['alpha'] = 5.0
        # elif experiment_args["pde"] == 'diffusion':
        #     if experiment_args["pde_params"] and len(experiment_args["pde_params"]) > 1:
        #         pde_params_dict['tau'] = float(experiment_args["pde_params"][1])
        #     else:
        #         pde_params_dict['tau'] = 4.0
        # elif experiment_args["pde"] == 'reaction_diffusion':
        #     if experiment_args["pde_params"]:
        #         if len(experiment_args["pde_params"]) > 1:
        #             pde_params_dict['alpha'] = float(experiment_args["pde_params"][1])
        #         if len(experiment_args["pde_params"]) > 0:
        #             pde_params_dict['tau'] = float(experiment_args["pde_params"][-1])
        #     else:
        #         pde_params_dict['alpha'] = 5.0
        #         pde_params_dict['tau'] = 4.0
        pde_params_dict = pde_params

        # Get learning rate
        lr = float(experiment_args["opt_params"][0]) if experiment_args["opt_params"] else 1e-3
        
        # Prepare save directory
        save_dir = None
        if experiment_args["save_model"]:
            folder = os.path.join(folder, f'set_{args.set_idx}', args.hc, f'seed_{initial_seed}')
            save_dir = folder
        
        # Run MultiAdam experiment
        try:
            run_multiadam_experiment(
                pde_type=experiment_args["pde"],
                pde_params=pde_params_dict,
                opt_name='multiadam',
                n_colloc=experiment_args["num_res"],
                hidden_layers=f"{experiment_args['num_neurons']}*{experiment_args['num_layers']}",
                lr=lr,
                iterations=experiment_args["epochs"],
                seed=initial_seed,
                data_seed=args.set_idx,  # Use set_idx as data seed for reproducibility
                device=str(args.device),
                save_dir=save_dir,
                exp_name=f"{experiment_args['pde']}_multiadam_seed{initial_seed}",
                log_every=100,
                plot_every=2000,
            )
        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            raise e
            
        print(f"\n{'='*60}")
        print(f"MultiAdam benchmark experiment completed")
        print(f"{'='*60}\n")
        
    else:
        # Original training code path for all other optimizers
        print(f"\n{'='*60}")
        print(f"Using original training implementation for {experiment_args['opt']}")
        print(f"{'='*60}\n")

        with wandb.init(project=experiment_args["wandb_project"], config=experiment_args):
            # initialize model
            model = PINN(in_dim=2, hidden_dim=experiment_args["num_neurons"], out_dim=1,
                         num_layer=experiment_args["num_layers"]).to(experiment_args["device"])
            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"The totel number of model parameters is {total_params}.")
            folder = os.path.join(folder, f'set_{args.set_idx}', args.hc)
            # train the model
            # print(experiment_args["new_data"], type(experiment_args["new_data"]))
            try:
                train(model,
                      proj_name=experiment_args["wandb_project"],
                      pde_name=experiment_args["pde"],
                      pde_params=experiment_args["pde_params"],
                      loss_name=experiment_args["loss"],
                      opt_name=experiment_args["opt"],
                      opt_params_list=experiment_args["opt_params"],
                      n_x=experiment_args["num_x"],
                      n_t=experiment_args["num_t"],
                      n_res=experiment_args["num_res"],
                      num_epochs=experiment_args["epochs"],
                      device=experiment_args["device"],
                      folder=folder,
                      dataset_path=dataset_path,
                      new_data=experiment_args["new_data"],
                      set_idx=args.set_idx,
                      sample_seed=args.set_idx,
                      initial_seed=args.initial_seed,
                      hc=args.hc,
                      cl=args.cl,
                      L=args.L, alm_L=args.alm_L, alm_beta=args.alm_beta, alm_iter=args.alm_iter, alm_mu=args.alm_mu,
                      alm_hc=args.alm_hc, weight_decay=args.alm_weight_decay
                      )
            # log error and traceback info to W&B, and exit gracefully
            except Exception as e:
                traceback.print_exc(file=sys.stderr)
                raise e

            if experiment_args["save_model"]:
                save_path = os.path.join(folder, f"seed_{initial_seed}.pt")
                torch.save(model.state_dict(), save_path)
                print(save_path)

if __name__ == "__main__":
    main()