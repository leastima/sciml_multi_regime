import argparse
import time
import os
from trainer import Trainer

os.environ["DDEBACKEND"] = "pytorch"

import numpy as np
import torch
import deepxde as dde

# Use absolute imports from multiadam's perspective
from multiadam_src.model.laaf import DNN_GAAF, DNN_LAAF
from multiadam_src.optimizer import MultiAdam, LR_Adaptor, LR_Adaptor_NTK, Adam_LBFGS
from multiadam_src.pde.convection import Convection1D
from multiadam_src.pde.reaction_diffusion import ReactionDiffusion1D, Reaction1D, Diffusion1D
from multiadam_src.utils.args import parse_hidden_layers, parse_loss_weight
from multiadam_src.utils.callbacks import (
    TesterCallback,
    PlotCallback,
    LossCallback,
    ModelSaveCallback,
    ModelCheckpointCallback,
)
from multiadam_src.utils.rar import rar_wrapper

# Updated PDE list to include reaction-diffusion variants
pde_list = [Convection1D]
# pde_list = [ReactionDiffusion1D]

def run_multiadam_experiment(
    pde_type="convection",
    pde_params=None,
    opt_name="multiadam",
    n_colloc=10000,
    hidden_layers="50*4",
    lr=1e-3,
    iterations=20000,
    seed=1234,
    data_seed=42,
    device="0",
    save_dir=None,
    exp_name=None,
    log_every=100,
    plot_every=2000,
):
    """
    Wrapper function to run MultiAdam experiments compatible with run_experiment.py API.
    
    Args:
        pde_type: Type of PDE ('convection', 'reaction', 'diffusion', 'reaction_diffusion')
        pde_params: Dictionary with PDE parameters (e.g., {'beta': 2.0} for convection)
        opt_name: Optimizer name ('multiadam', 'lbfgs', 'adam', 'lra', 'ntk', etc.)
        n_colloc: Number of collocation points
        hidden_layers: Network architecture as string (e.g., "50*4")
        lr: Learning rate
        iterations: Number of training iterations
        seed: Model initialization seed
        data_seed: Data generation seed (for reproducibility)
        device: GPU device ID
        save_dir: Directory to save models and results
        exp_name: Experiment name for logging
        log_every: Logging frequency
        plot_every: Plotting frequency
    
    Returns:
        Trained model
    """
    # Set data generation seed
    if data_seed is not None:
        dde.config.set_random_seed(data_seed)
        print(f"Data generation seed set to: {data_seed}")
    
    # Set model initialization seed
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        print(f"Model initialization seed set to: {seed}")
    
    # Initialize trainer
    if exp_name is None:
        date_str = time.strftime("%m.%d-%H.%M.%S", time.localtime())
        exp_name = f"{date_str}-{pde_type}_{opt_name}"
    
    trainer = Trainer(exp_name, device)
    
    # Map PDE type to class
    pde_map = {
        'convection': Convection1D,
        'reaction': Reaction1D,
        'diffusion': Diffusion1D,
        'reaction_diffusion': ReactionDiffusion1D,
    }
    
    if pde_type not in pde_map:
        raise ValueError(f"Unknown PDE type: {pde_type}. Available: {list(pde_map.keys())}")
    
    pde_class = pde_map[pde_type]
    
    # Prepare PDE kwargs
    pde_kwargs = {'n_colloc': n_colloc}
    if pde_params:
        pde_kwargs.update(pde_params)
    
    def get_model():
        """Create and configure the model"""
        # Set model seed again in case of multiple tasks
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Create PDE
        pde = pde_class(**pde_kwargs)
        
        # Create network
        hidden_layer_sizes = parse_hidden_layers(type('Args', (), {'hidden_layers': hidden_layers})())
        net = dde.nn.FNN(
            [pde.input_dim] + hidden_layer_sizes + [pde.output_dim],
            "tanh",
            "Glorot normal",
        ).float()
        
        # Configure optimizer
        if opt_name == "multiadam":
            opt = MultiAdam(
                net.parameters(),
                lr=lr,
                betas=(0.99, 0.99),
                loss_group_idx=[pde.num_pde],
            )
        elif opt_name == "lbfgs":
            opt = Adam_LBFGS(
                net.parameters(),
                switch_epoch=10000,
                adam_lr=lr,
                lbfgs_lr=1.0,
                lbfgs_max_iter=50,
            )
        elif opt_name == "adam":
            opt = torch.optim.Adam(net.parameters(), lr)
        elif opt_name == "lra":
            base_opt = torch.optim.Adam(net.parameters(), lr)
            loss_weights = np.ones(pde.num_loss)
            opt = LR_Adaptor(base_opt, loss_weights, pde.num_pde)
        elif opt_name == "ntk":
            base_opt = torch.optim.Adam(net.parameters(), lr)
            loss_weights = np.ones(pde.num_loss)
            opt = LR_Adaptor_NTK(base_opt, loss_weights, pde)
        else:
            raise ValueError(f"Unknown optimizer: {opt_name}")
        
        # Create model
        loss_weights = [1.0, 1.0, 1.0]  # PDE, IC, BC
        model = pde.create_model(net)
        model.compile(opt, loss_weights=loss_weights, metrics=["l2 relative error"])
        
        return model
    
    # Setup callbacks
    callbacks = [
        TesterCallback(log_every=log_every),
        PlotCallback(log_every=plot_every, fast=True),
        LossCallback(verbose=True),
    ]
    
    if save_dir:
        callbacks.extend([
            ModelSaveCallback(
                save_dir=save_dir,
                experiment_name=exp_name,
                verbose=True,
            ),
            ModelCheckpointCallback(
                save_dir=save_dir,
                experiment_name=exp_name,
                checkpoint_iterations=[2000, 5000, 10000, 15000, 20000],
                verbose=True,
            ),
        ])
    
    # Add task and train
    trainer.add_task(
        get_model,
        {
            "iterations": iterations,
            "display_every": log_every,
            "callbacks": callbacks,
        },
    )
    
    trainer.setup(__file__, seed)
    trainer.train_all()
    trainer.summary()
    
    return trainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PINNBench trainer")
    parser.add_argument("--name", type=str, default="benchmark")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--hidden-layers", type=str, default="100*5")
    parser.add_argument("--loss-weight", type=str, default="")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--iter", type=int, default=20000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--plot-every", type=int, default=2000)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--method", type=str, default="multiadam")
    parser.add_argument("--beta", type=float, default=40)
    parser.add_argument("--n_colloc", type=int, default=20000)
    # Reaction-diffusion parameters
    parser.add_argument("--alpha", type=float, default=5.0, help="Growth rate")
    parser.add_argument("--tau", type=float, default=4.0, help="Spreading speed")
    parser.add_argument(
        "--zeta",
        type=float,
        default=None,
        help="Initial sharpness (default: 1/(2*(pi/4)^2))",
    )
    # Legacy parameters for backward compatibility
    parser.add_argument(
        "--nu", type=float, default=1.0, help="Diffusion coefficient (legacy)"
    )
    parser.add_argument(
        "--rho", type=float, default=1.0, help="Reaction coefficient (legacy)"
    )
    parser.add_argument(
        "--model-save-dir",
        type=str,
        default=None,
        help="Custom directory to save trained models",
    )
    parser.add_argument(
        "--data-seed",
        type=int,
        default=42,
        help="Fixed seed for data generation (ensures same training data)",
    )
    parser.add_argument(
        "--model-seed",
        type=int,
        default=None,
        help="Seed for model initialization (for CKA analysis)",
    )

    command_args = parser.parse_args()

    # Handle seeds properly for CKA analysis
    data_seed = command_args.data_seed  # Fixed seed for data generation
    model_seed = (
        command_args.model_seed
        if command_args.model_seed is not None
        else command_args.seed
    )

    # Set data generation seed first (this affects training data sampling)
    if data_seed is not None:
        dde.config.set_random_seed(data_seed)
        print(f"Data generation seed set to: {data_seed}")

    # Legacy seed handling for backward compatibility
    seed = command_args.seed
    if seed is not None and command_args.model_seed is None:
        dde.config.set_random_seed(seed)
        print(f"Using legacy seed for both data and model: {seed}")

    date_str = time.strftime("%m.%d-%H.%M.%S", time.localtime())
    trainer = Trainer(f"{date_str}-{command_args.name}", command_args.device)

    for pde_config in pde_list:

        def get_model_dde():
            # Set model initialization seed (only affects weights, not data)
            if model_seed is not None:
                torch.manual_seed(model_seed)
                np.random.seed(model_seed)
                print(f"Model initialization seed set to: {model_seed}")

            if isinstance(pde_config, tuple):
                # Add command line arguments to the kwargs
                kwargs = pde_config[1].copy()
                if pde_config[0] == Convection1D:
                    kwargs["beta"] = command_args.beta
                    kwargs["n_colloc"] = command_args.n_colloc
                elif pde_config[0] == ReactionDiffusion1D:
                    kwargs["alpha"] = command_args.alpha
                    kwargs["tau"] = command_args.tau
                    if command_args.zeta is not None:
                        kwargs["zeta"] = command_args.zeta
                    kwargs["n_colloc"] = command_args.n_colloc
                elif pde_config[0] == Reaction1D:
                    kwargs["alpha"] = command_args.alpha
                    if command_args.zeta is not None:
                        kwargs["zeta"] = command_args.zeta
                    kwargs["n_colloc"] = command_args.n_colloc
                elif pde_config[0] == Diffusion1D:
                    kwargs["tau"] = command_args.tau
                    if command_args.zeta is not None:
                        kwargs["zeta"] = command_args.zeta
                    kwargs["n_colloc"] = command_args.n_colloc
                pde = pde_config[0](**kwargs)
            else:
                # For classes without constructor arguments
                if pde_config == Convection1D:
                    pde = pde_config(
                        beta=command_args.beta, n_colloc=command_args.n_colloc
                    )
                elif pde_config == ReactionDiffusion1D:
                    kwargs = {
                        "alpha": command_args.alpha,
                        "tau": command_args.tau,
                        "n_colloc": command_args.n_colloc,
                    }
                    if command_args.zeta is not None:
                        kwargs["zeta"] = command_args.zeta
                    pde = pde_config(**kwargs)
                elif pde_config == Reaction1D:
                    kwargs = {
                        "alpha": command_args.alpha,
                        "n_colloc": command_args.n_colloc,
                    }
                    if command_args.zeta is not None:
                        kwargs["zeta"] = command_args.zeta
                    pde = pde_config(**kwargs)
                elif pde_config == Diffusion1D:
                    kwargs = {
                        "tau": command_args.tau,
                        "n_colloc": command_args.n_colloc,
                    }
                    if command_args.zeta is not None:
                        kwargs["zeta"] = command_args.zeta
                    pde = pde_config(**kwargs)
                else:
                    pde = pde_config()

            # pde.training_points()
            if command_args.method == "gepinn":
                pde.use_gepinn()

            net = dde.nn.FNN(
                [pde.input_dim] + parse_hidden_layers(command_args) + [pde.output_dim],
                "tanh",
                "Glorot normal",
            )
            if command_args.method == "laaf":
                net = DNN_LAAF(
                    len(parse_hidden_layers(command_args)) - 1,
                    parse_hidden_layers(command_args)[0],
                    pde.input_dim,
                    pde.output_dim,
                )
            elif command_args.method == "gaaf":
                net = DNN_GAAF(
                    len(parse_hidden_layers(command_args)) - 1,
                    parse_hidden_layers(command_args)[0],
                    pde.input_dim,
                    pde.output_dim,
                )
            net = net.float()

            # loss_weights = parse_loss_weight(command_args)
            loss_weights = [1.0, 1.0, 1.0]  # PDE, IC, and Periodic BC
            if loss_weights is None:
                loss_weights = np.ones(pde.num_loss)
            else:
                loss_weights = np.array(loss_weights)

            opt = torch.optim.Adam(net.parameters(), command_args.lr)
            if command_args.method == "multiadam":
                opt = MultiAdam(
                    net.parameters(),
                    lr=1e-3,
                    betas=(0.99, 0.99),
                    loss_group_idx=[pde.num_pde],
                )
            elif command_args.method == "lra":
                opt = LR_Adaptor(opt, loss_weights, pde.num_pde)
            elif command_args.method == "ntk":
                opt = LR_Adaptor_NTK(opt, loss_weights, pde)
            elif command_args.method == "lbfgs":
                opt = Adam_LBFGS(
                    net.parameters(),
                    switch_epoch=10000,  # We'll tune this next
                    adam_lr=command_args.lr,  # Use --lr for Adam (e.g., 1e-3)
                    lbfgs_lr=1.0,  # Fixed for L-BFGS
                    lbfgs_max_iter=50,  # Tune as below
                    # Add other params as needed
                )
            model = pde.create_model(net)

            model.compile(opt, loss_weights=loss_weights, metrics=["l2 relative error"])
            if command_args.method == "rar":
                model.train = rar_wrapper(pde, model, {"interval": 1000, "count": 1})
            # the trainer calls model.train(**train_args)
            return model

        def get_model_others():
            model = None
            # create a model object which support .train() method, and param @model_save_path is required
            # create the object based on command_args and return it to be trained
            # schedule the task using trainer.add_task(get_model_other, {training args})
            return model

        # Create callbacks list
        callbacks = [
            TesterCallback(log_every=command_args.log_every),
            PlotCallback(log_every=command_args.plot_every, fast=True),
            LossCallback(verbose=True),
        ]

        # Add model save callback if custom save directory is specified
        if command_args.model_save_dir:
            callbacks.append(
                ModelSaveCallback(
                    save_dir=command_args.model_save_dir,
                    experiment_name=command_args.name,
                    verbose=True,
                )
            )

            # Add checkpoint callback for model similarity analysis
            # Save checkpoints at key epochs: 2000 (LBFGS switch point), 5000, 10000, 15000
            callbacks.append(
                ModelCheckpointCallback(
                    save_dir=command_args.model_save_dir,
                    experiment_name=command_args.name,
                    checkpoint_iterations=[2000, 5000, 10000, 15000, 20000, 25000],
                    verbose=True,
                )
            )

        trainer.add_task(
            get_model_dde,
            {
                "iterations": command_args.iter,
                "display_every": command_args.log_every,
                "callbacks": callbacks,
            },
        )

    trainer.setup(__file__, seed)
    trainer.set_repeat(command_args.repeat)
    trainer.train_all()
    trainer.summary()
