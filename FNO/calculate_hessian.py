import os, sys, time
import argparse
import torch
import numpy as np
import logging
from utils import logging_utils
logging_utils.config_logger()
from utils.YParams import YParams
from utils.trainer import HessianCalculator_Pyhessian
from utils.data_utils import get_data_loader
from utils.domains import DomainXY

def main():
    # parsers
    parser = argparse.ArgumentParser(description='Calculate Hessian matrix properties for neural operators')
    parser.add_argument("--yaml_config", default='./config/operators_poisson_1M.yaml', type=str, help='Path to YAML config file')
    parser.add_argument("--config", default='default', type=str, help='Config name from YAML file')
    parser.add_argument("--checkpoint_path", required=True, type=str, help='Path to model checkpoint (.tar file)')
    parser.add_argument("--root_dir", default='./', type=str, help='Root directory to store results')
    parser.add_argument("--run_num", default='0', type=str, help='Sub run config identifier')
    
    # Hessian computation parameters
    parser.add_argument("--method", default='pyhessian_eigenvalues', type=str, 
                       choices=['pyhessian_eigenvalues', 'pyhessian_trace', 'pyhessian_density', 
                               'pyhessian_all', 'pyhessian_sharpness'],
                       help='Method for Hessian computation')
    parser.add_argument("--max_batches", default=4, type=int, 
                       help='Maximum number of batches to process (use -1 for all batches)')
    parser.add_argument("--batch_size", default=128, type=int, help='Batch size for measuring Hessian')
    parser.add_argument("--layerwise", action='store_true', 
                       help='Compute Hessian properties for each layer of the model (can be combined with any method)')
    #parser.add_argument("--seed", default=0, type=int, help='Random seed')
    
    # PyHessian specific parameters
    parser.add_argument("--top_n", default=1, type=int, help='Number of top eigenvalues to compute (PyHessian)')
    parser.add_argument("--max_iter", default=100, type=int, help='Maximum iterations for PyHessian methods')
    parser.add_argument("--tol", default=1e-3, type=float, help='Convergence tolerance for PyHessian methods')
    parser.add_argument("--use_single_batch", action='store_true', 
                       help='Use single batch for PyHessian computation (faster but less accurate)')
    parser.add_argument("--lanczos_iter", default=100, type=int, help='Number of Lanczos iterations for eigenvalue density')
    parser.add_argument("--slq_runs", default=1, type=int, help='Number of SLQ runs for eigenvalue density')

    # Model parameters (override config if needed)
    parser.add_argument("--target_batch_size", default=128, type=int, help='Batch size for used in training, not for measuring Hessian')
    parser.add_argument("--subsample", default=1, type=int, help='Subsample parameter')
    parser.add_argument("--target_seed", default=0, type=int, help='Random seed used for training experiments')
    parser.add_argument("--expt_max_epochs", default=1000, type=int, help='Maximum number of epochs for training experiments')
    parser.add_argument("--ckpt_epoch", default=-1, type=int, help='Number of epochs for the checkpoint to be loaded')
    
    args = parser.parse_args()
    
    # yaml_config
    if "16M" in args.config:
        args.yaml_config = './config/operators_poisson_16M.yaml'
    else:
        args.yaml_config = './config/operators_poisson_1M.yaml'
    
    # Check if checkpoint exists
    logging.info(f"\n\n\n Executing the script with the following arguments: {args} \n")

    if not os.path.exists(args.checkpoint_path):
        logging.error(f"Checkpoint file not found: {args.checkpoint_path}")
        sys.exit(1)
    
    logging.info(f"Loading checkpoint from: {args.checkpoint_path}")
    logging.info(f"Using method: {args.method}")
    
    # Initialize parameters
    params = YParams(os.path.abspath(args.yaml_config), args.config)
    
    # Override parameters if provided
    if args.batch_size is not None:
        params['batch_size'] = args.batch_size
    if args.subsample is not None:
        params['subsample'] = args.subsample
    params['seed'] = args.target_seed
    params['target_seed'] = args.target_seed
    params['expt_max_epochs'] = args.expt_max_epochs
    params['target_batch_size'] = args.target_batch_size

    # Set validation batch size if not already set
    if not hasattr(params, 'valid_batch_size'):
        params['valid_batch_size'] = params.batch_size
    
    logging.info(f"Using batch_size: {params.batch_size}")
    logging.info(f"Using subsample: {params.subsample}")
    
    # Set device
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        logging.info(f"Using GPU: {device}")
    else:
        device = torch.device('cpu')
        logging.info("Using CPU")
    
    # Choose the appropriate calculator based on method
    use_pyhessian = args.method.startswith('pyhessian')
    
    if use_pyhessian:
        # Initialize HessianCalculator_Pyhessian and load checkpoint
        logging.info("Initializing HessianCalculator_Pyhessian...")
        hessian_calc = HessianCalculator_Pyhessian(params)
    else:
        # Initialize original HessianCalculator and load checkpoint
        raise NotImplementedError("Original HessianCalculator not implemented")
    
    ## ckpt.tar -> ckpt_best.tar
    logging.info("Loading model checkpoint...")
    metadata = hessian_calc.restore_checkpoint(args.checkpoint_path)
    if metadata['epoch'] < params.expt_max_epochs - 5:
        args.checkpoint_path = args.checkpoint_path.replace('ckpt.tar', 'ckpt_best.tar')
        metadata = hessian_calc.restore_checkpoint(args.checkpoint_path)
        logging.info(f"Using best checkpoint from epoch {metadata['epoch']}, iteration {metadata['iters']}")
    else:
        logging.info(f"Loaded checkpoint from epoch {metadata['epoch']}, iteration {metadata['iters']}")
    
    # Print model architecture summary
    hessian_calc.print_model_architecture_summary()
    
    # Load data
    logging.info("Loading data...")
    data_loader, dataset, sampler = get_data_loader(
        params, params.train_path, distributed=False, train=True, pack=params.pack_data
    )
    logging.info(f"Using training data: {len(dataset)} samples")
    
    # Prepare output directory (match training log structure)
    # Extract parameters for directory structure
    lr = getattr(params, 'lr', 0.001)  # Default values if not found
    epochs = getattr(params, 'expt_max_epochs', 1000)
    
    # Create directory structure matching training logs
    # Format: hessian_analysis/expts_eps{epochs}/{config}/train/bsz{batch_size}_lr{lr}_subsample{subsample}/seed{seed}_ckpt_epoch{ckpt_epoch}/
    if args.ckpt_epoch == -1:
        ckpt_epoch = ''
    else:
        ckpt_epoch = f'_ckpt_epoch{args.ckpt_epoch}'
    
    # Add layerwise suffix to method name if needed
    method_name = args.method
    if args.layerwise:
        method_name += '_layerwise'
        
    output_dir = os.path.join(
        args.root_dir, 
        'hessian_analysis',
        f'expts_eps{epochs}',
        args.config,
        f'hessian_analysis_{method_name}_topN{args.top_n}',  # Distinguish between methods
        f'bsz{params.target_batch_size}_lr{lr}_subsample{params.subsample}',
        f'seed{params.target_seed}{ckpt_epoch}'
    )
    # IF the output directory already exists, then skip the computation
    if os.path.exists(output_dir):
        logging.info(f"Output directory already exists: {output_dir}")
        sys.exit(0)
    else:
        os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Output directory: {output_dir}")
    
    
    # Calculate Hessian properties
    logging.info("Starting Hessian computation...")
    start_time = time.time()
    
    # Set max_batches to None if user wants all batches
    max_batches = None if args.max_batches == -1 else args.max_batches
    
    # Determine if we should use layerwise analysis
    use_layerwise = args.layerwise
    
    if use_pyhessian:
        # PyHessian methods
        common_kwargs = {
            'max_batches': max_batches,
            'use_single_batch': args.use_single_batch,
            'max_iter': args.max_iter,
            'tol': args.tol,
            'layerwise': use_layerwise
        }
        
        if args.method == 'pyhessian_eigenvalues':
            if use_layerwise:
                results = hessian_calc.compute_layerwise_hessian(
                    data_loader,
                    method='eigenvalues',
                    top_n=args.top_n,
                    **{k: v for k, v in common_kwargs.items() if k != 'layerwise'}
                )
            else:
                results = hessian_calc.compute_eigenvalues(
                    data_loader, 
                    top_n=args.top_n,
                    **{k: v for k, v in common_kwargs.items() if k != 'layerwise'}
                )
        elif args.method == 'pyhessian_trace':
            if use_layerwise:
                results = hessian_calc.compute_layerwise_hessian(
                    data_loader,
                    method='trace',
                    **{k: v for k, v in common_kwargs.items() if k != 'layerwise'}
                )
            else:
                results = hessian_calc.compute_trace(
                    data_loader,
                    **{k: v for k, v in common_kwargs.items() if k != 'layerwise'}
                )
        elif args.method == 'pyhessian_density':
            # keep only args accepted by compute_eigenvalue_density
            density_kwargs = {
                k: v
                for k, v in common_kwargs.items()
                if k in ['max_batches', 'use_single_batch']
            }
            density_kwargs.update({
                'iter': args.lanczos_iter,
                'n_v': args.slq_runs
            })
            if use_layerwise:
                results = hessian_calc.compute_layerwise_hessian(
                    data_loader,
                    method='density',
                    **density_kwargs
                )
            else:
                results = hessian_calc.compute_eigenvalue_density(
                    data_loader,
                    **density_kwargs
                )
        elif args.method == 'pyhessian_all':
            results = hessian_calc.analyze_hessian_properties(
                data_loader,
                method='all',
                top_n=args.top_n,
                iter=args.lanczos_iter,
                n_v=args.slq_runs,
                **common_kwargs
            )
        elif args.method == 'pyhessian_sharpness':
            results = hessian_calc.compute_sharpness_metrics(
                data_loader,
                top_n=args.top_n,
                iter=args.lanczos_iter,
                n_v=args.slq_runs,
                **common_kwargs
            )
    else:
        pass
    
    computation_time = time.time() - start_time
    logging.info(f"Hessian computation completed in {computation_time:.2f} seconds")
    
    # Display results
    print("\n" + "="*60)
    print(f"HESSIAN ANALYSIS RESULTS ({'PyHessian' if use_pyhessian else 'Original'})")
    print("="*60)
    print(f"Method: {args.method}")
    if use_layerwise:
        print(f"Layerwise analysis: Enabled")
    print(f"Model parameters: {hessian_calc.n_params:,}")
    print(f"Computation time: {computation_time:.2f} seconds")
    print(f"Checkpoint epoch: {metadata['epoch']}")
    if use_pyhessian:
        print(f"Use single batch: {args.use_single_batch}")
        print(f"Max iterations: {args.max_iter}")
        print(f"Tolerance: {args.tol}")
    print("-"*60)
    
    def print_results_recursive(results, indent=0):
        """Recursively print results with proper indentation"""
        prefix = "  " * indent
        for key, value in results.items():
            if isinstance(value, dict):
                print(f"{prefix}{key}:")
                print_results_recursive(value, indent + 1)
            elif isinstance(value, (list, np.ndarray)) and len(value) > 10:
                print(f"{prefix}{key}: [array with {len(value)} elements]")
            elif isinstance(value, float):
                print(f"{prefix}{key}: {value:.8f}")
            else:
                print(f"{prefix}{key}: {value}")
    
    print_results_recursive(results)
    
    # Save detailed layer information to files
    print("Saving layer information and analysis results...")
    
    # Save layer dimensions and Hessian metrics
    if use_layerwise and 'layerwise' in results:
        # Save with layerwise Hessian results
        hessian_calc.save_layerwise_info_to_file(output_dir, results)
    else:
        # Save just layer dimensions info
        hessian_calc.save_layerwise_info_to_file(output_dir)
    
    # Save results to file
    output_file = os.path.join(output_dir, f"hessian_{args.method}_results.txt")
    with open(output_file, 'w') as f:
        f.write(f"HESSIAN ANALYSIS RESULTS ({'PyHessian' if use_pyhessian else 'Original'})\n")
        f.write("="*60 + "\n")
        f.write(f"Checkpoint: {args.checkpoint_path}\n")
        f.write(f"Method: {args.method}\n")
        if use_layerwise:
            f.write(f"Layerwise analysis: Enabled\n")
        f.write(f"Model parameters: {hessian_calc.n_params:,}\n")
        f.write(f"Computation time: {computation_time:.2f} seconds\n")
        f.write(f"Checkpoint epoch: {metadata['epoch']}\n")
        f.write(f"Batches processed: {max_batches if max_batches else 'all'}\n")
        
        if use_pyhessian:
            f.write(f"Use single batch: {args.use_single_batch}\n")
            f.write(f"Max iterations: {args.max_iter}\n")
            f.write(f"Tolerance: {args.tol}\n")
            if args.method in ['pyhessian_eigenvalues', 'pyhessian_all', 'pyhessian_sharpness']:
                f.write(f"Top eigenvalues: {args.top_n}\n")
            if args.method in ['pyhessian_density', 'pyhessian_all', 'pyhessian_sharpness']:
                f.write(f"Lanczos iterations: {args.lanczos_iter}\n")
                f.write(f"SLQ runs: {args.slq_runs}\n")
        else:
            pass    
        
        f.write("-"*60 + "\n")
        
        def write_results_recursive(results, f, indent=0):
            """Recursively write results with proper indentation"""
            prefix = "  " * indent
            for key, value in results.items():
                if isinstance(value, dict):
                    f.write(f"{prefix}{key}:\n")
                    write_results_recursive(value, f, indent + 1)
                elif isinstance(value, (list, np.ndarray)) and len(value) > 10:
                    f.write(f"{prefix}{key}: [array with {len(value)} elements]\n")
                elif isinstance(value, float):
                    f.write(f"{prefix}{key}: {value:.8f}\n")
                else:
                    f.write(f"{prefix}{key}: {value}\n")
        
        write_results_recursive(results, f)
    
    logging.info(f"Results saved to: {output_file}")

if __name__ == '__main__':
    main()
