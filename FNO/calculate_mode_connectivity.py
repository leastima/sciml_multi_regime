#!/usr/bin/env python3
"""
Mode Connectivity Analysis Script

This script calculates mode connectivity between different neural network checkpoints
using Bézier curves to assess the smoothness of the loss landscape between models.

Mode connectivity formula:
mc(θ, θ') = 1/2(L(θ) + L(θ')) - L(γ_φ(t*))

Where t* = argmin |1/2(L(θ) + L(θ')) - L(γ_φ(t))|

Interpretation:
- mc < 0: Poor connectivity (loss barrier between models)
- mc > 0: Suspicious connectivity (lower loss regions, may indicate poor training)
- mc ≈ 0: Good connectivity (well-connected models)
"""

import os, sys, time
import argparse
import torch
import numpy as np
import logging
from pathlib import Path
from utils import logging_utils
logging_utils.config_logger()
from utils.YParams import YParams
from utils.trainer import ModeConnectivityCalculator
from utils.data_utils import get_data_loader
from utils.domains import DomainXY

def main():
    # parsers
    parser = argparse.ArgumentParser(description='Calculate Mode Connectivity between neural network models')
    parser.add_argument("--yaml_config", default='./config/operators_poisson_1M.yaml', type=str, help='Path to YAML config file')
    parser.add_argument("--config", default='default', type=str, help='Config name from YAML file')
    parser.add_argument("--checkpoint_a", required=True, type=str, help='Path to first model checkpoint (.tar file)')
    parser.add_argument("--checkpoint_b", required=True, type=str, help='Path to second model checkpoint (.tar file)')
    parser.add_argument("--root_dir", default='./', type=str, help='Root directory to store results')
    parser.add_argument("--run_num", default='0', type=str, help='Sub run config identifier')
    
    # Mode connectivity computation parameters
    parser.add_argument("--curve_type", default='linear', type=str, 
                       choices=['linear', 'bezier'],
                       help='Type of curve to use for interpolation')
    parser.add_argument("--optimize_control_point", action='store_true',
                       help='Optimize Bézier curve control point (only for bezier curve)')
    parser.add_argument("--num_curve_points", default=21, type=int, 
                       help='Number of points to evaluate along the curve')
    parser.add_argument("--max_batches", default=5, type=int, 
                       help='Maximum number of batches to process (use -1 for all batches)')
    parser.add_argument("--use_validation", action='store_true', 
                       help='Use validation data instead of training data')
    parser.add_argument("--comprehensive", action='store_true',
                       help='Run comprehensive analysis with multiple curve types')
    
    # Advanced options
    parser.add_argument("--pairwise", nargs='+', metavar='CHECKPOINT',
                       help='Compute pairwise connectivity between multiple checkpoints')
    parser.add_argument("--save_plots", action='store_true',
                       help='Save loss landscape plots')
    parser.add_argument("--batch_size", default=128, type=int, help='Batch size for data loading')
    parser.add_argument("--seed", default=0, type=int, help='Random seed')

    # Model parameters (override config if needed)
    parser.add_argument("--target_batch_size", default=128, type=int, help='Batch size for used in training, not for measuring mode connectivity')
    parser.add_argument("--subsample", default=1, type=int, help='Subsample parameter')
    parser.add_argument("--seed_a", default=0, type=int, help='seed_a')
    parser.add_argument("--seed_b", default=0, type=int, help='seed_b')
    parser.add_argument("--expt_max_epochs", default=1000, type=int, help='Maximum number of epochs for training experiments')
    
    args = parser.parse_args()
    
    # Check if checkpoints exist
    logging.info(f"\n\n\n Executing the script with the following arguments: {args} \n")

    if not os.path.exists(args.checkpoint_a):
        logging.error(f"Checkpoint A file not found: {args.checkpoint_a}")
        sys.exit(1)
    
    if not os.path.exists(args.checkpoint_b):
        logging.error(f"Checkpoint B file not found: {args.checkpoint_b}")
        sys.exit(1)
    
    logging.info(f"Loading checkpoint A from: {args.checkpoint_a}")
    logging.info(f"Loading checkpoint B from: {args.checkpoint_b}")
    logging.info(f"Using curve type: {args.curve_type}")
    
    # Initialize parameters
    params = YParams(os.path.abspath(args.yaml_config), args.config)
    
    # Override parameters if provided
    if args.batch_size is not None:
        params['batch_size'] = args.batch_size
    if args.subsample is not None:
        params['subsample'] = args.subsample
    params['seed'] = args.seed
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

    # Initialize ModeConnectivityCalculator
    logging.info("Initializing ModeConnectivityCalculator...")
    mode_calc = ModeConnectivityCalculator(params)
    
    # Load data
    logging.info("Loading data...")
    if args.use_validation:
        data_loader, dataset, sampler = get_data_loader(
            params, params.val_path, distributed=False, train=False, pack=params.pack_data
        )
        logging.info(f"Using validation data: {len(dataset)} samples")
    else:
        data_loader, dataset, sampler = get_data_loader(
            params, params.train_path, distributed=False, train=False, pack=params.pack_data
        )
        logging.info(f"Using training data: {len(dataset)} samples")
    
    # Prepare output directory (match training log structure)
    # Extract parameters for directory structure
    lr = getattr(params, 'lr', 0.001)  # Default values if not found
    epochs = getattr(params, 'expt_max_epochs', 1000)
    
    # Create directory structure matching training logs
    # Format: mode_connectivity_analysis/expts_eps{epochs}/{config}/mode_connectivity/bsz{batch_size}_lr{lr}_subsample{subsample}/seed{seed}/
    checkpoint_a_name = Path(args.checkpoint_a).stem.replace('ckpt', '').replace('_best', '').strip('_')
    checkpoint_b_name = Path(args.checkpoint_b).stem.replace('ckpt', '').replace('_best', '').strip('_')
    connectivity_type = f"{checkpoint_a_name}_vs_{checkpoint_b_name}_{args.curve_type}"
    
    output_dir = os.path.join(
        args.root_dir, 
        'mode_connectivity_analysis',
        f'expts_eps{epochs}',
        args.config,
        connectivity_type, 
        f'bsz{params.target_batch_size}_lr{lr}_subsample{params.subsample}',
        f'seed{args.seed_a}_vs_{args.seed_b}'
    )
    os.makedirs(output_dir, exist_ok=True)
    
    logging.info(f"Output directory: {output_dir}")
    
    # Check if output directory already exists and has results
    results_file = os.path.join(output_dir, f"mode_connectivity_{args.curve_type}_results.txt")
    if os.path.exists(results_file):
        logging.info(f"Results file already exists: {results_file}")
        logging.info("Skipping computation.")
        sys.exit(0)
    
    # Load models
    logging.info("Loading model checkpoints...")
    mode_calc.load_models(args.checkpoint_a, args.checkpoint_b)
    
    # Calculate Mode Connectivity properties
    logging.info("Starting mode connectivity computation...")
    start_time = time.time()
    
    # Set max_batches to None if user wants all batches
    max_batches = args.max_batches if args.max_batches != -1 else None
    
    try:
        # Check if doing pairwise analysis
        if args.pairwise:
            logging.info("Running pairwise mode connectivity analysis...")
            checkpoint_paths = args.pairwise
            
            # Compute pairwise connectivity
            pairwise_results = mode_calc.compute_pairwise_connectivity(
                checkpoint_paths, data_loader, max_batches=max_batches)
            
            # Save pairwise results
            import json
            pairwise_file = os.path.join(output_dir, f'pairwise_results.json')
            with open(pairwise_file, 'w') as f:
                # Convert numpy arrays for JSON serialization
                serializable_results = {
                    'connectivity_matrix': pairwise_results['connectivity_matrix'].tolist(),
                    'checkpoint_paths': pairwise_results['checkpoint_paths']
                }
                json.dump(serializable_results, f, indent=2)
            
            logging.info(f"Pairwise results saved to: {pairwise_file}")
            
            # Print connectivity matrix
            print("\nPairwise Connectivity Matrix:")
            matrix = pairwise_results['connectivity_matrix']
            n_models = len(checkpoint_paths)
            
            # Print header
            print("     ", end="")
            for i in range(n_models):
                print(f"{i:8d}", end="")
            print()
            
            # Print matrix
            for i in range(n_models):
                print(f"{i:3d}: ", end="")
                for j in range(n_models):
                    if i == j:
                        print("    -   ", end="")
                    else:
                        print(f"{matrix[i, j]:8.4f}", end="")
                print()
            
            return
        
        # Single pair analysis
        logging.info(f"Analyzing connectivity between:")
        logging.info(f"  Model A: {args.checkpoint_a}")
        logging.info(f"  Model B: {args.checkpoint_b}")
        
        # Run analysis
        if args.comprehensive:
            logging.info("Running comprehensive mode connectivity analysis...")
            # Warn user about computational cost
            if max_batches is None or max_batches > 20:
                logging.warning(f"Comprehensive analysis with {max_batches if max_batches else 'all'} batches may take significant time.")
            
            results = mode_calc.analyze_connectivity_properties(data_loader, max_batches=max_batches)
            
            # Save results for each method
            for method, method_results in results.items():
                method_results_file = os.path.join(output_dir, f"mode_connectivity_{method}_results.txt")
                with open(method_results_file, 'w') as f:
                    f.write("MODE CONNECTIVITY ANALYSIS RESULTS\n")
                    f.write("="*60 + "\n")
                    f.write(f"Checkpoint A: {args.checkpoint_a}\n")
                    f.write(f"Checkpoint B: {args.checkpoint_b}\n")
                    f.write(f"Method: {method}\n")
                    f.write(f"Curve Type: {method_results.get('curve_type', 'N/A')}\n")
                    f.write(f"Data type: {'validation' if args.use_validation else 'training'}\n")
                    f.write(f"Batches processed: {max_batches if max_batches else 'all'}\n")
                    f.write("-"*60 + "\n")
                    
                    for key, value in method_results.items():
                        if isinstance(value, (float, np.floating)):
                            f.write(f"{key}: {value:.8f}\n")
                        elif not isinstance(value, np.ndarray):
                            f.write(f"{key}: {value}\n")
                
                # Plot if requested
                if args.save_plots:
                    plots_dir = os.path.join(output_dir, 'plots')
                    os.makedirs(plots_dir, exist_ok=True)
                    plot_path = os.path.join(plots_dir, f'{method}_landscape.png')
                    fig = mode_calc.plot_loss_landscape(method_results, plot_path)
                    if fig:
                        import matplotlib.pyplot as plt
                        plt.close(fig)
        
        else:
            logging.info(f"Running mode connectivity analysis with {args.curve_type} curve...")
            results = mode_calc.compute_mode_connectivity(
                data_loader,
                curve_type=args.curve_type,
                optimize_control_point=args.optimize_control_point,
                num_curve_points=args.num_curve_points,
                max_batches=max_batches
            )
            
            # Save results
            with open(results_file, 'w') as f:
                f.write("MODE CONNECTIVITY ANALYSIS RESULTS\n")
                f.write("="*60 + "\n")
                f.write(f"Checkpoint A: {args.checkpoint_a}\n")
                f.write(f"Checkpoint B: {args.checkpoint_b}\n")
                f.write(f"Curve Type: {results.get('curve_type', 'N/A')}\n")
                f.write(f"Data type: {'validation' if args.use_validation else 'training'}\n")
                f.write(f"Batches processed: {max_batches if max_batches else 'all'}\n")
                f.write(f"Curve points evaluated: {args.num_curve_points}\n")
                if args.optimize_control_point:
                    f.write(f"Control point optimization: enabled\n")
                f.write("-"*60 + "\n")
                
                for key, value in results.items():
                    if isinstance(value, (float, np.floating)):
                        f.write(f"{key}: {value:.8f}\n")
                    elif not isinstance(value, np.ndarray):
                        f.write(f"{key}: {value}\n")
            
            # Plot if requested
            if args.save_plots:
                plots_dir = os.path.join(output_dir, 'plots')
                os.makedirs(plots_dir, exist_ok=True)
                plot_path = os.path.join(plots_dir, f'{args.curve_type}_landscape.png')
                fig = mode_calc.plot_loss_landscape(results, plot_path)
                if fig:
                    import matplotlib.pyplot as plt
                    plt.close(fig)
        
        computation_time = time.time() - start_time
        logging.info(f"Mode connectivity computation completed in {computation_time:.2f} seconds")
        
        # Display results
        print("\n" + "="*60)
        print("MODE CONNECTIVITY ANALYSIS RESULTS")
        print("="*60)
        print(f"Curve type: {args.curve_type}")
        print(f"Computation time: {computation_time:.2f} seconds")
        print("-"*60)
        
        if not args.comprehensive:
            mc = results['mode_connectivity']
            print(f"Mode Connectivity: {mc:.8f}")
            print(f"Loss A: {results['loss_a']:.8f}")
            print(f"Loss B: {results['loss_b']:.8f}")
            print(f"Optimal t*: {results['optimal_t']:.3f}")
            print(f"Loss at t*: {results['optimal_loss']:.8f}")
            print(f"Parameter distance: {results['parameter_distance']:.8f}")
            
            # Interpretation
            print("\nInterpretation:")
            if mc < -0.001:
                print("❌ POOR CONNECTIVITY (mc < 0)")
                print("   Loss barrier exists between models")
            elif mc > 0.001:
                print("⚠️  SUSPICIOUS CONNECTIVITY (mc > 0)")
                print("   Lower loss regions exist between models")
            else:
                print("✅ GOOD CONNECTIVITY (mc ≈ 0)")
                print("   Models are well-connected")
        
        logging.info(f"Results saved to: {results_file}")
    
    except Exception as e:
        logging.error(f"Error during analysis: {str(e)}")
        raise
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETED")
    print("="*60)

if __name__ == "__main__":
    main()
