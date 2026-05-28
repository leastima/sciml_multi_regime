#!/usr/bin/env python3
"""
CKA (Centered Kernel Alignment) Similarity Analysis Script

This script calculates CKA similarity between different neural network checkpoints
to assess the similarity of learned representations across different models.

CKA formula:
cka = HSIC(K, L) / sqrt(HSIC(K, K) * HSIC(L, L))

Where:
- HSIC is the Hilbert-Schmidt Independence Criterion
- K and L are the Gram matrices of representations X and Y respectively

Interpretation:
- CKA > 0.9: Very high similarity (nearly identical representations)
- CKA 0.7-0.9: High similarity (very similar representations)
- CKA 0.5-0.7: Moderate similarity
- CKA 0.3-0.5: Low similarity (quite different representations)
- CKA < 0.3: Very low similarity (very different representations)
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
from utils.trainer import CKACalculator
from utils.data_utils import get_data_loader
from utils.domains import DomainXY

def main():
    # parsers
    parser = argparse.ArgumentParser(description='Calculate CKA Similarity between neural network models')
    parser.add_argument("--yaml_config", default='./config/operators_poisson_1M.yaml', type=str, help='Path to YAML config file')
    parser.add_argument("--config", default='default', type=str, help='Config name from YAML file')
    parser.add_argument("--checkpoint_a", required=True, type=str, help='Path to first model checkpoint (.tar file)')
    parser.add_argument("--checkpoint_b", required=True, type=str, help='Path to second model checkpoint (.tar file)')
    parser.add_argument("--root_dir", default='./', type=str, help='Root directory to store results')
    parser.add_argument("--run_num", default='0', type=str, help='Sub run config identifier')
    
    # CKA computation parameters
    parser.add_argument("--kernel", default='linear', type=str, 
                       choices=['linear', 'rbf'],
                       help='Kernel type for CKA computation')
    parser.add_argument("--sigma", default=1.0, type=float,
                       help='RBF kernel bandwidth (only used for RBF kernel)')
    parser.add_argument("--max_batches", default=5, type=int, 
                       help='Maximum number of batches to process (use -1 for all batches)')
    parser.add_argument("--use_validation", action='store_true', 
                       help='Use validation data instead of training data')
    parser.add_argument("--comprehensive", action='store_true',
                       help='Run comprehensive analysis with multiple kernels')
    
    # Advanced options
    parser.add_argument("--pairwise", nargs='+', metavar='CHECKPOINT',
                       help='Compute pairwise CKA between multiple checkpoints')
    parser.add_argument("--save_plots", action='store_true',
                       help='Save CKA heatmap plots')
    parser.add_argument("--layer_analysis", action='store_true',
                       help='Perform layer-wise CKA analysis')
    parser.add_argument("--batch_size", default=128, type=int, help='Batch size for measuring CKA')
    parser.add_argument("--seed", default=0, type=int, help='Random seed')

    # Model parameters (override config if needed)
    parser.add_argument("--target_batch_size", default=128, type=int, help='Batch size for used in training, not for measuring CKA')
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
    logging.info(f"Using kernel: {args.kernel}")
    if args.kernel == 'rbf':
        logging.info(f"RBF sigma: {args.sigma}")
    
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

    # Initialize CKACalculator
    logging.info("Initializing CKACalculator...")
    cka_calc = CKACalculator(params)
    
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
    # Format: cka_analysis/expts_eps{epochs}/{config}/cka_similarity/bsz{batch_size}_lr{lr}_subsample{subsample}/seed{seed}/
    checkpoint_a_name = Path(args.checkpoint_a).stem.replace('ckpt', '').replace('_best', '').strip('_')
    checkpoint_b_name = Path(args.checkpoint_b).stem.replace('ckpt', '').replace('_best', '').strip('_')
    cka_type = f"{checkpoint_a_name}_vs_{checkpoint_b_name}_{args.kernel}"
    
    output_dir = os.path.join(
        args.root_dir, 
        'cka_analysis',
        f'expts_eps{epochs}',
        args.config,
        cka_type, 
        f'bsz{params.target_batch_size}_lr{lr}_subsample{params.subsample}',
        f'seed{args.seed_a}_vs_{args.seed_b}'
    )
    os.makedirs(output_dir, exist_ok=True)
    
    logging.info(f"Output directory: {output_dir}")
    
    # Check if output directory already exists and has results
    results_file = os.path.join(output_dir, f"cka_{args.kernel}_results.txt")
    if os.path.exists(results_file):
        logging.info(f"Results file already exists: {results_file}")
        logging.info("Skipping computation.")
        sys.exit(0)
    
    # Load models
    logging.info("Loading model checkpoints...")
    cka_calc.load_models(args.checkpoint_a, args.checkpoint_b)
    
    # Calculate CKA Similarity properties
    logging.info("Starting CKA similarity computation...")
    start_time = time.time()
    
    # Set max_batches to None if user wants all batches
    max_batches = args.max_batches if args.max_batches != -1 else None
    
    try:
        # Check if doing pairwise analysis
        if args.pairwise:
            logging.info("Running pairwise CKA similarity analysis...")
            checkpoint_paths = args.pairwise
            
            # Compute pairwise CKA
            pairwise_results = cka_calc.compute_pairwise_cka(
                checkpoint_paths, data_loader, kernel=args.kernel, 
                sigma=args.sigma, max_batches=max_batches)
            
            # Save pairwise results
            import json
            pairwise_file = os.path.join(output_dir, f'pairwise_cka_results.json')
            with open(pairwise_file, 'w') as f:
                # Convert numpy arrays for JSON serialization
                serializable_results = {
                    'cka_matrix': pairwise_results['cka_matrix'].tolist(),
                    'checkpoint_paths': pairwise_results['checkpoint_paths']
                }
                json.dump(serializable_results, f, indent=2)
            
            logging.info(f"Pairwise results saved to: {pairwise_file}")
            
            # Print CKA matrix
            print("\nPairwise CKA Similarity Matrix:")
            matrix = pairwise_results['cka_matrix']
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
                        print("  1.000 ", end="")
                    else:
                        print(f"{matrix[i, j]:8.4f}", end="")
                print()
            
            # Save heatmap if requested
            if args.save_plots:
                plots_dir = os.path.join(output_dir, 'plots')
                os.makedirs(plots_dir, exist_ok=True)
                plot_path = os.path.join(plots_dir, f'pairwise_cka_heatmap.png')
                
                # Create checkpoint names
                checkpoint_names = [f"Model_{i}" for i in range(n_models)]
                fig = cka_calc.plot_cka_heatmap(matrix, checkpoint_names, plot_path)
                if fig:
                    import matplotlib.pyplot as plt
                    plt.close(fig)
            
            return
        
        # Single pair analysis
        logging.info(f"Analyzing CKA similarity between:")
        logging.info(f"  Model A: {args.checkpoint_a}")
        logging.info(f"  Model B: {args.checkpoint_b}")
        
        # Run analysis
        if args.comprehensive:
            logging.info("Running comprehensive CKA similarity analysis...")
            # Warn user about computational cost
            if max_batches is None or max_batches > 20:
                logging.warning(f"Comprehensive analysis with {max_batches if max_batches else 'all'} batches may take significant time.")
            
            results = cka_calc.analyze_cka_properties(data_loader, max_batches=max_batches)
            
            # Save results for each method
            for method, method_results in results.items():
                method_results_file = os.path.join(output_dir, f"cka_{method}_results.txt")
                with open(method_results_file, 'w') as f:
                    f.write("CKA SIMILARITY ANALYSIS RESULTS\n")
                    f.write("="*60 + "\n")
                    f.write(f"Checkpoint A: {args.checkpoint_a}\n")
                    f.write(f"Checkpoint B: {args.checkpoint_b}\n")
                    f.write(f"Method: {method}\n")
                    f.write(f"Data type: {'validation' if args.use_validation else 'training'}\n")
                    f.write(f"Batches processed: {max_batches if max_batches else 'all'}\n")
                    f.write("-"*60 + "\n")
                    
                    if method == 'linear':
                        for key, value in method_results.items():
                            if isinstance(value, (float, np.floating)):
                                f.write(f"{key}: {value:.8f}\n")
                            elif not isinstance(value, (list, np.ndarray)):
                                f.write(f"{key}: {value}\n")
                    else:  # RBF results
                        f.write("RBF Kernel Results:\n")
                        for sigma_key, sigma_results in method_results.items():
                            f.write(f"\n{sigma_key}:\n")
                            for key, value in sigma_results.items():
                                if isinstance(value, (float, np.floating)):
                                    f.write(f"  {key}: {value:.8f}\n")
                                elif not isinstance(value, (list, np.ndarray)):
                                    f.write(f"  {key}: {value}\n")
        
        # Layer-wise analysis if requested
        elif args.layer_analysis:
            logging.info("Running layer-wise CKA analysis...")
            results = cka_calc.compute_layerwise_cka(data_loader, max_batches=max_batches)
            
            # Save layer-wise results
            layerwise_file = os.path.join(output_dir, f"cka_layerwise_results.txt")
            with open(layerwise_file, 'w') as f:
                f.write("LAYER-WISE CKA SIMILARITY ANALYSIS RESULTS\n")
                f.write("="*60 + "\n")
                f.write(f"Checkpoint A: {args.checkpoint_a}\n")
                f.write(f"Checkpoint B: {args.checkpoint_b}\n")
                f.write(f"Data type: {'validation' if args.use_validation else 'training'}\n")
                f.write(f"Batches processed: {max_batches if max_batches else 'all'}\n")
                f.write("-"*60 + "\n")
                
                for layer_name, layer_results in results.items():
                    f.write(f"\nLayer: {layer_name}\n")
                    for key, value in layer_results.items():
                        if isinstance(value, (float, np.floating)):
                            f.write(f"  {key}: {value:.8f}\n")
                        elif not isinstance(value, (list, np.ndarray)):
                            f.write(f"  {key}: {value}\n")
        
        else:
            # Standard single kernel analysis
            logging.info(f"Running CKA similarity analysis with {args.kernel} kernel...")
            results = cka_calc.compute_cka_between_models(
                data_loader,
                kernel=args.kernel,
                sigma=args.sigma,
                max_batches=max_batches
            )
            
            # Save results
            with open(results_file, 'w') as f:
                f.write("CKA SIMILARITY ANALYSIS RESULTS\n")
                f.write("="*60 + "\n")
                f.write(f"Checkpoint A: {args.checkpoint_a}\n")
                f.write(f"Checkpoint B: {args.checkpoint_b}\n")
                f.write(f"Kernel Type: {results.get('kernel_type', 'N/A')}\n")
                if args.kernel == 'rbf':
                    f.write(f"RBF Sigma: {results.get('sigma', 'N/A')}\n")
                f.write(f"Data type: {'validation' if args.use_validation else 'training'}\n")
                f.write(f"Batches processed: {max_batches if max_batches else 'all'}\n")
                f.write("-"*60 + "\n")
                
                for key, value in results.items():
                    if isinstance(value, (float, np.floating)):
                        f.write(f"{key}: {value:.8f}\n")
                    elif not isinstance(value, (list, np.ndarray)):
                        f.write(f"{key}: {value}\n")
        
        computation_time = time.time() - start_time
        logging.info(f"CKA similarity computation completed in {computation_time:.2f} seconds")
        
        # Display results
        print("\n" + "="*60)
        print("CKA SIMILARITY ANALYSIS RESULTS")
        print("="*60)
        print(f"Kernel type: {args.kernel}")
        if args.kernel == 'rbf':
            print(f"RBF sigma: {args.sigma}")
        print(f"Computation time: {computation_time:.2f} seconds")
        print("-"*60)
        
        if not args.comprehensive and not args.layer_analysis:
            cka_score = results['cka_score']
            print(f"CKA Similarity: {cka_score:.8f}")
            print(f"Number of samples: {results['num_samples']}")
            print(f"Representation A shape: {results['repr_a_shape']}")
            print(f"Representation B shape: {results['repr_b_shape']}")
            print(f"Mean cosine similarity: {results['mean_cosine_similarity']:.8f}")
            
            # Interpretation
            print("\nInterpretation:")
            if cka_score > 0.9:
                print("🔍 VERY HIGH CKA (>0.9)")
                print("   Representations are nearly identical")
            elif cka_score > 0.7:
                print("✅ HIGH CKA (0.7-0.9)")
                print("   Representations are very similar")
            elif cka_score > 0.5:
                print("⚠️  MODERATE CKA (0.5-0.7)")
                print("   Representations show some similarity")
            elif cka_score > 0.3:
                print("❌ LOW CKA (0.3-0.5)")
                print("   Representations are quite different")
            else:
                print("💥 VERY LOW CKA (<0.3)")
                print("   Representations are very different")
        
        logging.info(f"Results saved to: {results_file}")
    
    except Exception as e:
        logging.error(f"Error during analysis: {str(e)}")
        raise
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETED")
    print("="*60)

if __name__ == "__main__":
    main() 