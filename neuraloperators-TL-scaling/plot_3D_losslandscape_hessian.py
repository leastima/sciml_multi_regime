import os, sys, time
import argparse
import torch
import numpy as np
import logging
from utils import logging_utils
logging_utils.config_logger()
from utils.YParams import YParams
from utils.trainer import LossLandscapeCalculator
from utils.data_utils import get_data_loader

def main():
    # parsers
    parser = argparse.ArgumentParser(description='Calculate 3D loss landscape using Hessian eigenvectors')
    parser.add_argument("--yaml_config", default='./config/operators_poisson_1M.yaml', type=str, help='Path to YAML config file')
    parser.add_argument("--config", default='default', type=str, help='Config name from YAML file')
    parser.add_argument("--checkpoint_path", required=True, type=str, help='Path to model checkpoint (.tar file)')
    parser.add_argument("--root_dir", default='./', type=str, help='Root directory to store results')

    # Loss landscape computation parameters
    parser.add_argument("--max_batches", default=4, type=int,
                       help='Maximum number of batches to process per point (use -1 for all batches)')
    parser.add_argument("--batch_size", default=128, type=int, help='Batch size for loss evaluation')

    # Landscape grid parameters
    parser.add_argument("--grid", default=25, type=int,
                       help='Grid resolution (number of points in each direction)')
    parser.add_argument("--radius", default=0.5, type=float,
                       help='Radius of exploration in parameter space')

    # Direction generation parameters
    parser.add_argument("--use_hessian_directions", action='store_true', default=True,
                       help='Use Hessian top-2 eigenvectors as directions')
    parser.add_argument("--log_scale", action='store_true', default=True,
                       help='Use log10 scale for loss values')

    # Plotting parameters
    parser.add_argument("--plot_3d", action='store_true', default=True,
                       help='Generate 3D surface plot')
    parser.add_argument("--plot_2d", action='store_true', default=True,
                       help='Generate 2D contour plot')
    parser.add_argument("--plot_interactive", action='store_true', default=True,
                       help='Generate interactive 3D plot (Plotly HTML)')
    parser.add_argument("--plot_combined", action='store_true', default=True,
                       help='Generate combined 2D+3D visualization')
    parser.add_argument("--plot_multiview", action='store_true', default=True,
                       help='Generate multi-angle view plot')
    parser.add_argument("--elevation", default=30, type=int, help='Elevation angle for 3D plot')
    parser.add_argument("--azimuth", default=135, type=int, help='Azimuth angle for 3D plot')
    parser.add_argument("--cmap", default='RdYlBu_r', type=str, help='Colormap for plots')

    # Model parameters (override config if needed)
    parser.add_argument("--target_batch_size", default=128, type=int, help='Batch size used in training')
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
    logging.info(f"\n\n\n Executing Hessian-based loss landscape script with the following arguments: {args} \n")

    if not os.path.exists(args.checkpoint_path):
        logging.error(f"Checkpoint file not found: {args.checkpoint_path}")
        sys.exit(1)

    logging.info(f"Loading checkpoint from: {args.checkpoint_path}")

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

    # Initialize LossLandscapeCalculator (needed for plotting methods)
    logging.info("Initializing LossLandscapeCalculator (Hessian-based)...")
    landscape_calc = LossLandscapeCalculator(params)

    # Prepare output directory
    lr = getattr(params, 'lr', 0.001)
    epochs = getattr(params, 'expt_max_epochs', 1000)
    # Encode radius in folder name for disambiguation
    radius_tag = f"r{args.radius:.3f}".rstrip('0').rstrip('.')
    
    if args.ckpt_epoch == -1:
        ckpt_epoch = ''
    else:
        ckpt_epoch = f'_ckpt_epoch{args.ckpt_epoch}'

    # Results directory - save numerical results
    results_dir = os.path.join(
        args.root_dir,
        'loss_landscape_hessian',
        f'expts_eps{epochs}',
        args.config,
        f'landscape_res{args.grid}_{radius_tag}',
        f'bsz{params.target_batch_size}_lr{lr}_subsample{params.subsample}',
        f'seed{params.target_seed}{ckpt_epoch}'
    )

    # Plots directory - save visualizations
    plots_dir = os.path.join(
        args.root_dir,
        'plots',
        'Losslandscape_Hessian',
        f'expts_eps{epochs}',
        args.config,
        f'landscape_res{args.grid}_{radius_tag}',
        f'bsz{params.target_batch_size}_lr{lr}_subsample{params.subsample}',
        f'seed{params.target_seed}{ckpt_epoch}'
    )

    # Create directories
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    logging.info(f"Results directory: {results_dir}")
    logging.info(f"Plots directory: {plots_dir}")

    # Check if results already exist
    npz_file = os.path.join(results_dir, "loss_landscape_results.npz")
    results_exist = os.path.exists(npz_file)

    if results_exist:
        logging.info(f"Results already exist at {npz_file}")
        logging.info(f"Loading existing results and regenerating plots...")

        # Load existing results
        data = np.load(npz_file)
        results = {
            'alphas': data['alphas'],
            'betas': data['betas'],
            'Z': data['Z'],
            'log_scale': bool(data['log_scale']),
            'radius': float(data['radius']),
            'grid_size': int(data['grid_size']),
            'use_hessian_directions': bool(data['use_hessian_directions'])
        }
        computation_time = float(data.get('computation_time', 0))

        used_radius = results['radius']

        # Display loaded results
        print("\n" + "="*60)
        print("LOADED HESSIAN-BASED LOSS LANDSCAPE RESULTS")
        print("="*60)
        print(f"Checkpoint: {args.checkpoint_path}")
        print(f"Grid size: {results['grid_size']}x{results['grid_size']}")
        print(f"Radius: {results['radius']}")
        print(f"Log10 scale: {results['log_scale']}")
        print(f"Use Hessian directions: {results['use_hessian_directions']}")
        print(f"Original computation time: {computation_time:.2f} seconds")
        print("-"*60)
        print(f"Loss statistics:")
        print(f"  Min: {results['Z'].min():.6f}")
        print(f"  Max: {results['Z'].max():.6f}")
        print(f"  Mean: {results['Z'].mean():.6f}")
        print(f"  Std: {results['Z'].std():.6f}")
        print("="*60)
        print("\nRegenerating plots with updated settings...")

    else:
        logging.info("No existing results found. Computing loss landscape...")

        # Load model checkpoint for computation
        logging.info("Loading model checkpoint...")
        landscape_calc.load_checkpoint(args.checkpoint_path)

        # Load data for computation
        logging.info("Loading data...")
        data_loader, dataset, sampler = get_data_loader(
            params, params.train_path, distributed=False, train=True, pack=params.pack_data
        )
        logging.info(f"Using training data: {len(dataset)} samples")

        # Calculate loss landscape
        logging.info("Starting loss landscape computation...")
        start_time = time.time()

        # Set max_batches to None if user wants all batches
        max_batches = None if args.max_batches == -1 else args.max_batches

        # Compute loss landscape
        results = landscape_calc.compute_loss_landscape_2d(
            data_loader=data_loader,
            checkpoint_path=None,  # Already loaded
            radius=args.radius,
            grid=args.grid,
            log_scale=args.log_scale,
            max_batches=max_batches,
            use_hessian_directions=args.use_hessian_directions
        )
        used_radius = results['radius']

        computation_time = time.time() - start_time
        logging.info(f"Loss landscape computation completed in {computation_time:.2f} seconds")

        # Display results
        print("\n" + "="*60)
        print("HESSIAN-BASED LOSS LANDSCAPE RESULTS")
        print("="*60)
        print(f"Checkpoint: {args.checkpoint_path}")
        print(f"Grid size: {args.grid}x{args.grid}")
        print(f"Radius: {used_radius} (requested {args.radius})")
        print(f"Log10 scale: {args.log_scale}")
        print(f"Use Hessian directions: {args.use_hessian_directions}")
        print(f"Computation time: {computation_time:.2f} seconds")
        print(f"Total evaluations: {args.grid * args.grid}")
        print(f"Batches per evaluation: {max_batches if max_batches else 'all'}")
        print("-"*60)
        print(f"Loss statistics:")
        print(f"  Min: {results['Z'].min():.6f}")
        print(f"  Max: {results['Z'].max():.6f}")
        print(f"  Mean: {results['Z'].mean():.6f}")
        print(f"  Std: {results['Z'].std():.6f}")
        print("="*60)

        # Save results to file
        output_file = os.path.join(results_dir, "loss_landscape_results.txt")
        with open(output_file, 'w') as f:
            f.write("HESSIAN-BASED LOSS LANDSCAPE RESULTS\n")
            f.write("="*60 + "\n")
            f.write(f"Checkpoint: {args.checkpoint_path}\n")
            f.write(f"Grid size: {args.grid}x{args.grid}\n")
            f.write(f"Radius: {used_radius} (requested {args.radius})\n")
            f.write(f"Log10 scale: {args.log_scale}\n")
            f.write(f"Use Hessian directions: {args.use_hessian_directions}\n")
            f.write(f"Computation time: {computation_time:.2f} seconds\n")
            f.write(f"Total evaluations: {args.grid * args.grid}\n")
            f.write(f"Batches per evaluation: {max_batches if max_batches else 'all'}\n")
            f.write("-"*60 + "\n")
            f.write(f"Loss statistics:\n")
            f.write(f"  Min: {results['Z'].min():.8f}\n")
            f.write(f"  Max: {results['Z'].max():.8f}\n")
            f.write(f"  Mean: {results['Z'].mean():.8f}\n")
            f.write(f"  Std: {results['Z'].std():.8f}\n")

        logging.info(f"Results saved to: {output_file}")

        # Save numpy arrays for later analysis
        np.savez(
            npz_file,
            alphas=results['alphas'],
            betas=results['betas'],
            Z=results['Z'],
            log_scale=results['log_scale'],
            radius=results['radius'],
            grid_size=results['grid_size'],
            use_hessian_directions=results['use_hessian_directions'],
            computation_time=computation_time
        )
        logging.info(f"Numpy arrays saved to: {npz_file}")

    # Generate plots - save to plots directory
    if args.plot_2d:
        logging.info("Generating 2D contour plot...")
        plot_2d_path = os.path.join(plots_dir, "loss_landscape_2d.pdf")
        try:
            fig = landscape_calc.plot_loss_landscape_2d(
                results,
                save_path=plot_2d_path
            )
            if fig is not None:
                logging.info(f"2D plot saved to: {plot_2d_path}")
        except Exception as e:
            logging.warning(f"Could not generate 2D plot: {e}")

    if args.plot_3d:
        logging.info("Generating 3D surface plot...")
        plot_3d_path = os.path.join(plots_dir, "loss_landscape_3d.pdf")
        try:
            fig = landscape_calc.plot_loss_landscape_3d(
                results,
                save_path=plot_3d_path,
                elev=args.elevation,
                azim=args.azimuth,
                cmap=args.cmap
            )
            if fig is not None:
                logging.info(f"3D plot saved to: {plot_3d_path}")
                # Also save with different viewing angles
                plot_3d_path_alt = os.path.join(plots_dir, "loss_landscape_3d_alt_view.pdf")
                fig_alt = landscape_calc.plot_loss_landscape_3d(
                    results,
                    save_path=plot_3d_path_alt,
                    elev=60,
                    azim=45,
                    cmap=args.cmap
                )
        except Exception as e:
            logging.warning(f"Could not generate 3D plot: {e}")

    # NEW: Generate interactive 3D plot
    if args.plot_interactive:
        logging.info("Generating interactive 3D plot (Plotly)...")
        plot_interactive_path = os.path.join(plots_dir, "loss_landscape_3d_interactive.html")
        try:
            fig = landscape_calc.plot_loss_landscape_3d_interactive(
                results,
                save_path=plot_interactive_path,
                show=False  # Don't auto-open browser
            )
            if fig is not None:
                logging.info(f"Interactive 3D plot saved to: {plot_interactive_path}")
                logging.info(f"  👉 Open this file in a web browser to interact with the 3D visualization!")
        except Exception as e:
            logging.warning(f"Could not generate interactive 3D plot: {e}")
    
    # NEW: Generate combined visualization
    if args.plot_combined:
        logging.info("Generating combined 2D+3D visualization...")
        plot_combined_path = os.path.join(plots_dir, "loss_landscape_combined.pdf")
        try:
            fig = landscape_calc.plot_loss_landscape_combined(
                results,
                save_path=plot_combined_path
            )
            if fig is not None:
                logging.info(f"Combined plot saved to: {plot_combined_path}")
        except Exception as e:
            logging.warning(f"Could not generate combined plot: {e}")

    # NEW: Generate multi-view plot
    if args.plot_multiview:
        logging.info("Generating multi-angle view plot...")
        plot_multiview_path = os.path.join(plots_dir, "loss_landscape_multiview.pdf")
        # Include config and subsample in the View2 filename to disambiguate runs
        plot_view2_path = os.path.join(
            plots_dir,
            f"loss_landscape_view2_{args.config}_subsample{args.subsample}.pdf"
        )
        try:
            fig = landscape_calc.plot_loss_landscape_multiview(
                results,
                save_path=plot_multiview_path,
                single_view_path=plot_view2_path
            )
            if fig is not None:
                logging.info(f"Multi-view plot saved to: {plot_multiview_path}")
                logging.info(f"View 2 plot saved to: {plot_view2_path}")
        except Exception as e:
            logging.warning(f"Could not generate multi-view plot: {e}")

    logging.info("\nHessian-based loss landscape computation completed successfully!")

if __name__ == '__main__':
    main()
