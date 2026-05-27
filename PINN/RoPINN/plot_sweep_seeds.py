import os
import numpy as np
import matplotlib.pyplot as plt
import re
import glob
from matplotlib.colors import Normalize, LogNorm
import argparse
import matplotlib.ticker as mticker

ticks_fontsize = 24
label_fontsize = 32
default_figsize = (10, 7)

plt.rcParams['xtick.labelsize'] = ticks_fontsize
plt.rcParams['ytick.labelsize'] = ticks_fontsize
plt.rcParams['axes.labelsize'] = label_fontsize

# Define parameters - these should match those used in parallel_sweep.sh
betas = [5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 50, 70, 100, 150]
n_collocs = [10, 50, 100, 150, 200, 250, 500, 1000, 2000, 5000, 10000, 15000, 20000, 25000]
models = ["PINN"]  # We can also include "KAN", "QRes" later
seeds = [123]  # Seeds used in the sweep
initial_region = 1e-5  # Current sweep only used 1e-5
sample_num = 1  # Current sweep only used 1

# Get script directory for relative paths
script_dir = os.path.dirname(os.path.abspath(__file__))
log_dir = os.path.join(script_dir, "scratch", "ropinn_runs")
result_dir = os.path.join(script_dir, "results")

# Create results directory if it doesn't exist
os.makedirs(result_dir, exist_ok=True)

def parse_log_file(log_file):
    """Parse the RoPINN log file to extract training loss and test error."""
    training_loss = None
    test_error = None
    
    if not os.path.exists(log_file):
        return training_loss, test_error
    
    with open(log_file, 'r') as f:
        content = f.read()
        
        # Find final training loss - match "Train Loss: X.XXXXX" format
        train_loss_matches = re.findall(r"Train Loss: (\d+\.\d+e?[-+]?\d*)", content)
        if train_loss_matches:
            training_loss = float(train_loss_matches[-1])  # Take the last one if multiple matches
        
        # Find test error - using relative L2 error as the test metric
        test_error_matches = re.findall(r"relative L2 error: (\d+\.\d+e?[-+]?\d*)", content)
        if test_error_matches:
            test_error = float(test_error_matches[-1])  # Take the last one if multiple matches
            
    return training_loss, test_error

def get_robust_norm(data, log_scale=False, percentile_low=5, percentile_high=95):
    """
    Creates a robust normalization based on percentiles to handle outliers
    
    Args:
        data: numpy array with potentially NaN values
        log_scale: whether to use logarithmic scale
        percentile_low: lower percentile cutoff (default 5%)
        percentile_high: upper percentile cutoff (default 95%)
    
    Returns:
        norm: A matplotlib normalization object
    """
    valid_data = data[~np.isnan(data)]
    
    if len(valid_data) == 0:
        return Normalize(vmin=0, vmax=1)
    
    # Get percentile values
    vmin = max(0, np.nanpercentile(data, percentile_low))
    vmax = np.nanpercentile(data, percentile_high)
    
    # Ensure we have a reasonable range
    if vmax <= vmin:
        vmax = vmin * 1.1 + 0.01
    
    # Create appropriate normalization
    if log_scale and vmin > 0:
        return LogNorm(vmin=vmin, vmax=vmax)
    else:
        return Normalize(vmin=vmin, vmax=vmax)

def create_heatmap(data, norm, title, colorbar_label, filename):
    """Create a single heatmap without numbers inside cells."""
    if np.count_nonzero(~np.isnan(data)) == 0:
        print(f"Skipping {title.lower()}: no valid data.")
        return

    fig, ax = plt.subplots(figsize=default_figsize)
    ax.set_facecolor("#f0f0f0")

    pcm = ax.imshow(
        np.ma.masked_invalid(data).T,
        aspect="auto",
        cmap="viridis_r",
        origin="lower",
        norm=norm,
        interpolation="nearest"
    )

    ax.set_xlim(-0.5, len(betas) - 0.5)
    ax.set_ylim(-0.5, len(n_collocs) - 0.5)
    ax.set_xticks(np.arange(len(betas)))
    ax.set_xticklabels(betas)
    ax.set_yticks(np.arange(len(n_collocs)))
    ax.set_yticklabels(n_collocs)
    ax.set_xlabel("β (Convection Coefficient)")
    ax.set_ylabel("Collocation Points")
    ax.set_title(title, fontsize=label_fontsize)

    cbar = fig.colorbar(pcm)
    cbar.ax.tick_params(labelsize=ticks_fontsize)
    cbar.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
    cbar.set_label(colorbar_label, fontsize=label_fontsize)

    fig.tight_layout()
    plt.savefig(filename, format="pdf", bbox_inches="tight")
    plt.close(fig)

def create_heatmaps_for_model(model):
    """Create heatmaps for a specific model."""
    # Initialize arrays for all seeds
    training_losses = np.full((len(betas), len(n_collocs), len(seeds)), np.nan)
    test_errors = np.full((len(betas), len(n_collocs), len(seeds)), np.nan)
    
    # Parse logs for all seeds
    for i, beta in enumerate(betas):
        for j, n_colloc in enumerate(n_collocs):
            for k, seed in enumerate(seeds):
                # File pattern includes seed and other parameters
                # For RoPINN, there's no comparison - just using the model name directly
                log_file = f"{log_dir}/{model}_b{beta}_n{n_colloc}_s{seed}_ir1e-5_sn{sample_num}.out"
                
                # Debug output to check if files exist
                if not os.path.exists(log_file):
                    print(f"Warning: File not found: {log_file}")
                    continue
                    
                train_loss, test_err = parse_log_file(log_file)
                if train_loss is not None:
                    training_losses[i, j, k] = train_loss
                if test_err is not None:
                    test_errors[i, j, k] = test_err
    
    # Calculate mean across seeds
    mean_training_loss = np.nanmean(training_losses, axis=2)
    mean_test_error = np.nanmean(test_errors, axis=2)
    
    # Print stats
    print(f"Model: {model}")
    print(f"Found data for {np.count_nonzero(~np.isnan(mean_training_loss))}/{mean_training_loss.size} configurations")
    print(f"Test error data found for {np.count_nonzero(~np.isnan(mean_test_error))}/{mean_test_error.size} configurations")
    
    # Create robust normalizations
    norm_loss = get_robust_norm(mean_training_loss, log_scale=False)
    norm_error = get_robust_norm(mean_test_error, log_scale=False)
    
    # Print normalization ranges for reference
    print(f"Training loss range: {norm_loss.vmin:.6f} to {norm_loss.vmax:.6f}")
    print(f"Test error range: {norm_error.vmin:.6f} to {norm_error.vmax:.6f}")
    
    # Create individual heatmaps
    output_prefix = os.path.join(result_dir, f"{model}_grid_study")
    
    create_heatmap(
        data=mean_training_loss,
        norm=norm_loss,
        title="",
        colorbar_label="Training Loss",
        filename=f"{output_prefix}_training_loss.pdf"
    )
    
    create_heatmap(
        data=mean_test_error,
        norm=norm_error,
        title="",
        colorbar_label="Relative L2 Error",
        filename=f"{output_prefix}_test_error.pdf"
    )
    
    print(f"Saved heatmaps to {output_prefix}_*.pdf")
    
    return mean_training_loss, mean_test_error

def analyze_region_optimization():
    """
    Analyze and plot the optimization of initial regions across different settings.
    For RoPINN, this can show how the region optimization performed for different
    betas and number of collocation points.
    """
    print("\nAnalyzing region optimization is not implemented in this version.")
    # Placeholder for future implementation

def main():
    # Process all models and collect their data
    all_model_data = {}
    
    for model in models:
        print(f"\nProcessing model: {model}")
        train_loss, test_error = create_heatmaps_for_model(model)
        all_model_data[model] = {
            'train_loss': train_loss,
            'test_error': test_error
        }
    
    print(f"\nPlotting complete. Results saved to {result_dir}")

if __name__ == "__main__":
    main()