'''This script is for visualizing one optimizer only, without comparing to others.'''
import numpy as np
import matplotlib.pyplot as plt
import re
import os
from matplotlib.colors import Normalize, LogNorm

# Define parameters
betas = [5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 50, 70, 100, 150]
n_collocs = [10, 50, 100, 150, 200, 250, 500, 1000, 2000, 5000, 10000, 15000, 20000, 25000]
method = "multiadam"  # Only MultiAdam
seeds = [123]  # Seeds used in the sweep
# seeds = [123, 234, 345, 456, 567]  # Seeds used in the sweep
log_dir = "/scratch/kenzhong/pinnacle/convection/multiadam/results"

# Helper function for robust normalization
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

# Initialize arrays for all seeds
training_losses = np.full((len(betas), len(n_collocs), len(seeds)), np.nan)
test_losses = np.full((len(betas), len(n_collocs), len(seeds)), np.nan)
test_errors = np.full((len(betas), len(n_collocs), len(seeds)), np.nan)

# Parse logs for all seeds
for i, beta in enumerate(betas):
    for j, n_colloc in enumerate(n_collocs):
        for k, seed in enumerate(seeds):
            # File pattern includes seed
            log_file = f"{log_dir}/{method}_b{beta}_n{n_colloc}_s{seed}.out"
            if os.path.exists(log_file):
                with open(log_file, "r") as f:
                    content = f.read()
                    
                    # Parse the "Best model at step X:" section
                    best_model_pattern = r"Best model at step \d+:\s+train loss: ([\d.e+-]+)\s+test loss: ([\d.e+-]+)\s+test metric: \[([\d.e+-]+)\]"
                    matches = re.findall(best_model_pattern, content)
                    
                    if matches:
                        # Get the last match (most recent best model)
                        train_loss, test_loss, test_metric = matches[-1]
                        training_losses[i, j, k] = float(train_loss)
                        test_losses[i, j, k] = float(test_loss)
                        test_errors[i, j, k] = float(test_metric)

# Calculate mean across seeds
mean_training_loss = np.nanmean(training_losses, axis=2)
mean_test_loss = np.nanmean(test_losses, axis=2)
mean_test_error = np.nanmean(test_errors, axis=2)

# Print stats
print(f"Method: {method}")
print(f"Found training loss data for {np.count_nonzero(~np.isnan(mean_training_loss))}/{mean_training_loss.size} configurations")
print(f"Found test loss data for {np.count_nonzero(~np.isnan(mean_test_loss))}/{mean_test_loss.size} configurations")
print(f"Found test error data for {np.count_nonzero(~np.isnan(mean_test_error))}/{mean_test_error.size} configurations")

# Create robust normalizations
norm_train_loss = get_robust_norm(mean_training_loss, log_scale=False)
norm_test_loss = get_robust_norm(mean_test_loss, log_scale=False)
norm_test_error = get_robust_norm(mean_test_error, log_scale=False)

# Print normalization ranges for reference
print(f"Training loss range: {norm_train_loss.vmin:.6f} to {norm_train_loss.vmax:.6f}")
print(f"Test loss range: {norm_test_loss.vmin:.6f} to {norm_test_loss.vmax:.6f}")
print(f"Test error range: {norm_test_error.vmin:.6f} to {norm_test_error.vmax:.6f}")

# Create heatmap visualization with three subplots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 7))

def create_heatmap(ax, data, norm, title, label, values_title):
    """Helper function to create consistent heatmaps"""
    if np.count_nonzero(~np.isnan(data)) > 0:
        pcm = ax.pcolormesh(
            range(len(betas)+1),
            range(len(n_collocs)+1),
            data.T,
            cmap='viridis_r',
            norm=norm,
            edgecolors='w',
            linewidth=0.5
        )
        
        # Add values to each cell
        for i in range(len(betas)):
            for j in range(len(n_collocs)):
                if not np.isnan(data[i, j]):
                    value = data[i, j]
                    # Format based on value magnitude for better readability
                    if value < 0.001:
                        text = f"{value:.2e}"
                    else:
                        text = f"{value:.4f}"
                    
                    # Add indicator for values outside the robust range
                    if value > norm.vmax:
                        text = f"↑{text}"
                    elif value < norm.vmin:
                        text = f"↓{text}"
                        
                    # Calculate normalized value for text color
                    norm_value = np.clip((value - norm.vmin) / (norm.vmax - norm.vmin), 0, 1)
                    ax.text(i + 0.5, j + 0.5, text, 
                            ha="center", va="center", fontsize=8,
                            color="white" if norm_value > 0.5 else "black")
        
        # Set square aspect ratio and formatting
        ax.set_aspect('equal')
        ax.set_xlabel('β (Convection Coefficient)', fontsize=12)
        ax.set_ylabel('Number of Collocation Points', fontsize=12)
        ax.set_title(title, fontsize=14)
        
        # Set proper tick positions and labels
        ax.set_xticks(np.arange(0.5, len(betas), 1))
        ax.set_yticks(np.arange(0.5, len(n_collocs), 1))
        ax.set_xticklabels(betas)
        ax.set_yticklabels(n_collocs)
        
        cbar = fig.colorbar(pcm, ax=ax, label=label)
        cbar.ax.text(0.5, 1.05, f'max: {np.nanmax(data):.3e}', 
                    ha='center', va='bottom', transform=cbar.ax.transAxes, fontsize=8)
        
        # Display statistics on the plot
        textstr = f"Average {values_title}: {np.nanmean(data):.4f}\n"
        textstr += f"Based on {len(seeds)} seeds"
        
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
    else:
        ax.text(0.5, 0.5, f"No valid {values_title.lower()} data", 
                ha='center', va='center', transform=ax.transAxes)

# Create the three heatmaps
create_heatmap(ax1, mean_training_loss, norm_train_loss, 
               'MultiAdam Training Loss (5-95 percentile scale)', 
               'Training Loss', 'Training Loss')

create_heatmap(ax2, mean_test_loss, norm_test_loss,
               'MultiAdam Test Loss (5-95 percentile scale)',
               'Test Loss', 'Test Loss')

create_heatmap(ax3, mean_test_error, norm_test_error,
               'MultiAdam Test Error (5-95 percentile scale)',
               'Test Error', 'Test Error')

plt.tight_layout()
plt.savefig(f'/scratch/kenzhong/pinnacle/convection/lbfgs/multiadam_only_grid_study.png', dpi=300, bbox_inches='tight')
plt.show()
