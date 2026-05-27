import numpy as np
import matplotlib.pyplot as plt
import re
import os
from matplotlib.colors import Normalize, LogNorm
import matplotlib as mpl
from matplotlib.colors import TwoSlopeNorm

# Define parameters for RoPINN
betas = [5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 50, 70, 100, 150]
n_collocs = [10, 50, 100, 150, 200, 250, 500, 1000, 2000, 5000, 10000, 15000, 20000, 25000]
model = "PINN"  # As set in the parallel_sweep.sh script
seeds = [0, 1, 2]  # Updated seeds used in RoPINN sweep
log_dir = "/scratch/kenzhong/ropinn_runs"  # Updated log directory

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
test_errors_l1 = np.full((len(betas), len(n_collocs), len(seeds)), np.nan)
test_errors_l2 = np.full((len(betas), len(n_collocs), len(seeds)), np.nan)

# Parse logs for all seeds
for i, beta in enumerate(betas):
    for j, n_colloc in enumerate(n_collocs):
        for k, seed in enumerate(seeds):
            # Updated file pattern for RoPINN logs
            log_file = f"{log_dir}/conv_{model}_b{beta}_n{n_colloc}_s{seed}.out"
            if os.path.exists(log_file):
                with open(log_file, "r") as f:
                    content = f.read()
                    # Need to update these patterns to match RoPINN output format
                    # Extract training loss
                    loss_matches = re.findall(r"loss: ([\d.e+-]+)", content)
                    if loss_matches:
                        training_losses[i, j, k] = float(loss_matches[-1])
                    
                    # Extract L1 error - RoPINN reports this differently
                    l1_matches = re.findall(r"relative L1 error: ([\d.e+-]+)", content)
                    if l1_matches:
                        test_errors_l1[i, j, k] = float(l1_matches[-1])
                    
                    # Extract L2 error
                    l2_matches = re.findall(r"relative L2 error: ([\d.e+-]+)", content)
                    if l2_matches:
                        test_errors_l2[i, j, k] = float(l2_matches[-1])

# Calculate mean across seeds
mean_training_loss = np.nanmean(training_losses, axis=2)
mean_test_error_l1 = np.nanmean(test_errors_l1, axis=2)
mean_test_error_l2 = np.nanmean(test_errors_l2, axis=2)

# Print stats
print(f"Model: {model} (RoPINN)")
print(f"Found data for {np.count_nonzero(~np.isnan(mean_training_loss))}/{mean_training_loss.size} configurations")
print(f"L1 error data found for {np.count_nonzero(~np.isnan(mean_test_error_l1))}/{mean_test_error_l1.size} configurations")
print(f"L2 error data found for {np.count_nonzero(~np.isnan(mean_test_error_l2))}/{mean_test_error_l2.size} configurations")

# Create robust normalizations
norm_loss = get_robust_norm(mean_training_loss, log_scale=False)
norm_error_l1 = get_robust_norm(mean_test_error_l1, log_scale=False)
norm_error_l2 = get_robust_norm(mean_test_error_l2, log_scale=False)

# Print normalization ranges for reference
print(f"Training loss range: {norm_loss.vmin:.6f} to {norm_loss.vmax:.6f}")
print(f"L1 error range: {norm_error_l1.vmin:.6f} to {norm_error_l1.vmax:.6f}")
print(f"L2 error range: {norm_error_l2.vmin:.6f} to {norm_error_l2.vmax:.6f}")

# Create heatmap visualization - simplified to just show L1 and L2 errors
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# L1 Error heatmap
if np.count_nonzero(~np.isnan(mean_test_error_l1)) > 0:
    pcm1 = ax1.pcolormesh(
        range(len(betas)+1),
        range(len(n_collocs)+1),
        mean_test_error_l1.T,
        cmap='viridis_r',
        norm=norm_error_l1,
        edgecolors='w',
        linewidth=0.5
    )
    
    # Add values to each cell
    for i in range(len(betas)):
        for j in range(len(n_collocs)):
            if not np.isnan(mean_test_error_l1[i, j]):
                value = mean_test_error_l1[i, j]
                if value < 0.001:
                    text = f"{value:.2e}"
                else:
                    text = f"{value:.4f}"
                
                # Add indicator for values outside the robust range
                if value > norm_error_l1.vmax:
                    text = f"↑{text}"
                elif value < norm_error_l1.vmin:
                    text = f"↓{text}"
                    
                # Calculate normalized value for text color
                norm_value = np.clip((value - norm_error_l1.vmin) / (norm_error_l1.vmax - norm_error_l1.vmin), 0, 1)
                ax1.text(i + 0.5, j + 0.5, text, 
                        ha="center", va="center", fontsize=8,
                        color="white" if norm_value > 0.5 else "black")
    
    ax1.set_aspect('equal')
    ax1.set_xlabel('β (Convection Coefficient)', fontsize=12)
    ax1.set_ylabel('Number of Collocation Points', fontsize=12)
    ax1.set_title('RoPINN L1 Error (5-95 percentile scale)', fontsize=14)
    
    ax1.set_xticks(np.arange(0.5, len(betas), 1))
    ax1.set_yticks(np.arange(0.5, len(n_collocs), 1))
    ax1.set_xticklabels(betas)
    ax1.set_yticklabels(n_collocs)
    
    cbar = fig.colorbar(pcm1, ax=ax1, label='L1 Error')
    cbar.ax.text(0.5, 1.05, f'max: {np.nanmax(mean_test_error_l1):.3e}', 
                ha='center', va='bottom', transform=cbar.ax.transAxes, fontsize=8)
    
    # Display statistics
    textstr = f"Average L1 Error: {np.nanmean(mean_test_error_l1):.4f}\n"
    textstr += f"Based on {len(seeds)} seeds"
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
else:
    ax1.text(0.5, 0.5, "No valid L1 error data", 
             ha='center', va='center', transform=ax1.transAxes)

# L2 Error heatmap
if np.count_nonzero(~np.isnan(mean_test_error_l2)) > 0:
    pcm2 = ax2.pcolormesh(
        range(len(betas)+1),
        range(len(n_collocs)+1),
        mean_test_error_l2.T,
        cmap='viridis_r',
        norm=norm_error_l2,
        edgecolors='w',
        linewidth=0.5
    )
    
    # Add values to each cell
    for i in range(len(betas)):
        for j in range(len(n_collocs)):
            if not np.isnan(mean_test_error_l2[i, j]):
                value = mean_test_error_l2[i, j]
                if value < 0.001:
                    text = f"{value:.2e}"
                else:
                    text = f"{value:.4f}"
                
                # Add indicator for values outside the robust range
                if value > norm_error_l2.vmax:
                    text = f"↑{text}"
                elif value < norm_error_l2.vmin:
                    text = f"↓{text}"
                    
                # Calculate normalized value for text color
                norm_value = np.clip((value - norm_error_l2.vmin) / (norm_error_l2.vmax - norm_error_l2.vmin), 0, 1)
                ax2.text(i + 0.5, j + 0.5, text, 
                        ha="center", va="center", fontsize=8,
                        color="white" if norm_value > 0.5 else "black")
    
    ax2.set_aspect('equal')
    ax2.set_xlabel('β (Convection Coefficient)', fontsize=12)
    ax2.set_ylabel('Number of Collocation Points', fontsize=12)
    ax2.set_title('RoPINN L2 Error (5-95 percentile scale)', fontsize=14)
    
    ax2.set_xticks(np.arange(0.5, len(betas), 1))
    ax2.set_yticks(np.arange(0.5, len(n_collocs), 1))
    ax2.set_xticklabels(betas)
    ax2.set_yticklabels(n_collocs)
    
    cbar = fig.colorbar(pcm2, ax=ax2, label='L2 Error')
    cbar.ax.text(0.5, 1.05, f'max: {np.nanmax(mean_test_error_l2):.3e}', 
                ha='center', va='bottom', transform=cbar.ax.transAxes, fontsize=8)
    
    # Display statistics
    textstr = f"Average L2 Error: {np.nanmean(mean_test_error_l2):.4f}\n"
    textstr += f"Based on {len(seeds)} seeds"
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
else:
    ax2.text(0.5, 0.5, "No valid L2 error data", 
             ha='center', va='center', transform=ax2.transAxes)

plt.suptitle(f"RoPINN Performance on Convection Equation (5-95 percentile scale)", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(f'{log_dir}/ropinn_error_grid.png', dpi=300, bbox_inches='tight')
plt.close()

# Create line plots for various cross-sections
# Plot effect of beta for different n_colloc values
plt.figure(figsize=(10, 6))
for j, n_colloc in enumerate(n_collocs):
    if np.count_nonzero(~np.isnan(mean_test_error_l2[:, j])) > 0:
        plt.plot(betas, mean_test_error_l2[:, j], 'o-', label=f'n_colloc = {n_colloc}')
        
plt.xlabel('Beta (Convection Speed)', fontsize=14)
plt.ylabel('L2 Error', fontsize=14)
plt.title('Effect of Beta on L2 Error', fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(f'{log_dir}/ropinn_beta_effect.png', dpi=300)
plt.close()

# Plot effect of n_colloc for different beta values
plt.figure(figsize=(10, 6))
for i, beta in enumerate(betas):
    if np.count_nonzero(~np.isnan(mean_test_error_l2[i, :])) > 0:
        plt.plot(n_collocs, mean_test_error_l2[i, :], 'o-', label=f'beta = {beta}')
        
plt.xlabel('Number of Collocation Points', fontsize=14)
plt.ylabel('L2 Error', fontsize=14)
plt.title('Effect of Collocation Points on L2 Error', fontsize=16)
plt.grid(True, alpha=0.3)
plt.xscale('log')
plt.legend()
plt.tight_layout()
plt.savefig(f'{log_dir}/ropinn_colloc_effect.png', dpi=300)
plt.close()

print(f"Plotting completed. Results saved to {log_dir}")