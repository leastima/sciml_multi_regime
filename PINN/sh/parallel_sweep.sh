#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "$SCRIPT_DIR"

# Activate the appropriate virtual environment
# source /path/to/your/venv/bin/activate
# echo "Please ensure your virtual environment is activated."

# Clean up orphaned tracking files
echo "Cleaning up orphaned tracking files from previous runs..."
for tracking_file in /tmp/gpu_*_pid_*; do
  if [ -f "$tracking_file" ]; then
    pid=$(echo "$tracking_file" | grep -o "pid_[0-9]*" | sed 's/pid_//')
    if ! ps -p "$pid" >/dev/null; then
      echo "Removing orphaned tracking file: $tracking_file (PID $pid not running)"
      rm "$tracking_file"
    else
      echo "Found tracking file for running process: $tracking_file (PID $pid)"
    fi
  fi
done

# Parameters for the sweep
pdes=("convection")            # Focus on convection equation for beta parameter
optimizers=("lbfgs" "adam_lbfgs" "adam_lbfgs_nncg") # Options: lbfgs, adam_lbfgs, adam_lbfgs_nncg

seeds=(123 234 345)  # Multiple seeds for statistical significance
betas=(5 30)
n_collocs=(10000 20000)

# ============ Optimizer Parameter Defaults ============
# Standard values for consistency across all LBFGS-based optimizers
LBFGS_LR=1.0
LBFGS_HISTORY_SIZE=100
LBFGS_MAX_ITER=50
LBFGS_MAX_EVAL=75  # Typically 1.5x max_iter
ADAM_LR=1e-3
ADAM_SWITCH_EPOCH=5000  # For Adam+LBFGS and Adam+LBFGS+NNCG
LBFGS_SWITCH_EPOCH=10000  # For Adam+LBFGS+NNCG
NNCG_SWITCH_EPOCH=15000   # For Adam+LBFGS+NNCG
# seeds=(123 234 345) # Single seed for testing
# betas=(5 6 7 8 9 10 15 20 25 30 50 70 100 150)
# n_collocs=(10 50 100 150 200 250 500 1000 2000 5000 10000 15000 20000 25000)

num_layers=4
num_neurons=50
num_x=257
num_t=101
epochs=25000
wandb_project="pinn_multiadam_sweep"

# ==========================================
# SPECIFY WHICH GPUS TO USE HERE
available_gpus=(2 3 4 5 6 7)
# ==========================================

# Output directory
OUTPUT_DIR="./results_nncg"
# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

pids=()

# GPU tracking array
declare -A gpu_jobs
for gpu in "${available_gpus[@]}"; do # Assuming 8 GPUs available, adjust as needed
  gpu_jobs[$gpu]=0
done

# Function to check if GPU is free
is_gpu_free() {
  local gpu=$1

  # Simple allocation: check if we have reached the max jobs per GPU
  # You can adjust the limit (currently 1) if you want multiple jobs per GPU
  if [ "${gpu_jobs[$gpu]}" -ge 1 ]; then
    return 1
  fi

  return 0
}

# Function to launch a job on a GPU
launch_job() {
  local pde=$1
  local opt=$2
  local seed=$3
  local beta=$4
  local num_res=$5
  local gpu=$6

  # Set PDE parameters based on PDE type - JSON format
  local pde_params=""
  if [ "$pde" == "convection" ]; then
    pde_params="{\"beta\":$beta,\"diffusion_coefficient\":0.1}"
  elif [ "$pde" == "reaction" ]; then
    pde_params="{\"rho\":$beta}"
  elif [ "$pde" == "wave" ]; then
    pde_params="{\"beta\":$beta,\"c\":1.0}"
  fi

  # Determine optimizer parameters - JSON format
  # Using consistent LBFGS parameters across all variants for fair comparison
  local opt_params=""
  if [[ "$opt" == "adam_lbfgs_nncg" ]]; then
    opt_params="{\"switch_epochs\":$ADAM_SWITCH_EPOCH,\"adam_lr\":$ADAM_LR,\"lbfgs_lr\":$LBFGS_LR,\"lbfgs_history_size\":$LBFGS_HISTORY_SIZE,\"lbfgs_max_iter\":$LBFGS_MAX_ITER,\"switch_epoch_lbfgs\":$LBFGS_SWITCH_EPOCH,\"switch_epoch_nncg\":$NNCG_SWITCH_EPOCH,\"precond_update_freq\":100,\"nncg_rank\":50,\"nncg_mu\":0.01,\"nncg_cg_tol\":1e-5,\"nncg_use_double\":false,\"nncg_dynamic_damping\":true,\"nncg_verbose\":true}"
  elif [[ "$opt" == "adam_lbfgs" ]]; then
    opt_params="{\"switch_epochs\":$ADAM_SWITCH_EPOCH,\"adam_lr\":$ADAM_LR,\"lbfgs_lr\":$LBFGS_LR,\"lbfgs_history_size\":$LBFGS_HISTORY_SIZE,\"lbfgs_max_iter\":$LBFGS_MAX_ITER}"
  elif [[ "$opt" == "lbfgs" ]]; then
    opt_params="{\"lr\":$LBFGS_LR,\"history_size\":$LBFGS_HISTORY_SIZE,\"max_iter\":$LBFGS_MAX_ITER,\"max_eval\":$LBFGS_MAX_EVAL}"
  elif [[ "$opt" == "adam" ]]; then
    opt_params="{\"lr\":$ADAM_LR,\"weight_decay\":0.0}"
  elif [[ "$opt" == "multiadam" ]]; then
    opt_params="{\"lr\":$ADAM_LR,\"betas\":[0.99,0.99],\"loss_group_idx\":[1,2]}"
  elif [[ "$opt" == *"gd"* ]]; then
    opt_params="{\"gd_lr\":0.01,\"gd_max_iter\":10000}"
  fi

  echo "Starting pde=$pde, opt=$opt, seed=$seed, beta=$beta, num_res=$num_res on GPU $gpu"

  # Ensure output directories exist
  mkdir -p "${OUTPUT_DIR}/${pde}/models"
  mkdir -p "${OUTPUT_DIR}/${pde}/results"

  # Launch experiment - use eval to properly handle quoted JSON strings
  eval "WANDB_MODE=offline CUDA_VISIBLE_DEVICES=$gpu python run_experiment.py \
    --initial_seed $seed \
    --pde $pde \
    --pde_params '$pde_params' \
    --opt $opt \
    --opt_params '$opt_params' \
    --num_layers $num_layers \
    --num_neurons $num_neurons \
    --loss mse \
    --num_x $num_x \
    --num_t $num_t \
    --num_res $num_res \
    --epochs $epochs \
    --wandb_project $wandb_project \
    --device 0 \
    --save_path ${OUTPUT_DIR}/${pde}/models \
    --save_model \
    --new_data \
    --set_idx 0 \
    > ${OUTPUT_DIR}/${pde}/results/${pde}_${opt}_b${beta}_s${seed}_r${num_res}.out 2>&1 &"

  # Store PID and increment GPU counter
  local pid=$!
  pids+=($pid)
  gpu_jobs[$gpu]=$((gpu_jobs[$gpu] + 1))

  # Create tracking file
  touch "/tmp/gpu_${gpu}_pid_${pid}"

  echo "Launched pde=$pde, opt=$opt, seed=$seed, beta=$beta, num_res=$num_res on GPU $gpu (PID $pid)"
  echo "GPU $gpu now has ${gpu_jobs[$gpu]} jobs"
}

# Function to clean up and update job tracking
cleanup_jobs() {
  # Get original PIDs
  local old_pids=("${pids[@]}")

  # Update PID list with only running processes
  pids=()
  for pid in "${old_pids[@]}"; do
    if kill -0 $pid 2>/dev/null; then
      pids+=($pid)
    else
      # Track finished jobs by GPU
      for gpu in "${available_gpus[@]}"; do
        if [ -f "/tmp/gpu_${gpu}_pid_${pid}" ]; then
          gpu_jobs[$gpu]=$((gpu_jobs[$gpu] - 1))
          echo "Process $pid on GPU $gpu finished. Count now ${gpu_jobs[$gpu]}"
          rm "/tmp/gpu_${gpu}_pid_${pid}"
        fi
      done
    fi
  done

  # Verify GPU counts against actual running processes
  verify_gpu_counts
}

# Function to verify and correct GPU counts
verify_gpu_counts() {
  for gpu in "${available_gpus[@]}"; do
    local actual_count=$(ls -1 /tmp/gpu_${gpu}_pid_* 2>/dev/null | wc -l)

    if [ "$actual_count" != "${gpu_jobs[$gpu]}" ]; then
      echo "WARNING: GPU $gpu count mismatch! Counter: ${gpu_jobs[$gpu]}, Files: $actual_count"
      gpu_jobs[$gpu]=$actual_count
    fi
  done
}

# Build a list of all configurations
configs=()
for pde in "${pdes[@]}"; do
  for opt in "${optimizers[@]}"; do
    for seed in "${seeds[@]}"; do
      for beta in "${betas[@]}"; do
        for num_res in "${n_collocs[@]}"; do
          configs+=("$pde $opt $seed $beta $num_res")
        done
      done
    done
  done
done

# Initial cleanup
cleanup_jobs

# Count configurations
total_configs=${#configs[@]}
completed_configs=0
echo "Total configurations to run: $total_configs"

# Scan GPUs and report status
echo "Initial GPU status:"
for gpu in "${available_gpus[@]}"; do
  if is_gpu_free $gpu; then
    echo "GPU $gpu is available"
  else
    echo "GPU $gpu is NOT available"
  fi
done

# Allocate jobs to GPUs
echo "Starting job allocation - one job per GPU..."
while [ $completed_configs -lt $total_configs ]; do
  # Clean up before allocation
  cleanup_jobs

  # Try to allocate jobs to free GPUs
  made_progress=false

  for gpu in "${available_gpus[@]}"; do
    # Skip if all configs completed
    if [ $completed_configs -ge $total_configs ]; then
      break
    fi

    # Check if GPU is free
    if is_gpu_free $gpu; then
      IFS=' ' read -ra config <<<"${configs[$completed_configs]}"
      launch_job "${config[0]}" "${config[1]}" "${config[2]}" "${config[3]}" "${config[4]}" "$gpu"
      completed_configs=$((completed_configs + 1))
      made_progress=true
    fi
  done

  # Wait if no progress made
  if [ "$made_progress" = false ]; then
    echo "Waiting for GPUs to free up... ($completed_configs/$total_configs complete)"
    sleep 60
  fi
done

# Wait for final jobs to complete
echo "All jobs allocated! Waiting for ${#pids[@]} jobs to finish..."
while [ ${#pids[@]} -gt 0 ]; do
  sleep 30
  cleanup_jobs
  echo "Still waiting for ${#pids[@]} jobs to finish..."
done

echo "All jobs completed successfully."

