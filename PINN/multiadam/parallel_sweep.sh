#!/bin/bash
# Parallel hyperparameter sweep script for PINN training
# This script distributes training jobs across multiple GPUs
#
# Usage: bash parallel_sweep.sh
# 
# Configuration:
#   - Modify the hyperparameter arrays below to customize the sweep
#   - Set GPUS array to specify which GPUs to use
#   - Adjust MAX_JOBS_PER_GPU to control GPU utilization
#   - Models and logs are saved to ./models and ./results directories

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ==================== CONFIGURATION ====================
# GPU Configuration - specify which GPUs to use
GPUS=(0) # Default: use only GPU 0. Example: GPUS=(0 1 2 3) to use 4 GPUs

# Maximum number of jobs per GPU (default: 2)
MAX_JOBS_PER_GPU=2

# Hyperparameter configurations
betas=(5 6 7 8 9 10 15 20 25 30 50 70 100 150)
n_collocs=(10 50 100 150 200 250 500 1000 2000 5000 10000 15000 20000 25000)
model_seeds=(123) # Different model initialization seeds for CKA analysis
data_seed=42      # Fixed data seed - all models use same training data

# Training method(s) to use
methods=("multiadam")

# Neural network architecture
hidden_layers="50*4"

# Learning rate (can be adjusted per method in launch_job function)
default_lr="1e-3"

# Training iterations
iterations=20000

# ==================== END CONFIGURATION ====================

# Set up relative paths for model and log storage
MODEL_SAVE_DIR="${SCRIPT_DIR}/models"
OUTPUT_LOG_DIR="${SCRIPT_DIR}/results"

mkdir -p "$MODEL_SAVE_DIR"
mkdir -p "$OUTPUT_LOG_DIR"

echo "Script directory: $SCRIPT_DIR"
echo "Models will be saved to: $MODEL_SAVE_DIR"
echo "Training logs will be saved to: $OUTPUT_LOG_DIR"

# Clean up orphaned tracking files at script startup
echo "Cleaning up orphaned tracking files from previous runs..."
TEMP_DIR="${TMPDIR:-/tmp}"
for tracking_file in "${TEMP_DIR}"/gpu_*_pid_*; do
  if [ -f "$tracking_file" ]; then
    # Extract PID from filename
    pid=$(echo "$tracking_file" | grep -o "pid_[0-9]*" | sed 's/pid_//')

    # Check if process is still running
    if ! ps -p "$pid" >/dev/null; then
      echo "Removing orphaned tracking file: $tracking_file (PID $pid not running)"
      rm "$tracking_file"
    else
      echo "Found tracking file for running process: $tracking_file (PID $pid)"
    fi
  fi
done

echo "Using GPUs: ${GPUS[@]}"
echo "Maximum jobs per GPU: $MAX_JOBS_PER_GPU"

pids=()

# Simple GPU tracking array
declare -A gpu_jobs
for gpu in "${GPUS[@]}"; do
  gpu_jobs[$gpu]=0
done

# Function to check if GPU is free for our use - ALLOWS 1 TASK ON designated GPUs ONLY
is_gpu_free() {
  local gpu=$1

  # Check if this GPU is in our allowed list
  local gpu_allowed=false
  for allowed_gpu in "${GPUS[@]}"; do
    if [ "$gpu" -eq "$allowed_gpu" ]; then
      gpu_allowed=true
      break
    fi
  done

  if [ "$gpu_allowed" = false ]; then
    return 1 # GPU not in our allowed list
  fi

  # Allow up to MAX_JOBS_PER_GPU of our tasks on allowed GPUs, ignore other users
  if [ "${gpu_jobs[$gpu]}" -ge "$MAX_JOBS_PER_GPU" ]; then
    return 1 # Already have maximum tasks running
  fi

  return 0 # GPU is available for another of our tasks
}

# Function to launch a job on a GPU
launch_job() {
  local method=$1
  local beta=$2
  local n_colloc=$3
  local model_seed=$4
  local gpu=$5

  # Create experiment name
  local exp_name="${method}_b${beta}_n${n_colloc}_s${model_seed}"

  # Set learning rate based on method
  if [ "$method" == "lbfgs" ]; then
    lr="1e-3"
  else
    lr="1e-3"
  fi

  echo "Launching job with experiment name: $exp_name"
  echo "Using fixed data_seed=$data_seed for consistent training data"
  echo "Model will be saved to: $MODEL_SAVE_DIR/$exp_name/"

  # Launch Python in background with custom model save directory
  CUDA_VISIBLE_DEVICES=$gpu python "$SCRIPT_DIR/benchmark.py" \
    --method "$method" \
    --name "$exp_name" \
    --beta "$beta" \
    --n_colloc "$n_colloc" \
    --hidden-layers "$hidden_layers" \
    --lr "$lr" \
    --iter "$iterations" \
    --data-seed "$data_seed" \
    --model-seed "$model_seed" \
    --model-save-dir "$MODEL_SAVE_DIR" \
    >"$OUTPUT_LOG_DIR/${exp_name}.out" 2>&1 &

  # Store the PID and increment GPU counter
  local pid=$!
  pids+=($pid)
  gpu_jobs[$gpu]=$((gpu_jobs[$gpu] + 1))

  # Create tracking file for this job immediately (using system temp directory)
  TEMP_DIR="${TMPDIR:-/tmp}"
  touch "${TEMP_DIR}/gpu_${gpu}_pid_${pid}"

  echo "Launched job: method=$method, beta=$beta, n_colloc=$n_colloc, model_seed=$model_seed on GPU $gpu (PID $pid)"
  echo "GPU $gpu now has ${gpu_jobs[$gpu]} active job(s)"
}

# Function to clean up and update job tracking
cleanup_jobs() {
  # Get original PIDs for comparison
  local old_pids=("${pids[@]}")

  # Update PID list with only running processes
  pids=()
  for pid in "${old_pids[@]}"; do
    if kill -0 $pid 2>/dev/null; then # Check if process is still running
      pids+=($pid)
    else
      # Track finished jobs by GPU
      TEMP_DIR="${TMPDIR:-/tmp}"
      for gpu in "${GPUS[@]}"; do
        if [ -f "${TEMP_DIR}/gpu_${gpu}_pid_${pid}" ]; then
          gpu_jobs[$gpu]=$((gpu_jobs[$gpu] - 1))
          echo "Process $pid on GPU $gpu finished. Count now ${gpu_jobs[$gpu]}"
          rm "${TEMP_DIR}/gpu_${gpu}_pid_${pid}"
        fi
      done
    fi
  done

  # Verify GPU counts against actual running processes
  verify_gpu_counts
}

# Function to verify and correct GPU counts
verify_gpu_counts() {
  TEMP_DIR="${TMPDIR:-/tmp}"
  for gpu in "${GPUS[@]}"; do
    local actual_count=$(ls -1 "${TEMP_DIR}"/gpu_${gpu}_pid_* 2>/dev/null | wc -l)

    if [ "$actual_count" != "${gpu_jobs[$gpu]}" ]; then
      echo "WARNING: GPU $gpu count mismatch! Counter: ${gpu_jobs[$gpu]}, Files: $actual_count"
      gpu_jobs[$gpu]=$actual_count
    fi
  done
}

# Build a list of all configurations
configs=()
for method in "${methods[@]}"; do
  for beta in "${betas[@]}"; do
    for n_colloc in "${n_collocs[@]}"; do
      for model_seed in "${model_seeds[@]}"; do
        configs+=("$method $beta $n_colloc $model_seed")
      done
    done
  done
done

# Initial cleanup
cleanup_jobs

# Count of configurations
total_configs=${#configs[@]}
completed_configs=0
echo "Total configurations to run: $total_configs"

# Scan all GPUs and report their status before starting
echo "Initial GPU status:"
for gpu in "${GPUS[@]}"; do
  if is_gpu_free $gpu; then
    echo "GPU $gpu is available"
  else
    echo "GPU $gpu is NOT available"
  fi
done

# Single-pass allocation - distribute jobs across GPUs
echo "Starting job allocation - one job per GPU..."
while [ $completed_configs -lt $total_configs ]; do
  # Clean up before each round of allocation
  cleanup_jobs

  # Try to allocate jobs to any free GPUs
  made_progress=false

  for gpu in "${GPUS[@]}"; do
    # Skip if we've completed all configs
    if [ $completed_configs -ge $total_configs ]; then
      break
    fi

    # Check if this GPU is free
    if is_gpu_free $gpu; then
      IFS=' ' read -ra config <<<"${configs[$completed_configs]}"
      launch_job "${config[0]}" "${config[1]}" "${config[2]}" "${config[3]}" "$gpu"
      completed_configs=$((completed_configs + 1))
      made_progress=true
    fi
  done

  # If we couldn't allocate any jobs, wait briefly
  if [ "$made_progress" = false ]; then
    echo "Waiting for GPUs to free up... ($completed_configs/$total_configs complete)"
    sleep 20
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
