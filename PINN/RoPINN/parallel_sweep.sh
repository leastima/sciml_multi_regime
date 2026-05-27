#!/bin/bash
# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Activate virtual environment (assuming it's in a standard location relative to project)
# Adjust this path based on where your virtual environment actually is
# source "${SCRIPT_DIR}/../../ropinn_env/bin/activate"

cd "$SCRIPT_DIR"

# ============================================
# USER CONFIGURATION: Specify GPUs to use
# ============================================
# List the GPU IDs you want to use for this sweep
# Example: AVAILABLE_GPUS=(0 1 2 3) to use GPUs 0-3
# Example: AVAILABLE_GPUS=(5 7) to use only GPUs 5 and 7
AVAILABLE_GPUS=(0 1)

# Maximum number of jobs per GPU (default: 2)
MAX_JOBS_PER_GPU=2

# Parameters for the sweep
betas=(5 6 7 8 9 10 15 20 25 30 50 70 100 150)
n_collocs=(10 50 100 150 200 250 500 1000 2000 5000 10000 15000 20000 25000)
# n_collocs=(2 16 128 1024 8192 16384)
models=("PINN")
seeds=(123)  # Seeds for statistical robustness
initial_regions=(1e-5)
sample_nums=(1)
past_iterations=(5)
pids=()

# Create results directory if it doesn't exist
mkdir -p ./results
mkdir -p ./scratch/ropinn_runs

# Simple GPU tracking array
declare -A gpu_jobs
for gpu in "${AVAILABLE_GPUS[@]}"; do
  gpu_jobs[$gpu]=0
done

echo "Using GPUs: ${AVAILABLE_GPUS[*]}"
echo "Max jobs per GPU: $MAX_JOBS_PER_GPU"

# Function to check if GPU has capacity for more jobs
is_gpu_free() {
  local gpu=$1
  
  # Check if this GPU has fewer jobs than the maximum allowed
  if [ "${gpu_jobs[$gpu]}" -lt "$MAX_JOBS_PER_GPU" ]; then
    return 0  # GPU has capacity
  else
    return 1  # GPU is at capacity
  fi
}

# Function to launch a job on a GPU
launch_job() {
  local model=$1
  local beta=$2
  local n_colloc=$3
  local seed=$4
  local initial_region=$5
  local sample_num=$6
  local past_iterations=$7
  local gpu=$8
  
  echo "Starting model=$model, beta=$beta, n_colloc=$n_colloc, seed=$seed, initial_region=$initial_region, sample_num=$sample_num on GPU $gpu"
  
  # Launch Python in background
  CUDA_VISIBLE_DEVICES=$gpu python convection_region_optimization.py \
    --model "$model" \
    --device "cuda:0" \
    --beta "$beta" \
    --n_colloc "$n_colloc" \
    --seed "$seed" \
    --initial_region "$initial_region" \
    --sample_num "$sample_num" \
    --past_iterations "$past_iterations" \
    > "./scratch/ropinn_runs/${model}_b${beta}_n${n_colloc}_s${seed}_ir${initial_region}_sn${sample_num}.out" 2>&1 &
  
  # Store the PID and increment GPU counter
  local pid=$!
  pids+=($pid)
  gpu_jobs[$gpu]=$((gpu_jobs[$gpu] + 1))
  echo "Launched model=$model, beta=$beta, n_colloc=$n_colloc, seed=$seed on GPU $gpu (PID $pid)"
  echo "GPU $gpu now has ${gpu_jobs[$gpu]} jobs"
  
  # Important: Sleep to allow gpustat to catch up
  sleep 15
}

# Function to clean up and update job tracking
cleanup_jobs() {
  # Get original PIDs for comparison
  local old_pids=("${pids[@]}")
  
  # Update PID list with only running processes
  pids=()
  for pid in "${old_pids[@]}"; do
    if kill -0 $pid 2>/dev/null; then   # Check if process is still running
      pids+=($pid)
    else
      echo "Process $pid has finished"
      
      # Track finished jobs by GPU
      for gpu in "${AVAILABLE_GPUS[@]}"; do
        if [ -f "/tmp/gpu_${gpu}_pid_${pid}" ]; then
          gpu_jobs[$gpu]=$((gpu_jobs[$gpu] - 1))
          echo "Decreased GPU $gpu count to ${gpu_jobs[$gpu]}"
          rm "/tmp/gpu_${gpu}_pid_${pid}"
        fi
      done
    fi
  done
  
  echo "Updated active job count: ${#pids[@]}"
}

# Clear old tracking files
rm -f /tmp/gpu_*_pid_*
echo "Old tracking files cleared."

# Launch jobs with more reliable tracking
for model in "${models[@]}"; do
  for beta in "${betas[@]}"; do
    for n_colloc in "${n_collocs[@]}"; do
      for seed in "${seeds[@]}"; do
        for initial_region in "${initial_regions[@]}"; do
          for sample_num in "${sample_nums[@]}"; do
            # Only use one past_iterations value to reduce sweep size
            past_iteration=${past_iterations[0]}
            
            # Track whether this configuration was successfully launched
            config_launched=false
            max_attempts=8
            attempt=1
            
            while [ $attempt -le $max_attempts ] && [ "$config_launched" = false ]; do
              # Find free GPU from available list
              gpu_assigned=-1
              for gpu in "${AVAILABLE_GPUS[@]}"; do
                if is_gpu_free $gpu; then
                  gpu_assigned=$gpu
                  break
                fi
              done
              
              if [ $gpu_assigned -ne -1 ]; then
                # Launch job on assigned GPU
                launch_job "$model" "$beta" "$n_colloc" "$seed" "$initial_region" "$sample_num" "$past_iteration" "$gpu_assigned"
                
                # Create tracking file for this job
                touch "/tmp/gpu_${gpu_assigned}_pid_${pids[-1]}"
                
                config_launched=true
              else
                # No GPU available
                echo "No GPUs free, waiting... (attempt $attempt of $max_attempts)"
                
                if [ ${#pids[@]} -gt 0 ]; then
                  # Wait for any job to finish
                  sleep 30
                  cleanup_jobs
                else
                  echo "No jobs running to wait for, sleeping for 30 seconds..."
                  sleep 30
                fi
                
                attempt=$((attempt+1))
              fi
            done
            
            # If failed to launch after all attempts
            if [ "$config_launched" = false ]; then
              echo "WARNING: Failed to allocate GPU for model=$model, beta=$beta, n_colloc=$n_colloc, seed=$seed, initial_region=$initial_region, sample_num=$sample_num after $max_attempts attempts"
              echo "$(date): Failed to allocate GPU for model=$model, beta=$beta, n_colloc=$n_colloc, seed=$seed, initial_region=$initial_region, sample_num=$sample_num" >> ./scratch/ropinn_runs/skipped_configs.log
            fi
            
            # Limit concurrent jobs
            if [ ${#pids[@]} -ge 14 ]; then
              echo "Reached limit of 14 concurrent jobs, waiting for one to finish..."
              sleep 30
              cleanup_jobs
            fi
          done
        done
      done
    done
  done
done

# Wait for remaining jobs
echo "Sweep complete! Waiting for ${#pids[@]} remaining jobs to finish..."
while [ ${#pids[@]} -gt 0 ]; do
  sleep 60
  cleanup_jobs
  echo "Still waiting for ${#pids[@]} jobs to finish..."
done

echo "All jobs completed. Check skipped_configs.log for any configurations that couldn't run."