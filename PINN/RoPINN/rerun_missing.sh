#!/bin/bash
source /jumbo/yaoqingyang/kenzhong/ropinn_env/bin/activate
cd /jumbo/yaoqingyang/kenzhong/RoPINN

# Define missing combinations with seed values
declare -a missing=(
  "5 128 1"
  "7 128 1"
  "7 128 2"
  "7 16384 1"
  "7 16384 2"
  "15 128 1"
  "15 1024 2"
  "15 16384 1"
  "15 16384 2"
  "30 2 0"
  "30 16 1"
  "30 128 0"
  "30 8192 1"
  "30 16384 1"
  "70 2 0"
  "70 2 1"
  "70 16 0"
  "70 16 1"
  "70 16 2"
  "70 128 0"
  "70 128 1"
  "70 128 2"
  "70 8192 1"
  "70 8192 2"
  "70 16384 1"
  "70 16384 2"
)

model="PINN"
pids=()

# Ensure output directories exist
mkdir -p ./results/
mkdir -p /scratch/kenzhong/ropinn_runs/

# Function to run a job on a GPU
run_job() {
  local beta=$1
  local n_colloc=$2
  local seed=$3
  local gpu=$4
  echo "Starting beta=$beta, n_colloc=$n_colloc, seed=$seed on GPU $gpu"
  CUDA_VISIBLE_DEVICES=$gpu python convection_region_optimization.py \
    --model $model \
    --beta "$beta" \
    --seed "$seed" \
    --n_colloc "$n_colloc" \
    --device "cuda:0" \
    > /scratch/kenzhong/ropinn_runs/conv_${model}_b${beta}_n${n_colloc}_s${seed}.out 2>&1
  
  # Create a dedicated directory for this run's results
  mkdir -p ./results/conv_${model}_b${beta}_n${n_colloc}_s${seed}/
  # Copy result files to the dedicated directory
  cp ./results/convection_${model}_region*.* ./results/conv_${model}_b${beta}_n${n_colloc}_s${seed}/
  
  echo "Finished beta=$beta, n_colloc=$n_colloc, seed=$seed on GPU $gpu"
}

# Function to check if GPU is free
check_gpu_usage() {
  # Print current usage
  echo "Current GPU usage:"
  gpustat
  
  # Find least used GPU
  local min_usage=9999
  local best_gpu=-1
  
  for gpu in 0 1 2 3 4 5 6 7; do
    # Get memory usage for this GPU
    local gpu_info=$(gpustat | grep "^\[$gpu\]")
    if [[ $gpu_info =~ ([0-9]+)\ /\ ([0-9]+)\ MB ]]; then
      local used=${BASH_REMATCH[1]}
      local total=${BASH_REMATCH[2]}
      
      # If this is the least used GPU so far, record it
      if [ "$used" -lt "$min_usage" ]; then
        min_usage=$used
        best_gpu=$gpu
      fi
    fi
  done
  
  echo "Selected GPU $best_gpu with lowest memory usage ($min_usage MB)"
  return $best_gpu
}

# Process each missing experiment
for combo in "${missing[@]}"; do
  read -r beta n_colloc seed <<< "$combo"
  
  # Check if already completed (file exists)
  if [ -f "/scratch/kenzhong/ropinn_runs/conv_${model}_b${beta}_n${n_colloc}_s${seed}.out" ]; then
    echo "Output file for beta=$beta, n_colloc=$n_colloc, seed=$seed already exists, skipping"
    continue
  fi
  
  # Find best GPU to use
  check_gpu_usage
  gpu=$?
  
  # Run the job
  run_job $beta $n_colloc $seed $gpu &
  pids+=($!)
  echo "Launched beta=$beta, n_colloc=$n_colloc, seed=$seed on GPU $gpu (PID ${pids[-1]})"
  
  # Wait a bit to let the job start and memory usage stabilize
  sleep 6
  
  # Keep no more than 8 jobs running at once
  if [ ${#pids[@]} -ge 8 ]; then
    echo "Reached limit of 8 concurrent jobs, waiting for one to finish..."
    wait -n
    pids=($(ps -p "${pids[@]}" -o pid= 2>/dev/null))
    echo "Job finished, continuing with ${#pids[@]} active jobs"
  fi
done

# Wait for all remaining jobs
wait
echo "Missing experiments completed!"

# Verify all missing files now exist
still_missing=0
for combo in "${missing[@]}"; do
  read -r beta n_colloc seed <<< "$combo"
  if [ ! -f "/scratch/kenzhong/ropinn_runs/conv_${model}_b${beta}_n${n_colloc}_s${seed}.out" ]; then
    echo "Still missing: beta=$beta, n_colloc=$n_colloc, seed=$seed"
    still_missing=$((still_missing + 1))
  fi
done

if [ $still_missing -eq 0 ]; then
  echo "All missing experiments successfully completed!"
else
  echo "$still_missing experiments still missing."
fi