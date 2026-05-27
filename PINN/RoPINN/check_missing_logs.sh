#!/bin/bash

log_dir="/scratch/kenzhong/ropinn_runs"
betas=(2 5 7 15 30 70)
n_collocs=(2 16 128 1024 8192 16384)
seeds=(0 1 2)

for beta in "${betas[@]}"; do
  for n in "${n_collocs[@]}"; do
    for seed in "${seeds[@]}"; do
      file="$log_dir/conv_PINN_b${beta}_n${n}_s${seed}.out"
      if [ ! -f "$file" ]; then
        echo "MISSING: beta=$beta, n_colloc=$n, seed=$seed"
      fi
    done
  done
done