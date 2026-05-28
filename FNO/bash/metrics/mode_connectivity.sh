#!/bin/bash
source ~/.bashrc
conda activate sciml_diagnosis

src_path=$(pwd)

# Define the 5 seeds
seeds=(2025 2024 2023 2022 2021)

for SLURM_ARRAY_TASK_ID in {3..210..1}
    do
        cfg=$(sed -n "$SLURM_ARRAY_TASK_ID"p ${src_path}/config/txt_files_metrics/mode_connectivity.txt)
        
        config=$(echo $cfg | cut -f 1 -d ' ')
        max_epochs=$(echo $cfg | cut -f 2 -d ' ')
        bsz=$(echo $cfg | cut -f 3 -d ' ')
        subsample=$(echo $cfg | cut -f 4 -d ' ')
        curve_type=$(echo $cfg | cut -f 5 -d ' ')
        max_batches=$(echo $cfg | cut -f 6 -d ' ')
        
        # Generate all pairwise combinations of seeds (10 combinations total)
        for i in {0..3}
            do
                for j in $(seq $((i+1)) 4)
                    do
                        seed_a=${seeds[$i]}
                        seed_b=${seeds[$j]}
                        
                        # Construct checkpoint paths
                        checkpoint_a="results_sameiteration/expts_eps${max_epochs}/${config}/train/bsz${bsz}_lr0.001_subsample${subsample}/seed${seed_a}/checkpoints/ckpt.tar"
                        checkpoint_b="results_sameiteration/expts_eps${max_epochs}/${config}/train/bsz${bsz}_lr0.001_subsample${subsample}/seed${seed_b}/checkpoints/ckpt.tar"
                        
                        # Check if both checkpoints exist before running
                        if [ -f "$checkpoint_a" ] && [ -f "$checkpoint_b" ]; then
                            echo "Running mode connectivity analysis: ${config} - seed${seed_a} vs seed${seed_b} - ${curve_type}"
                            
                            CUDA_VISIBLE_DEVICES=6 python calculate_mode_connectivity.py \
                                --config=$config \
                                --checkpoint_a=$checkpoint_a \
                                --checkpoint_b=$checkpoint_b \
                                --curve_type=$curve_type \
                                --max_batches=$max_batches \
                                --seed_a=$seed_a \
                                --seed_b=$seed_b \
                                --target_batch_size=$bsz \
                                --subsample=$subsample \
                                --expt_max_epochs $max_epochs
                        else
                            echo "Skipping: One or both checkpoints not found"
                            echo "  Checkpoint A: $checkpoint_a"
                            echo "  Checkpoint B: $checkpoint_b"
                        fi
                    done
            done
    done 

#--use_validation \
