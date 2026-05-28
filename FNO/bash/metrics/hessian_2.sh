#!/bin/bash
source ~/.bashrc
conda activate sciml_diagnosis

src_path=$(pwd)

for SLURM_ARRAY_TASK_ID in {77..148..1}
    do
        cfg=$(sed -n "$SLURM_ARRAY_TASK_ID"p ${src_path}/config/txt_files_metrics/hessian.txt)
        
        config=$(echo $cfg | cut -f 1 -d ' ')
        max_epochs=$(echo $cfg | cut -f 2 -d ' ')
        bsz=$(echo $cfg | cut -f 3 -d ' ')
        subsample=$(echo $cfg | cut -f 4 -d ' ')
        method=$(echo $cfg | cut -f 5 -d ' ')
        max_batches=$(echo $cfg | cut -f 6 -d ' ')
        
        for seed in 2025 2024 2023 2022 2021
            do
                CUDA_VISIBLE_DEVICES=1 python calculate_hessian.py  \
                    --config=$config \
                    --checkpoint_path results_sameiteration/expts_eps${max_epochs}/${config}/train/bsz${bsz}_lr0.001_subsample${subsample}/seed${seed}/checkpoints/ckpt.tar \
                    --method $method \
                    --max_batches $max_batches \
                    --target_seed $seed \
                    --expt_max_epochs $max_epochs \
                    --subsample $subsample \
                    --target_batch_size $bsz \
                    --layerwise 
            done
    done

#--use_validation \