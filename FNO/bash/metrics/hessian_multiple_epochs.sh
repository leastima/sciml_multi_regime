#!/bin/bash
source ~/.bashrc
conda activate sciml_diagnosis

src_path=$(pwd)

for SLURM_ARRAY_TASK_ID in {217..225..1}
    do
        cfg=$(sed -n "$SLURM_ARRAY_TASK_ID"p ${src_path}/config/txt_files_metrics/hessian.txt)
        
        config=$(echo $cfg | cut -f 1 -d ' ')
        max_epochs=$(echo $cfg | cut -f 2 -d ' ')
        bsz=$(echo $cfg | cut -f 3 -d ' ')
        subsample=$(echo $cfg | cut -f 4 -d ' ')
        method=$(echo $cfg | cut -f 5 -d ' ')
        max_batches=$(echo $cfg | cut -f 6 -d ' ')

        step=$((max_epochs/50))
        if [ $step -lt 1 ]; then
            step=1
        fi
        epoch_list=""
        for ((e=0; e<max_epochs; e+=step)); do
            epoch_list="$epoch_list $e"
        done

        for seed in 2025 2024 2023 2022 2021
            do
                for ckpt_epoch in $epoch_list
                    do
                        CUDA_VISIBLE_DEVICES=3 python calculate_hessian.py  \
                            --config=$config \
                            --checkpoint_path results_all_downloaded/expts_eps${max_epochs}/${config}/train/bsz${bsz}_lr0.001_subsample${subsample}/seed${seed}/checkpoints/ckpt_${ckpt_epoch}.tar \
                            --method $method \
                            --max_batches $max_batches \
                            --target_seed $seed \
                            --expt_max_epochs $max_epochs \
                            --subsample $subsample \
                            --target_batch_size $bsz \
                            --ckpt_epoch $ckpt_epoch 
                    done
            done
    done

#--use_validation \