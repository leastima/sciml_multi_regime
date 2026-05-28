#!/bin/bash
source ~/.bashrc
conda activate sciml_diagnosis

src_path=$(pwd)

for SLURM_ARRAY_TASK_ID in 16 21
    do
        cfg=$(sed -n "$SLURM_ARRAY_TASK_ID"p ${src_path}/config/txt_files/advdiff_epochs.txt)
        
        run_num=$(echo $cfg | cut -f 1 -d ' ')
        max_epochs=$(echo $cfg | cut -f 2 -d ' ')
        batch_size=$(echo $cfg | cut -f 3 -d ' ')
        lr=$(echo $cfg | cut -f 4 -d ' ')
        subsample=$(echo $cfg | cut -f 5 -d ' ')
        scratch=$(echo $cfg | cut -f 6 -d ' ')
        config_file=$(echo $cfg | cut -f 7 -d ' ')
        config=$(echo $cfg | cut -f 8 -d ' ')
        
        for seed in  2024 2023 2022 2021 #2025
            do
                CUDA_VISIBLE_DEVICES=5 python train.py  \
                                        --yaml_config=$config_file \
                                        --config=$config \
                                        --run_num=$run_num \
                                        --root_dir=$scratch \
                                        --batch_size=$batch_size \
                                        --subsample=$subsample \
                                        --lr=$lr \
                                        --seed=$seed \
                                        --max_epochs=$max_epochs 
            done
    done










