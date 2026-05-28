#!/bin/bash
source ~/.bashrc
conda activate sciml_diagnosis

src_path=$(pwd)
cfg_file="${src_path}/config/txt_files/poisson_epoch_setting.txt"

awk 'found {
        if ($0 ~ /^[[:space:]]*$/) next;
        if ($0 ~ /^#/) next;
        print
     }
     /^###### initialized/ {found=1; next}' "$cfg_file" | while read -r run_num max_epochs batch_size lr subsample scratch config_file config
    do
        if [[ -z "$run_num" ]]; then
            continue
        fi

        scratch="./results_sameiteration/"
        seed=2021

        python initalization.py  \
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
