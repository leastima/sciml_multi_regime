#!/bin/bash
#source ~/.bashrc
source /jumbo/yaoqingyang/yuanzhehu/anaconda3/bin/activate
conda activate sciml_diagnosis

src_path=$(pwd)


##results_sameiteration
###

for SLURM_ARRAY_TASK_ID in {127..134..1}
    do
        cfg=$(sed -n "$SLURM_ARRAY_TASK_ID"p ${src_path}/config/txt_files_metrics/loss_landscape_hessian.txt)

        config=$(echo $cfg | cut -f 1 -d ' ')
        max_epochs=$(echo $cfg | cut -f 2 -d ' ')
        bsz=$(echo $cfg | cut -f 3 -d ' ')
        subsample=$(echo $cfg | cut -f 4 -d ' ')
        grid=$(echo $cfg | cut -f 5 -d ' ')
        max_batches=$(echo $cfg | cut -f 6 -d ' ')
        radius=$(echo $cfg | cut -f 7 -d ' ')

        for seed in 2021
            do
                CUDA_VISIBLE_DEVICES=6 python plot_3D_losslandscape_hessian.py  \
                    --config=$config \
                    --checkpoint_path results_sameiteration/expts_eps${max_epochs}/${config}/train/bsz${bsz}_lr0.001_subsample${subsample}/seed${seed}/checkpoints/ckpt_best.tar \
                    --grid $grid \
                    --max_batches $max_batches \
                    --radius $radius \
                    --target_seed $seed \
                    --expt_max_epochs $max_epochs \
                    --subsample $subsample \
                    --target_batch_size $bsz \
                    --use_hessian_directions \
                    --log_scale
            done
    done
