#!/bin/bash
source /jumbo/yaoqingyang/yuanzhehu/anaconda3/bin/activate
conda activate sciml_diagnosis

src_path=$(pwd)
config_file="${src_path}/config/txt_files_metrics/hessian_density.txt"
seed=2021
root_dir="${src_path}/hessian_analysis/density_init"

while IFS= read -r cfg
do
    [[ -z "${cfg}" || "${cfg}" == \#* ]] && continue
    read -r config max_epochs bsz subsample method max_batches lanczos_iter slq_runs <<< "${cfg}"

    checkpoint_path="${src_path}/results_sameiteration/expts_eps${max_epochs}/${config}/initialization/bsz${bsz}_lr0.001_subsample${subsample}/seed${seed}/checkpoints/ckpt_init.tar"
    if [ ! -f "${checkpoint_path}" ]; then
        echo "Skipping: checkpoint not found: ${checkpoint_path}"
        continue
    fi

    CUDA_VISIBLE_DEVICES=2 python calculate_hessian.py \
        --config="${config}" \
        --checkpoint_path "${checkpoint_path}" \
        --method "${method}" \
        --max_batches "${max_batches}" \
        --target_seed "${seed}" \
        --expt_max_epochs "${max_epochs}" \
        --subsample "${subsample}" \
        --target_batch_size "${bsz}" \
        --lanczos_iter "${lanczos_iter}" \
        --slq_runs "${slq_runs}" \
        --root_dir "${root_dir}"
done < "${config_file}"
