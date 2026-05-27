#!/bin/bash
# =====================================================
# 智能GPU批量实验脚本 for Wave PDE (限制并发数 N=2)
# - GPU 占用率 < 50% 才会被分配
# - ALM 参数固定
# - 最多同时运行 N=2 个任务
# =====================================================

# ============ 用户配置区 ============
device_ids=(2)   # 可用 GPU 列表
c_list=(0.1 0.5 1 2 3 4 5 6)
numres_list=(5 10 50 100 250 500 1000 2000 5000 10000)
seednum=5
save_path="output"
pde="wave"
pde_params_template='{"beta":2, "c":%g}'
hc="none"

# 固定 ALM 参数（示例命令）
L=1
alm_L=100
alm_beta=2
alm_iter=10
alm_weight_decay=0

MAX_JOBS=2      # 最多同时运行的任务数
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

# ============ 工具函数 ============
# 获取当前正在运行的后台任务数量
get_running_jobs() {
  jobs -rp | wc -l
}

# 获取显存占用率 < 50% 的 GPU
get_free_gpu() {
  free_gpu=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | \
    awk -v devices="${device_ids[*]}" '
      BEGIN {
        split(devices, arr, " ");
        min_usage=999999; free_id=-1;
        idx=0;
      }
      {
        used=$1; total=$2;
        usage=used/total*100;
        mem[idx]=usage;
        idx++;
      }
      END {
        for (i in arr) {
          id=arr[i];
          if (mem[id] < 50 && mem[id] < min_usage) {
            min_usage=mem[id];
            free_id=id;
          }
        }
        print free_id;
      }')
  echo $free_gpu
}

# ============ 主循环 ============
exp_count=0
seed_list=$(seq 0 $((seednum - 1)))

for seed in $seed_list; do
  for c in "${c_list[@]}"; do
    for numres in "${numres_list[@]}"; do

      # 控制最大并发任务数
      while [ "$(get_running_jobs)" -ge "$MAX_JOBS" ]; do
        echo "[INFO] 当前已有 $(get_running_jobs) 个任务在运行，等待空位..."
        sleep 100
      done

      # 获取可用 GPU（占用率 < 50）
      device_id=$(get_free_gpu)
      while [ -z "$device_id" ] || [ "$device_id" -lt 0 ]; do
        echo "[INFO] 无空闲 GPU（<50% 占用率），等待中..."
        sleep 100
        device_id=$(get_free_gpu)
      done

      # 构造 PDE 参数 JSON
      pde_params=$(printf "$pde_params_template" "$c")

      echo "=========================================================="
      echo "实验 #$exp_count"
      echo "Seed=$seed | c=$c | num_res=$numres | GPU=$device_id"
      echo "=========================================================="

      # 构建命令
      cmd="python run_experiment.py \
        --save_path $save_path \
        --pde $pde \
        --pde_params '$pde_params' \
        --new_data \
        --save_model \
        --initial_seed $seed \
        --hc $hc \
        --L $L \
        --alm_L $alm_L \
        --alm_beta $alm_beta \
        --alm_iter $alm_iter \
        --alm_weight_decay $alm_weight_decay \
        --device $device_id \
        --num_res $numres"

      log_file="$LOG_DIR/seed${seed}_c${c}_numres${numres}.log"

      echo "[RUNNING] $cmd > $log_file 2>&1 &"
      eval "$cmd > $log_file 2>&1 &"
      exp_count=$((exp_count + 1))

      # 稍等以避免GPU争用
      sleep 5
    done
  done
done

wait
echo "✅ 所有实验已完成！"
