#!/bin/bash
# =====================================================
# 智能GPU批量实验脚本 (限制并发数 N=2)
# - 若系统无GPU则自动使用CPU
# - GPU 占用率 < 50% 才会被分配
# - 最多同时运行 N=2 个任务
# =====================================================

# ============ 用户配置区 ============
device_ids=(0)   # 可用 GPU 列表
BETA_list=(5 6 7 8 9 10 15 20 25 30 50 70 100 150)
BETA_list=(30)
numres_list=(10 50 100 150 200 250 500 1000 2000 5000 10000 15000 20000 25000)
numres_list=(10000)
seednum=1
save_path="output"
pde="convection"
pde_params_template='{"beta":%d}'
opt="lbfgs"  # Options: lbfgs, adam_lbfgs, adam_lbfgs_nncg
hc="none"

# 固定 ALM 参数
L=1
alm_beta=1.2
alm_iter=50
MAX_JOBS=5      # 最多同时运行的任务数
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

# ============ Optimizer Parameter Defaults ============
# Standard values for consistency across all LBFGS-based optimizers
LBFGS_LR=1.0
LBFGS_HISTORY_SIZE=100
LBFGS_MAX_ITER=10000
ADAM_LR=1e-3
ADAM_SWITCH_EPOCH=5000  # For Adam+LBFGS
LBFGS_SWITCH_EPOCH=10000  # For Adam+LBFGS+NNCG
NNCG_SWITCH_EPOCH=15000   # For Adam+LBFGS+NNCG

# 检查是否存在 GPU（或 nvidia-smi）
if ! command -v nvidia-smi &> /dev/null; then
  echo "[WARN] nvidia-smi 未找到，切换到 CPU 模式运行。"
  USE_CPU=true
else
  gpu_count=$(nvidia-smi -L 2>/dev/null | wc -l)
  if [ "$gpu_count" -eq 0 ]; then
    echo "[WARN] 未检测到可用 GPU，切换到 CPU 模式运行。"
    USE_CPU=true
  else
    USE_CPU=false
  fi
fi

# ============ 工具函数 ============
# 获取当前正在运行的后台任务数量
get_running_jobs() {
  jobs -rp | wc -l
}

# 获取显存占用率 < 50% 的 GPU
get_free_gpu() {
  if [ "$USE_CPU" = true ]; then
    echo -1
    return
  fi

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
  for beta in "${BETA_list[@]}"; do
    for numres in "${numres_list[@]}"; do

      # 控制最大并发任务数
      while [ "$(get_running_jobs)" -ge "$MAX_JOBS" ]; do
#        echo "[INFO] 当前已有 $(get_running_jobs) 个任务在运行，等待空位..."
        sleep 10
      done

      # GPU / CPU 选择逻辑
      if [ "$USE_CPU" = true ]; then
        device_id="cpu"
      else
        device_id=$(get_free_gpu)
        while [ -z "$device_id" ] || [ "$device_id" -lt 0 ]; do
#          echo "[INFO] 无空闲 GPU（<50% 占用率），等待中..."
          sleep 15
          device_id=$(get_free_gpu)
        done
      fi

      # 动态参数调整
      if (( $(echo "$beta < 50" | bc -l) )); then
        alm_L=500
        alm_weight_decay=0
      else
        alm_L=100
        alm_weight_decay=0.001
      fi

      # 构造 PDE 参数 JSON
      pde_params=$(printf "$pde_params_template" "$beta")

      # 设置优化器参数 (显式指定以保证一致性和可重复性)
      if [[ "$opt" == "lbfgs" ]]; then
        opt_params="{\"lr\":$LBFGS_LR,\"history_size\":$LBFGS_HISTORY_SIZE,\"max_iter\":$LBFGS_MAX_ITER}"
      elif [[ "$opt" == "adam_lbfgs" ]]; then
        opt_params="{\"switch_epochs\":$ADAM_SWITCH_EPOCH,\"adam_lr\":$ADAM_LR,\"lbfgs_lr\":$LBFGS_LR,\"lbfgs_history_size\":$LBFGS_HISTORY_SIZE,\"lbfgs_max_iter\":$LBFGS_MAX_ITER}"
      elif [[ "$opt" == "adam_lbfgs_nncg" ]]; then
        opt_params="{\"switch_epochs\":$ADAM_SWITCH_EPOCH,\"adam_lr\":$ADAM_LR,\"lbfgs_history_size\":$LBFGS_HISTORY_SIZE,\"lbfgs_max_iter\":$LBFGS_MAX_ITER,\"switch_epoch_lbfgs\":$LBFGS_SWITCH_EPOCH,\"switch_epoch_nncg\":$NNCG_SWITCH_EPOCH,\"precond_update_freq\":100,\"nncg_rank\":50,\"nncg_mu\":0.01,\"nncg_cg_tol\":1e-5,\"nncg_use_double\":false,\"nncg_dynamic_damping\":true,\"nncg_verbose\":true}"
      else
        opt_params="{}"
      fi

      echo "=========================================================="
      echo "实验 #$exp_count"
      echo "Seed=$seed | beta=$beta | num_res=$numres | Device=$device_id"
      echo "alm_L=$alm_L | alm_weight_decay=$alm_weight_decay"
      echo "=========================================================="

      # 构建命令
      cmd="python run_experiment.py \
        --save_path $save_path \
        --pde $pde \
        --pde_params '$pde_params' \
        --opt $opt \
        --opt_params '$opt_params' \
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
        --num_res $numres \
        --epochs 1 \
        --cl \
        --set_idx 2"

      log_file="$LOG_DIR/seed${seed}_beta${beta}_numres${numres}.log"

      echo "[RUNNING] $cmd > $log_file 2>&1 &"
      eval "$cmd > $log_file 2>&1 &"
      exp_count=$((exp_count + 1))

      sleep 5
    done
  done
done

wait
echo "✅ 所有实验已完成！"
