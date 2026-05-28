# cmds=(
#     "python run_exp.py --model ResNet18 --dataset MNIST --epochs 20"
#     "python run_exp.py --model ResNet18 --dataset CIFAR-10 --epochs 200 --visualize"
#     "python run_exp.py --model ResNet18 --dataset CIFAR-100 --epochs 200"
# )

# for cmd in "${cmds[@]}"; do
#     echo "Running: $cmd"
#     eval "$cmd"
# done

export CUDA_VISIBLE_DEVICES=4

# seeds=(0 1 2)
# lrs=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8)
# batch_size=64

# for seed in "${seeds[@]}"
# do
#     for lr in "${lrs[@]}"
#     do
#         python train_trajectories.py --seed $seed --lr $lr --batch_size $batch_size
#     done
# done

seed=0
batch_sizes=(64)
for batch_size in "${batch_sizes[@]}"
do
    python run_exp.py --batch_size $batch_size --epochs 200 
done