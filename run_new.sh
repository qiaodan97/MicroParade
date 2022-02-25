# ===================
# Runs the experiment on RecSys dataset
# Author: @liamhebert
# ===================


#!/usr/bin/env bash
[ -z "${exp_name}" ] && exp_name="micro_parade"
[ -z "${seed}" ] && seed="1"
[ -z "${arch}" ] && arch="--hidden_dim 769 --transformer_nheads 4 --num_transformer_layers 2 --dropout 0.35  --n_layers 10"
[ -z "${batch_size}" ] && batch_size="8"         
[ -z "${epoch}" ] && epoch="10"
[ -z "${lr}" ] && lr="5e-5"
[ -z "${ckpt_path}" ] && ckpt_path="../../checkpoints/"

echo -e "\n\n"
echo "=====================================ARGS======================================"
echo "arg0: $0"
echo "exp_name: ${exp_name}"
echo "ckpt_path ${ckpt_path}"
echo "arch: ${arch}"
echo "batch_size: ${batch_size}"
echo "lr ${lr}"
echo "seed: ${seed}"
echo "epoch: ${epoch}"
echo "==============================================================================="

n_gpu=$(nvidia-smi -L | wc -l) 
max_epochs=$((epoch+1))
export TRANSFORMERS_OFFLINE=0

default_root_dir=exps/$exp_name/$seed
mkdir -p $default_root_dir

python micro_parade_new.py --num_workers 8 --seed $seed --batch_size $batch_size \
      --gpus $n_gpu --precision 32 $arch \
      --default_root_dir $default_root_dir \
      --check_val_every_n_epoch 1 --val_check_interval 1.0 \
      --max_epochs $max_epochs \
      --peak_lr $peak_lr --end_lr $end_lr --progress_bar_refresh_rate 10 --gradient_clip_val 5.0

# validate and test on every checkpoint
checkpoint_dir=$default_root_dir/lightning_logs/checkpoints/
echo "=====================================EVAL======================================"
for file in `ls $checkpoint_dir/*.ckpt`
do
      echo -e "\n\n\n ckpt:"
      echo "$file"
      echo -e "\n\n\n"
      python micro_parade_new.py --num_workers 8 --seed $seed --batch_size $batch_size \
      --gpus $n_gpu --precision 32 $arch \
      --default_root_dir $default_root_dir \
      --check_val_every_n_epoch 1 --val_check_interval 1.0 \
      --max_epochs $max_epochs --validate \
      --peak_lr $peak_lr --end_lr $end_lr --progress_bar_refresh_rate 10 --gradient_clip_val 5.0

      python micro_parade_new.py --num_workers 8 --seed $seed --batch_size $batch_size \
      --gpus $n_gpu --precision 32 $arch \
      --default_root_dir $default_root_dir \
      --check_val_every_n_epoch 1 --val_check_interval 1.0 \
      --max_epochs $max_epochs --test \
      --peak_lr $peak_lr --end_lr $end_lr --progress_bar_refresh_rate 10 --gradient_clip_val 5.0

done
echo "==============================================================================="

