#!/bin/sh

#SBATCH --job-name=cruncher
#SBATCH --time 1-00:00:00
#SBATCH --gres=gpu:A6000:4
#SBATCH --mem=96GB
#SBATCH --cpus-per-task 16

eval "$(conda shell.bash hook)"
conda activate ~/envs/sandbox
cd ~/sangwu/finebooru

export TQDM_MININTERVAL=42

accelerate launch --config_file 4.yml --main_process_port 29542 train.py \
    --name imagenet-fsq-vqvae \
    --dataset hub://activeloop/imagenet-train \
    --backbone convolution \
    --lr 3e-4 \
    --epochs 16 \
    --features 192 \
    --batch_size 16 \
    --heads 6 \
    --strides 8 \
    --padding 0 \
    --depth 6 \
    --patch 8 \
    --pages 65536 \
    --codes 16 \
    --gradient_accumulation_steps 1 \
    --compile True \
    --log_every_n_steps 2048 \
    --quantiser fsq \
    --save True \
    --push True

exit 0
