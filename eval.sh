#!/bin/sh

#SBATCH --job-name=eval
#SBATCH --time 1-00:00:00
#SBATCH --gres=gpu:A6000
#SBATCH --mem=96GB
#SBATCH --cpus-per-task 16

eval "$(conda shell.bash hook)"
conda activate ~/envs/sandbox
cd ~/sangwu/finebooru

export TQDM_MININTERVAL=42
# accelerate launch --config_file 4.yml --main_process_port 29542 train.py \
python eval.py \
    --name RE-N-Y/imagenet-fsq-vqvae \
    --dataset hub://activeloop/imagenet-val \

exit 0
