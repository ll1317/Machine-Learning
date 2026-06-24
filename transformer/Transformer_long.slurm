#!/bin/bash
#SBATCH --job-name=TF # create a short name for your job
#SBATCH --partition=A100
#SBATCH --mem=64G
#SBATCH --nodes=1 # node count
#SBATCH --ntasks=1 # total number of tasks across all nodes
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=6
#SBATCH -o logs/transformer_long.log

unset LD_LIBRARY_PATH

source activate maha_norm

python train_transformer.py --horizon 365 --seeds 42 52 62 72 82 --epochs 200 --patience 20 --device cuda