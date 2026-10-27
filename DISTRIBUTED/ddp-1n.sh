#!/bin/bash
#SBATCH -J ddp-1n
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH -p short
#SBATCH -N 1
#SBATCH --gres=gpu:a100:2
#SBATCH -c 64
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --time=03:00:00

module load cesga/2022 python/3.10.8
export HF_HOME=$STORE/hf_cache
source $STORE/venvs/bertlab/bin/activate

torchrun --standalone --nproc-per-node=$SLURM_GPUS_ON_NODE bert_lab.py --log-dir runs/$SLURM_JOB_NAME "$@"
