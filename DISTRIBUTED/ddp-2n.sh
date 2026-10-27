#!/bin/bash
#SBATCH -J ddp-2n
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH -p short
#SBATCH -N 2
#SBATCH --gres=gpu:a100:2
#SBATCH -c 64
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --time=03:00:00

module load cesga/2022 python/3.10.8
export HF_HOME=$STORE/hf_cache
source $STORE/venvs/bertlab/bin/activate

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

srun torchrun --nnodes=$SLURM_NNODES \
    --nproc-per-node=$SLURM_GPUS_ON_NODE \
    --rdzv-id=$SLURM_JOB_ID \
    --rdzv-backend=c10d \
    --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT \
    bert_lab.py --log-dir runs/$SLURM_JOB_NAME "$@"
