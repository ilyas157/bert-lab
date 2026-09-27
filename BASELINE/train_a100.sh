#!/bin/bash
#SBATCH -J bertlab
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH -p short
#SBATCH --gres=gpu:a100
#SBATCH -c 32
#SBATCH --mem=32G
#SBATCH --time=03:00:00

module load cesga/2022 python/3.10.8
export HF_HOME=$STORE/hf_cache
source $STORE/venvs/bertlab/bin/activate
python -u bert_lab.py --log-dir runs/$SLURM_JOB_NAME "$@"
