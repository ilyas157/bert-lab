#!/bin/bash
#SBATCH -J bertlab
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --mem=12G
#SBATCH --time=00-0:10:00
#SBATCH --gres=gpu:t4
#SBATCH -p viz

module load cesga/2022 python/3.10.8
export HF_HOME=$STORE/hf_cache
source $STORE/venvs/bertlab/bin/activate
python -u bert_lab.py "$@" 
