#!/bin/bash
#SBATCH --job-name=FHAI
#SBATCH --output=out
#SBATCH --gres=gpu:1

cd /local/scratch/cfikes/FHAI_3/Conditional_MNIST
pip install --no-index -r requirements.txt
python3.8 Experiment_6-22.py