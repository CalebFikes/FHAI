#!/bin/bash
#SBATCH --job-name=FHAI
#SBATCH --output=out_slurmtest
#SBATCH --gres=gpu:1


cd /local/scratch/cfikes/FHAI_3/Conditional_MNIST
python3.8 Experiment_6-22.py