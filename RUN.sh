#!/bin/bash
#SBATCH --job-name=slurmtest
#SBATCH --output=out_slurmtest
#SBATCH --gres=gpu:1


cd /path/to/working/directory/
source venv/bin/activate
python Experiment 6-22.py