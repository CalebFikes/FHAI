#!/bin/bash
#SBATCH --job-name=FHAI
#SBATCH --output=out
#SBATCH --gres=gpu:1

cd /local/scratch/cfikes/FHAI_3/Conditional_MNIST
pip install --no-index -r requirements.txt
pip install copy, logging, random, math, matplotlib==3.4.2, numpy==1.19.5, pandas==1.3.0
pip install seaborn==0.11.1, scikit-learn==0.24.2, torch==1.8.1, torchvision==0.9.1
pip install , tqdm==4.61.1, torch-utils-data==0.2.0, imbalanced-learn==0.8.0
python3.8 Experiment_6-22.py