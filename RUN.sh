#!/bin/bash
#SBATCH --job-name=FHAI
#SBATCH --output=out
#SBATCH --gres=gpu:1

# Create and activate the virtual environment
python3.8 -m venv $HOME/myenv
source $HOME/myenv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install required packages
pip install --user copy logging random math matplotlib==3.4.2 numpy==1.19.5 pandas==1.3.0 seaborn==0.11.1 scikit-learn==0.24.2 torch==1.8.1 torchvision==0.9.1 tqdm==4.61.1 torch-utils-data==0.2.0 imbalanced-learn==0.8.0

# Change to the project directory
cd /local/scratch/cfikes/FHAI_3/Conditional_MNIST

# Run the Python script
python3.8 Experiment_6-22.py