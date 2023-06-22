#!/bin/bash
#SBATCH --job-name=FHAI
#SBATCH --output=out
#SBATCH --gres=gpu:1

# Create and activate the virtual environment
virtualenv -p python3 venv
source venv/bin/activate

# Install required packages
pip install copy logging random math matplotlib matplotlib.pyplot numplotnumpy==1.19.5 pandas==1.3.0 
pip install  torch scikit-learn==0.24.2 torchvision==0.9.1 tqdm==4.61.1 torch-utils-data==0.2.0 
pip install imbalanced-learn==0.8.0 seaborn==0.11.1 
# Change to the project directory
cd /local/scratch/cfikes/FHAI_3/Conditional_MNIST

# Run the Python script
python3.8 Experiment_6-22.py