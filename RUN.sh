#!/bin/bash
#SBATCH --job-name=FHAI
#SBATCH --output=/path/to/output/out_slurmtest
#SBATCH --gres=gpu:1

# Load any necessary modules
module load python3

# Set the absolute path to the Python interpreter
PYTHON_INTERPRETER= python3.8

# Set the absolute path to the project directory
PROJECT_DIR=/local/scratch/cfikes/FHAI_3/Conditional_MNIST

# Install pipreqs
$PYTHON_INTERPRETER -m pip install pipreqs

# Generate requirements.txt
cd $PROJECT_DIR
$PYTHON_INTERPRETER -m pipreqs --force .

# Install required packages
$PYTHON_INTERPRETER -m pip install --no-index -r requirements.txt

# Run the Python script
$PYTHON_INTERPRETER $PROJECT_DIR/Experiment_6-22.py