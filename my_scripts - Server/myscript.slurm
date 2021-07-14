#!/bin/bash

#SBATCH --job-name=test-job

#SBATCH --account=thinklab-ckpt

#SBATCH --partition=ckpt

#SBATCH --nodes=1

#SBATCH --ntasks-per-node=16

#SBATCH --mem=100G

#SBATCH --time=48:00:00

#SBATCH --mail-type=ALL

#SBATCH --mail-user=guanxy@uw.edu

#SBATCH --chdir=/gscratch/thinklab/sb3-seir2

bash my_scripts/hyperparameters.sh