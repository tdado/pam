#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --mem=8G

module load 2022
module load CUDA/11.8.0

source ~/fallon/bin/activate
python /home/tdado/pam/main_sty.py
python /home/tdado/pam/lin_sty.py