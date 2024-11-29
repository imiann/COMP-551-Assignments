#!/bin/bash
#SBATCH --nodes 1
#SBATCH --gres=gpu:1 # request a GPU
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=%N-%j.out
#SBATCH --acount=fall2024-comp551

module load cuda/cuda-12.6
module load miniconda/miniconda-fall2024
source ~/.bashrc
conda activate test-jf

echo "TEST" > test_out.txt
nvidia-smi >> test_out.txt
python cifar10.py