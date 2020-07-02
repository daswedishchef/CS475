#!/bin/bash
#SBATCH -J CarloSim
#SBATCH -A cs475-575
#SBATCH -p class
#SBATCH --gres=gpu:1
#SBATCH -o matrixmul.out
#SBATCH -e matrixmul.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=carlsosp@oregonstate.edu
for t in 16 32 64 128
do
    /usr/local/apps/cuda/cuda-10.1/bin/nvcc -DBLOCKSIZE=$t -o carlox carlox.cu
    ./carlox
done