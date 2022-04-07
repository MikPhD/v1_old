#!/bin/sh
#SBATCH -J v1_gru
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --gres-flags=enforce-binding
#SBATCH --partition=all
#SBATCH -t 48:00:00
#SBATCH -o logs/case.out
#SBATCH -e logs/case.err
#SBATCH --mail-user=mik.quattromini@gmail.com
#SBATCH --mail-type=begin,end,fail

srun python Main.py -e 100 -r False -tcase 150 -vcas 150 -n_out 10

