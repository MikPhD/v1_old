#!/bin/sh
#SBATCH -J v1_set_opt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --gres-flags=enforce-binding
#SBATCH --partition=all
#SBATCH -t 48:00:00
#SBATCH -o logs/case.out
#SBATCH -e logs/case.err

srun python Main.py -e 5000 -r False -tcase 110 -vcas 110 -n_out 1000

