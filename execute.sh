#!/bin/sh
#SBATCH -J v1_optuna
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

srun python Main.py -e 250 -r False -tcase 40 50 60 70 80 90 100 120 130 140 150 -vcas 110 -n_out 1000

