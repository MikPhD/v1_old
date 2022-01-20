#!/bin/sh
#SBATCH -J casev2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --gres-flags=enforce-binding
#SBATCH --partition=all
#SBATCH -t 24:00:00
#SBATCH -o logs/case.out
#SBATCH -e logs/case.err

srun python Main.py -e 1000000
