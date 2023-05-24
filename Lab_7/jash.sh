#!/bin/bash
#SBATCH --job-name=dlops_lab_7
#SBATCH --partition=gpu2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8gb
#SBATCH --gres=gpu
#SBATCH --time=02:00:00
#SBATCH --output=first_%j.log


module load python/3.8
#module load anaconda/3


#<Executable PATH> INPUT OUTPUT
python3 M22CS061_Lab_Assignment7.py