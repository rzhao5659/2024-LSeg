#! /bin/bash
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --job-name=lseg_ml
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8 
#SBATCH --mem=32G
#SBATCH -o logs/output_%j.txt
#SBATCH -e logs/error_%j.txt 

source activate vml2
module load cuda/12.1
python train.py