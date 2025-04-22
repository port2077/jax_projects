#!/bin/bash
#SBATCH --job-name=nano_gpt    
#SBATCH --output=train_%j.log         
#SBATCH --error=train_%j.err          
#SBATCH --nodes=1                     
#SBATCH --ntasks=1                    
#SBATCH --nodelist=gpu11
#SBATCH --gres=gpu:1                  
#SBATCH --partition=gpu.q               
#SBATCH --time=02:00:00              
#SBATCH --mem=20G                     

export PYTHONUNBUFFERED=TRUE

source $(conda info --base)/etc/profile.d/conda.sh
conda activate  ~/conda_envs/mljax


cd ~/jax_logs/nano-gpt

python attention.py --steps=5000

