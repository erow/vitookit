#!/bin/bash
# Parameters
#SBATCH --partition=3090_risk
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --job-name=vitrun
#SBATCH --mem=100GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --open-mode=append
#SBATCH --error=outputs/slurm/%j_%t_log.err
#SBATCH --output=outputs/slurm/%j_%t_log.out

ARGS=${@}
PORT=$((29401 + $RANDOM % 100))
#conda activate 

nvidia-smi
vitrun --nproc_per_node=$SLURM_GPUS_ON_NODE --master_port $PORT --node-rank $SLURM_PROCID $ARGS
