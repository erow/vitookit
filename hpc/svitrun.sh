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

export RDZV_HOST=$(hostname)
export RDZV_PORT=$((29400 + $RANDOM % 100))

echo dist $RDZV_HOST:$RDZV_PORT rdzv_id=$SLURM_JOB_ID $SLURM_JOB_NUM_NODES x $SLURM_GPUS_PER_NODE

echo "================= SLURM JOB INFO ================"
env

echo "================= GPU INFO       ================"
nvidia-smi

echo "================= SLURM JOB START ================"
#conda activate 

# vitrun --nproc_per_node=$SLURM_GPUS_ON_NODE --master_port $PORT --node-rank $SLURM_PROCID $ARGS

vitrun --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=$SLURM_GPUS_PER_NODE \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$RDZV_HOST:$RDZV_PORT" \
    $ARGS
