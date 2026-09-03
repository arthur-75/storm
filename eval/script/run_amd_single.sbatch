#!/bin/bash
#SBATCH --constraint=HPDA
#SBATCH -t 6:00:00
#SBATCH -J eval8_single_mircal
#SBATCH --array=0-10
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --exclusive
#SBATCH --cpus-per-task=8 
#SBATCH --mem=61G
#SBATCH --gres=gpu:1         
#SBATCH --output=storm/train/logs/%x/slurm-%x-%j.out
#SBATCH --error=storm/train/logs/%x/slurm-%x-%j.err

set -euo pipefail

echo "Job started on $(hostname)"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "SLURM_ARRAY_JOB_ID=$SLURM_ARRAY_JOB_ID"
echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"

# Always work from the BEAM project root
cd "$HOME/projects/beam"

module purge
module load anaconda3

conda activate beamS


echo "Working directory: $(pwd)"
echo "Shard ID: ${SLURM_ARRAY_TASK_ID}"
echo "Number of shards: ${NUM_SHARDS}"

python storm/eval/run_retrieval_pipeline.py \
    --config storm/eval/configs/experiment_miracl.yaml \
    #--shard-id "${SLURM_ARRAY_TASK_ID}"
    # --merge-corpus-shards