#!/bin/bash
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH -t 24:00:00
#SBATCH --array=0-1%1
#SBATCH -t 24:00:00
#SBATCH -J q_8_exp
#SBATCH --output=storm/train/logs/%x/slurm-%x-%j.out
#SBATCH --error=storm/train/logs/%x/slurm-%x-%j.err

set -euo pipefail
# ── Version ────────────────────────────────────────────
echo "============================================"
echo " Training Version : q8_eps "
echo " Script           : $0"
echo " Date             : $(date '+%Y-%m-%d %H:%M:%S')"
echo "Job name          : ${SLURM_JOB_NAME}"
echo "SLURM_ARRAY_JOB_ID=$SLURM_ARRAY_JOB_ID"   # the 4950860 part
echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"   # the _0, _1, _2 part
echo "============================================"
echo "Job started on $(hostname)"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-}"
echo "ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-}"
echo "HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"

# Run from where you submitted the job
cd "${SLURM_SUBMIT_DIR}"


# mkdir -p logs

# --- CRITICAL ENVIRONMENT SETUP ---
module purge
module load anaconda3

# Activate your AMD-compatible environment (See Step 2 below!)
conda activate beamS_amd       

# export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

# After activation, CONDA_PREFIX points to the env
# export JAVA_HOME="$HOME/.conda/envs/beam/lib/jvm"
# Optional: reduce noisy warnings / threading surprises
export TOKENIZERS_PARALLELISM=false
# "HSA_OVERRIDE" is sometimes needed for older code, but PyTorch usually handles it.
# If you get memory errors, uncomment this:
# export PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:512

export OMP_NUM_THREADS=24



export PYTORCH_ALLOC_CONF=expandable_segments:True

python -u storm/train/run_qwen_multi.py  --config storm/train/config/config_qwen8b.yaml