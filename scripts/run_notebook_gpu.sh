#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:v100l:1
#SBATCH --account=rrg-pmkim
#SBATCH --job-name=reserve-gpu-node
#SBATCH --export=ALL
#SBATCH --mail-type=BEGIN
#SBATCH --mail-user=alexey.strokach@kimlab.org
#SBATCH --output=/scratch/strokach/reserve-gpu-node-%N-%j.log

unset XDG_RUNTIME_DIR

set -ev

mkdir -p ${SLURM_TMPDIR}/conda/envs/default
tar -xzf ~/datapkg-data-dir/conda-envs/default/default-v33.tar.gz -C ${SLURM_TMPDIR}/conda/envs/default

mkdir -p /dev/shm/conda/envs/
pushd /dev/shm/conda/envs/
ln -s ${SLURM_TMPDIR}/conda/envs/default
popd

source /dev/shm/conda/envs/default/bin/activate
conda-unpack
# source /dev/shm/env/bin/deactivate

# conda activate base
# jupyter lab --ip 0.0.0.0 --no-browser

NOTEBOOK_STEM=$(basename ${NOTEBOOK_PATH%%.ipynb})
NOTEBOOK_DIR=$(dirname ${NOTEBOOK_PATH})
OUTPUT_TAG="${SLURM_JOB_NODELIST}-${SLURM_JOB_ID}-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"

mkdir -p "${NOTEBOOK_DIR}/${NOTEBOOK_STEM}"
papermill --no-progress-bar --log-output --kernel python3 "${NOTEBOOK_PATH}" "${NOTEBOOK_DIR}/${NOTEBOOK_STEM}-${OUTPUT_TAG}.ipynb"

# sleep 72h
