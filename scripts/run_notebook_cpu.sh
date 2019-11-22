#!/bin/bash
#SBATCH --array=1-100
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --mem=62G
#SBATCH --account=def-pmkim
#SBATCH --job-name=run-notebook-cpu
#SBATCH --export=ALL
#SBATCH --mail-type=BEGIN
#SBATCH --mail-user=alexey.strokach@kimlab.org
#SBATCH --output=/lustre04/scratch/strokach/run-notebook-cpu-%N-%j.log

unset XDG_RUNTIME_DIR

mkdir ${SLURM_TMPDIR}/env
pushd ~/datapkg_input_dir/conda-envs/
tar -xzf defaults-v008.tar.gz -C ${SLURM_TMPDIR}/env
popd

pushd /dev/shm
ln -s ${SLURM_TMPDIR}/env
popd

source /dev/shm/env/bin/activate
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

