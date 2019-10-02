#!/bin/bash

#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --array=0-100
#SBATCH --account=def-pmkim
#SBATCH --job-name=run-notebook
#SBATCH --export=ALL
#SBATCH --mail-type=BEGIN
#SBATCH --mail-user=alexey.strokach@kimlab.org

set -ev

pushd ~/datapkg_input_dir/conda-envs/
mkdir /dev/shm/env
tar -xzf defaults-v2.tar.gz -C /dev/shm/env
source /dev/shm/env/bin/activate
conda-unpack
popd

pushd ~/scratch/workspace/proteinsolver/notebooks/
export START_BATCH_IDX=20000
jupyter nbconvert --to=html --execute --allow-errors --ExecutePreprocessor.timeout=100000000 --ExecutePreprocessor.kernel_name=python3 01-generate_difficult_sudokus.ipynb
popd

