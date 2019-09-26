#!/bin/bash
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --array=1-50
#SBATCH --account=def-pmkim
#SBATCH --job-name=reserve-node
#SBATCH --export=ALL
#SBATCH --mail-type=BEGIN
#SBATCH --mail-user=alexey.strokach@kimlab.org
#SBATCH --output=/scratch/p/pmkim/strokach/reserve-node.log
#SBATCH --comment=interactive

/scratch/p/pmkim/strokach/.conda/envs/ci-datapkg-adjacency-net-v2-test81/bin/python 01-generate_difficult_sudokus.py --host-name "$(hostname --short)-$SLURM_ARRAY_TASK_ID"
