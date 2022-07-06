#!/bin/bash
#SBATCH -J bsm
#SBATCH -n 1
#SBATCH --mem-per-cpu 7gb
#SBATCH -C centos7
#SBATCH -t 12:00:00 # max time limit of 1/2 day, check time limit with sinfo
#SBATCH -p sched_mit_hill # partition from which slurm will select the requested amt of nodes

cd $SLURM_SUBMIT_DIR

echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Working Directory = $(pwd)"
echo ""
echo "Number of Nodes Allocated = $SLURM_JOB_NUM_NODES"
echo "Number of Tasks Allocated = $SLURM_NTASKS"

module load python/3.9.4
module list

noises=(0.2 0.3 0.4)

date
for noise in ${noises[@]};
do
    python -u beadspringsim.py --perturb rand --noise $noise --outdir /nobackup1/iamlll > bsmodel.out & 
done
wait
date
