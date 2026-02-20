#!/bin/bash

echo "Strong scaling: multinode scalability study"

N_STEPS=500
GRID_SIZE_X=15000
GRID_SIZE_Y=15000
OMP_THREADS=14
CORES_PER_NODE=112
NTASKS_PER_NODE=$((CORES_PER_NODE / OMP_THREADS))

for NODES in 1 2 4 8 16; do
    TOTAL_TASKS=$((NODES * NTASKS_PER_NODE))
    
    JOB_NAME="strong_scale_${NODES}n_${TOTAL_TASKS}t"

    sbatch --nodes=${NODES} --ntasks-per-node=${NTASKS_PER_NODE} --cpus-per-task=${OMP_THREADS} --job-name=${JOB_NAME} --export=ALL,GRID_SIZE_X=${GRID_SIZE_X},GRID_SIZE_Y=${GRID_SIZE_Y},N_STEPS=${N_STEPS},OMP_THREADS=${OMP_THREADS},JOB_NAME=${JOB_NAME},TOTAL_TASKS=${TOTAL_TASKS} go_dcgp.sh
done

echo "All Strong Scaling jobs submitted."