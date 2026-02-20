#!/bin/bash

echo "Weak scaling: multinode scalability study with constant workload per resource"

N_STEPS=500        
TASKS_PER_NODE=8   
OMP_THREADS=14
CPUS_PER_TASK=${OMP_THREADS}
LOCAL_SIZE=4000
GRID_SIZE_X=${LOCAL_SIZE}
GRID_SIZE_Y=${LOCAL_SIZE}          

for NODES in 1 2 4 8 16; do
    TOTAL_TASKS=$((NODES * TASKS_PER_NODE))

    GRID_SIZE_X=$((NODES * LOCAL_SIZE))

    if [ $NODES -eq 2 ]; then
        GRID_SIZE_X=$(( (LOCAL_SIZE) * 2 ))
        GRID_SIZE_Y=$(( LOCAL_SIZE ))
    elif [ $NODES -eq 4 ]; then
        GRID_SIZE_X=$(( LOCAL_SIZE * 2 ))
        GRID_SIZE_Y=$(( LOCAL_SIZE * 2 ))
    elif [ $NODES -eq 8 ]; then
        GRID_SIZE_X=$(( LOCAL_SIZE * 4 ))
        GRID_SIZE_Y=$(( LOCAL_SIZE * 2 ))
    elif [ $NODES -eq 16 ]; then
        GRID_SIZE_X=$(( LOCAL_SIZE * 4 ))
        GRID_SIZE_Y=$(( LOCAL_SIZE * 4 ))
    fi    

    JOB_NAME="weak_scale_${NODES}n_${TOTAL_TASKS}t"

    sbatch --nodes=${NODES} \
           --ntasks=${TOTAL_TASKS} \
           --ntasks-per-node=${TASKS_PER_NODE} \
           --cpus-per-task=${CPUS_PER_TASK} \
           --job-name=${JOB_NAME} \
           --export=ALL,GRID_SIZE_X=${GRID_SIZE_X},GRID_SIZE_Y=${GRID_SIZE_Y},N_STEPS=${N_STEPS},OMP_THREADS=${OMP_THREADS},JOB_NAME=${JOB_NAME},TOTAL_TASKS=${TOTAL_TASKS} \
           go_dcgp.sh

    echo "Submitting job with ${NODES} nodes, ${TOTAL_TASKS} total tasks, grid size ${GRID_SIZE_X}x${GRID_SIZE_Y}"
done

echo "All Weak Scaling jobs submitted."