#!/bin/bash
#SBATCH --mem=0
#SBATCH --partition dcgp_usr_prod
#SBATCH -A uTS25_Tornator_0
#SBATCH -t 00:10:00
#SBATCH --exclusive

EXEC=./stencil_template_parallel

# =======================================================
module purge
module load openmpi/4.1.6--gcc--12.2.0

export OMP_NUM_THREADS=$OMP_THREADS
export OMP_PLACES=cores
export OMP_PROC_BIND=close


if [[ ${TOTAL_TASKS} -eq 1 ]]; then
    ${EXEC} -n ${N_STEPS} -x ${GRID_SIZE_X} -y ${GRID_SIZE_Y} -p 1
else
    mpirun -np ${TOTAL_TASKS} ${EXEC} -n ${N_STEPS} -x ${GRID_SIZE_X} -y ${GRID_SIZE_Y} -p 1
fi
