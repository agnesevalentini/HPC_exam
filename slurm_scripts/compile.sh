#!/bin/bash
#SBATCH --job-name=build
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:10:00
#SBATCH --partition=dcgp_usr_prod
#SBATCH -A uTS25_Tornator_0


module load openmpi/4.1.6--gcc--12.2.0
make