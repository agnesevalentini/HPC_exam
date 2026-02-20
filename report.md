Splitting computation between internal and boundary nodes because we want to optimize for overlapping computation with communication. The pattern is:
1. Start MPI_Isend/Irecv (non-blocking halo exchange)
2. Compute internal nodes (while MPI transfers happen in background)
3. Wait for MPI communication to finish
4. Compute boundary nodes (now that neighbor data is available)