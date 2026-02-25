# Exam report
The aim of this exam project is to parallelise the code to solve the heat equation. The discretisation is done via finite differences in what is called "stencil method".
What was needed to add in the code was mainly in the loop that governs the iterations in time. 
I 
Splitting computation between internal and boundary nodes because we want to optimize for overlapping computation with communication. The pattern is:
1. Start MPI_Isend/Irecv (non-blocking halo exchange)
2. Compute internal nodes (while MPI transfers happen in background)
3. Wait for MPI communication to finish
4. Compute boundary nodes (now that neighbor data is available)



Commands to login into Leonardo
step ssh login agnes.valentini@gmail.com --provisioner cineca-hpc
Manually delete all the keys for Leonardo present in .ssh/known_hosts
ssh avalent5@login.leonardo.cineca.it and press yes to add the key
Copy the data with scp on Leonardo and fix directories


Strong Scaling (15000×15000 fixed):

Tests: "How fast can I solve THIS specific problem with more resources?"
Metric: Speedup = Time(1 node) / Time(N nodes)

Weak Scaling (variable grid):
Here it does not matter the dimension of the problem but just how we manage the scaling, in a way that the ratio dimension of grid^2 / number of nodes remains constant.
Tests: "Can I maintain efficiency when solving proportionally larger problems?"
Metric: Parallel efficiency = Time(1 node) / Time(N nodes) (should ideally stay ~1.0)