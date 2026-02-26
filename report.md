# Exam report
The aim of this exam project is to parallelise the code to solve the heat equation. The discretisation is done via finite differences in what is called "stencil method".
The matrix where the heat equation is solved is going to be divided into tasks for MPI and each task is split into different threads using OpenMP. This is what is called hybrid optimization.

What was requested to add in the code was mainly in the loop that governs the iterations in time.
It starts with injecting the energy in the grid points randomly chosen. The grid is decomposed across MPI processes and each process allocates memory for both the grid and the halos. The halos are needed because we want to recompose the matrix after having parallelised the computation. Each process needs boundary data from its neighbours to compute the 5 point stencil, so there are north and south halos where data from halos is added in a row at the top and one at the bottom of the original matrix so that we can access it directly with a pointer. For east and west halos the data from the contiguous processes (tasks) is copied into separate buffers. 
Then we are ready for the halo data exchange, which happens to be asynchronous using non-blocking MPI communication: the north, south, east and west receives neighbours are posted and also the sends using MPI_Irecv and MPI_Isend. Communication happens in the background while computation proceeds. 
While MPI communication is in transit, update_plane_internal() computes all internal grid points (excludes boundaries and halos) and uses a 5-point stencil: each point is updated based on itself and its 4 neighbors. This achieves computation-communication overlap for better performance.
MPI_Waitall() blocks until all receive operations complete. Data from west/east buffers is copied into the halo regions whereas north/south data is already in place. After halo data arrives, update_plane_boundary() computes all boundary grid points using the fresh halo data. It also applies periodic boundary conditions if enabled.
MPI_Waitall() on send requests ensures buffers can be reused. Plane indices are swapped (current = !current) for the next iteration to enable using double-buffering.

For the OpenMP part, it was used in the for loops with schedule static for a balanced workload distribution and in the energy reduction with collapse to merge nested loops into one iteration space and reduction to gain thread-safe accumulation.
Besides, to get better parallelism, MPI_Thread_funneled enables only the main thread to call MPI functions; the splitting of the planes between old and new prevents race conditions.

Time measurement were added in the code with MPI_Wtime() to compute the computation, communication and waiting time. 
The time was extracted by using MPI_Reduce and doing a mean along all the tasks/processes.

The tests that must be conducted are weak scaling and strong scaling, and a thread scaling as well. 
For the thread/openmp scaling, it was used 1 MPI task and the number of threads was scaled in a series like 1,2,4,8,16,32,56,84,112 so to fill the entire Leonardo node. The test was done with a grid of dimension 15000x15000 with 1000 steps. 
Then, for the MPI scaling 8 MPI tasks and 14 openmp threads per node were chosen. The strong scaling is useful to understand how fast can I solve THIS specific problem with more resources and the speedup is Time(1 node) / Time(N nodes). The grid is again 15000x15000 with 1000 steps and the number of nodes is scaled with the series 1,2,4,8,16,32. As for the weak scaling, it does not matter the dimension of the problem but just how we manage the scaling, in a way that the ratio dimension of grid^2 / number of nodes remains constant.
It is useful to understant if the efficiency can be maintained when solving proportionally larger problems. The efficiency is Time(1 node) / Time(N nodes) (should ideally stay ~1.0). In the test, the grid grows from 4000x4000 up to 16000x160000 by changing the dimension and the 




## Commands to login into Leonardo
```step ssh login agnes.valentini@gmail.com --provisioner cineca-hpc```

Manually delete all the keys for Leonardo present in .ssh/known_hosts

```ssh avalent5@login.leonardo.cineca.it -y```

Copy the data with scp on Leonardo and fix directories
