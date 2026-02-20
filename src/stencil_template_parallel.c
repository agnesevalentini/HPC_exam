
/*
 *
 *  mysizex   :   local x-extendion of your patch
 *  mysizey   :   local y-extension of your patch
 *
 *  Optimization Strategy:
 *  - Split stencil update into internal and boundary nodes
 *  - Internal nodes are computed while MPI communication is in transit
 *  - Boundary nodes are computed after halo exchange completes
 *  - This overlaps computation with communication for better performance
 *
 */


#include "stencil_template_parallel.h"


// Communication-computation overlap optimization implemented:
// 1. Start non-blocking MPI_Isend/Irecv for halo exchange
// 2. Compute internal nodes (don't need neighbor data) - OVERLAPS with communication
// 3. Wait for halo data to arrive (MPI_Waitall on receives)
// 4. Compute boundary nodes (now have fresh halo data)

// ------------------------------------------------------------------
// ------------------------------------------------------------------

int main(int argc, char **argv)
{
  MPI_Comm myCOMM_WORLD;  
  int  Rank, Ntasks;
  int neighbours[4];

  int  Niterations;
  int  periodic;
  vec2_t S, N;
  
  int      Nsources;
  int      Nsources_local;
  vec2_t  *Sources_local;
  double   energy_per_source;

  plane_t   planes[2];  
  buffers_t buffers[2];
  
  int output_energy_stat_perstep;
  
  /* initialize MPI envrionment */
  {
    int level_obtained;
    
    // NOTE: change MPI_FUNNELED if appropriate
    // MPI_thread_funneled means that only the main thread will make MPI calls
    MPI_Init_thread( &argc, &argv, MPI_THREAD_FUNNELED, &level_obtained );
    if ( level_obtained < MPI_THREAD_FUNNELED ) {
      printf("MPI_thread level obtained is %d instead of %d\n",
	     level_obtained, MPI_THREAD_FUNNELED );
      MPI_Finalize();
      exit(1); }
    
    MPI_Comm_rank(MPI_COMM_WORLD, &Rank); 
    MPI_Comm_size(MPI_COMM_WORLD, &Ntasks);
    MPI_Comm_dup(MPI_COMM_WORLD, &myCOMM_WORLD);
  }
  
  
  /* argument checking and setting */
  int ret = initialize ( &myCOMM_WORLD, Rank, Ntasks, argc, argv, &S, &N, &periodic, &output_energy_stat_perstep,
			 neighbours, &Niterations,
			 &Nsources, &Nsources_local, &Sources_local, &energy_per_source,
			 &planes[0], &buffers[0] );

  if ( ret )
    {
      printf("task %d is opting out with termination code %d\n",
	     Rank, ret );
      
      MPI_Finalize();
      return 0;
    }
  
  
  /* ============================================================
   * NUMA Optimization: Touch-by-all first-touch policy
   * Each OpenMP thread touches its portion of memory using
   * the same parallel decomposition as the computation loops.
   * This ensures pages are allocated on the NUMA node closest
   * to the thread that will use them, minimizing remote accesses
   * ============================================================ */
  // dunno if we should use it
  initialize_first_touch( &planes[OLD] );
  initialize_first_touch( &planes[NEW] );
  
  
  int current = OLD;
  double t1 = MPI_Wtime();   /* take wall-clock time */
  
  // Timing variables for performance analysis
  double total_communication_time = 0.0; 
  double total_computation_time = 0.0; 
  double total_waiting_time = 0.0;
  double total_energy_injection_time = 0.0;
  //double total_energy_stat_time = 0.0;
  
  // MPI Request arrays for non-blocking communication
  MPI_Request send_requests[4];
  MPI_Request recv_requests[4];
  int num_send_requests = 0;
  int num_recv_requests = 0;
  
  // Save initial state (iteration 0) for the plot of the grid evolution
  //save_grid_snapshot(0, &planes[current], S, N, Rank, &myCOMM_WORLD);
  
  for (int iter = 0; iter < Niterations; ++iter)
    
    {
      /* new energy from sources */
      double energy_start = MPI_Wtime();
      inject_energy( periodic, Nsources_local, Sources_local, energy_per_source, &planes[current], N );
      total_energy_injection_time += MPI_Wtime() - energy_start;


      /* -------------------------------------- */

      // [A] fill the buffers, and/or make the buffers' pointers pointing to the correct position
      uint sizex = planes[current].size[_x_] + 2;
      uint sizey = planes[current].size[_y_] + 2;
      double *data = planes[current].data;

      #define IDX(i, j) ((j) * sizex + (i))

      // For NORTH: send top row (j=1), receive into j=0
      // For SOUTH: send bottom row (j=sizey-2), receive into j=sizey-1
      // For EAST: send rightmost column (i=sizex-2), receive into i=sizex-1
      // For WEST: send leftmost column (i=1), receive into i=0

      // NORTH and SOUTH buffers can point directly to the rows
      buffers[SEND][NORTH] = &data[IDX(0, 1)];        // first internal row to send
      buffers[RECV][NORTH] = &data[IDX(0, 0)];        // top halo row to receive into
      buffers[SEND][SOUTH] = &data[IDX(0, sizey - 2)];  // last internal row to send
      buffers[RECV][SOUTH] = &data[IDX(0, sizey - 1)];  // bottom halo row to receive into

      // EAST and WEST need actual buffers (data is strided)
      // Fill EAST buffer (rightmost column)
      if (neighbours[EAST] != MPI_PROC_NULL) {
        #pragma GCC unroll 4
          for (uint j = 0; j < sizey; j++) {
              buffers[SEND][EAST][j] = data[IDX(sizex - 2, j)];
          }
      }

      // Fill WEST buffer (leftmost column)
      if (neighbours[WEST] != MPI_PROC_NULL) {
        #pragma GCC unroll 4
          for (uint j = 0; j < sizey; j++) {
              buffers[SEND][WEST][j] = data[IDX(1, j)];
          }
      }

      #undef IDX
    
      // [B] perform the halo communications
      //     Using non-blocking Isend / Irecv to enable overlapping communication with computation
      
      double communication_start = MPI_Wtime();  // here we set the start time for communication
      num_send_requests = 0;
      num_recv_requests = 0;
      
      // Post all receives first 
      // NORTH receive
      if (neighbours[NORTH] != MPI_PROC_NULL) {
          MPI_Irecv(buffers[RECV][NORTH], sizex, MPI_DOUBLE, neighbours[NORTH], 1,
                   myCOMM_WORLD, &recv_requests[num_recv_requests++]);
      }
      
      // SOUTH receive
      if (neighbours[SOUTH] != MPI_PROC_NULL) {
          MPI_Irecv(buffers[RECV][SOUTH], sizex, MPI_DOUBLE, neighbours[SOUTH], 0,
                   myCOMM_WORLD, &recv_requests[num_recv_requests++]);
      }
      
      // EAST receive
      if (neighbours[EAST] != MPI_PROC_NULL) {
          MPI_Irecv(buffers[RECV][EAST], sizey, MPI_DOUBLE, neighbours[EAST], 3,
                   myCOMM_WORLD, &recv_requests[num_recv_requests++]);
      }
      
      // WEST receive
      if (neighbours[WEST] != MPI_PROC_NULL) {
          MPI_Irecv(buffers[RECV][WEST], sizey, MPI_DOUBLE, neighbours[WEST], 2,
                   myCOMM_WORLD, &recv_requests[num_recv_requests++]);
      }
      
      // Post all sends after
      // NORTH send
      if (neighbours[NORTH] != MPI_PROC_NULL) {
          MPI_Isend(buffers[SEND][NORTH], sizex, MPI_DOUBLE, neighbours[NORTH], 0,
                   myCOMM_WORLD, &send_requests[num_send_requests++]);
      }
      
      // SOUTH send
      if (neighbours[SOUTH] != MPI_PROC_NULL) {
          MPI_Isend(buffers[SEND][SOUTH], sizex, MPI_DOUBLE, neighbours[SOUTH], 1,
                   myCOMM_WORLD, &send_requests[num_send_requests++]);
      }
      
      // EAST send
      if (neighbours[EAST] != MPI_PROC_NULL) {
          MPI_Isend(buffers[SEND][EAST], sizey, MPI_DOUBLE, neighbours[EAST], 2,
                   myCOMM_WORLD, &send_requests[num_send_requests++]);
      }
      
      // WEST send
      if (neighbours[WEST] != MPI_PROC_NULL) {
          MPI_Isend(buffers[SEND][WEST], sizey, MPI_DOUBLE, neighbours[WEST], 3,
                   myCOMM_WORLD, &send_requests[num_send_requests++]);
      }
      
      double communication_end = MPI_Wtime();
      total_communication_time += (communication_end - communication_start);

      /* --------------------------------------  */
      /* STEP 1: Update INTERNAL grid points (overlaps with communication) */
      /* Internal points don't depend on halo data, so compute them while data is in transit */
      
      double internal_start = MPI_Wtime();
      update_plane_internal( N, &planes[current], &planes[!current] );
      double internal_end = MPI_Wtime();
      total_computation_time += (internal_end - internal_start);
      
      /* --------------------------------------  */
      /* STEP 2: Wait for halo data to arrive */
      
      double waiting_start = MPI_Wtime();
      
      // Wait for all receives to complete
      if (num_recv_requests > 0) {
          MPI_Waitall(num_recv_requests, recv_requests, MPI_STATUSES_IGNORE);
      }
      
      // [C] Copy halo data from buffers (EAST and WEST)
      // NORTH and SOUTH: data was received directly into halo rows (no copy needed)
      #define IDX(i, j) ((j) * sizex + (i))
      
      if (neighbours[EAST] != MPI_PROC_NULL) {
          for (uint j = 0; j < sizey; j++) {
              data[IDX(sizex - 1, j)] = buffers[RECV][EAST][j];
          }
      }

      if (neighbours[WEST] != MPI_PROC_NULL) {
          for (uint j = 0; j < sizey; j++) {
              data[IDX(0, j)] = buffers[RECV][WEST][j];
          }
      }
      
      #undef IDX
      
      double waiting_end = MPI_Wtime();
      total_waiting_time += (waiting_end - waiting_start);
      
      /* --------------------------------------  */
      /* STEP 3: Update BOUNDARY grid points (requires fresh halo data) */
      
      double boundary_start = MPI_Wtime();
      update_plane_boundary( periodic, N, &planes[current], &planes[!current] );
      double boundary_end = MPI_Wtime();
      total_computation_time += (boundary_end - boundary_start);
      
      /* Wait for all sends to complete (to ensure buffers can be reused) */
      if (num_send_requests > 0) {
          MPI_Waitall(num_send_requests, send_requests, MPI_STATUSES_IGNORE);
      }


      /* output if needed */
      //if ( output_energy_stat_perstep ) {
      //    double stat_start = MPI_Wtime();
	    //    output_energy_stat ( iter, &planes[!current], (iter+1) * Nsources*energy_per_source, Rank, &myCOMM_WORLD );
      //    total_energy_stat_time += MPI_Wtime() - stat_start;
      //}
      
      /* Save grid snapshots at specific iterations for visualization */
      // Save at iterations: 10, 50, 100, 250, 500, and every 250 iterations after
      //if (iter == 9 || iter == 49 || iter == 99 || iter == 249 || iter == 499 || 
      //    (iter >= 250 && (iter+1) % 250 == 0)) {
      //    snapshot_start = MPI_Wtime();
      //    //save_grid_snapshot(iter+1, &planes[!current], S, N, Rank, &myCOMM_WORLD);
      //    total_snapshot_time += MPI_Wtime() - snapshot_start;
      //} 
      
      
      /* swap plane indexes for the new iteration */
      current = !current;
      
    }
  
  // Save final state
  //snapshot_start = MPI_Wtime();
  //save_grid_snapshot(Niterations, &planes[!current], S, N, Rank, &myCOMM_WORLD);
  //total_snapshot_time += MPI_Wtime() - snapshot_start;

  t1 = MPI_Wtime() - t1; // compute total execution time

  output_energy_stat ( -1, &planes[!current], Niterations * Nsources*energy_per_source, Rank, &myCOMM_WORLD );

  double computation_time_mean, communication_time_mean, waiting_time_mean, energy_injection_time_mean;

  MPI_Reduce(&total_computation_time, &computation_time_mean, 1, MPI_DOUBLE, MPI_SUM, 0, myCOMM_WORLD);
  MPI_Reduce(&total_communication_time, &communication_time_mean, 1, MPI_DOUBLE, MPI_SUM, 0, myCOMM_WORLD);
  MPI_Reduce(&total_waiting_time, &waiting_time_mean, 1, MPI_DOUBLE, MPI_SUM, 0, myCOMM_WORLD);
  MPI_Reduce(&total_energy_injection_time, &energy_injection_time_mean, 1, MPI_DOUBLE, MPI_SUM, 0, myCOMM_WORLD);

  // add the code to print the time to post process them
  if (Rank == 0 || Ntasks == 1) {
    const char *job_name = getenv("JOB_NAME");
    
  }


  /*
  // Print timing statistics to check the overlap of communication and computation
  if (Rank == 0) {
      printf("\n=== Performance Statistics (Communication-Computation Overlap) ===\n");
      fflush(stdout);
      printf("Total execution time:      %f seconds (100.00%%)\n", t1);
      printf("-----------------------------------------------------------\n");
      printf("Computation time:          %f seconds (%.2f%%)\n", 
             total_computation_time, 100.0 * total_computation_time / t1);
      printf("  Communication setup:     %f seconds (%.2f%%)\n", 
             total_communication_time, 100.0 * total_communication_time / t1);
      printf("  Waiting for halos:       %f seconds (%.2f%%)\n", 
             total_waiting_time, 100.0 * total_waiting_time / t1);
      printf("Energy injection:          %f seconds (%.2f%%)\n", 
             total_energy_injection_time, 100.0 * total_energy_injection_time / t1);
      //printf("Grid snapshots (I/O):      %f seconds (%.2f%%)\n", 
      //       total_snapshot_time, 100.0 * total_snapshot_time / t1);
      printf("Energy statistics:         %f seconds (%.2f%%)\n", 
             total_energy_stat_time, 100.0 * total_energy_stat_time / t1);
      printf("-----------------------------------------------------------\n");
      printf("Overlap efficiency:        %.2f%%\n", 
             100.0 * total_computation_time / (total_computation_time + total_waiting_time + total_communication_time));
      printf("===============================================================\n\n");
      fflush(stdout);
  }
  */
  
  memory_release( buffers, planes ); // free all allocated memory
  
   
  MPI_Finalize();   // finalize MPI environment
  return 0;
}


/* ==========================================================================
   =                                                                        =
   =   initialization                                                       =
   ========================================================================== */


uint simple_factorization( uint, int *, uint ** );

int initialize_sources( int       ,
			int       ,
			MPI_Comm  *,
			uint      [2],
			int       ,
			int      *,
			vec2_t  ** );


int memory_allocate ( const int       *,
		      const vec2_t     ,
		            buffers_t *,
		            plane_t   * );
		      

int initialize ( MPI_Comm *Comm,
		 int      Me,                  // the rank of the calling process
		 int      Ntasks,              // the total number of MPI ranks
		 int      argc,                // the argc from command line
		 char   **argv,                // the argv from command line
		 vec2_t  *S,                   // the size of the plane
		 vec2_t  *N,                   // two-uint array defining the MPI tasks' grid
		 int     *periodic,            // periodic-boundary tag
		 int     *output_energy_stat,  // whether to output energy statistics at each step
		 int     *neighbours,          // four-int array that gives back the neighbours of the calling task
		 int     *Niterations,         // how many iterations
		 int     *Nsources,            // how many heat sources
		 int     *Nsources_local,      // how many heat sources are local to the calling task
		 vec2_t **Sources_local,       // the array of heat sources local to the calling task
		 double  *energy_per_source,   // how much heat per source
		 plane_t *planes,              // the two planes (OLD and NEW) that we will use for the computation
		 buffers_t *buffers            // the communication buffers
		 )
{
  int halt = 0;
  int ret;
  int verbose = 0;
  
  // ··································································
  // set default values

  (*S)[_x_]         = 10000;
  (*S)[_y_]         = 10000;
  *periodic         = 0;
  *Nsources         = 4;
  *Nsources_local   = 0;
  *Sources_local    = NULL;
  *Niterations      = 1000;
  *energy_per_source = 1.0;

  if ( planes == NULL ) {
    // manage the situation
    printf("Task %d: planes is NULL\n", Me);
    return 1;
  }

  planes[OLD].size[0] = 0; // initializing width of the plane to 0
  planes[OLD].size[1] = 0; // initializing height of the plane to 0 both for old and new planes
  planes[NEW].size[0] = 0; // in this way we suppose that the grid could be rectangular
  planes[NEW].size[1] = 0;
  
  for ( int i = 0; i < 4; i++ )  // initializing neighbours to MPI_PROC_NULL, meaning that by default we suppose that there are no neighbours
    neighbours[i] = MPI_PROC_NULL;

  for ( int b = 0; b < 2; b++ )  // initializing communication buffers to NULL
    for ( int d = 0; d < 4; d++ )
      buffers[b][d] = NULL;
  
  // ··································································
  // process the command line
  // 
  while ( 1 )
  {
    int opt;
    while((opt = getopt(argc, argv, ":hx:y:e:E:n:o:p:v:")) != -1)
      {
	switch( opt )
	  {
	  case 'x': (*S)[_x_] = (uint)atoi(optarg);
	    break;

	  case 'y': (*S)[_y_] = (uint)atoi(optarg);
	    break;

	  case 'e': *Nsources = atoi(optarg);
	    break;

	  case 'E': *energy_per_source = atof(optarg);
	    break;

	  case 'n': *Niterations = atoi(optarg);
	    break;

	  case 'o': *output_energy_stat = (atoi(optarg) > 0);
	    break;

	  case 'p': *periodic = (atoi(optarg) > 0);
	    break;

	  case 'v': verbose = atoi(optarg);
	    break;

	  case 'h': {
	    if ( Me == 0 )   // only the master task prints the help message
	      printf( "\nvalid options are ( values btw [] are the default values ):\n"
		      "-x    x size of the plate [10000]\n"
		      "-y    y size of the plate [10000]\n"
		      "-e    how many energy sources on the plate [4]\n"
		      "-E    how many energy sources on the plate [1.0]\n"
		      "-n    how many iterations [1000]\n"
		      "-p    whether periodic boundaries applies  [0 = false]\n\n"
          "-o    whether to output energy statistics at each step [0 = false]\n"
          "-v    verbosity level [0]\n\n"
		      );
	    halt = 1; }
	    break;
	    
	    
	  case ':': printf( "option -%c requires an argument\n", optopt);
	    break;
	    
	  case '?': printf(" -------- help unavailable ----------\n");
	    break;
	  }
      }

    if ( opt == -1 )
      break;
  }

  if ( halt )
    return 1;
  
  
  // ··································································
  /*
   * here we should check for all the params being meaningful
   *
   */
  if ( (*S)[_x_] < 1 || (*S)[_y_] < 1 )
    {
      printf("error: invalid grid size\n");
      return 1;
    }

  if ( *Niterations < 1 )
    {
      printf("error: invalid number of iterations\n");
      return 1;
    }

  if ( *Nsources < 1 )
    {
      printf("error: invalid number of heat sources\n");
      return 1;
    }

  if ( *energy_per_source < 0 )
    {
      printf("error: invalid energy per source\n");
      return 1;
    }

  if ( *output_energy_stat < 0 )
    {
      printf("error: invalid output energy stat flag\n");
      return 1;
    }

  if ( *periodic < 0 )
    {
      printf("error: invalid periodic flag\n");
      return 1;
    }
  

  
  // ··································································
  /*
   * find a suitable domain decomposition
   * very simple algorithm, you may want to
   * substitute it with a better one
   *
   * the plane Sx x Sy will be solved with a grid
   * of Nx x Ny MPI tasks
   */

  vec2_t Grid;

  // the formfactor is the ratio between the two dimensions of the grid, it is used to decide whether to decompose in 1D or 2D
  double formfactor = ((*S)[_x_] >= (*S)[_y_] ? (double)(*S)[_x_]/(*S)[_y_] : (double)(*S)[_y_]/(*S)[_x_] );
  
  // if the number of tasks is less than or equal to the formfactor, we decompose in 1D, otherwise we decompose in 2D
  int    dimensions = 2 - (Ntasks <= ((int)formfactor+1) );

  
  if ( dimensions == 1 )
    {
      if ( (*S)[_x_] >= (*S)[_y_] )
	Grid[_x_] = Ntasks, Grid[_y_] = 1;
      else
	Grid[_x_] = 1, Grid[_y_] = Ntasks;
    }
  else
    {
      int   Nf;
      uint *factors;
      uint  first = 1;
      ret = simple_factorization( Ntasks, &Nf, &factors );
      
      for ( int i = 0; (i < Nf) && ((Ntasks/first)/first > formfactor); i++ )
	first *= factors[i];

      if ( (*S)[_x_] > (*S)[_y_] )
	Grid[_x_] = Ntasks/first, Grid[_y_] = first;
      else
	Grid[_x_] = first, Grid[_y_] = Ntasks/first;
    }

  (*N)[_x_] = Grid[_x_];
  (*N)[_y_] = Grid[_y_];
  

  // ··································································
  // my coordinates in the grid of processors
  // Me is the rank of the current calling process
  int X = Me % Grid[_x_];
  int Y = Me / Grid[_x_];

  // ··································································
  // find my neighbours
  //
  // condition ? value_if_true : value_if_false
  // processes go from 0 to Ntasks-1

  if ( Grid[_x_] > 1 )
    {  
      if ( *periodic ) {       
	neighbours[EAST]  = Y*Grid[_x_] + (Me + 1 ) % Grid[_x_];
	neighbours[WEST]  = (X%Grid[_x_] > 0 ? Me-1 : (Y+1)*Grid[_x_]-1); }
  // neighbours[WEST] = (X > 0 ? Me-1 : (Y+1)*Grid[_x_]-1); }
      
      else {
	neighbours[EAST]  = ( X < Grid[_x_]-1 ? Me+1 : MPI_PROC_NULL );
	neighbours[WEST]  = ( X > 0 ? (Me-1)%Ntasks : MPI_PROC_NULL ); }  
  // neighbours[WEST]  = ( X > 0 ? Me-1 : MPI_PROC_NULL ); }
    }

  if ( Grid[_y_] > 1 )
    {
      if ( *periodic ) {      
	neighbours[NORTH] = (Ntasks + Me - Grid[_x_]) % Ntasks;
	neighbours[SOUTH] = (Ntasks + Me + Grid[_x_]) % Ntasks; }

      else {    
	neighbours[NORTH] = ( Y > 0 ? Me - Grid[_x_]: MPI_PROC_NULL );
	neighbours[SOUTH] = ( Y < Grid[_y_]-1 ? Me + Grid[_x_] : MPI_PROC_NULL ); }
    }

  // ··································································
  // the size of my patch
  //

  /*
   * every MPI task determines the size sx x sy of its own domain
   * REMIND: the computational domain will be embedded into a frame
   *         that is (sx+2) x (sy+2)
   *         the outern frame will be used for halo communication or
   *         to apply the boundary conditions
   */
  
  vec2_t mysize;
  uint s = (*S)[_x_] / Grid[_x_];
  uint r = (*S)[_x_] % Grid[_x_];
  mysize[_x_] = s + (X < r);
  s = (*S)[_y_] / Grid[_y_];
  r = (*S)[_y_] % Grid[_y_];
  mysize[_y_] = s + (Y < r);

  planes[OLD].size[0] = mysize[0];
  planes[OLD].size[1] = mysize[1];
  planes[NEW].size[0] = mysize[0];
  planes[NEW].size[1] = mysize[1];
  

  if ( verbose > 0 )
    {
      if ( Me == 0 ) {
	printf("Tasks are decomposed in a grid %d x %d\n\n",
		 Grid[_x_], Grid[_y_] );
	fflush(stdout);
      }

      MPI_Barrier(*Comm);
      
      for ( int t = 0; t < Ntasks; t++ )
	{
	  if ( t == Me )
	    {
	      printf("Task %4d :: "
		     "\tgrid coordinates : %3d, %3d\n"
		     "\tneighbours: N %4d    E %4d    S %4d    W %4d\n",
		     Me, X, Y,
		     neighbours[NORTH], neighbours[EAST],
		     neighbours[SOUTH], neighbours[WEST] );
	      fflush(stdout);
	    }

	  MPI_Barrier(*Comm);
	}
      
    }

  
  // ··································································
  // allocate the needed memory
  //
  ret = memory_allocate(neighbours, mysize, buffers, planes );
  if ( ret != 0 ) {
    printf("Task %d: memory allocation failed\n", Me);
    return ret;
  }
  

  // ··································································
  // allocate the heat sources
  //
  ret = initialize_sources( Me, Ntasks, Comm, mysize, *Nsources, Nsources_local, Sources_local );
  
  
  return 0;  
}


uint simple_factorization( uint A, int *Nfactors, uint **factors )
/*
 * rought factorization;
 * assumes that A is small, of the order of <~ 10^5 max,
 * since it represents the number of tasks
 #
 */
{
  int N = 0;
  int f = 2;
  uint _A_ = A;

  while ( f < A )
    {
      while( _A_ % f == 0 ) {
	N++;
	_A_ /= f; }

      f++;
    }

  *Nfactors = N;
  uint *_factors_ = (uint*)malloc( N * sizeof(uint) );

  N   = 0;
  f   = 2;
  _A_ = A;

  while ( f < A )
    {
      while( _A_ % f == 0 ) {
	_factors_[N++] = f;
	_A_ /= f; }
      f++;
    }

  *factors = _factors_;
  return 0;
}

// initialize heat sources by randomly assigning them to some MPI tasks
int initialize_sources( int       Me,
			int       Ntasks,
			MPI_Comm *Comm,
			vec2_t    mysize,
			int       Nsources,
			int      *Nsources_local,
			vec2_t  **Sources )

{

  srand(time(NULL) ^ Me);
  int *tasks_with_sources = (int*)malloc( Nsources * sizeof(int) );
  
  if ( Me == 0 )
    {
      for ( int i = 0; i < Nsources; i++ )
	tasks_with_sources[i] = rand() % Ntasks;
    }
  
  // broadcast the tasks that have sources to all the processes
  MPI_Bcast( tasks_with_sources, Nsources, MPI_INT, 0, *Comm );

  int nlocal = 0;
  for ( int i = 0; i < Nsources; i++ )
    nlocal += (tasks_with_sources[i] == Me);
  *Nsources_local = nlocal;
  
  if ( nlocal > 0 )
    {
      vec2_t * restrict helper = (vec2_t*)malloc( nlocal * sizeof(vec2_t) );      
      for ( int s = 0; s < nlocal; s++ )
	{
	  helper[s][_x_] = 1 + rand() % mysize[_x_];
	  helper[s][_y_] = 1 + rand() % mysize[_y_];
	}

      *Sources = helper;
    }
  
  free( tasks_with_sources );

  return 0;
}



int memory_allocate ( const int       *neighbours  ,
		      const vec2_t     N           ,
		            buffers_t *buffers_ptr ,
		            plane_t   *planes_ptr
		      )

{
    /*
      here you allocate the memory buffers that you need to
      (i)  hold the results of your computation
      (ii) communicate with your neighbours

      The memory layout that I propose to you is as follows:

      (i) --- calculations
      you need 2 memory regions: the "OLD" one that contains the
      results for the step (i-1)th, and the "NEW" one that will contain
      the updated results from the step ith.

      Then, the "NEW" will be treated as "OLD" and viceversa.

      These two memory regions are indexed by *plane_ptr:

      plane_ptr[0] ==> the "OLD" region
      plane_ptr[1] ==> the "NEW" region


      (ii) --- communications

      you may need two buffers (one for sending and one for receiving)
      for each one of your neighbours, that are at most 4:
      north, south, east and west.      

      To them you need to communicate at most mysizex or mysizey
      double data.

      These buffers are indexed by the buffer_ptr pointer so
      that 

      (*buffers_ptr)[SEND][ {NORTH,...,WEST} ] = .. some memory regions
      (*buffers_ptr)[RECV][ {NORTH,...,WEST} ] = .. some memory regions
      
      --->> Of course you can change this layout as you prefer
      
     */

  if (planes_ptr == NULL ) {
    // an invalid pointer has been passed
    // manage the situation
    printf("Error: planes_ptr is NULL\n");
    return 1;
  }

  if (buffers_ptr == NULL ) {
    // an invalid pointer has been passed
    // manage the situation
    printf("Error: buffers_ptr is NULL\n");
    return 1;
  }

  // ··················································
  // allocate memory for data
  // we allocate the space needed for the plane plus a contour frame
  // that will contains data form neighbouring MPI tasks
  unsigned int frame_size = (planes_ptr[OLD].size[_x_]+2) * (planes_ptr[OLD].size[_y_]+2);

  planes_ptr[OLD].data = (double*)malloc( frame_size * sizeof(double) );
  if ( planes_ptr[OLD].data == NULL ) {
    printf("Malloc fail: planes_prt[OLD].data is NULL\n");
    return 1;
  }
  memset ( planes_ptr[OLD].data, 0, frame_size * sizeof(double) );


  planes_ptr[NEW].data = (double*)malloc( frame_size * sizeof(double) );
  if ( planes_ptr[NEW].data == NULL ) {
    printf("Malloc fail: planes_prt[NEW].data is NULL\n");
    return 1;
  }
  memset ( planes_ptr[NEW].data, 0, frame_size * sizeof(double) );


  // ··················································
  // buffers for north and south communication 
  // are not really needed
  //
  // in fact, they are already contiguous, just the
  // first and last line of every rank's plane
  //
  // you may just make some pointers pointing to the
  // correct positions
  //

  // or, if you prefer, just go on and allocate buffers
  // also for north and south communications

  // ··················································
  // allocate buffers for EAST and WEST (columns need explicit copying)
  //

  // Allocate EAST buffers (both send and receive)
  if ( neighbours[EAST] != MPI_PROC_NULL )
    {
      buffers_ptr[SEND][EAST] = (double*)malloc( (N[_y_] + 2) * sizeof(double) );
      if ( buffers_ptr[SEND][EAST] == NULL ) {
        printf("Malloc fail: buffers_ptr[SEND][EAST] is NULL\n");
        return 1;
      }
      
      buffers_ptr[RECV][EAST] = (double*)malloc( (N[_y_] + 2) * sizeof(double) );
      if ( buffers_ptr[RECV][EAST] == NULL ) {
        printf("Malloc fail: buffers_ptr[RECV][EAST] is NULL\n");
        return 1;
      }
    }

  // Allocate WEST buffers (both send and receive)
  if ( neighbours[WEST] != MPI_PROC_NULL )
    {
      buffers_ptr[SEND][WEST] = (double*)malloc( (N[_y_] + 2) * sizeof(double) );
      if ( buffers_ptr[SEND][WEST] == NULL ) {
        printf("Malloc fail: buffers_ptr[SEND][WEST] is NULL\n");
        return 1;
      }
      
      buffers_ptr[RECV][WEST] = (double*)malloc( (N[_y_] + 2) * sizeof(double) );
      if ( buffers_ptr[RECV][WEST] == NULL ) {
        printf("Malloc fail: buffers_ptr[RECV][WEST] is NULL\n");
        return 1;
      }
    }

  // NORTH and SOUTH buffers will be set to point directly to the data
  // (no allocation needed, just pointers set during communication)
  buffers_ptr[SEND][NORTH] = NULL;
  buffers_ptr[RECV][NORTH] = NULL;
  buffers_ptr[SEND][SOUTH] = NULL;
  buffers_ptr[RECV][SOUTH] = NULL;

  // ··················································

  
  return 0;
}



// function to free all allocated memory
 int memory_release ( buffers_t *buffers,
		      plane_t   *planes
		     )
  
{

  if ( planes != NULL )
    {
      if ( planes[OLD].data != NULL )
	      free (planes[OLD].data);
      
      if ( planes[NEW].data != NULL )
	      free (planes[NEW].data);
    }

  if ( buffers != NULL )
    {
      // Free EAST buffers
      if ( buffers[SEND][EAST] != NULL )
	      free (buffers[SEND][EAST]);
        
      if ( buffers[RECV][EAST] != NULL )
	      free (buffers[RECV][EAST]);
      
      // Free WEST buffers
      if ( buffers[SEND][WEST] != NULL )
	      free (buffers[SEND][WEST]);

      if ( buffers[RECV][WEST] != NULL )
	      free (buffers[RECV][WEST]);
    }

      
  return 0;
}



int output_energy_stat ( int step, plane_t *plane, double budget, int Me, MPI_Comm *Comm )
{

  double system_energy = 0;
  double tot_system_energy = 0;
  get_total_energy ( plane, &system_energy );
  
  MPI_Reduce ( &system_energy, &tot_system_energy, 1, MPI_DOUBLE, MPI_SUM, 0, *Comm );
  
  if ( Me == 0 )
    {
      if ( step >= 0 )
	{
	  printf(" [ step %4d ] ", step );
	  fflush(stdout);
	}
      
      printf( "total injected energy is %g, "
	      "system energy is %g "
	      "( in avg %g per grid point)\n",
	      budget,
	      tot_system_energy,
	      tot_system_energy / (plane->size[_x_]*plane->size[_y_]) );
    }
  
  return 0;
}


/* ==========================================================================
   =                                                                        =
   =   Grid Snapshot Saving for Visualization                               =
   ========================================================================== */
// keep this function in case I need to have the gif
// DON'T run it in Leonardo because it is costly
int save_grid_snapshot( int step, plane_t *plane, vec2_t S, vec2_t Grid, int Me, MPI_Comm *Comm )
{
  int Ntasks;
  MPI_Comm_size(*Comm, &Ntasks);
  
  // Get local grid size (without halo)
  uint local_sizex = plane->size[_x_];
  uint local_sizey = plane->size[_y_];
  
  // Total grid size with halos
  uint full_sizex = local_sizex + 2;
  //uint full_sizey = local_sizey + 2;
  
  // Extract local data (without halo)
  double *local_data = (double*)malloc(local_sizex * local_sizey * sizeof(double));
  
  #define IDX(i, j) ((j) * full_sizex + (i))
  for (uint j = 0; j < local_sizey; j++) {
    for (uint i = 0; i < local_sizex; i++) {
      local_data[j * local_sizex + i] = plane->data[IDX(i+1, j+1)];
    }
  }
  #undef IDX
  
  // Gather all data to rank 0
  if (Me == 0) {
    // Allocate full grid
    double *full_grid = (double*)malloc(S[_x_] * S[_y_] * sizeof(double));
    
    // Calculate sizes and displacements for each process
    int *recvcounts = (int*)malloc(Ntasks * sizeof(int));
    int *displs = (int*)malloc(Ntasks * sizeof(int));
    
    int offset = 0;
    for (int rank = 0; rank < Ntasks; rank++) {
      int X = rank % Grid[_x_];
      int Y = rank / Grid[_x_];
      
      uint s = S[_x_] / Grid[_x_];
      uint r = S[_x_] % Grid[_x_];
      uint rank_sizex = s + (X < r);
      
      s = S[_y_] / Grid[_y_];
      r = S[_y_] % Grid[_y_];
      uint rank_sizey = s + (Y < r);
      
      recvcounts[rank] = rank_sizex * rank_sizey;
      displs[rank] = offset;
      offset += recvcounts[rank];
    }
    
    // Gather data
    MPI_Gatherv(local_data, local_sizex * local_sizey, MPI_DOUBLE,
                full_grid, recvcounts, displs, MPI_DOUBLE,
                0, *Comm);
    
    // Reconstruct 2D grid from gathered data
    double *output_grid = (double*)malloc(S[_x_] * S[_y_] * sizeof(double));
    
    offset = 0;
    for (int rank = 0; rank < Ntasks; rank++) {
      int X = rank % Grid[_x_];
      int Y = rank / Grid[_x_];
      
      uint s = S[_x_] / Grid[_x_];
      uint r = S[_x_] % Grid[_x_];
      uint rank_sizex = s + (X < r);
      
      s = S[_y_] / Grid[_y_];
      r = S[_y_] % Grid[_y_];
      uint rank_sizey = s + (Y < r);
      
      // Calculate offset in global grid
      uint global_x_offset = 0;
      uint global_y_offset = 0;
      for (int i = 0; i < X; i++) {
        s = S[_x_] / Grid[_x_];
        r = S[_x_] % Grid[_x_];
        global_x_offset += s + (i < r);
      }
      for (int j = 0; j < Y; j++) {
        s = S[_y_] / Grid[_y_];
        r = S[_y_] % Grid[_y_];
        global_y_offset += s + (j < r);
      }
      
      // Copy data to output grid
      for (uint j = 0; j < rank_sizey; j++) {
        for (uint i = 0; i < rank_sizex; i++) {
          uint global_idx = (global_y_offset + j) * S[_x_] + (global_x_offset + i);
          output_grid[global_idx] = full_grid[offset + j * rank_sizex + i];
        }
      }
      offset += rank_sizex * rank_sizey;
    }
    
    // Write to binary file
    char filename[256];
    snprintf(filename, sizeof(filename), "grid_snapshot_%04d.bin", step);
    FILE *fp = fopen(filename, "wb");
    if (fp) {
      fwrite(&S[_x_], sizeof(uint), 1, fp);
      fwrite(&S[_y_], sizeof(uint), 1, fp);
      fwrite(output_grid, sizeof(double), S[_x_] * S[_y_], fp);
      fclose(fp);
      printf("Saved snapshot: %s\n", filename);
    }
    
    free(full_grid);
    free(output_grid);
    free(recvcounts);
    free(displs);
  } else {
    // Other ranks send their data
    MPI_Gatherv(local_data, local_sizex * local_sizey, MPI_DOUBLE,
                NULL, NULL, NULL, MPI_DOUBLE,
                0, *Comm);
  }
  
  free(local_data);
  return 0;
}

