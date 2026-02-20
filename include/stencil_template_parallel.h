/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */


#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <getopt.h>
#include <time.h>
#include <math.h>

#include <omp.h>
#include <mpi.h>


#define NORTH 0
#define SOUTH 1
#define EAST  2
#define WEST  3

#define SEND 0
#define RECV 1

#define OLD 0
#define NEW 1

#define _x_ 0
#define _y_ 1

typedef unsigned int uint;

typedef uint    vec2_t[2];
typedef double *restrict buffers_t[4];

typedef struct {
    double   * restrict data;
    vec2_t     size;
} plane_t;



extern int inject_energy ( const int      ,
                           const int      ,
			   const vec2_t  *,
			   const double   ,
                                 plane_t *,
                           const vec2_t   );


extern int update_plane_internal ( const vec2_t   ,
                                    const plane_t *,
                                          plane_t * );

extern int update_plane_boundary ( const int      ,
                                    const vec2_t   ,
                                    const plane_t *,
                                          plane_t * );

extern int update_plane ( const int      ,
                          const vec2_t   ,
                          const plane_t *,
                                plane_t * );


extern int get_total_energy( plane_t *,
                             double  * );

extern int initialize_first_touch( plane_t * );

extern int save_grid_snapshot( int             ,
                               plane_t        *,
                               vec2_t          ,
                               vec2_t          ,
                               int             ,
                               MPI_Comm       * );

int initialize ( MPI_Comm *,
                 int       ,
		 int       ,
		 int       ,
		 char    **,
                 vec2_t   *,
                 vec2_t   *,                 
		 int      *,
                 int      *,
		 int      *,
		 int      *,
		 int      *,
		 int      *,
                 vec2_t  **,
                 double   *,
                 plane_t  *,
                 buffers_t * );


int memory_release (buffers_t *, plane_t *);


int output_energy_stat ( int      ,
                         plane_t *,
                         double   ,
                         int      ,
                         MPI_Comm *);



inline int inject_energy ( const int      periodic,
                           const int      Nsources,
			   const vec2_t  *Sources,
			   const double   energy,
                                 plane_t *plane,
                           const vec2_t   N
                           )
{
    const uint register sizex = plane->size[_x_]+2;
    double * restrict data = plane->data;
    
   #define IDX( i, j ) ( (j)*sizex + (i) )
    for (int s = 0; s < Nsources; s++)
        {
            int x = Sources[s][_x_];
            int y = Sources[s][_y_];
            
            data[ IDX(x,y) ] += energy;
            
            if ( periodic )
                {
                    if ( (N[_x_] == 1)  )
                        {
                            // propagate the boundaries if needed
                            // Only one task in x-direction, so wrap left-right within this task
                            // If source is at left edge, copy to right halo
                            if (x == 1)
                                data[ IDX(plane->size[_x_], y) ] = data[ IDX(1, y) ];
                            // If source is at right edge, copy to left halo
                            if (x == plane->size[_x_])
                                data[ IDX(0, y) ] = data[ IDX(plane->size[_x_], y) ];
                        }
                    
                    if ( (N[_y_] == 1) )
                        {
                            // propagate the boundaries if needed
                            // Only one task in y-direction, so wrap top-bottom within this task
                            // If source is at top edge, copy to bottom halo
                            if (y == 1)
                                data[ IDX(x, plane->size[_y_]) ] = data[ IDX(x, 1) ];
                            // If source is at bottom edge, copy to top halo
                            if (y == plane->size[_y_])
                                data[ IDX(x, 0) ] = data[ IDX(x, plane->size[_y_]) ];
                        }
                }                
        }
 #undef IDX
    
  return 0;
}



// Update all internal nodes (excluding boundaries and halos)
// This function can be called immediately after starting MPI communication
// because internal nodes don't depend on halo data from neighbors
inline int update_plane_internal ( 
                          const vec2_t   N,         // the grid of MPI tasks
                          const plane_t *oldplane,
                                plane_t *newplane
                          )
{
    // sizes including the halos
    uint register fxsize = oldplane->size[_x_]+2;
    
    // sizes excluding the halos
    uint register xsize = oldplane->size[_x_];
    uint register ysize = oldplane->size[_y_];
    
   #define IDX( i, j ) ( (j)*fxsize + (i) )

    double * restrict old = oldplane->data;
    double * restrict new = newplane->data;
    
    // Optimized stencil constants
    double alpha = 0.6;
    double weight = (1.0 - alpha) * 0.25;
    
    // Process internal nodes only (not on any boundary)

    #pragma omp parallel for schedule(static)
    for (uint j = 2; j < ysize; j++) {
        for (uint i = 2; i < xsize; i++) {                
            double result = old[ IDX(i,j) ] * alpha;
            result += (old[IDX(i-1, j)] + old[IDX(i+1, j)] + 
                      old[IDX(i, j-1)] + old[IDX(i, j+1)]) * weight;
            new[ IDX(i,j) ] = result; 
        }
    }
    
    #undef IDX
    return 0;
}



// Update boundary nodes (after MPI communication completes)
// This must be called AFTER halo exchange is complete
inline int update_plane_boundary ( 
                          const int      periodic,
                          const vec2_t   N,
                          const plane_t *oldplane,
                                plane_t *newplane
                          )
{
    // sizes including the halos
    uint register fxsize = oldplane->size[_x_]+2;
    
    // sizes excluding the halos
    uint register xsize = oldplane->size[_x_];
    uint register ysize = oldplane->size[_y_];
    
    #define IDX( i, j ) ( (j)*fxsize + (i) )
    
    double * restrict old = oldplane->data;
    double * restrict new = newplane->data;

    // Optimized stencil constants
    double alpha = 0.6;
    double weight = (1.0 - alpha) * 0.25;
    
    uint j_top = 1;
    uint j_bottom = ysize;
    uint i_left = 1;
    uint i_right = xsize;

    #pragma omp parallel
    {
        // Top & bottom rows (process entire width including corners)
        #pragma omp for schedule(static)  // static schedule for regular grid
        for (uint i = 1; i <= xsize; i++) {
            // top row j=1
            double result1 = old[ IDX(i, j_top) ] * alpha;
            result1 += (old[IDX(i-1, j_top)] + old[IDX(i+1, j_top)] + 
                       old[IDX(i, j_top-1)] + old[IDX(i, j_top+1)]) * weight;
            new[ IDX(i, j_top) ] = result1;

            // bottom row j=ysize
            double result2 = old[ IDX(i, j_bottom) ] * alpha;
            result2 += (old[IDX(i-1, j_bottom)] + old[IDX(i+1, j_bottom)] + 
                       old[IDX(i, j_bottom-1)] + old[IDX(i, j_bottom+1)]) * weight;
            new[ IDX(i, j_bottom) ] = result2;
        }

        // Left & right columns (skip corners already processed)
        #pragma omp for schedule(static)
        for (uint j = 2; j < ysize; j++) {
            // left column i=1
            double result1 = old[ IDX(i_left, j) ] * alpha;
            result1 += (old[IDX(i_left-1, j)] + old[IDX(i_left+1, j)] + 
                       old[IDX(i_left, j-1)] + old[IDX(i_left, j+1)]) * weight;
            new[ IDX(i_left, j) ] = result1;

            // right column i=xsize
            double result2 = old[ IDX(i_right, j) ] * alpha;
            result2 += (old[IDX(i_right-1, j)] + old[IDX(i_right+1, j)] + 
                       old[IDX(i_right, j-1)] + old[IDX(i_right, j+1)]) * weight;
            new[ IDX(i_right, j) ] = result2;
        }
    }

    // Handle periodic boundary conditions
    if ( periodic ) {
        if ( N[_x_] == 1 ) {
            // propagate the boundaries as needed
            #pragma GCC unroll 4
            for (uint j = 1; j <= ysize; j++) {
                // first column of padding gets values from the last column of the actual grid
                new[ IDX(0, j) ] = new[ IDX(xsize, j) ];
                // last column of padding gets values from the first column of the actual grid
                new[ IDX(xsize+1, j) ] = new[ IDX(1, j) ];
            }
        }
  
        if ( N[_y_] == 1 ) {
            // propagate the boundaries as needed
            #pragma GCC unroll 4
            for (uint i = 1; i <= xsize; i++) {
                // first line of padding gets values from the last line of the actual grid
                new[ IDX(i, 0) ] = new[ IDX(i, ysize) ];
                // last line of padding gets values from the first line of the actual grid
                new[ IDX(i, ysize+1) ] = new[ IDX(i, 1) ];
            }
        }
    }
    
    #undef IDX
    return 0;
}

/* original version 
inline int update_plane ( const int      periodic, 
                          const vec2_t   N,
                          const plane_t *oldplane,
                                plane_t *newplane
                          )
{
    update_plane_internal(N, oldplane, newplane);
    update_plane_boundary(periodic, N, oldplane, newplane);
    return 0;
}
*/


inline int get_total_energy( plane_t *plane,
                             double  *energy )
/*
 * NOTE: this routine a good candiadate for openmp
 *       parallelization
 */
{

    const int register xsize = plane->size[_x_];
    const int register ysize = plane->size[_y_];
    const int register fsize = xsize+2;

    double * restrict data = plane->data;
    
   #define IDX( i, j ) ( (j)*fsize + (i) )

   #if defined(LONG_ACCURACY)    
    long double totenergy = 0;
   #else
    double totenergy = 0;    
   #endif

    // HINT: you may attempt to
    //       (i)  manually unroll the loop
    //       (ii) ask the compiler to do it
    // for instance
    // #pragma GCC unroll 4
    #pragma omp parallel for collapse(2) reduction(+:totenergy) 
    for ( int j = 1; j <= ysize; j++ )
        for ( int i = 1; i <= xsize; i++ )
            totenergy += data[ IDX(i, j) ];

    
   #undef IDX

    *energy = (double)totenergy;
    return 0;
}



// Touch-by-all policy for NUMA optimization
// Each thread in the parallel region touches (writes to) its portion of memory
// using the SAME parallel decomposition as the computation loops.
// This triggers first-touch allocation on the appropriate NUMA node.
inline int initialize_first_touch( plane_t *plane )
{
    const uint register fxsize = plane->size[_x_] + 2;
    const uint register xsize = plane->size[_x_];
    const uint register ysize = plane->size[_y_];
    
    double * restrict data = plane->data;
    
   #define IDX( i, j ) ( (j)*fxsize + (i) )
    
    // Touch memory using the SAME pattern as the computation
    // This ensures NUMA-local allocation for maximum performance
    
    #pragma omp parallel
    {
        // Touch internal nodes (same pattern as update_plane_internal)
        #pragma omp for schedule(static)
        for (uint j = 2; j < ysize; j++) {
            for (uint i = 2; i < xsize; i++) {
                data[ IDX(i, j) ] = 0.0;
            }
        }
        
        // Touch boundary nodes (same pattern as update_plane_boundary)
        // Top & bottom rows (including corners)
        #pragma omp for schedule(static)
        for (uint i = 1; i <= xsize; i++) {
            data[ IDX(i, 1) ] = 0.0;        // top row
            data[ IDX(i, ysize) ] = 0.0;    // bottom row
        }
        
        // Left & right columns (excluding corners to avoid duplicate touches)
        #pragma omp for schedule(static)
        for (uint j = 2; j < ysize; j++) {
            data[ IDX(1, j) ] = 0.0;        // left column
            data[ IDX(xsize, j) ] = 0.0;    // right column
        }
        
        // Touch halo regions (these are shared by all threads typically)
        #pragma omp for schedule(static)
        for (uint i = 0; i < fxsize; i++) {
            data[ IDX(i, 0) ] = 0.0;        // top halo row
            data[ IDX(i, ysize+1) ] = 0.0;  // bottom halo row
        }
        
        // Left and right halo columns
        #pragma omp single
        {
            for (uint j = 0; j <= ysize+1; j++) {
                data[ IDX(0, j) ] = 0.0;           // left halo
                data[ IDX(xsize+1, j) ] = 0.0;     // right halo
            }
        }
    }
    
   #undef IDX
    
    return 0;
}


