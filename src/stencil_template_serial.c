/*
 *
 *  mysizex   :   local x-extendion of your patch
 *  mysizey   :   local y-extension of your patch
 *
 */


#include "stencil_template_serial.h"
//#include <types.h>
typedef unsigned int uint;

int dump ( const double *, const uint [2], const char *, double *, double * );

// ------------------------------------------------------------------
// ------------------------------------------------------------------

int main(int argc, char **argv)
{

  int  Niterations;
  int  periodic;
  uint S[2];
  
  int     Nsources;
  uint   *Sources;
  double  energy_per_source;

  double *planes[2];
  
  double injected_heat = 0;

  int injection_frequency;
  int output_energy_at_steps = 0;
   
  /* argument checking and setting */
  initialize ( argc, argv, &S[0], &periodic, &Niterations,
	       &Nsources, &Sources, &energy_per_source, &planes[0],
	       &output_energy_at_steps, &injection_frequency );
  
  
  int current = OLD; // the current plane

  // could be commented because it's useless 
  if ( injection_frequency > 1 )
    inject_energy( periodic, Nsources, Sources, energy_per_source, S, planes[current] );
  
  for (int iter = 0; iter < Niterations; iter++)
    
    {    
      /* new energy from sources */

      if ( iter % injection_frequency == 0 )
      {
        inject_energy( periodic, Nsources, Sources, energy_per_source, S, planes[current] );
        injected_heat += Nsources*energy_per_source; // the thing written on the terminal
      }
                  
      /* update grid points */
      update_plane(periodic, S, planes[current], planes[!current] );

      if ( output_energy_at_steps )
      {
        double system_heat;
        get_total_energy( S, planes[!current], &system_heat);
                
        printf("step %d :: injected energy is %g, updated system energy is %g\n", iter, 
        injected_heat, system_heat );

        char filename[100];
        sprintf( filename, "plane_%05d.bin", iter );
        dump( planes[!current], S, filename, NULL, NULL );
          
      }

      /* swap planes for the new iteration */
      current = !current;
      
    }
  
  
  /* get final heat in the system */
  
  double system_heat;
  get_total_energy( S, planes[!current], &system_heat);

  printf("injected energy is %g, system energy is %g\n",
	 injected_heat, system_heat );
  
  memory_release( planes[OLD], Sources );
  return 0;
}


/* ==========================================================================
   =                                                                        =
   =   routines called within the integration loop                          =
   ========================================================================== */





/* ==========================================================================
   =                                                                        =
   =   initialization                                                       =
   ========================================================================== */



int memory_allocate ( const uint [2],
		      double ** );


int initialize_sources( uint      [2],
			int       ,
			uint     ** );

int initialize ( int      argc,                // the argc from command line
		 char   **argv,                // the argv from command line
		 uint     *S,                   // two-uint array defining the x,y dimensions of the grid
		 int     *periodic,            // periodic-boundary tag
		 int     *Niterations,         // how many iterations
		 int     *Nsources,            // how many heat sources
		 uint    **Sources,             // the array of heat sources
		 double  *energy_per_source,   // how much heat per source
		 double **planes,
		 int     *output_energy_at_steps,
		 int     *injection_frequency
		 )
{
  int ret;
  
  // ··································································
  // set default values

  S[_x_]            = 1000;
  S[_y_]            = 1000;
  *periodic         = 0;
  *Nsources         = 2;   // was 1
  *Niterations      = 99;   // n
  *output_energy_at_steps = 0;
  *energy_per_source = 1.0;
  *injection_frequency = *Niterations;

  double freq = 0;
  
  // ··································································
  // process the commadn line
  // 
  while ( 1 )
  {
    int opt;
    while((opt = getopt(argc, argv, ":x:y:e:E:f:n:p:o:")) != -1)
      {
	switch( opt )
	  {
	  case 'x': S[_x_] = (uint)atoi(optarg);
	    break;

	  case 'y': S[_y_] = (uint)atoi(optarg);
	    break;

	  case 'e': *Nsources = atoi(optarg);
	    break;

	  case 'E': *energy_per_source = atof(optarg);
	    break;

	  case 'n': *Niterations = atoi(optarg);
	    break;

	  case 'p': *periodic = (atoi(optarg) > 0);
	    break;

	  case 'o': *output_energy_at_steps = (atoi(optarg) > 0);
	    break;

	  case 'f': freq = atof(optarg);
	    break;
	    
	  case 'h': printf( "valid options are ( values btw [] are the default values ):\n"
			    "-x    x size of the plate [1000]\n"
			    "-y    y size of the plate [1000]\n"
			    "-e    how many energy sources on the plate [1]\n"
			    "-E    how much energy per source on the plate [1.0]\n"
			    "-f    the frequency of energy injection [0.0]\n"
			    "-n    how many iterations [100]\n"
			    "-p    whether periodic boundaries applies  [0 = false]\n"
			    "-o    whether to print the energy budgets at every step [0 = false]\n"
			    );
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
  // injection frequency
  if ( freq == 0 )
    *injection_frequency = 1;  // inject at every step
  else
    {
      freq = (freq > 1.0 ? 1.0 : freq );
      *injection_frequency = freq * *Niterations; // inject at this frequency, e.g. 1 means at every step
    }

  // ··································································
  // here we should check for all the parameters being meaningful
  if ( S[_x_] < 1 || S[_y_] < 1 )
    {
      printf("error: invalid grid size\n");
      return 1;
    }

  if ( *Nsources < 1 )
    {
      printf("error: invalid number of heat sources\n");
      return 1;
    }

  if ( *Niterations < 1 )
    {
      printf("error: invalid number of iterations\n");
      return 1;
    }

  if ( *energy_per_source < 0 )
    {
      printf("error: invalid energy per source\n");
      return 1;
    }

  if ( *injection_frequency < 1 || *injection_frequency > *Niterations )
    {
      printf("error: invalid injection frequency\n");
      return 1;
    }




  // ··································································
  // allocate the needed memory
  //
  ret = memory_allocate( S, planes );   
  if (ret == 1)
    {
      printf("error allocating memory for the planes\n");
      return ret;
    }

  // ··································································
  // allocate the heat sources
  //
  ret = initialize_sources( S, *Nsources, Sources );
  
  
  return 0;  
}


int memory_allocate ( const uint      size[2],
		            double **planes_ptr )
/*
 * allocate the memory for the planes
 * we need 2 planes: the first contains the
 * current data, the second the updated data
 *
 * in the integration loop then the roles are
 * swapped at every iteration
 *
 */
{
  if (planes_ptr == NULL )
    // an invalid pointer has been passed
    // manage the situation
    return 1;

  unsigned int bytes = (size[_x_]+2)*(size[_y_]+2);

  planes_ptr[OLD] = (double*)malloc( 2*bytes*sizeof(double) );
  memset ( planes_ptr[OLD], 0, 2*bytes*sizeof(double) ); // allocating memory for two planes
  planes_ptr[NEW] = planes_ptr[OLD] + bytes;
      
  return 0;
}



int initialize_sources( uint      size[2],
			int       Nsources,
			uint     **Sources )
/*
 * randomly spread heat sources
 */
{
  *Sources = (uint*)malloc( Nsources * 2 *sizeof(uint) );
  for ( int s = 0; s < Nsources; s++ )
  {
    (*Sources)[s*2] = 1+ rand() % size[_x_];
    (*Sources)[s*2+1] = 1+ rand() % size[_y_];
  }

  return 0;
}



int memory_release ( double *data, uint *sources )
  
{
  if( data != NULL )
    free( data );

  if( sources != NULL )
    free( sources );

  
  
  return 0;
}



int dump ( const double *data, const uint size[2], const char *filename, double *min, double *max )
{
  if ( (filename != NULL) && (filename[0] != '\0') )
    {
      FILE *outfile = fopen( filename, "w" );
      if ( outfile == NULL )
	return 2;
      
      float *array = (float*)malloc( size[0] * sizeof(float) );
      
      double _min_ = DBL_MAX;
      double _max_ = 0;

      for ( int j = 0; j < size[1]; j++ )
	{
	  /*
	  float y = (float)j / size[1];
	  fwrite ( &y, sizeof(float), 1, outfile );
	  */
	  
	  const double * restrict line = data + j*size[0];
	  for ( int i = 0; i < size[0]; i++ ) {
	    array[i] = (float)line[i];
	    _min_ = ( line[i] < _min_? line[i] : _min_ );
	    _max_ = ( line[i] > _max_? line[i] : _max_ ); }
	  
	  fwrite( array, sizeof(float), size[0], outfile );
	}
      
      free( array );
      
      fclose( outfile );

      if ( min != NULL )
	*min = _min_;
      if ( max != NULL )
	*max = _max_;
    }

  else return 1;
  
  return 0;
}

