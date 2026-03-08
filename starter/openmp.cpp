/* ------------
 * The code is adapted from the XSEDE online course Applications of Parallel Computing. 
 * The copyright belongs to all the XSEDE and the University of California Berkeley staff
 * that moderate this online course as well as the University of Toronto CSC367 staff.
 * This code is provided solely for the use of students taking the CSC367 course at 
 * the University of Toronto.
 * Copying for purposes other than this use is expressly prohibited. 
 * All forms of distribution of this code, whether as given or with 
 * any changes, are expressly prohibited. 
 * -------------
*/

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "common.h"
#include "omp.h"

// constants for spatial binning (must match common.cpp; do not change common files)
#define CUTOFF 0.01
#define DENSITY 0.0005

//
//  benchmarking program
//
int main( int argc, char **argv )
{   
    int navg,nabsavg=0,numthreads; 
    double dmin, absmin=1.0,davg,absavg=0.0;
	
    if( find_option( argc, argv, "-h" ) >= 0 )
    {
        printf( "Options:\n" );
        printf( "-h to see this help\n" );
        printf( "-n <int> to set number of particles\n" );
        printf( "-o <filename> to specify the output file name\n" );
        printf( "-s <filename> to specify a summary file name\n" ); 
        printf( "-no turns off all correctness checks and particle output\n");   
        return 0;
    }

    int n = read_int( argc, argv, "-n", 1000 );
    char *savename = read_string( argc, argv, "-o", NULL );
    char *sumname = read_string( argc, argv, "-s", NULL );

    FILE *fsave = savename ? fopen( savename, "w" ) : NULL;
    FILE *fsum = sumname ? fopen ( sumname, "a" ) : NULL;      

    particle_t *particles = (particle_t*) malloc( n * sizeof(particle_t) );
    set_size( n );
    init_particles( n, particles );

    // same cell list setup as serial for O(n) force computation
    double simulationSize = sqrt( DENSITY * n );
    int numberCellsX = (int)( simulationSize / CUTOFF );
    int numberCellsY = (int)( simulationSize / CUTOFF );
    if ( numberCellsX < 1 ) numberCellsX = 1;
    if ( numberCellsY < 1 ) numberCellsY = 1;
    double cellWidthX = simulationSize / numberCellsX;
    double cellWidthY = simulationSize / numberCellsY;
    int totalCells = numberCellsX * numberCellsY;

    int *cellFirst = (int*) malloc( totalCells * sizeof(int) );
    int *nextInCell = (int*) malloc( n * sizeof(int) );

    //
    //  simulate a number of time steps
    //
    double simulation_time = read_timer( );

    #pragma omp parallel private(dmin)
    {
        numthreads = omp_get_num_threads();
        for ( int step = 0; step < NSTEPS; step++ )
        {
            navg = 0;
            davg = 0.0;
            dmin = 1.0;

            // clear cell lists then bin particles (serial binning is O(n), avoids lock contention)
            #pragma omp single
            {
                for ( int cellIndex = 0; cellIndex < totalCells; cellIndex++ )
                    cellFirst[cellIndex] = -1;
                for ( int particleIndex = 0; particleIndex < n; particleIndex++ )
                {
                    double x = particles[particleIndex].x;
                    double y = particles[particleIndex].y;
                    int cellX = (int)( x / cellWidthX );
                    int cellY = (int)( y / cellWidthY );
                    if ( cellX >= numberCellsX ) cellX = numberCellsX - 1;
                    if ( cellY >= numberCellsY ) cellY = numberCellsY - 1;
                    if ( cellX < 0 ) cellX = 0;
                    if ( cellY < 0 ) cellY = 0;
                    int cellIndex = cellY * numberCellsX + cellX;
                    nextInCell[particleIndex] = cellFirst[cellIndex];
                    cellFirst[cellIndex] = particleIndex;
                }
            }
            #pragma omp barrier

            //
            //  compute forces (each particle written by one thread only; use reduction for stats)
            //  static scheduling gives contiguous chunks per thread for better cache locality
            //
            #pragma omp for schedule(static) reduction(+:navg) reduction(+:davg)
            for ( int particleIndex = 0; particleIndex < n; particleIndex++ )
            {
                particles[particleIndex].ax = 0;
                particles[particleIndex].ay = 0;

                double x = particles[particleIndex].x;
                double y = particles[particleIndex].y;
                int centerCellX = (int)( x / cellWidthX );
                int centerCellY = (int)( y / cellWidthY );
                if ( centerCellX >= numberCellsX ) centerCellX = numberCellsX - 1;
                if ( centerCellY >= numberCellsY ) centerCellY = numberCellsY - 1;
                if ( centerCellX < 0 ) centerCellX = 0;
                if ( centerCellY < 0 ) centerCellY = 0;

                for ( int offsetY = -1; offsetY <= 1; offsetY++ )
                {
                    int cellY = centerCellY + offsetY;
                    if ( cellY < 0 || cellY >= numberCellsY ) continue;
                    for ( int offsetX = -1; offsetX <= 1; offsetX++ )
                    {
                        int cellX = centerCellX + offsetX;
                        if ( cellX < 0 || cellX >= numberCellsX ) continue;
                        int cellIndex = cellY * numberCellsX + cellX;
                        for ( int neighborIndex = cellFirst[cellIndex]; neighborIndex != -1; neighborIndex = nextInCell[neighborIndex] )
                            apply_force( particles[particleIndex], particles[neighborIndex], &dmin, &davg, &navg );
                    }
                }
            }

            //
            //  move particles
            //
            #pragma omp for schedule(static)
            for ( int i = 0; i < n; i++ ) 
                move( particles[i] );

            if ( find_option( argc, argv, "-no" ) == -1 ) 
            {
                //
                //  compute statistical data
                //
                #pragma omp master
                if ( navg ) { 
                    absavg += davg / navg;
                    nabsavg++;
                }

                #pragma omp critical
                {
                    if ( dmin < absmin ) absmin = dmin; 
                }

                //
                //  save if necessary
                //
                #pragma omp single
                if ( fsave && ( step % SAVEFREQ ) == 0 )
                    save( fsave, n, particles );
            }
        }
    }
    simulation_time = read_timer( ) - simulation_time;
    
    printf( "n = %d,threads = %d, simulation time = %g seconds", n, numthreads, simulation_time);

    if ( find_option( argc, argv, "-no" ) == -1 )
    {
      if ( nabsavg ) absavg /= nabsavg;
    // 
    //  -the minimum distance absmin between 2 particles during the run of the simulation
    //  -A Correct simulation will have particles stay at greater than 0.4 (of cutoff) with typical values between .7-.8
    //  -A simulation were particles don't interact correctly will be less than 0.4 (of cutoff) with typical values between .01-.05
    //
    //  -The average distance absavg is ~.95 when most particles are interacting correctly and ~.66 when no particles are interacting
    //
    printf( ", absmin = %lf, absavg = %lf", absmin, absavg);
    if (absmin < 0.4) printf ("\nThe minimum distance is below 0.4 meaning that some particle is not interacting");
    if (absavg < 0.8) printf ("\nThe average distance is below 0.8 meaning that most particles are not interacting");
    }
    printf("\n");
    
    //
    // Printing summary data
    //
    if ( fsum )
        fprintf( fsum, "%d %d %g\n", n, numthreads, simulation_time );

    //
    // Clearing space
    //
    free( cellFirst );
    free( nextInCell );
    if ( fsum )
        fclose( fsum );

    free( particles );
    if ( fsave )
        fclose( fsave );
    
    return 0;
}
