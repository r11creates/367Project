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

#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <string.h>
#include <vector>
#include "common.h"

// constants for spatial binning (must match common.cpp; do not change common files)
#define CUTOFF 0.01
#define DENSITY 0.0005

//
//  benchmarking program
//
int main( int argc, char **argv )
{    
    int navg, nabsavg=0;
    double dmin, absmin=1.0,davg,absavg=0.0;
    double reducedDavg,reducedDmin;
    int reducedNavg; 
 
    //
    //  process command line parameters
    //
    if( find_option( argc, argv, "-h" ) >= 0 )
    {
        printf( "Options:\n" );
        printf( "-h to see this help\n" );
        printf( "-n <int> to set the number of particles\n" );
        printf( "-o <filename> to specify the output file name\n" );
        printf( "-s <filename> to specify a summary file name\n" );
        printf( "-no turns off all correctness checks and particle output\n");
        return 0;
    }
    
    int n = read_int( argc, argv, "-n", 1000 );
    char *savename = read_string( argc, argv, "-o", NULL );
    char *sumname = read_string( argc, argv, "-s", NULL );
    
    //
    //  set up MPI
    //
    int numberOfProcessors, rank;
    MPI_Init( &argc, &argv );
    MPI_Comm_size( MPI_COMM_WORLD, &numberOfProcessors );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    
    //
    //  allocate generic resources
    //
    FILE *fsave = savename && rank == 0 ? fopen( savename, "w" ) : NULL;
    FILE *fsum = sumname && rank == 0 ? fopen ( sumname, "a" ) : NULL;

    // create mpi datatype for particle (6 doubles: x, y, vx, vy, ax, ay)
    MPI_Datatype PARTICLE;
    MPI_Type_contiguous( 6, MPI_DOUBLE, &PARTICLE );
    MPI_Type_commit( &PARTICLE );
    
    //
    //  initialize and distribute the particles
    //
    particle_t *allParticles = (particle_t*) malloc( n * sizeof(particle_t) );
    set_size( n );
    if( rank == 0 )
        init_particles( n, allParticles );
    MPI_Bcast( allParticles, n, PARTICLE, 0, MPI_COMM_WORLD );

    // spatial grid setup (same formula as common.cpp)
    double simulationSize = sqrt( DENSITY * n );
    int numberCells = (int)( simulationSize / CUTOFF );
    if ( numberCells < 1 ) numberCells = 1;
    double cellWidth = simulationSize / numberCells;

    // 1d strip decomposition along y-axis: evenly divide cell rows among processors
    int *processorRowStart = (int*) malloc( (numberOfProcessors + 1) * sizeof(int) );
    for ( int i = 0; i <= numberOfProcessors; i++ )
        processorRowStart[i] = (int)( (long long)i * numberCells / numberOfProcessors );

    int myFirstRow = processorRowStart[rank];
    int myLastRow = processorRowStart[rank + 1];
    int myNumberOfRows = myLastRow - myFirstRow;

    double myYStart = myFirstRow * cellWidth;
    double myYEnd = myLastRow * cellWidth;
    // last processor owns up to the full simulation boundary
    if ( rank == numberOfProcessors - 1 )
        myYEnd = simulationSize;

    // neighbor processor ranks (-1 means no neighbor in that direction)
    int neighborAbove = ( rank > 0 && myNumberOfRows > 0 ) ? rank - 1 : -1;
    int neighborBelow = ( rank < numberOfProcessors - 1 && myNumberOfRows > 0 ) ? rank + 1 : -1;

    // ghost row counts: one row of cells above and below for boundary force computation
    int ghostRowsAbove = ( neighborAbove >= 0 ) ? 1 : 0;
    int ghostRowsBelow = ( neighborBelow >= 0 ) ? 1 : 0;
    int totalLocalRows = myNumberOfRows + ghostRowsAbove + ghostRowsBelow;
    int totalLocalCells = totalLocalRows * numberCells;

    // offset to convert global cell row index to local cell row index
    int globalToLocalRowOffset = myFirstRow - ghostRowsAbove;

    // distribute initial particles to owning processors based on y-position
    std::vector<particle_t> localParticles;
    for ( int i = 0; i < n; i++ )
    {
        double y = allParticles[i].y;
        bool belongsToMe = false;
        if ( rank == numberOfProcessors - 1 )
            belongsToMe = ( y >= myYStart && y <= myYEnd );
        else
            belongsToMe = ( y >= myYStart && y < myYEnd );
        if ( belongsToMe )
            localParticles.push_back( allParticles[i] );
    }
    free( allParticles );

    // preallocate cell list and combined particle buffer
    int *cellFirst = (int*) malloc( totalLocalCells * sizeof(int) );
    int cellListCapacity = 2 * n / numberOfProcessors + 1000;
    int *nextInCell = (int*) malloc( cellListCapacity * sizeof(int) );
    particle_t *combinedParticles = (particle_t*) malloc( cellListCapacity * sizeof(particle_t) );

    // reusable buffers for ghost exchange
    std::vector<particle_t> sendBufferAbove, sendBufferBelow;
    std::vector<particle_t> receiveBufferAbove, receiveBufferBelow;

    // reusable buffers for particle migration after move
    std::vector<particle_t> migrateBufferUp, migrateBufferDown;
    std::vector<particle_t> receiveFromAbove, receiveFromBelow;

    // for saving: gather all particles on rank 0 (only when correctness checks are on)
    int *gatherCounts = NULL;
    int *gatherDisplacements = NULL;
    particle_t *gatherBuffer = NULL;
    if ( find_option( argc, argv, "-no" ) == -1 )
    {
        gatherCounts = (int*) malloc( numberOfProcessors * sizeof(int) );
        gatherDisplacements = (int*) malloc( numberOfProcessors * sizeof(int) );
        gatherBuffer = (particle_t*) malloc( n * sizeof(particle_t) );
    }

    //
    //  simulate a number of time steps
    //
    double simulation_time = read_timer( );
    for( int step = 0; step < NSTEPS; step++ )
    {
        navg = 0;
        dmin = 1.0;
        davg = 0.0;

        int localCount = (int)localParticles.size();

        // clear cell list
        for ( int cellIndex = 0; cellIndex < totalLocalCells; cellIndex++ )
            cellFirst[cellIndex] = -1;

        // grow buffers if local count exceeds capacity
        if ( localCount > cellListCapacity )
        {
            cellListCapacity = localCount * 2;
            nextInCell = (int*) realloc( nextInCell, cellListCapacity * sizeof(int) );
            combinedParticles = (particle_t*) realloc( combinedParticles, cellListCapacity * sizeof(particle_t) );
        }

        // copy local particles into combined buffer and bin into cells
        memcpy( combinedParticles, localParticles.data(), localCount * sizeof(particle_t) );
        for ( int i = 0; i < localCount; i++ )
        {
            int cellX = (int)( combinedParticles[i].x / cellWidth );
            int cellY = (int)( combinedParticles[i].y / cellWidth );
            if ( cellX >= numberCells ) cellX = numberCells - 1;
            if ( cellX < 0 ) cellX = 0;
            if ( cellY >= numberCells ) cellY = numberCells - 1;
            if ( cellY < 0 ) cellY = 0;
            int localRow = cellY - globalToLocalRowOffset;
            if ( localRow >= 0 && localRow < totalLocalRows )
            {
                int flatIndex = localRow * numberCells + cellX;
                nextInCell[i] = cellFirst[flatIndex];
                cellFirst[flatIndex] = i;
            }
        }

        // prepare ghost particles: extract boundary cell rows using cell list
        sendBufferAbove.clear();
        sendBufferBelow.clear();
        if ( neighborAbove >= 0 )
        {
            // my first owned row is ghost data for the processor above me
            int localRow = myFirstRow - globalToLocalRowOffset;
            for ( int cx = 0; cx < numberCells; cx++ )
            {
                int flatIndex = localRow * numberCells + cx;
                for ( int j = cellFirst[flatIndex]; j != -1; j = nextInCell[j] )
                    sendBufferAbove.push_back( combinedParticles[j] );
            }
        }
        if ( neighborBelow >= 0 )
        {
            // my last owned row is ghost data for the processor below me
            int localRow = ( myLastRow - 1 ) - globalToLocalRowOffset;
            for ( int cx = 0; cx < numberCells; cx++ )
            {
                int flatIndex = localRow * numberCells + cx;
                for ( int j = cellFirst[flatIndex]; j != -1; j = nextInCell[j] )
                    sendBufferBelow.push_back( combinedParticles[j] );
            }
        }

        // exchange ghost particle counts with neighbors
        int sendCountAbove = (int)sendBufferAbove.size();
        int sendCountBelow = (int)sendBufferBelow.size();
        int receiveCountAbove = 0, receiveCountBelow = 0;

        if ( neighborAbove >= 0 )
            MPI_Sendrecv( &sendCountAbove, 1, MPI_INT, neighborAbove, 0,
                          &receiveCountAbove, 1, MPI_INT, neighborAbove, 0,
                          MPI_COMM_WORLD, MPI_STATUS_IGNORE );
        if ( neighborBelow >= 0 )
            MPI_Sendrecv( &sendCountBelow, 1, MPI_INT, neighborBelow, 0,
                          &receiveCountBelow, 1, MPI_INT, neighborBelow, 0,
                          MPI_COMM_WORLD, MPI_STATUS_IGNORE );

        int ghostCount = receiveCountAbove + receiveCountBelow;
        int totalCount = localCount + ghostCount;

        // grow buffers again if needed after receiving ghost counts
        if ( totalCount > cellListCapacity )
        {
            cellListCapacity = totalCount * 2;
            nextInCell = (int*) realloc( nextInCell, cellListCapacity * sizeof(int) );
            combinedParticles = (particle_t*) realloc( combinedParticles, cellListCapacity * sizeof(particle_t) );
        }

        receiveBufferAbove.resize( receiveCountAbove );
        receiveBufferBelow.resize( receiveCountBelow );

        // exchange ghost particles using non-blocking communication
        MPI_Request ghostRequests[4];
        int numberOfGhostRequests = 0;

        if ( neighborAbove >= 0 )
        {
            if ( sendCountAbove > 0 )
                MPI_Isend( sendBufferAbove.data(), sendCountAbove, PARTICLE,
                           neighborAbove, 1, MPI_COMM_WORLD,
                           &ghostRequests[numberOfGhostRequests++] );
            if ( receiveCountAbove > 0 )
                MPI_Irecv( receiveBufferAbove.data(), receiveCountAbove, PARTICLE,
                           neighborAbove, 1, MPI_COMM_WORLD,
                           &ghostRequests[numberOfGhostRequests++] );
        }
        if ( neighborBelow >= 0 )
        {
            if ( sendCountBelow > 0 )
                MPI_Isend( sendBufferBelow.data(), sendCountBelow, PARTICLE,
                           neighborBelow, 1, MPI_COMM_WORLD,
                           &ghostRequests[numberOfGhostRequests++] );
            if ( receiveCountBelow > 0 )
                MPI_Irecv( receiveBufferBelow.data(), receiveCountBelow, PARTICLE,
                           neighborBelow, 1, MPI_COMM_WORLD,
                           &ghostRequests[numberOfGhostRequests++] );
        }

        if ( numberOfGhostRequests > 0 )
            MPI_Waitall( numberOfGhostRequests, ghostRequests, MPI_STATUSES_IGNORE );

        // append ghost particles to combined buffer and bin into cells
        if ( receiveCountAbove > 0 )
            memcpy( combinedParticles + localCount,
                    receiveBufferAbove.data(),
                    receiveCountAbove * sizeof(particle_t) );
        if ( receiveCountBelow > 0 )
            memcpy( combinedParticles + localCount + receiveCountAbove,
                    receiveBufferBelow.data(),
                    receiveCountBelow * sizeof(particle_t) );

        for ( int g = 0; g < ghostCount; g++ )
        {
            int idx = localCount + g;
            int cellX = (int)( combinedParticles[idx].x / cellWidth );
            int cellY = (int)( combinedParticles[idx].y / cellWidth );
            if ( cellX >= numberCells ) cellX = numberCells - 1;
            if ( cellX < 0 ) cellX = 0;
            if ( cellY >= numberCells ) cellY = numberCells - 1;
            if ( cellY < 0 ) cellY = 0;
            int localRow = cellY - globalToLocalRowOffset;
            if ( localRow >= 0 && localRow < totalLocalRows )
            {
                int flatIndex = localRow * numberCells + cellX;
                nextInCell[idx] = cellFirst[flatIndex];
                cellFirst[flatIndex] = idx;
            }
        }

        //
        //  compute forces for local particles using cell list (O(n/p) expected)
        //
        for ( int i = 0; i < localCount; i++ )
        {
            localParticles[i].ax = 0;
            localParticles[i].ay = 0;

            int cellX = (int)( localParticles[i].x / cellWidth );
            int cellY = (int)( localParticles[i].y / cellWidth );
            if ( cellX >= numberCells ) cellX = numberCells - 1;
            if ( cellX < 0 ) cellX = 0;
            if ( cellY >= numberCells ) cellY = numberCells - 1;
            if ( cellY < 0 ) cellY = 0;

            // check 3x3 neighborhood of cells
            for ( int offsetY = -1; offsetY <= 1; offsetY++ )
            {
                int localRow = ( cellY - globalToLocalRowOffset ) + offsetY;
                if ( localRow < 0 || localRow >= totalLocalRows ) continue;
                for ( int offsetX = -1; offsetX <= 1; offsetX++ )
                {
                    int neighborCellX = cellX + offsetX;
                    if ( neighborCellX < 0 || neighborCellX >= numberCells ) continue;
                    int flatIndex = localRow * numberCells + neighborCellX;
                    for ( int j = cellFirst[flatIndex]; j != -1; j = nextInCell[j] )
                        apply_force( localParticles[i], combinedParticles[j],
                                     &dmin, &davg, &navg );
                }
            }
        }

        //
        //  move particles
        //
        for ( int i = 0; i < localCount; i++ )
            move( localParticles[i] );

        // migrate particles that moved outside this processor's y-range
        migrateBufferUp.clear();
        migrateBufferDown.clear();
        int writeIndex = 0;
        for ( int i = 0; i < localCount; i++ )
        {
            double y = localParticles[i].y;
            if ( neighborAbove >= 0 && y < myYStart )
                migrateBufferUp.push_back( localParticles[i] );
            else if ( neighborBelow >= 0 && y >= myYEnd )
                migrateBufferDown.push_back( localParticles[i] );
            else
                localParticles[writeIndex++] = localParticles[i];
        }
        localParticles.resize( writeIndex );

        // exchange migration counts with neighbors
        int migrateUpCount = (int)migrateBufferUp.size();
        int migrateDownCount = (int)migrateBufferDown.size();
        int receiveFromAboveCount = 0, receiveFromBelowCount = 0;

        if ( neighborAbove >= 0 )
            MPI_Sendrecv( &migrateUpCount, 1, MPI_INT, neighborAbove, 2,
                          &receiveFromAboveCount, 1, MPI_INT, neighborAbove, 2,
                          MPI_COMM_WORLD, MPI_STATUS_IGNORE );
        if ( neighborBelow >= 0 )
            MPI_Sendrecv( &migrateDownCount, 1, MPI_INT, neighborBelow, 2,
                          &receiveFromBelowCount, 1, MPI_INT, neighborBelow, 2,
                          MPI_COMM_WORLD, MPI_STATUS_IGNORE );

        receiveFromAbove.resize( receiveFromAboveCount );
        receiveFromBelow.resize( receiveFromBelowCount );

        // exchange migrating particles using non-blocking communication
        MPI_Request migrateRequests[4];
        int numberOfMigrateRequests = 0;

        if ( neighborAbove >= 0 )
        {
            if ( migrateUpCount > 0 )
                MPI_Isend( migrateBufferUp.data(), migrateUpCount, PARTICLE,
                           neighborAbove, 3, MPI_COMM_WORLD,
                           &migrateRequests[numberOfMigrateRequests++] );
            if ( receiveFromAboveCount > 0 )
                MPI_Irecv( receiveFromAbove.data(), receiveFromAboveCount, PARTICLE,
                           neighborAbove, 3, MPI_COMM_WORLD,
                           &migrateRequests[numberOfMigrateRequests++] );
        }
        if ( neighborBelow >= 0 )
        {
            if ( migrateDownCount > 0 )
                MPI_Isend( migrateBufferDown.data(), migrateDownCount, PARTICLE,
                           neighborBelow, 3, MPI_COMM_WORLD,
                           &migrateRequests[numberOfMigrateRequests++] );
            if ( receiveFromBelowCount > 0 )
                MPI_Irecv( receiveFromBelow.data(), receiveFromBelowCount, PARTICLE,
                           neighborBelow, 3, MPI_COMM_WORLD,
                           &migrateRequests[numberOfMigrateRequests++] );
        }

        if ( numberOfMigrateRequests > 0 )
            MPI_Waitall( numberOfMigrateRequests, migrateRequests, MPI_STATUSES_IGNORE );

        // add received migrant particles to local list
        localParticles.insert( localParticles.end(),
                               receiveFromAbove.begin(), receiveFromAbove.end() );
        localParticles.insert( localParticles.end(),
                               receiveFromBelow.begin(), receiveFromBelow.end() );

        if( find_option( argc, argv, "-no" ) == -1 )
        {
          
          MPI_Reduce(&davg,&reducedDavg,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
          MPI_Reduce(&navg,&reducedNavg,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);
          MPI_Reduce(&dmin,&reducedDmin,1,MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD);

 
          if (rank == 0){
            //
            // Computing statistical data
            //
            if (reducedNavg) {
              absavg +=  reducedDavg/reducedNavg;
              nabsavg++;
            }
            if (reducedDmin < absmin) absmin = reducedDmin;
          }

          //
          //  save if necessary
          //
          if ( fsave && ( step % SAVEFREQ ) == 0 )
          {
              int currentLocalCount = (int)localParticles.size();
              MPI_Gather( &currentLocalCount, 1, MPI_INT,
                          gatherCounts, 1, MPI_INT, 0, MPI_COMM_WORLD );
              if ( rank == 0 )
              {
                  gatherDisplacements[0] = 0;
                  for ( int i = 1; i < numberOfProcessors; i++ )
                      gatherDisplacements[i] = gatherDisplacements[i - 1] + gatherCounts[i - 1];
              }
              MPI_Gatherv( localParticles.data(), currentLocalCount, PARTICLE,
                           gatherBuffer, gatherCounts, gatherDisplacements, PARTICLE,
                           0, MPI_COMM_WORLD );
              if ( rank == 0 )
                  save( fsave, n, gatherBuffer );
          }
        }
    }
    simulation_time = read_timer( ) - simulation_time;
  
    if (rank == 0) {  
      printf( "n = %d, simulation time = %g seconds", n, simulation_time);

      if( find_option( argc, argv, "-no" ) == -1 )
      {
        if (nabsavg) absavg /= nabsavg;
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
      if( fsum)
        fprintf(fsum,"%d %d %g\n",n,numberOfProcessors,simulation_time);
    }
  
    //
    //  release resources
    //
    free( processorRowStart );
    free( cellFirst );
    free( nextInCell );
    free( combinedParticles );
    if ( gatherCounts ) free( gatherCounts );
    if ( gatherDisplacements ) free( gatherDisplacements );
    if ( gatherBuffer ) free( gatherBuffer );
    if ( fsum )
        fclose( fsum );
    if( fsave )
        fclose( fsave );
    
    MPI_Finalize( );
    
    return 0;
}
