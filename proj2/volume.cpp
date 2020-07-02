#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <time.h>

#define XMIN     -1.
#define XMAX      1.
#define YMIN     -1.
#define YMAX      1.

float
Height( int iu, int iv, int NUMNODES )	// iu,iv = 0 .. NUMNODES-1
{
	int N = 2;
	float x = -1.  +  2.*(float)iu /(float)(NUMNODES-1);	// -1. to +1.
	float y = -1.  +  2.*(float)iv /(float)(NUMNODES-1);	// -1. to +1.

	float xn = pow( fabs(x), (double)N );
	float yn = pow( fabs(y), (double)N );
	float r = 1. - xn - yn;
	if( r < 0. )
	        return 0.;
	float height = pow( 1. - xn - yn, 1./(float)N );
	return height;
}

void driver(int NUMT, int NUMNODES){
	double time0 = omp_get_wtime( );
	omp_set_num_threads( NUMT );
	double volume = 0;
	double fullTileArea = (  ( ( XMAX - XMIN )/(float)(NUMNODES-1) ) *( ( YMAX - YMIN )/(float)(NUMNODES-1) )  );
	double halftile = fullTileArea/2.;
	double quartile = halftile/2.;
    #pragma omp parallel for reduction(+:volume)
	for( int i = 0; i < NUMNODES*NUMNODES; i++ )
    {
        int iu = i % NUMNODES;
        int iv = i / NUMNODES;
        double z = Height( iu, iv, NUMNODES);
		//corner case
		if((iu == 0 || iu == NUMNODES-1)&&(iv == 0 || iv == NUMNODES-1)){
			volume += z*quartile;
		}//edge case
		else if(((iu == 0 || iu == NUMNODES-1)&&(iv != 0 || iv != NUMNODES-1))||((iu != 0 || iu != NUMNODES-1)&&(iv == 0 || iv == NUMNODES-1))){
			volume += z*halftile;
		}
		else{
			volume += z*fullTileArea;
		}
		//fprintf(stderr,"x: %d, y: %d,z: %lf, volume: %lf\n",iu,iv,z,volume);
    }
	double time1 = omp_get_wtime( );
	double megaTiles = (double)NUMNODES*NUMNODES / ( time1 - time0 ) / 1000000.;
	fprintf(stderr,"Performance %lf\n", megaTiles);
}


int main( int argc, char *argv[ ] )
{
	for(int i = 1; i<9; i+=i){
		fprintf(stderr,"----%d Threads----\n",i);
		for(int j = 100; j < 1001; j+=100){
			fprintf(stderr,"-%d NumTrials-    ",j);
			driver(i,j);
		}
	}
}