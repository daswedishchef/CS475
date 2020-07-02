//--------------------------
//Compile with gcc -o <executable> -fopenmp multiply.c
//Run with a single argument specifying the size of the arrays
//It will create and print to a file called results.txt
//It will append to this file upon addiitonal runs
//--------------------------
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define NUMTRIES        20

double multiply(int SIZE, int NUMT, FILE *output){
        double megasum = 0.;
	// inialize the arrays:
        float *A = malloc(SIZE*sizeof(float));
        float *B = malloc(SIZE*sizeof(float));
        float *C = malloc(SIZE*sizeof(float));
        //give them random values
        int i, t;
        srand(time(NULL));
	for( i = 0; i < SIZE; i++ )
	{
		A[ i ] = (float)(rand()%100)/10;
		B[ i ] = (float)(rand()%100)/10;
	}
        printf("Testing with %d threads, and a datasize of %d %d times\n",NUMT,SIZE,NUMTRIES);
        //I used the multiplication loop from lecture to keep results consistent
        omp_set_num_threads( NUMT );
        double maxMegaMults = 0.;
        double timesum = 0.;
        for( t = 0; t < NUMTRIES; t++ )
        {
                double time0 = omp_get_wtime( );

                #pragma omp parallel for
                for( i = 0; i < SIZE; i++ )
                {
                        C[i] = A[i] * B[i];
                }

                double time1 = omp_get_wtime( );
                double megaMults = (double)SIZE/(time1-time0)/1000000.;
                megasum += megaMults;
                timesum += (time1-time0);
                if( megaMults > maxMegaMults )
                        maxMegaMults = megaMults;
        }
        //get averages
        megasum = megasum/NUMTRIES;
        timesum = timesum/NUMTRIES;
        //print to file
        fprintf(output,"Threads = %d \nData Size = %d\nAverage Time = %1.6lf\nPeak Performance = %8.2lf MegaMults/Sec\nAverage Performance = %8.2lf MegaMults/Sec\n\n",NUMT,SIZE,timesum,maxMegaMults,megasum);
        free(A);
        free(B);
        free(C);
        return megasum;
}

int main(int argc, char *argv[])
{
#ifndef _OPENMP
        fprintf( stderr, "OpenMP is not supported here -- sorry.\n" );
        return 1;
#endif
        int SIZE = atoi(argv[1]);
        double onethrd = 0.;
        double fourthrd = 0.;
        double s = 0.;
        FILE *output = fopen("results.txt", "a");
        if(output == NULL){
                output = fopen("results.txt", "w");
        }
        //run first test with 1 thread
        fprintf(output,"/////////////////////////\n--- Test 1 ---\n");
        onethrd = multiply(SIZE,1,output);
        //run second test with 4 threads
        fprintf(output,"--- Test 2 ---\n");
        fourthrd = multiply(SIZE,4,output);
        s = fourthrd/onethrd;
        float fp = (4./3.)*(1.-(1./s));
        fprintf(output,"--- Results ---\nSpeedup S = %lf\nParallel Fraction = %f\n/////////////////////////\n\n",s,fp);
        fclose(output);
        return 0;
}