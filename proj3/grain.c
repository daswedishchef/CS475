#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

int	NowYear;		// 2020 - 2025
int	NowMonth;		// 0 - 11

float	NowPrecip;		// inches of rain per month
float	NowTemp;		// temperature this month
float	NowHeight;		// grain height in inches
float	nextHeight;
int	NowNumDeer;		// number of deer in the current population
int     nextDeer;
float   NowDew;
float   nextDew;

const float GRAIN_GROWS_PER_MONTH =		9.0;
const float ONE_DEER_EATS_PER_MONTH =		1.0;

const float AVG_PRECIP_PER_MONTH =		7.0;	// average
const float AMP_PRECIP_PER_MONTH =		6.0;	// plus or minus
const float RANDOM_PRECIP =			2.0;	// plus or minus noise

const float AVG_TEMP =				60.0;	// average
const float AMP_TEMP =				20.0;	// plus or minus
const float RANDOM_TEMP =			10.0;	// plus or minus noise

const float MIDTEMP =				40.0;
const float MIDPRECIP =				10.0;

unsigned int seed = 0;

omp_lock_t	Lock;
int		NumInThreadTeam;
int		NumAtBarrier;
int		NumGone;

// specify how many threads will be in the barrier:
//	(also init's the Lock)

void InitBarrier( int n )
{
        NumInThreadTeam = n;
        NumAtBarrier = 0;
	omp_init_lock( &Lock );
}


// have the calling thread wait here until all the other threads catch up:

void WaitBarrier( )
{
        omp_set_lock( &Lock );
        {
                NumAtBarrier++;
                if( NumAtBarrier == NumInThreadTeam )
                {
                        NumGone = 0;
                        NumAtBarrier = 0;
                        // let all other threads get back to what they were doing
			// before this one unlocks, knowing that they might immediately
			// call WaitBarrier( ) again:
                        while( NumGone != NumInThreadTeam-1 );
                        omp_unset_lock( &Lock );
                        return;
                }
        }
        omp_unset_lock( &Lock );

        while( NumAtBarrier != 0 );	// this waits for the nth thread to arrive

        #pragma omp atomic
        NumGone++;			// this flags how many threads have returned
}

float Ranf( unsigned int *seedp,  float low, float high )
{
        float r = (float) rand_r( seedp );              // 0 - RAND_MAX

        return(   low  +  r * ( high - low ) / (float)RAND_MAX   );
}

void getit(){
        double ang = (  30.*(float)NowMonth + 15.  ) * ( M_PI / 180. );
        float temp = AVG_TEMP - AMP_TEMP * cos( ang );
        NowTemp = temp + Ranf( &seed, -RANDOM_TEMP, RANDOM_TEMP );
        float precip = AVG_PRECIP_PER_MONTH + AMP_PRECIP_PER_MONTH * sin( ang );
        NowPrecip = precip + Ranf( &seed,  -RANDOM_PRECIP, RANDOM_PRECIP );
        if( NowPrecip < 0. )
	        NowPrecip = 0.;
        NowTemp = (5./9.)*(NowTemp-32);
}


float SQR( float x )
{
        return x*x;
}

void getheight(){
        float tempFactor = exp(   -SQR(  ( NowTemp - MIDTEMP ) / 10.  )   );
        float precipFactor = exp(   -SQR(  ( NowPrecip - MIDPRECIP ) / 10.  )   );
        nextHeight = NowHeight;
        nextHeight += tempFactor * precipFactor * GRAIN_GROWS_PER_MONTH;
        nextHeight += (NowDew / 5);
        nextHeight -= (float)NowNumDeer * ONE_DEER_EATS_PER_MONTH;
        if(nextHeight < 0.){
                nextHeight = 0.;
        }
}

void height(){
        NowHeight = nextHeight;
}

void getdeer(){
        if((float)NowNumDeer >= NowHeight){
                nextDeer = NowNumDeer - 1;
        }
        else{
                nextDeer = NowNumDeer + 1;
        }
        if(nextDeer < 0){
                nextDeer = 0;
        }
}

void deer(){
        NowNumDeer = nextDeer;
}

void getdew(){
        nextDew = (NowTemp / 100) * NowPrecip + Ranf( &seed,  -RANDOM_PRECIP, RANDOM_PRECIP );
        if(nextDew < 0.){
                nextDew = 0.;
        }
}

void dew(){
        NowDew = nextDew;
}

void GrainDeer(){
        while( NowYear < 2026 )
        {
        getdeer();
	// DoneComputing barrier:
	WaitBarrier();
        deer();
	// DoneAssigning barrier:
	WaitBarrier();

	// DonePrinting barrier:
	WaitBarrier();
        }
}

void Grain(){
        while( NowYear < 2026 )
        {
        getheight();
	// DoneComputing barrier:
	WaitBarrier();
        height();
	// DoneAssigning barrier:
	WaitBarrier();

	// DonePrinting barrier:
	WaitBarrier();
        }
}

void Watcher(){
        FILE *grass = fopen("grass.txt", "w");
        FILE *temp = fopen("Temp.txt", "w");
        FILE *precip = fopen("prec.txt", "w");
        FILE *dew = fopen("dew.txt", "w");
        FILE *deer = fopen("deer.txt","w");
        while( NowYear < 2026 )
        {
        
	// DoneComputing barrier:
	WaitBarrier();
        
	// DoneAssigning barrier:
	WaitBarrier();
        fprintf(grass,"%lf\n",NowHeight);
        fprintf(temp,"%lf\n",NowTemp);
        fprintf(precip,"%lf\n",NowPrecip);
        fprintf(dew,"%lf\n",NowDew);
        fprintf(deer,"%d\n",NowNumDeer);
        //fprintf(output,"%d      %d      %lf     %lf     %lf     %d      %lf\n", NowMonth, NowYear, NowTemp, NowPrecip, NowHeight, NowNumDeer, NowDew);
        NowMonth += 1;
        if(NowMonth == 12){
                NowMonth = 0;
                NowYear += 1;
        }
        getit();
	// DonePrinting barrier:
	WaitBarrier();
        }
        fclose(deer);
        fclose(dew);
        fclose(precip);
        fclose(temp);
        fclose(grass);
}

void MyAgent(){
        while( NowYear < 2026 )
        {
        getdew();
	// DoneComputing barrier:
	WaitBarrier();
        dew();
	// DoneAssigning barrier:
	WaitBarrier();

	// DonePrinting barrier:
	WaitBarrier();
        }
}

int main(int argc, char *argv[]){
        NowMonth = 0;
        NowYear = 2020;
        NowNumDeer = 1;
        NowHeight = 1.;
        getit();
        omp_set_num_threads( 4 );	// same as # of sections
        InitBarrier(4);
        #pragma omp parallel sections
        {
                #pragma omp section
                {
                        GrainDeer( );
                }

                #pragma omp section
                {
                        Grain( );
                }

                #pragma omp section
                {
                        Watcher();
                }

                #pragma omp section
                {
                        MyAgent( );	// your own
                }
        }       // implied barrier -- all functions must return in order
        // to allow any of them to get past here
}