#include <xmmintrin.h>
#include <omp.h>
#include <stdio.h>
#define SSE_WIDTH		4
#define MAX 10000000
float a[MAX];
float b[MAX];

float SimdMulSum( float *a, float *b, int len )
{
	float sum[4] = { 0., 0., 0., 0. };
	int limit = ( len/SSE_WIDTH ) * SSE_WIDTH;
	register float *pa = a;
	register float *pb = b;

	__m128 ss = _mm_loadu_ps( &sum[0] );
	for( int i = 0; i < limit; i += SSE_WIDTH )
	{
		ss = _mm_add_ps( ss, _mm_mul_ps( _mm_loadu_ps( pa ), _mm_loadu_ps( pb ) ) );
		pa += SSE_WIDTH;
		pb += SSE_WIDTH;
	}
	_mm_storeu_ps( &sum[0], ss );

	for( int i = limit; i < len; i++ )
	{
		sum[0] += a[i] * b[i];
	}
	return sum[0] + sum[1] + sum[2] + sum[3];
}

float omul(float *a,float *b,int len){
	float sum = 0.;
	#pragma omp simd
	for(int i = 0; i < len; i++){
		sum += a[i]*b[i];
	}
	return sum;
}

float mul(float *a,float *b,int len){
	float sum = 0.;
	for(int i = 0; i < len; i++){
		sum += a[i]*b[i];
	}
	return sum;
}


int main(){
	for(int i = 0; i < MAX; i++){
		a[i] = (rand() % 100)/7;
		b[i] = (rand() % 100)/13;
	}
	printf("Array size: %d\n",MAX);
	double time0 = omp_get_wtime( );
	SimdMulSum(a,b,MAX);
	double time1 = omp_get_wtime( );
	double ssetime = time1-time0;
	double megaMults = (double)MAX/(ssetime)/1000000.;
	printf("SIMD SSE: %lf\nPerformance: %lf\n",ssetime,megaMults);
	time0 = omp_get_wtime( );
	mul(a,b,MAX);
	time1 = omp_get_wtime( );
	double mtime = time1-time0;
	megaMults = (double)MAX/(mtime)/1000000.;
	printf("Nothing: %lf\nPerformance: %lf\n",mtime,megaMults);
	__builtin_prefetch ( &a[MAX] );
	__builtin_prefetch ( &b[MAX] );
	time0 = omp_get_wtime( );
	omul(a,b,MAX);
	time1 = omp_get_wtime( );
	double otime = time1-time0;
	megaMults = (double)MAX/(otime)/1000000.;
	printf("OMP SIMD: %lf\nPerformance: %lf\n",otime,megaMults);
	printf("Speedup: %lf\n",otime/ssetime);

}