#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <fstream>
#include <iostream>
#include <utility>
#include <omp.h>
#ifdef WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif
#include <stdio.h>
#include <xmmintrin.h>
#include "cl.h"
#include "cl_platform.h"
#define SSE_WIDTH		4
#define MAX 32768

#ifndef LOCAL_SIZE
#define	LOCAL_SIZE	 128
#endif

#define	NUM_WORK_GROUPS		NUM_ELEMENTS/LOCAL_SIZE

const char *			CL_FILE_NAME = { "corr.cl" };
const float			TOL = 0.0001f;

void				Wait( cl_command_queue );
int				LookAtTheBits( float );

// opencl vendor ids:
#define ID_AMD          0x1002
#define ID_INTEL        0x8086
#define ID_NVIDIA       0x10de



void correlate(float *sig, float *sums, int Size){
    for( int shift = 0; shift < Size; shift++ )
    {
        float sum = 0.;
        for( int i = 0; i < Size; i++ )
        {
            sum += sig[i] * sig[i + shift];
        }
        sums[shift] = sum;
    }
}

void porrelate(float *sig, float *sums, int size,int t){
    omp_set_num_threads( t );
    #pragma omp parallel for
    for( int shift = 0; shift < size; shift++ )
    {
        float sum = 0.;
        for( int i = 0; i < size; i++ )
        {
            sum += sig[i] * sig[i + shift];
        }
        sums[shift] = sum;
    }
}

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

void simp( float *a,float *sums, int len )
{
    for(int shift = 0; shift < len;shift++){
        sums[shift] = SimdMulSum(&a[0],&a[0+shift],len);
    }
}

int gorrelate(float* sig, float *sums, int size){
	// see if we can even open the opencl kernel program
	// (no point going on if we can't):

	FILE *fp;
#ifdef WIN32
	errno_t err = fopen_s( &fp, CL_FILE_NAME, "r" );
	if( err != 0 )
#else
	fp = fopen( CL_FILE_NAME, "r" );
	if( fp == NULL )
#endif
	{
		fprintf( stderr, "Cannot open OpenCL source file '%s'\n", CL_FILE_NAME );
		return 1;
	}

	cl_int status;		// returned status from opencl calls
				// test against CL_SUCCESS

	// get the platform id:


	cl_platform_id platform;
	status = clGetPlatformIDs( 1, &platform, NULL );
	if( status != CL_SUCCESS )
		fprintf( stderr, "clGetPlatformIDs failed (2)\n" );
	
	// get the device id:
	

	cl_device_id device;
	status = clGetDeviceIDs( platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL );
	if( status != CL_SUCCESS )
		fprintf( stderr, "clGetDeviceIDs failed (2)\n" );


	// 2. allocate the host memory buffers:
	float *hA = new float[ size*2 ];
	float *hB = new float[ size ];
	// fill the host memory buffers:

	hA = sig;
    hB = sums;
	size_t asize = size * 2 * sizeof(float);
    size_t bsize = size * sizeof(float);

	// 3. create an opencl context:

	cl_context context = clCreateContext( NULL, 1, &device, NULL, NULL, &status );
	if( status != CL_SUCCESS )
		fprintf( stderr, "clCreateContext failed\n" );

	// 4. create an opencl command queue:

	cl_command_queue cmdQueue = clCreateCommandQueue( context, device, 0, &status );
	if( status != CL_SUCCESS )
		fprintf( stderr, "clCreateCommandQueue failed\n" );

	// 5. allocate the device memory buffers:

	cl_mem dA = clCreateBuffer( context, CL_MEM_READ_ONLY,  asize, NULL, &status );
	if( status != CL_SUCCESS )
		fprintf( stderr, "clCreateBuffer failed (1)\n" );

	cl_mem dB = clCreateBuffer( context, CL_MEM_WRITE_ONLY,  bsize, NULL, &status );
	if( status != CL_SUCCESS )
		fprintf( stderr, "clCreateBuffer failed (2)\n" );


	// 6. enqueue the 2 commands to write the data from the host buffers to the device buffers:

	status = clEnqueueWriteBuffer( cmdQueue, dA, CL_FALSE, 0, asize, hA, 0, NULL, NULL );
	if( status != CL_SUCCESS )
		fprintf( stderr, "clEnqueueWriteBuffer failed (1)\n" );

	Wait( cmdQueue );

	// 7. read the kernel code from a file:

	fseek( fp, 0, SEEK_END );
	size_t fileSize = ftell( fp );
	fseek( fp, 0, SEEK_SET );
	char *clProgramText = new char[ fileSize+1 ];		// leave room for '\0'
	size_t n = fread( clProgramText, 1, fileSize, fp );
	clProgramText[fileSize] = '\0';
	fclose( fp );
	if( n != fileSize )
		fprintf( stderr, "Expected to read %d bytes read from '%s' -- actually read %d.\n", fileSize, CL_FILE_NAME, n );

	// create the text for the kernel program:

	char *strings[1];
	strings[0] = clProgramText;
	cl_program program = clCreateProgramWithSource( context, 1, (const char **)strings, NULL, &status );
	if( status != CL_SUCCESS )
		fprintf( stderr, "clCreateProgramWithSource failed\n" );
	delete [ ] clProgramText;

	// 8. compile and link the kernel code:

	char *options = { "" };
	status = clBuildProgram( program, 1, &device, options, NULL, NULL );
	if( status != CL_SUCCESS )
	{
		size_t size;
		clGetProgramBuildInfo( program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &size );
		cl_char *log = new cl_char[ size ];
		clGetProgramBuildInfo( program, device, CL_PROGRAM_BUILD_LOG, size, log, NULL );
		fprintf( stderr, "clBuildProgram failed:\n%s\n", log );
		delete [ ] log;
	}

	// 9. create the kernel object:
	//set second argument to ArrayMult for multi-add and ArrayMultRed for multi-reduction
	cl_kernel kernel = clCreateKernel( program, "AutoCorrelate", &status );
	if( status != CL_SUCCESS )
		fprintf( stderr, "clCreateKernel failed\n" );

	// 10. setup the arguments to the kernel object:

	status = clSetKernelArg( kernel, 0, sizeof(cl_mem), &dA );
	if( status != CL_SUCCESS )
		fprintf( stderr, "clSetKernelArg failed (1)\n" );

	status = clSetKernelArg( kernel, 1, sizeof(cl_mem), &dB );
	if( status != CL_SUCCESS )
		fprintf( stderr, "clSetKernelArg failed (2)\n" );

	// 11. enqueue the kernel object for execution:

	size_t globalWorkSize[3] = { MAX*2, 1, 1 };
	size_t localWorkSize[3]  = { LOCAL_SIZE,   1, 1 };

	Wait( cmdQueue );
	

	status = clEnqueueNDRangeKernel( cmdQueue, kernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL );
	if( status != CL_SUCCESS )
		fprintf( stderr, "clEnqueueNDRangeKernel failed: %d\n", status );

	Wait( cmdQueue );

	// 12. read the results buffer back from the device to the host:

	status = clEnqueueReadBuffer( cmdQueue, dB, CL_TRUE, 0, MAX, sums, 0, NULL, NULL );
	if( status != CL_SUCCESS )
			fprintf( stderr, "clEnqueueReadBuffer failed\n" );


#ifdef WIN32
	Sleep( 2 );
#endif


	// 13. clean everything up:

	clReleaseKernel(        kernel   );
	clReleaseProgram(       program  );
	clReleaseCommandQueue(  cmdQueue );
	clReleaseMemObject(     dA  );
	clReleaseMemObject(     dB  );

	delete [ ] hA;
	delete [ ] hB;
	return 0;
}


int
LookAtTheBits( float fp )
{
	int *ip = (int *)&fp;
	return *ip;
}

void
Wait( cl_command_queue queue )
{
      cl_event wait;
      cl_int      status;

      status = clEnqueueMarker( queue, &wait );
      if( status != CL_SUCCESS )
	      fprintf( stderr, "Wait: clEnqueueMarker failed\n" );

      status = clWaitForEvents( 1, &wait );
      if( status != CL_SUCCESS )
	      fprintf( stderr, "Wait: clWaitForEvents failed\n" );
}



void runit(float *sig,float *sums, int size){
    double time1,time2,time3,time4,time5;
    double time0 = omp_get_wtime( );
	time0 = omp_get_wtime( );
    correlate(sig,sums,size);
    time1 = omp_get_wtime( );
    simp(sig,sums,size);
    time2 = omp_get_wtime( );
    porrelate(sig,sums,size,1);
    time3 = omp_get_wtime( );
    porrelate(sig,sums,size,8);
    time4 = omp_get_wtime( );
	gorrelate(sig,sums,size);
	time5 = omp_get_wtime( );
    float p1 = ((double)MAX*MAX)/(time1-time0)/1000000.;
    float p2 = ((double)MAX*MAX)/(time2-time1)/1000000.;
    float p3 = ((double)MAX*MAX)/(time3-time2)/1000000.;
    float p4 = ((double)MAX*MAX)/(time4-time3)/1000000.;
	float p5 = ((double)MAX*MAX)/(time5-time4)/1000000.;
    fprintf(stderr,"-Performance-\nSimple: %lf\nSIMD: %lf\nParallel(1): %lf\nParallel(8): %lf\nOpenCL: %lf\n",p1,p2,p3,p4,p5);
}

int main(){
    FILE *fp = fopen( "signal.txt", "r" );
    if( fp == NULL )
    {
        fprintf( stderr, "Cannot open file 'signal.txt'\n" );
        exit(1);
    }
    int Size;
    fscanf( fp, "%d", &Size );
    float *A =     new float[ 2*Size ];
    float *Sums  = new float[ 1*Size ];
    for( int i = 0; i < Size; i++ )
    {
        fscanf( fp, "%f", &A[i] );
        A[i+Size] = A[i];		// duplicate the array
    }
    fclose( fp );
    runit(A,Sums,Size);
    return 0;
}
