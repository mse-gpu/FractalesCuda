#include "Tools.hpp"

__global__ static void juliaAnimation(uchar4* ptrDevPixels, int w, int h, int N, DomaineMaths domainNew, CalibreurCudas calibreur);

__device__ static float julia(float x, float y, int N);

void launchJuliaAnimation(uchar4* ptrDevPixels, int w, int h, int N, const DomaineMaths& domainNew){
    dim3 blockPerGrid = dim3(32, 32, 1);
    dim3 threadPerBlock = dim3(16, 16, 1);

    //TODO Check the value 0.7f
    CalibreurCudas calibreur(0, N, 0.0f, 0.7f);
    juliaAnimation<<<blockPerGrid,threadPerBlock>>>(ptrDevPixels, w, h, N, domainNew, calibreur);
}

__global__ static void juliaAnimation(uchar4* ptrDevPixels, int w, int h, int N, DomaineMaths domainNew, CalibreurCudas calibreur){
    int i = threadIdx.y + blockIdx.y * blockDim.y;
    int j = threadIdx.x + blockIdx.x * blockDim.x;

    int nbThreadY = gridDim.y * blockDim.y;
    int nbThreadX = gridDim.x * blockDim.x;
    int nbThreadCuda = nbThreadY * nbThreadX;

    float dx = (float) (domainNew.dx / (float) w);
    float dy = (float) (domainNew.dy / (float) h);

    unsigned char r, g, b;
    int tid = j +  (i * nbThreadX);

    float x, y;

    while(tid < (w * h)){
	int pixelI = tid / w;
	int pixelJ = tid - w * pixelI;

	x = domainNew.x0 + pixelJ * dx;
	y = domainNew.y0 + pixelI * dy;

	float h = julia(x, y, N);
	if(h == 0){
	    r = 0;
	    g = 0;
	    b = 0;
	} else {
	    h = calibreur.calibrate(h);
	    HSB_TO_RVB(h, 1.0, 1.0, r, g, b);
	}

	ptrDevPixels[tid].x = r;
	ptrDevPixels[tid].y = g;
	ptrDevPixels[tid].z = b;
	ptrDevPixels[tid].w = 255;

	tid += nbThreadCuda;
    }
}

#define CREAL -0.745
#define CIMAG 0.1

__device__ static float julia(float x, float y, int N){
    float real = x;
    float imag = y;

    float n = 0;
    float norm;

    do{
	float tmpReal = real;
	real = real * real - imag * imag + CREAL;
	imag = tmpReal * imag + imag * tmpReal + CIMAG;

	++n;

	norm = sqrt(real * real + imag * imag);
    } while (norm <= 2.0 && n < N);

    return n == N ? 0 : n;
}
