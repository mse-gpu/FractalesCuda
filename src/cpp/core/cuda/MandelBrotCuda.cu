#include <iostream>
#include "GLImageCudas.h"
#include "ColorTools_Device.h"

__global__ static void mandelBrotAnimation(uchar4* ptrDevPixels, int w, int h, float t);

__device__ static float color(int w, int h, float x, float y, float t);
__device__ static float d(int w, int h, float x, float y);

void launchMandelBrotAnimation(uchar4* ptrDevPixels, int w, int h, float t){
    dim3 blockPerGrid = dim3(32, 32, 1);
    dim3 threadPerBlock = dim3(16, 16, 1);

    mandelBrotAnimation<<<blockPerGrid,threadPerBlock>>>(ptrDevPixels, w, h, t);
}

__global__ static void mandelBrotAnimation(uchar4* ptrDevPixels, int w, int h, float t){
    int i = threadIdx.y + blockIdx.y * blockDim.y;
    int j = threadIdx.x + blockIdx.x * blockDim.x;

    int nbThreadY = gridDim.y * blockDim.y;
    int nbThreadX = gridDim.x * blockDim.x;
    int nbThreadCuda = nbThreadY * nbThreadX;

    int tid = j +  (i * nbThreadX);
    int pixelI;
    int pixelJ;

    while(tid < (w * h)){
	pixelI = tid / w;
	pixelJ = tid - w * pixelI;

	float c = color(w, h, pixelI, pixelJ, t);

	ptrDevPixels[tid].x = c;
	ptrDevPixels[tid].y = c;
	ptrDevPixels[tid].z = c;
	ptrDevPixels[tid].w = 255;

	tid += nbThreadCuda;
    }
}

__device__ static float color(int w, int h, float x, float y, float t){
    return 128 + 127 * ((cos(d(w, h, x,y) / (float)10 -(t / (float)7))) / (d(w, h, x, y) / 10 + 1));
}

__device__ static float d(int w, int h, float x, float y){
    float fx = x - (w / 2);
    float fy = y - (h / 2);

    return sqrt(fx * fx + fy * fy);
}
