#include <stdlib.h>
#include <iostream>

#include "GLUTWindowManagers.h"
#include "cuda_gl_interop.h"

#include "MandelBrotImage.hpp"
#include "JuliaImage.hpp"

#include "cudaTools.h"
#include "deviceTools.h"

int bench(int argc, char** argv);
int launchApplication(int argc, char** argv);

int main(int argc, char** argv){
    //return launchApplication();
    return bench(argc, argv);
}

int launchApplication(int argc, char** argv){
    if (nbDeviceDetect() >= 1){
	int deviceId = 2;

	HANDLE_ERROR(cudaSetDevice(deviceId)); // active gpu of deviceId
	HANDLE_ERROR(cudaSetDeviceFlags(cudaDeviceMapHost)); // Not all gpu allow the use of mapMemory (avant prremier appel au kernel)
	HANDLE_ERROR(cudaGLSetGLDevice(deviceId));

	GLUTWindowManagers::init(argc, argv);
	GLUTWindowManagers* glutWindowManager = GLUTWindowManagers::getInstance();

	int w = 800;
	int h = 600;

	GLImageFonctionelCudaSelections* image;

	bool mandelbrot = false;

	if(mandelbrot){
	    std::cout << "Launch MandelBrot in Cuda" << std::endl;

	    float xMin = -1.3968;
	    float xMax = -1.3578;
	    float yMin = -0.03362;
	    float yMax = 0.0013973;

	    DomaineMaths domain(xMin, yMin, xMax - xMin, yMax - yMin);

	    image = new GLMandelBrotImage(w, h, domain);
	} else {
	    std::cout << "Launch Julia in Cuda" << std::endl;

	    float xMin = -1.7;
	    float xMax = +1.7;
	    float yMin = -1.1;
	    float yMax = +1.1;

	    DomaineMaths domain(xMin, yMin, xMax - xMin, yMax - yMin);

	    image = new GLJuliaImage(w, h, domain);
	}

	glutWindowManager->createWindow(image);
	glutWindowManager->runALL(); //Blocking

	delete image;

	return EXIT_SUCCESS;
    } else {
	return EXIT_FAILURE;
    }
}

#define DIM_H 2000
#define DIM_W 2000
#define TIMES 20

void launchMandelBrotAnimation(uchar4* ptrDevPixels, int w, int h, int N, const DomaineMaths& domainNew);
void launchJuliaAnimation(uchar4* ptrDevPixels, int w, int h, int N, const DomaineMaths& domainNew);

void bench(int N, uchar4* image){
    std::cout << "Start Benchmark with N=" << N << std::endl;

    CUevent start;
    CUevent stop;
    HANDLE_ERROR(cudaEventCreate(&start, CU_EVENT_DEFAULT));
    HANDLE_ERROR(cudaEventCreate(&stop, CU_EVENT_DEFAULT));
    HANDLE_ERROR(cudaEventRecord(start,0));

    float xMin = -1.3968;
    float xMax = -1.3578;
    float yMin = -0.03362;
    float yMax = 0.0013973;

    DomaineMaths domain1(xMin, yMin, xMax - xMin, yMax - yMin);

    for(int i = 0; i < TIMES; ++i){
	launchMandelBrotAnimation(image, DIM_W, DIM_H, N, domain1);
    }

    float elapsed = 0;
    HANDLE_ERROR(cudaEventRecord(stop,0));
    HANDLE_ERROR(cudaEventSynchronize(stop));
    HANDLE_ERROR(cudaEventElapsedTime(&elapsed, start, stop));

    std::cout << "MandelBrot Total (" << TIMES << " times) = " << elapsed << "ms" << std::endl;
    std::cout << "MandelBrot Mean  (" << TIMES << " times) = " << (elapsed / TIMES) << "ms" << std::endl;

    HANDLE_ERROR(cudaEventRecord(start,0));

    xMin = -1.3968;
    xMax = -1.3578;
    yMin = -0.03362;
    yMax = 0.0013973;

    DomaineMaths domain2(xMin, yMin, xMax - xMin, yMax - yMin);

    for(int i = 0; i < TIMES; ++i){
	launchJuliaAnimation(image, DIM_W, DIM_H, N, domain2);
    }

    elapsed = 0;
    HANDLE_ERROR(cudaEventRecord(stop,0));
    HANDLE_ERROR(cudaEventSynchronize(stop));
    HANDLE_ERROR(cudaEventElapsedTime(&elapsed, start, stop));

    std::cout << "Julia Total (" << TIMES << " times) = " << elapsed << "ms" << std::endl;
    std::cout << "Julia Mean  (" << TIMES << " times) = " << (elapsed / TIMES) << "ms" << std::endl;

    HANDLE_ERROR(cudaEventDestroy(start));
    HANDLE_ERROR(cudaEventDestroy(stop));
}

int bench(int argc, char** argv){
    std::cout << "Launch benchmark" << std::endl;

    if (nbDeviceDetect() >= 1){
	int deviceId = 1;

	HANDLE_ERROR(cudaSetDevice(deviceId)); // active gpu of deviceId
	HANDLE_ERROR(cudaSetDeviceFlags(cudaDeviceMapHost)); // Not all gpu allow the use of mapMemory (avant prremier appel au kernel)

	//Force the driver to run
	uchar4* image;
	HANDLE_ERROR(cudaMalloc(&image, DIM_W * DIM_H * sizeof(uchar4)));

	std::cout << "End of malloc" << std::endl;
	std::cout << "Size of the image: " << (DIM_W * DIM_H * sizeof(uchar4)) << std::endl;

	bench(10, image);
	bench(25, image);
	bench(50, image);
	bench(100, image);

	HANDLE_ERROR(cudaFree(image));

	return EXIT_SUCCESS;
    } else {
	return EXIT_FAILURE;
    }
}
