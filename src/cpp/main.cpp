#include <stdlib.h>
#include <iostream>

#include "GLUTWindowManagers.h"
#include "cuda_gl_interop.h"

#include "MandelBrotImage.hpp"
#include "JuliaImage.hpp"

#include "cudaTools.h"
#include "deviceTools.h"

int main(int argc, char** argv){
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
