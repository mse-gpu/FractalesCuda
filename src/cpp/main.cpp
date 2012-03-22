#include <stdlib.h>
#include <iostream>

#include "GLUTWindowManagers.h"
#include "MandelBrotImage.hpp"
#include "cuda_gl_interop.h"

#include "cudaTools.h"
#include "deviceTools.h"

int main(int argc, char** argv){
    if (nbDeviceDetect() >= 1){
	int deviceId = 2;

	std::cout << "Launch MandelBrot in Cuda" << std::endl;

	HANDLE_ERROR(cudaSetDevice(deviceId)); // active gpu of deviceId
	HANDLE_ERROR(cudaSetDeviceFlags(cudaDeviceMapHost)); // Not all gpu allow the use of mapMemory (avant prremier appel au kernel)
	HANDLE_ERROR(cudaGLSetGLDevice(deviceId));

	GLUTWindowManagers::init(argc, argv);
	GLUTWindowManagers* glutWindowManager = GLUTWindowManagers::getInstance();

	float xMin = -1.3968;
	float xMax = -1.3578;
	float yMin = -0.03362;
	float yMax = 0.0013973;

	DomaineMaths domain(xMin, yMin, xMax - xMin, yMax - yMin);

	int w = 800;
	int h = 600;

	GLMandelBrotImage* image = new GLMandelBrotImage(w, h, domain);

	glutWindowManager->createWindow(image);
	glutWindowManager->runALL(); //Blocking

	delete image;

	return EXIT_SUCCESS;
    } else {
	return EXIT_FAILURE;
    }
}
