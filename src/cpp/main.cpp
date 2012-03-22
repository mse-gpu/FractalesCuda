#include <stdlib.h>

#include "GLUTWindowManagers.h"
#include "RipplingImage.hpp"
#include "cuda_gl_interop.h"

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

	int w = 600;
	int h = 600;

	GLRipplingImage* image = new GLRipplingImage(w, h);

	glutWindowManager->createWindow(image);
	glutWindowManager->runALL(); //Blocking

	delete image;

	return EXIT_SUCCESS;
    } else {
	return EXIT_FAILURE;
    }
}
