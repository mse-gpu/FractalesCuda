#ifndef GL_MANDELBROT_IMAGE
#define GL_MANDELBROT_IMAGE

#include <iostream>
#include "cudaTools.h"

#include "GLImageCudas.h"

class GLMandelBrotImage : public GLImageCudas {
    public:
	GLMandelBrotImage(int dx, int dy);
	virtual ~GLMandelBrotImage();

    protected:
	virtual void performKernel(uchar4* ptrDevPixels, int w, int h);
	virtual void idleFunc();

    private:
	float t;
	float dt;
};

#endif
