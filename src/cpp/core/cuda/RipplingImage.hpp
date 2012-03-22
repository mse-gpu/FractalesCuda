#ifndef GL_MANDELBROT_IMAGE
#define GL_MANDELBROT_IMAGE

#include <iostream>
#include "cudaTools.h"

#include "GLImageCudas.h"

class GLRipplingImage : public GLImageCudas {
    public:
	GLRipplingImage(int dx, int dy);
	virtual ~GLRipplingImage();

    protected:
	virtual void performKernel(uchar4* ptrDevPixels, int w, int h);
	virtual void idleFunc();

    private:
	float t;
	float dt;
};

#endif
