#ifndef GL_MANDELBROT_IMAGE
#define GL_MANDELBROT_IMAGE

#include <iostream>
#include "cudaTools.h"

#include "DomaineMaths.h"
#include "GLImageFonctionelCudaSelections.h"

class GLMandelBrotImage : public GLImageFonctionelCudaSelections {
    public:
	GLMandelBrotImage(int dx, int dy, DomaineMaths domain);
	virtual ~GLMandelBrotImage();

    protected:
	virtual void performKernel(uchar4* ptrDevPixels, int w, int h, const DomaineMaths& domainNew);
	virtual void idleFunc();

    private:
	int N;
};

#endif
