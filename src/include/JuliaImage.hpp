#ifndef GL_JULIA_IMAGE
#define GL_JULIA_IMAGE

#include <iostream>
#include "cudaTools.h"

#include "DomaineMaths.h"
#include "GLImageFonctionelCudaSelections.h"

class GLJuliaImage : public GLImageFonctionelCudaSelections {
    public:
	GLJuliaImage(int dx, int dy, DomaineMaths domain);
	virtual ~GLJuliaImage();

    protected:
	virtual void performKernel(uchar4* ptrDevPixels, int w, int h, const DomaineMaths& domainNew);
	virtual void idleFunc();

    private:
	int N;
};

#endif
