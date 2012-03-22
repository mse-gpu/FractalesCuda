#include <cmath>
#include "MandelBrotImage.hpp"

extern void launchMandelBrotAnimation(uchar4* ptrDevPixels, int w, int h, int N, const DomaineMaths& domainNew);

GLMandelBrotImage::GLMandelBrotImage(int dx, int dy, DomaineMaths domain): acc(0), GLImageFonctionelCudaSelections(dx, dy, domain){
    N = 52;
}

GLMandelBrotImage::~GLMandelBrotImage(){
    //Nothing
}

void GLMandelBrotImage::performKernel(uchar4* ptrDevPixels, int w, int h, const DomaineMaths& domainNew){
    launchMandelBrotAnimation(ptrDevPixels, w, h, N, domainNew);
}

void GLMandelBrotImage::idleFunc(){
    ++N;
    updateView();
}
