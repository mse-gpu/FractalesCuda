#include "MandelBrotImage.hpp"

extern void launchMandelBrotAnimation(uchar4* ptrDevPixels, int w, int h, int N, const DomaineMaths& domainNew);

GLMandelBrotImage::GLMandelBrotImage(int dx, int dy, DomaineMaths domain): N(0), GLImageFonctionelCudaSelections(dx, dy, domain){
    //Nothing to init other than the initialization list
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
