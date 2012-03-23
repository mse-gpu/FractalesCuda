#include "JuliaImage.hpp"

extern void launchJuliaAnimation(uchar4* ptrDevPixels, int w, int h, int N, const DomaineMaths& domainNew);

GLJuliaImage::GLJuliaImage(int dx, int dy, DomaineMaths domain): N(0), GLImageFonctionelCudaSelections(dx, dy, domain){
    //Nothing to init other than the initialization list
}

GLJuliaImage::~GLJuliaImage(){
    //Nothing
}

void GLJuliaImage::performKernel(uchar4* ptrDevPixels, int w, int h, const DomaineMaths& domainNew){
    launchJuliaAnimation(ptrDevPixels, w, h, N, domainNew);
}

void GLJuliaImage::idleFunc(){
    ++N;
    updateView();
}
