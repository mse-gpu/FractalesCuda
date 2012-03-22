#include <cmath>
#include "MandelBrotImage.hpp"

extern void launchMandelBrotAnimation(uchar4* ptrDevPixels, int w, int h, float t);

GLMandelBrotImage::GLMandelBrotImage(int dx, int dy): GLImageCudas(dx, dy){
    t = 1;
    dt = 2 * (atan(1) * 4) / (float) 36;
}

GLMandelBrotImage::~GLMandelBrotImage(){
    //Nothing
}

void GLMandelBrotImage::performKernel(uchar4* ptrDevPixels, int w, int h){
    launchMandelBrotAnimation(ptrDevPixels, w, h, t);
}

void GLMandelBrotImage::idleFunc(){
    t += dt;
    updateView();
}
