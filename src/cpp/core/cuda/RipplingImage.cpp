#include <cmath>
#include "RipplingImage.hpp"

extern void useKernelAnimationHSB(uchar4* ptrDevPixels, int w, int h, float t);

GLRipplingImage::GLRipplingImage(int dx, int dy): GLImageCudas(dx, dy){
    t = 1;
    dt = 2 * (atan(1) * 4) / (float) 36;
}

GLRipplingImage::~GLRipplingImage(){
    //Nothing
}

void GLRipplingImage::performKernel(uchar4* ptrDevPixels, int w, int h){
    useKernelAnimationHSB(ptrDevPixels, w, h, t);
}

void GLRipplingImage::idleFunc(){
    t += dt;
    updateView();
}
