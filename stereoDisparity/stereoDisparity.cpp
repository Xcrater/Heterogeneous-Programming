/**
521288S:3 Asign 4 - Stereo Disparity C/C++ implementation
@file image_filtering.cpp
@author accioo
@version 0.1
@E-mail:engg.rajviky@gmail.com
**/
#define CL_HPP_MINIMUM_OPENCL_VERSION   120
#define CL_HPP_TARGET_OPENCL_VERSION    120

#include <iostream>
#include <vector>
#include "lodepng.h"
#include <assert.h>
#include <fstream>

#define Pr .299
#define Pg .587
#define Pb .114

const char *FILENAME_L = "im0.png";
const char *FILENAME_R = "im1.png";
const char *FILENAME_GRAYSCALE_L = "grey0.png";
const char *FILENAME_GRAYSCALE_R = "grey1.png";

using namespace std;

void decodeImage(std::vector<unsigned char> &imgRGB, unsigned &width, 
                unsigned &height, const char *filename){
    
    //Converts RGBA 8-bit per channel so 32-bit image into a std::vector with RGBARGBARGBA... 
    //format and stores in the disk with given filename
    unsigned error = lodepng::decode(imgRGB, width, height, filename);

    //if there's an error, display it
    if(error) std::cout << "[ERR]: Decoder error " << error << ": " << lodepng_error_text(error) << std::endl;
    else std::cout << "[STATUS]: Image " << filename << " loaded successfully ! " << std::endl;
}

void encodeImage(std::vector<unsigned char> &imgGrey, unsigned &width,
                unsigned &height, const char *filename){

    //converts and stores 8-bit grayscale (only one channel) raw pixel data into a PNG file 
    //on disk and it takes filename as output.
    unsigned error = lodepng::encode(filename, imgGrey, width, height, LCT_GREY); 

	//if there's an error, display it
	if (error) std::cout << "[ERR]: Encoder error " << error << ": " << lodepng_error_text(error) << std::endl;
	else std::cout << "[STATUS]: Image was successfully written to disk at: " << filename << std::endl;
}


void grayscaleConvert(){

    std::vector<unsigned char> imgRGB_L, imgRGB_R, imgGrey_L, imgGrey_R;
    unsigned width;
    unsigned height; 
    
    std::cout << "[STATUS]: Running grayscale image conversion ..." << std::endl;

    /* get decoded Image */
    decodeImage(imgRGB_L, width, height, FILENAME_L);
    decodeImage(imgRGB_R, width, height, FILENAME_R);

    imgRGB_L.shrink_to_fit();
    imgRGB_R.shrink_to_fit();

    /* RGBA to Graysacle */
    for (unsigned i = 0; i < (width*height*4); i+=4)
        imgGrey_L.push_back((imgRGB_L[i]*Pr)+(imgRGB_L[i+1])*Pg+(imgRGB_L[i+2]*Pb));	
    encodeImage(imgGrey_L, width, height, FILENAME_GRAYSCALE_L);
    for (unsigned i = 0; i < (width*height*4); i+=4)
        imgGrey_R.push_back((imgRGB_R[i]*Pr)+(imgRGB_R[i+1])*Pg+(imgRGB_R[i+2]*Pb));	
    encodeImage(imgGrey_R, width, height, FILENAME_GRAYSCALE_R);
}


void stereoDispariyt () {

}

int main(int argc, char** argv){
    grayscaleConvert();
    return 0;
}
