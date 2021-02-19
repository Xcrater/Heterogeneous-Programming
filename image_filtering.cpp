/**
521288s:3 Asign 3 - Load image, convert greyscale and apply 5x5 moving filter (convolution)
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
#include <CL/cl.hpp>
#include <CL/cl2.hpp>
#include <assert.h>
#include <fstream>

#define Pr .299
#define Pg .587
#define Pb .114
#define filterWidth 5
#define filterHeight 5

const char *FILENAME = "im0.png";
const char *FILENAME_GRAYSCALE_C = "grey.png";
const char *FILENAME_GRAYSCALE_OpenCL = "grey_OpenCL.png";
const char *FILENAME_FILTERED_OPENCL = "filtered_OpenCL.png";

using namespace std;

void decodeImage(std::vector<unsigned char> &imgRGB, unsigned &width, 
                unsigned &height, const char *filename){
    
    //Converts RGBA 8-bit per channel so 32-bit image into a std::vector with RGBARGBARGBA... 
    //format and stores in the disk with given filename
    unsigned error = lodepng::decode(imgRGB, width, height, filename);

    //if there's an error, display it
    if(error) std::cout << "[ERR]: Decoder error " << error << ": " << lodepng_error_text(error) << std::endl;
    else std::cout << endl << "[STATUS]: Image " << FILENAME << " loaded successfully ! " << std::endl;
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

void profilingInfo(cl::Event &event) {
    
    cl_ulong queue, submit, start, stop; 
    double  queueing, submitting, executing, full_time;
    cl_int err;

    err = event.getProfilingInfo(CL_PROFILING_COMMAND_START, &queue);
    if (err != CL_SUCCESS)
        cout << "[ERR]: Failed to get queue time for above command "<< err << endl;
    
    err = event.getProfilingInfo(CL_PROFILING_COMMAND_SUBMIT, &submit);
    if (err != CL_SUCCESS)
        cout << "[ERR]: Failed to get submit time for above command "<< err << endl;
    
    err = event.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
    if (err != CL_SUCCESS)
        cout << "[ERR]: Failed to get start time for above command "<< err << endl;
    
    err = event.getProfilingInfo(CL_PROFILING_COMMAND_END, &stop);
    if (err != CL_SUCCESS) 
        cout << "[ERR]: Failed to get stop time for above command "<< err << endl;
    
    queueing    =   (double)(submit  -   queue)/10e6;
    submitting  =   (double)(start   -   submit)/10e6;
    executing   =   (double)(stop    -   start)/10e6;
    full_time   =   (double)(stop    -   submit)/10e6;

    std::cout << " queue|submit|exec|full: " << queueing << "|" << submitting
    		  << "|" << executing <<"|" << full_time << endl;
}

void grayscaleFilter_OpenCL_impl(cl::Platform platform, cl::Device device){
    
    std::vector <unsigned char> imgRGB, imgGrey, imgFiltered;
    std::vector <cl::Event> exe_writeGreyImgEvent, 
                       exe_filterKerEvent,
                       exe_readFilterImgEvent;
    unsigned int filterSize =   filterWidth*filterHeight;
    float filter[filterSize] = {0, 0, -1, 0, 0,
                                0, 0, -1, 0, 0,
                                0, 0, 4, 0, 0,
                                0, 0, -1, 0, 0,
                                0, 0, -1, 0, 0,
                                };
    unsigned width;
    unsigned height;

    cl_int err;
    cl::Program::Sources sources;
    cl::Event greyKerEvent, 
              filterKerEvent, 
              readGreyEvent, 
              writeGreyEvent, 
              readFilterEvent,
              writeRgbEvent;

    /* Get decoded image */
    decodeImage(imgRGB, width, height, FILENAME);
    
    size_t gradSize = height*width;
    
    /* create context with GPU device */
    cl::Context context(device);

    /* create out-of-order queue (due to dependency) to push the commands for the device */
    cl::CommandQueue queue(context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE, &err);
    if (err != CL_SUCCESS) 
        cout << "[ERR]: Failed to create queue with device "<< device.getInfo<CL_DEVICE_NAME>() <<" ("<<err <<")"<< endl;
    
    /* create 2D image objects for images ,filter and size of filters */
    cl::ImageFormat formatRGB(CL_RGBA, CL_UNORM_INT8);
    cl::Image2D inputRgbObj(context, CL_MEM_READ_ONLY , formatRGB, width, height, 0, NULL, &err);
    if (err != CL_SUCCESS) 
        cout << "[ERR]: Image2D object 'inputRgbObj' failed to create " << err << endl;
    
    cl::ImageFormat formatGrey(CL_R, CL_UNORM_INT8);
    cl::Image2D outputGreyObj(context, CL_MEM_WRITE_ONLY, formatGrey, width, height, 0, NULL, &err);
    if (err != CL_SUCCESS) 
        cout << "[ERR]: Image2D object 'outputGreyObj' failed to create " << err << endl;

    cl::Image2D inputGreyObj(context, CL_MEM_READ_ONLY, formatGrey, width, height, 0, NULL, &err);
    if (err != CL_SUCCESS) 
        cout << "[ERR]: Image2D object 'inputGreyObj' failed to create " << err << endl;
    
    cl::Image2D outputFilterObj(context, CL_MEM_WRITE_ONLY, formatGrey, width, height, 0, NULL, &err);
    if (err != CL_SUCCESS) 
        cout << "[ERR]: Image2D object 'outputFilterObj' failed to create " << err << endl;

    /* Create buffer for filter and transfer it to the device */
    cl::Buffer filterBuf(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR , sizeof(float)*(filterSize), &filter, &err);
    if (err != CL_SUCCESS) 
        cout << "[ERR]: Failed to create buffer 'filterBuf' on device and tranfer the data " << err << endl;
    
    cl::size_t<3> origin;
    origin[0]=0;
    origin[1]=0;
    origin[2]=0;

    cl::size_t<3> region;
    region[0] = width;
    region[1] = height;
    region[2] = 1;

    err = queue.enqueueWriteImage(inputRgbObj, CL_TRUE, origin, region, 0, 0, (void *) imgRGB.data(), NULL, &writeRgbEvent);
    if (err != CL_SUCCESS)
        cout << "[ERR]: Image2D object 'inputRgbObj' failed to write " << endl;
    cout << "[Status]: Write RGB Image successful !-------------Profile time(ms)";
    profilingInfo(writeRgbEvent);

    err = queue.enqueueWriteBuffer(filterBuf, CL_TRUE, 0, filterSize*sizeof(float), &filter, NULL ,NULL) != CL_SUCCESS;
    //if (err != CL_SUCCESS) 
    //    cout << "[ERR]: Failed to write buffer 'filterBuf' to device " << err << endl;

    /* This piece of snippet is for image data as buffers */
/************
    size_t rgbSize = height*width*4;
    cl::Buffer inputRgbBuf(context, CL_MEM_READ_ONLY , rgbSize, NULL, &err);
    if (err != CL_SUCCESS)
        cout << "[ERR]: Buffer 'inputRgbBuf' failed to create" << endl;
    
    cl::Buffer outputGreyBuf(context, CL_MEM_WRITE_ONLY, gradSize, NULL, &err);
    if (err != CL_SUCCESS)
        cout << "[ERR]: Buffer 'outputGreyBuf' failed to create" << endl;
    
    if(queue.enqueueWriteBuffer(inputRgbBuf, CL_TRUE, 0, rgbSize, imgRGB.data(), NULL ,NULL) != CL_SUCCESS)
        cout << "[ERR]: Failed to write buffer 'inputRgbBuf' to device" << endl;
*************/

    /* Load kernel code */
    std::fstream grayscaleCL("grayscale.cl");
    std::string kernel_code;
    kernel_code.assign((std::istreambuf_iterator<char>(grayscaleCL)), std::istreambuf_iterator<char>());
    if(kernel_code.empty()){
        cout<<"[ERR]: Failed to load kernel"<<endl;
    }
    
    sources.push_back({kernel_code.c_str(),kernel_code.length()});

    /* Convert the source code into kernel program using program object */
    cl::Program program(context,sources);

    /* Builds the program and creates binary */
    if(program.build({1,device})!=CL_SUCCESS){
        std::cout<<"[ERR]: Error in Building the kernal program:"<<program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device)<< endl;
        exit(1);
    }

    /* Create kernel object, each kernel object corresponds to each kernel function */
    cl::Kernel kernelGrayscale = cl::Kernel(program,"grayscale");
    cl::Kernel kernelFilter = cl::Kernel(program,"filter");

    /* Set Arguments for Grayscale kernel*/
    kernelGrayscale.setArg(0,inputRgbObj);
    kernelGrayscale.setArg(1,outputGreyObj);

    /* Execute OpenCL kernel as data parallel with 2D work-item of size (width, height) */
    err = queue.enqueueNDRangeKernel(kernelGrayscale, cl::NullRange, cl::NDRange(width, height), cl::NullRange, NULL, &greyKerEvent);
    if (err != CL_SUCCESS)
        cout << "[ERR]: Failed to create memory model for the kernel Grayscale " << err << endl;
    
    unsigned char *flat_image_out = new unsigned char [gradSize];
    
    /* Transfer data back to host from device */
    err = queue.enqueueReadImage(outputGreyObj, CL_TRUE, origin, region, 0, 0, flat_image_out, NULL, &readGreyEvent);
    if (err != CL_SUCCESS)
        cout << "[ERR]: Failed to read buffer 'outputGreyBuf' from device " << err << endl; 
    cout << "[Status]: Read Grayscale Image successful !--------Profile time(ms)";
    profilingInfo(readGreyEvent);

    /* Set Argument for Filter kernel*/
    kernelFilter.setArg(0,inputGreyObj);
    kernelFilter.setArg(1,outputFilterObj);
    kernelFilter.setArg(2,filterBuf);
    kernelFilter.setArg(3,filterWidth);

    /* Write grayscale image data to the buffer */
    exe_writeGreyImgEvent.push_back(greyKerEvent);
    err = queue.enqueueWriteImage(inputGreyObj, CL_TRUE, origin, region, 0, 0, flat_image_out, &exe_writeGreyImgEvent, &writeGreyEvent) != CL_SUCCESS;
    if (err != CL_SUCCESS)
        cout << "[ERR]: Image2D object inputGreyObj failed to write " << err << endl;
    cout << "[Status]: Write Grayscale Image successful !-------Profile time(ms)";
    profilingInfo(writeGreyEvent);

    /* Execute OpenCL kernel as data parallel with 2D work-item of size (width, height) */
    exe_filterKerEvent.push_back(writeGreyEvent);
    err = queue.enqueueNDRangeKernel(kernelFilter, cl::NullRange, cl::NDRange(width, height), cl::NullRange, &exe_filterKerEvent, &filterKerEvent);
    if (err != CL_SUCCESS)
        cout << "[ERR]: Failed to create memory model for the filter kernel " << err << endl;
    
    unsigned char *flat_image_out_filter = new unsigned char [gradSize]; 

    /* Transfer data back to host from device */
    exe_readFilterImgEvent.push_back(filterKerEvent);
    err = queue.enqueueReadImage(outputFilterObj, CL_TRUE, origin, region, 0, 0, flat_image_out_filter, &exe_filterKerEvent, &readFilterEvent) != CL_SUCCESS;
    if(err != CL_SUCCESS)
    cout << "[ERR]: Failed to read buffer 'outputGreyBuf' from device " << err << endl; 
    cout << "[Status]: Read Filtered Image successful !---------Profile time(ms)";
    profilingInfo(readFilterEvent);
    
    cout << "[Status]: Grayscale Kernel Run successful !--------Profile time(ms)";
    profilingInfo(greyKerEvent);
    
    cout << "[Status]: Filter Kernel Run successful !-----------Profile time(ms)";
    profilingInfo(filterKerEvent);
    
    /* Encode the GPU grayscale image */
    for (unsigned i = 0;i < gradSize; i++)
        imgGrey.push_back(flat_image_out[i]);
    encodeImage(imgGrey, width, height, FILENAME_GRAYSCALE_OpenCL);

    /* Encode the GPU filtered image */
    for (unsigned i = 0;i < gradSize; i++)
        imgFiltered.push_back(flat_image_out_filter[i]);
    encodeImage(imgFiltered, width, height, FILENAME_FILTERED_OPENCL);

    queue.finish();
    queue.flush();
}

void grayscaleFilter_C_impl(){

    /* get decoded Image */
    std::vector<unsigned char> imgRGB, imgGrey;
    unsigned width;
    unsigned height; 
    decodeImage(imgRGB, width, height, FILENAME);

    imgRGB.shrink_to_fit();

    /* RGBA to Graysacle */
    for (unsigned i = 0; i < (width*height*4); i+=4)
        imgGrey.push_back((imgRGB[i]*Pr)+(imgRGB[i+1])*Pg+(imgRGB[i+2]*Pb));	
    encodeImage(imgGrey, width, height, FILENAME_GRAYSCALE_C);
}

void getAllDeviceInfo(){

    std::vector<cl::Platform> platforms;
    std::vector<cl::Device> devices;
    cl::Platform::get(&platforms);
    
    assert(platforms.size()>0);

    uint countPlatform = 0;
    uint countDevice;

    //List all available devices in each OpenCL-Platform

    for (auto platform : platforms){
        countPlatform++;
        devices.clear();
        platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
  
        assert(devices.size()>0);
  
        auto ret_num_devices = devices.size();
        cout<<endl<<"Number of devices in Platform "<<countPlatform<<" : "<<ret_num_devices<<endl;
        countDevice = 0;
  
        for (auto device : devices){
            countDevice++;
            auto device_name = device.getInfo<CL_DEVICE_NAME>();
            auto version = device.getInfo<CL_DRIVER_VERSION>();
            auto compute_units = device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
            auto work_item_dim = device.getInfo<CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS>();
            
            cout<<" "<<countPlatform<<"."<<countDevice<<" Device"<<endl; 
            cout<<"  "<<countPlatform<<"."<<countDevice<<" Name"<<"                     : "<<device_name<<endl;
            cout<<"  "<<countPlatform<<"."<<countDevice<<" Driver Version"<<"           : "<<version<<endl;
            cout<<"  "<<countPlatform<<"."<<countDevice<<" Max Compute Unit         : "<<compute_units<<endl;
            cout<<"  "<<countPlatform<<"."<<countDevice<<" Work Item Dimension      : "<<work_item_dim<<endl;
            cout<<"  "<<countPlatform<<"."<<countDevice<<" Local Memory Type        : "<<device.getInfo<CL_DEVICE_LOCAL_MEM_TYPE>() << endl;
			cout<<"  "<<countPlatform<<"."<<countDevice<<" Local Memory Size        : "<<device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>() << endl;
			cout<<"  "<<countPlatform<<"."<<countDevice<<" Max Frequency            : "<<device.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>() << endl;
			cout<<"  "<<countPlatform<<"."<<countDevice<<" Max Constant Buffer Size : "<<device.getInfo<CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE>() << endl;
			cout<<"  "<<countPlatform<<"."<<countDevice<<" Max Work Group Size      : "<<device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << endl;
			cout<<"  "<<countPlatform<<"."<<countDevice<<" Max Work Item Size       : "<<device.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>()[1] << endl;
        }
    }
    cout<<"------------------------------------------------------------------------------------------" << endl;
}


/** This Function is written in C style to understand querying in C as OpenCL C is supported extensively than C++**/
void getAllPlatformInfo(){

    cl_int err;
    cl_platform_id *platform_IDs = NULL;
    const int number_of_attributes = 5; 
    const cl_platform_info attributeTypes[5] = { CL_PLATFORM_PROFILE, CL_PLATFORM_VERSION,
        CL_PLATFORM_NAME, CL_PLATFORM_VENDOR, CL_PLATFORM_EXTENSIONS };
    const char* attributeNames[5] = { "Profile", "Version",
        "Name", "Vendor", "Extensions" };
    cl_uint ret_num_platforms;

    char* info;
    size_t info_size;
    
    /* List all available OpenCL-Platforms */
    err = clGetPlatformIDs(5, NULL, &ret_num_platforms); 
    platform_IDs = (cl_platform_id*) malloc(
        sizeof(cl_platform_id) * ret_num_platforms);
    
    err = clGetPlatformIDs(ret_num_platforms, platform_IDs, NULL); 

    if (err != CL_SUCCESS) {
        printf(" Failed to find any OpenCL platforms"); 
    }     

    printf("Number of Platforms: %d", ret_num_platforms);

    for (cl_uint i = 0; i < ret_num_platforms; i++){
        printf("\n %d. Platform \n", i+1);
        for (cl_uint j = 0; j < number_of_attributes; j++) {
            // get platform attribute value size
            clGetPlatformInfo(platform_IDs[i], attributeTypes[j], 0, NULL, &info_size);
            info = (char*) malloc(info_size);

            // get platform attribute value
            clGetPlatformInfo(platform_IDs[i], attributeTypes[j], info_size, info, NULL);
            printf("  %d.%d %-11s: %s \n", i+1, j+1, attributeNames[j], info);

            free(info);
        }
        printf("\n");
    }
    free(platform_IDs);
    printf("------------------------------------------------------------------------------------------\n");
}

int main(int argc, char** argv){

    cl_int err;
    std::vector<cl::Platform> platforms;
    std::vector<cl::Device> devices;

    //Queries the handler for first platform and GPU device 
    err = cl::Platform::get(&platforms);
    if (err != CL_SUCCESS) {
        cout <<"Failed to find any OpenCL platforms" << endl;
    }
    cl::Platform platform = platforms.front();
    
    //select GPU device
    err = platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    if (err != CL_SUCCESS) {
        cout <<"Failed to find any Devices in the first platform" << endl;
    }
    cl::Device device = devices.front();

    getAllPlatformInfo();
    getAllDeviceInfo();
    grayscaleFilter_C_impl();
    grayscaleFilter_OpenCL_impl(platform, device);
}