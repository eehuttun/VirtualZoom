#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <stdio.h>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/opencv_modules.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;

__global__ void distance_matching(float* i_img, uchar* d_zimg, float* remapX, float* remapY) 
{
	int idx = blockIdx.x * blockDim.x;
	int idy = blockIdx.y * blockDim.y;

	int w = 256;
	float distance = 100.0;
	float search_range = 0.75f;
	double disX; 
	double disY;
	disX = (double)(remapX[idx + w*idy] -(float)idx);
	disY = (double)(remapY[idx + w*idy] -(float)idy);
	distance = pow(disX, 2) + pow(disY, 2);
	if ( distance > 0 && distance < search_range) {
		i_img[idx + idy*w] *= (1-1/distance);
		i_img[idx + idy*w] += ((float)d_zimg[idx + idy*w])*(1/distance);
	}


}

__global__ void non_uniform_pixel_fusion(uchar* d_img,
										 uchar* d_zimg, 
									 	 uchar* r_img,
										 int img_w,
										 int img_h,
										 int zmd_w,
										 int zmd_h,
										 float* remapX,
										 float* remapY,
										 float* minimum,
										 float* maximum)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;

	if(idx > img_w || idy > img_h) return;

	uchar subpixels[32];
	float weights[32];

	for (int i = 0; i < 32; i++) {
		subpixels[i] = (uchar)0;
		weights[i] = (float)0;
	}

    double distance;
    float average;
    float total_weight;
    float search_range = 0.75f;

    //  go trough every pixel of the area found by matching the zoomed image

	weights[0] = 1.0f;
	subpixels[0] = d_img[idx + idy*img_w];

	int px_count = 1;
	distance = 100.0;
	double disX; 
	double disY;
	if( (float)idx >= minimum[0]-search_range && (float)idy >= minimum[1]+search_range &&
        (float)idx <= maximum[0]-search_range && (float)idy <= maximum[1]+search_range) {
		
		for(int k = 0; k < zmd_w; k++) {
			for(int l = 0; l < zmd_h; l++) {
				disX = (double)(remapX[k + zmd_w*l] -(float)idx);
				disY = (double)(remapY[k + zmd_w*l] -(float)idy);
				distance = pow(disX, 2) + pow(disY, 2);
				if ( distance > 0 && distance < search_range) {
					weights[px_count] = (float)(1/distance);
					subpixels[px_count] = d_zimg[k + zmd_w*l];
					px_count++;
				}
			}
		}
	}
	
    //  calculate the total weights of the pixels
	total_weight = 0;
    for(int j = 0; j < px_count; j++) {
        total_weight += weights[j];
    }

    //  count the average of the subpixels and set it as the result pixel
    if( px_count > 0 ) {
        int i;
		average = 0;
        for(i = 0; i < px_count; i++) {
            average += (float)(subpixels[i])*weights[i];
        }
        r_img[idx + img_w*idy] = (uchar)(average/total_weight);
        //r_img[idx + img_w*idy] = 255;
    }

}

extern void cuda_doStuff(Mat* img,
						 Mat& result,
						 std::vector<Mat>* images, 
						 float* remappedX,
						 float* remappedY, 
						 float* minimum,
						 float* maximum,
						 std::vector<Mat> H)
{

	cudaDeviceProp props;
	cudaGetDeviceProperties(&props, 0);
	printf("Global memory: %i\n", props.totalGlobalMem);
	printf("Warp size: %i\n", props.warpSize);
	printf("Threads per blk: %i\n", props.maxThreadsPerBlock);
	printf("Max block dim: %i\n", props.maxThreadsDim[0]);
	printf("Max grid dim: %i\n", props.maxGridSize[0]);

	int i_h = img->size().height;
	int i_w = img->size().width;
	size_t size = i_w*i_h;
	int z_h = images->at(0).size().height;
	int z_w = images->at(0).size().width;
	size_t zoomed_size = z_w*z_h;

	printf("image size: %i\n", size);
	printf("img_w: %i\timg_h: %i\tzmg_w: %i\tzmg_h: %i\n", i_w, i_h, z_w, z_h);
	printf("min: %f, %f\tmax: %f, %f\n", minimum[0], minimum[1], maximum[0], maximum[1]);

	uchar* d_img;
	cudaMalloc((void**)&d_img, sizeof(uchar)*size);
	uchar* r_img;
	cudaMalloc((void**)&r_img, sizeof(uchar)*size);
	uchar* d_zimg;
	cudaMalloc((void**)&d_zimg, sizeof(uchar)*size);
	uchar* h_img;
	if(img->isContinuous()) {
		h_img = img->data;
	}
	uchar* h_zimg;
	h_zimg = images->at(0).data;

	float* remapX;
	cudaMalloc(&remapX, sizeof(float)*size);

	float* remapY;
	cudaMalloc(&remapY, sizeof(float)*size);

	float* mini;
	cudaMalloc(&mini, sizeof(float)*2);

	float* maxi;
	cudaMalloc(&maxi, sizeof(float)*2);

	//memory copies to the device and prints errors
	cudaMemcpy(d_img, h_img, sizeof(uchar)*size, cudaMemcpyHostToDevice);
	printf("h_img -> d_img:\t%s\n", cudaGetErrorString(cudaGetLastError()));

	cudaThreadSynchronize();

	cudaMemcpy(d_zimg, h_zimg, sizeof(uchar)*zoomed_size, cudaMemcpyHostToDevice);
	printf("h_zimg -> d_zimg:\t%s\n", cudaGetErrorString(cudaGetLastError()));

	cudaMemcpy(remapX, remappedX, sizeof(float)*size, cudaMemcpyHostToDevice);
	printf("remappedX -> remapX:\t%s\n", cudaGetErrorString(cudaGetLastError()));

	cudaMemcpy(remapY, remappedY, sizeof(float)*size, cudaMemcpyHostToDevice);
	printf("remappedY -> remapY:\t%s\n", cudaGetErrorString(cudaGetLastError()));

	cudaMemcpy(mini, minimum, sizeof(float)*2, cudaMemcpyHostToDevice);
	printf("minimum -> mini:\t%s\n", cudaGetErrorString(cudaGetLastError()));

	cudaMemcpy(maxi, maximum, sizeof(float)*2, cudaMemcpyHostToDevice);
	printf("maximum -> maxi:\t%s\n", cudaGetErrorString(cudaGetLastError()));

	dim3 threadsPerBlock(32, 32);
	dim3 numberOfBlocks(ceil((float)i_w/(float)threadsPerBlock.x), ceil((float)i_h/(float)threadsPerBlock.y));

	non_uniform_pixel_fusion << <numberOfBlocks, threadsPerBlock >> >(d_img, d_zimg, r_img,
																	  i_w, i_h, z_w, z_h, 
																	  remapX, remapY, 
																	  mini, maxi);
	cudaError err = cudaGetLastError();
	cudaDeviceSynchronize();
	printf("all kernels done. Latest error: %s\n", cudaGetErrorString( err ));

	uchar* output;
	output = (uchar*) malloc(size);
	cudaMemcpy(output, r_img, sizeof(uchar)*size, cudaMemcpyDeviceToHost);
	Mat res = Mat(i_h, i_w, CV_8UC1, output, Mat::AUTO_STEP);
	result = res.clone();

	//free resources
	cudaFree(d_img);
	cudaFree(d_zimg);
	cudaFree(r_img);
	cudaFree(remapX);
	cudaFree(remapY);
}