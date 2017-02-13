Combines 2 or more pictures together to improve image quality. The idea is to use cameras with same resolution but 
different focal lenghts and combine these images.

The code matches a zoomed image (an image with big focal lenght) to a broader one with SIFT (Scale-Invariant Feature Transform) and 
takes the average pixel values from these pictures.

The process is sped with CUDA.  

Depedencies:

	- OpenCV library with Extra 2D Features Framework (xFeatures2D)
	- CUDA (NVIDIA GPU needed)

