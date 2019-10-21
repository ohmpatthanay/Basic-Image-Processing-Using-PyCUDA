# Image addition using Pycuda

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import cv2
import time # Import for execution time calculation

mod = SourceModule("""
    __global__ void addNum(float *result, float *a, float *b, int n){
        int tid = threadIdx.x + blockDim.x * blockIdx.x;
        while(tid < n){
            result[tid] = a[tid] + b[tid];
            if(result[tid] > 511) result[tid] = 511;
            tid = tid + blockDim.x * gridDim.x;
        }
    }
    """)

# Start the timer
start_time = time.time()

# read images as grayscale
img1 = cv2.imread("doggy.jpg", 0)
img2 = cv2.imread("Illuminati.png", 0)

# Size (262144 = 512*512)
newimg1 = img1.reshape(262144).astype(np.float32)
newimg2 = img2.reshape(262144).astype(np.float32)
n = newimg1.size

# make the result to have same size as newimg1
result = newimg1

# to kernel
func = mod.get_function("addNum")
func(cuda.Out(result),cuda.In(newimg1),cuda.In(newimg2),np.uint32(n),block=(1024,1,1),grid=(64,1,1))

# Show the result
result = np.reshape(result, (512, 512)).astype(np.uint8)
cv2.imshow("Dog", img1)
cv2.imshow("Illuminati", img2)
cv2.imshow("Added image_CUDA", result)

# Print execution time result
print ("time elapsed: {:.4f}s".format(time.time() - start_time))

cv2.waitKey(0)
cv2.destroyAllWindows()

