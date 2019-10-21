# Image convertion to grayscale using Pycuda

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import cv2
import time # Import for execution time calculation

mod = SourceModule\
  ("""
     #define INDEX(a, b) a*512+b

     __global__ void bgr2gray(float *result, float *bb, float *gg, float *rr){
        unsigned int i = threadIdx.x + (blockIdx.x * (blockDim.x*blockDim.y));
        unsigned int a = i/512;
        unsigned int b = i%512;
        result[INDEX(a, b)] = (0.299*rr[INDEX(a, b)] + 0.587*gg[INDEX(a, b)] + 0.114*bb[INDEX(a, b)]);
    }
    """)

# Start the timer
start_time = time.time()

# read the image
img = cv2.imread("doggy.jpg",1)

# Separate shades (262144 = 512*512)
b_img = img[:, :, 0].reshape(262144).astype(np.float32)
g_img = img[:, :, 1].reshape(262144).astype(np.float32)
r_img = img[:, :, 2].reshape(262144).astype(np.float32)
result = g_img

# to kernel
func = mod.get_function("bgr2gray")
func(cuda.Out(result),cuda.In(b_img),cuda.In(g_img),cuda.In(r_img),block=(1024,1,1),grid=(256,1,1))

# Show the result
result = np.reshape(result, (512, 512)).astype(np.uint8)
cv2.imshow("Original",img)
cv2.imshow("Grayscale_CUDA",result)

# Print execution time result
print ("time elapsed: {:.4f}s".format(time.time() - start_time))
cv2.waitKey(0)
cv2.destroyAllWindows()

