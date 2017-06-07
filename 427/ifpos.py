import pycuda.gpuarray as gpuarray
import pycuda.cumath as cumath
import pycuda.driver as cuda
import pycuda.curandom
import pycuda.autoinit
import random
import matplotlib.pyplot as plt
import timeit
import numpy as np
import pdb

# def number of arrays
test_array = np.array([-1, -2, 1, 2])
then_array = np.array([-9999])
else_array = np.array([-666])

test_gpu = gpuarray.to_gpu(test_array)

then_gpu = gpuarray.to_gpu(then_array)
else_gpu = gpuarray.to_gpu(else_array)
pos_array = gpuarray.if_positive(test_gpu, then_array, else_array)

p = pos_array.get()

print p