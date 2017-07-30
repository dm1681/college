#imports
import pycuda.gpuarray as gpuarray
import pycuda.cumath as cumath
import pycuda.driver as cuda
import pycuda.curandom
import pycuda.autoinit
import random
import time
import numpy as np
import pdb

# how many iterations?
# 10 million is good
N = 10000000

# def function for non-gpu calcs
def MC_cir(r, N):
	'''
	Params:
	------
	r = radius of circle with which to approx. pi
	N = # of iterations
	Output:
	------
	n_pi = pi after n iterations
	n_err = error (n_pi - pi)
	t_elapsed = how long it took
	'''
	t_start = time.time()
	n = 1
	C = 0
	S = 0
	while n <= N:
	    x = random.uniform(0,r)
	    y = random.uniform(0,r)
	    S+=1
	    if (x**2 + y**2) <= (r**2):
	        C+=1
	    n += 1
	n_pi = 4. * C/S
       
	t_end = time.time()
	t_elapsed = t_end - t_start
	n_err = n_pi - np.pi
	return n_pi, n_err, t_elapsed


# define randomness
rg = pycuda.curandom.XORWOWRandomNumberGenerator()

# generate x and y
x_rand = rg.gen_uniform(N, np.float32)
y_rand = rg.gen_uniform(N, np.float32)

# generate counting arrays
# z_count is for when there needs to be a 0 in c_count
s_count = gpuarray.zeros_like(x_rand)
z_count = gpuarray.zeros_like(x_rand)
c_count = gpuarray.zeros_like(x_rand)

# send to gpu
x_gpu = gpuarray.to_gpu(x_rand)
y_gpu = gpuarray.to_gpu(y_rand)
s_gpu = gpuarray.to_gpu(s_count)
c_gpu = gpuarray.to_gpu(c_count)
z_gpu = gpuarray.to_gpu(z_count)

start_time = time.time()

# do math
r_gpu = cumath.sqrt(x_gpu*x_gpu + y_gpu*y_gpu)

# square
s_gpu = s_gpu + 1

# circle
c_gpu = gpuarray.if_positive(1.-r_gpu, s_gpu, z_gpu)

# now sum
s_gpu_count = gpuarray.sum(s_gpu)
c_gpu_count = gpuarray.sum(c_gpu)

# pi estimate
pi_est_gpu = 4. * c_gpu_count / s_gpu_count

end_time = time.time()

# get it all back
r = r_gpu.get()
c = c_gpu.get()
s = s_gpu.get()
pi_est = pi_est_gpu.get()


pi_err = pi_est - np.pi

elapsed_time = end_time - start_time
print 'GPU:', pi_est, 'err:', pi_err, 'in:', elapsed_time, 'secs'

# now for cpu calcs

pi_cpu, pi_err_cpu, elapsed_time_cpu = MC_cir(1,N)

print 'CPU:', pi_cpu, 'err:', pi_err_cpu, 'in:',elapsed_time_cpu, 'secs'