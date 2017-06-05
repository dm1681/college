// adapted from some youtube video


#include <iostream>
#include <time.h>
#include <random>
#include <curand.h>
#include <math.h>
#include "kernels.cuh"

int main()
{
	unsigned int n = 256*256; // 32blocks, 32 threads each
	unsigned int m = 1000;
	int *h_count;
	int *d_count;
	curandState *d_state;
	float pi;

	// allocate memory
	h_count = (int*)malloc(n*sizeof(int));
	cudaMalloc((void**)&d_count, n*sizeof(int));
	cudaMalloc((void**)&d_state, n*sizeof(curandState));
	cudaMemset(d_count, 0, sizeof(int));

	// set up timing
	float gpu_elapse_time;
	cudaEvent_t gpu_state, gpu_stop;
	cudaEventCreate(&gpu_start);
	cudaEventCreate(&gpu_stop);
	cudaEventRecord(gpu_start, 0);


	// set kernel
	dim3 gridSize = 256;
	dim3 blockSize = 256;
	setup_kernel<<< gridSize, blockSize>>>(d_state);

	// monte carlo kernel
	monte_carlo_pi_kernel<<<gridSize, blockSize>>>(d_state, d_count, m);

	// copy results back to host
	cudaMemcpy(h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
	cudaEventRecord(gpu_stop, 0);
	cudaEventSynchronize(gpu_stop);
	cudaEventElapsedTime(&gpu_elapse_time, gpu_start, gpu_stop);
	cudaEventDestroy(gpu_start);
	cudaEventDestroy(gpu_stop);

	// display results and timings for gpu
	pi = *h_count*4.0/(n*m);
	std::cout<<"Approximate pi calculated on GPU is: "<<pi<<" and calculation took "<<gpu_elapse_time<<std::endl;

	//serial version
	clock_t cpu_start = clock();
	std::default_random_engine_generator;
	std::uniform_real_distribution<float> distribution(0, 1.0);
	int count = 0;
	for(unsigned int i=0;i<n;i++){
		int temp = 0;
		while(temp < m){
			float x = distribution(generator);
			float y = distribution(generator);
			float r = x*x + y*y;

			if(r <= 1){
				count++;
			}
			temp++

		}
	}
	clock_t cpu_stop = clock()
	pi = 4.0*count/(n*m)
	std::cout<<"Approximate pi calculated on CPU is: "<<pi<<" and calculation took "<<1000*(cpu_stop - cpu_start)/CLOCKS_PER_SEC<<'ms'<<std::endl;

	//delete memory

	free(h_count);
	cudaFree(d_count);
	cudaFree(d_state);

}