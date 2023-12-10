#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "config.h"
#include <cuda_runtime.h>
#include "cuda.h"

// Global variables for values and accels
vector3* values;
vector3** accels;

//Parallelizes computation fo acceleration of plantes, updates velocity and mass
__global__ void parallel_compute(vector3* values, vector3** accels, vector3* d_hVel, vector3* d_hPos, double* d_mass){
	int curr_thread_id = blockIdx.x * blockDim.x +threadIdx.x;
	int i = curr_thread_id / NUMENTITIES;
	int j = curr_thread_id % NUMENTITIES;

	accels[curr_thread_id] = &values[curr_thread_id*NUMENTITIES];

	if(curr_thread_id < NUMENTITIES*NUMENTITIES){
		if(i==j){
			FILL_VECTOR(accels[i][j],0,0,0);
		}
		else{
			vector3 distance;
			distance[0] = d_hPos[i][0]-d_hPos[j][0];
			distance[1] = d_hPos[i][1]-d_hPos[j][1];
			distance[2] = d_hPos[i][2]-d_hPos[j][2];
			double magnitude_sum = (distance[0]*distance[0])+(distance[1]*distance[1])+(distance[2]*distance[2]);
			double magnitude_sqrt = sqrt(magnitude_sum);
			double accel_mag = -1 * GRAV_CONSTANT * d_mass[j] / magnitude_sum;
			FILL_VECTOR(accels[i][j], accel_mag*distance[0]/magnitude_sqrt, accel_mag*distance[1]/magnitude_sqrt,accel_mag*distance[2]/magnitude_sqrt);
		}
		vector3 accel_sum{(double)*(accels[curr_thread_id])[0],(double) *(accels[curr_thread_id])[1],(double) *(accels[curr_thread_id])[2]};

		d_hVel[i][0] = d_hVel[i][0] + accel_sum[0] * INTERVAL;
		d_hPos[i][0] = d_hVel[i][0] * INTERVAL;
		d_hVel[i][1] = d_hVel[i][1] + accel_sum[1] * INTERVAL;
		d_hPos[i][1] = d_hVel[i][1] * INTERVAL;
		d_hVel[i][2] = d_hVel[i][2] + accel_sum[2] * INTERVAL;
		d_hPos[i][2] = d_hVel[i][2] * INTERVAL;
	}
}


void compute() {
	vector3 *d_hVel, *d_hPos;
	double *d_mass;

	cudaMallocManaged((void**) &d_hVel, (sizeof(vector3)*NUMENTITIES));
	cudaMallocManaged((void**) &d_hPos, (sizeof(vector3)*NUMENTITIES));
	cudaMallocManaged((void**) &d_mass, (sizeof(double)*NUMENTITIES));

	cudaMemcpy(d_hVel, hVel, sizeof(vector3)*NUMENTITIES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_hPos, hPos, sizeof(vector3)*NUMENTITIES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_mass, mass, sizeof(double)*NUMENTITIES, cudaMemcpyHostToDevice);

	cudaMallocManaged((void**) &values, sizeof(vector3)*NUMENTITIES*NUMENTITIES);
	cudaMallocManaged((void**) &accels, sizeof(vector3*)*NUMENTITIES);

	int block_size=256;
	int num_blocks=(NUMENTITIES+block_size-1)/block_size;

	parallel_compute<<<num_blocks, block_size>>>(values,accels,d_hVel, d_hPos,d_mass);

	cudaDeviceSynchronize();

	cudaMemcpy(hVel, d_hVel, sizeof(vector3)*NUMENTITIES, cudaMemcpyDefault);
	cudaMemcpy(hPos, d_hPos, sizeof(vector3)*NUMENTITIES, cudaMemcpyDefault);
	cudaMemcpy(mass, d_mass, sizeof(double)*NUMENTITIES, cudaMemcpyDefault);
	
	cudaFree(accels);
	cudaFree(values);
}
