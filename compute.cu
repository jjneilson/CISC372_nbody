#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "config.h"
#include <cuda_runtime.h>
#include "cuda.h"

// Globally defined variables for values and accels
vector3* values;
vector3** accels;
// vector3 *d_hVel, *d_hPos;
// double *d_mass;

// Parallelizes computation of acceleration of planets, updates their velocity and mass
__global__ void parallel_compute(vector3* values, vector3** accels, vector3* d_hVel, vector3* d_hPos, double* d_mass) {
	// Find the ID of the current thread block
	int curr_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
	int i = curr_thread_id / NUMENTITIES;
	int j = curr_thread_id % NUMENTITIES;

	// Create an accels list that holds locations of accels pointer locations
	accels[curr_thread_id] = &values[curr_thread_id*NUMENTITIES];

	// Fill with all 0's if i == j
	if(curr_thread_id < NUMENTITIES*NUMENTITIES) {
		if(i == j) {
			FILL_VECTOR(accels[i][j], 0, 0, 0);
		}
		else {
			// Find the distance between particles, the acceleration magnitude and the acceleration direction vector components
			vector3 distance;
			distance[0] = d_hPos[i][0] - d_hPos[j][0];
			distance[1] = d_hPos[i][1] - d_hPos[j][1];
			distance[2] = d_hPos[i][2] - d_hPos[j][2];
			double magnitude_sum = (distance[0] * distance[0]) + (distance[1] * distance[1]) + (distance[2] * distance[2]);
			double magnitude_sqrt = sqrt(magnitude_sum);
			double accel_mag = -1 * GRAV_CONSTANT * d_mass[j] / magnitude_sum;
			FILL_VECTOR(accels[i][j], accel_mag * distance[0] / magnitude_sqrt, accel_mag * distance[1] / magnitude_sqrt, accel_mag * distance[2] / magnitude_sqrt);
		}

		// Find the sum of all accelerations of the particle
		vector3 accel_sum = {(double) *(accels[curr_thread_id])[0], (double) *(accels[curr_thread_id])[1], (double) *(accels[curr_thread_id])[2]};
		
		// Update the velocity and acceleration of the particle
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
	
	// Allocate memory for velocity, mass, and position
	cudaMallocManaged((void**) &d_hVel, (sizeof(vector3) * NUMENTITIES));
	cudaMallocManaged((void**) &d_hPos, (sizeof(vector3) * NUMENTITIES));
	cudaMallocManaged((void**) &d_mass, (sizeof(double) * NUMENTITIES));

	// Copy data from host memory to device memory
	cudaMemcpy(d_hVel, hVel, sizeof(vector3) * NUMENTITIES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_hPos, hPos, sizeof(vector3) * NUMENTITIES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_mass, mass, sizeof(double) * NUMENTITIES, cudaMemcpyHostToDevice);

	// Allocate memory for acceleration 
	cudaMallocManaged((void**) &values, sizeof(vector3)*NUMENTITIES*NUMENTITIES);
	cudaMallocManaged((void**) &accels, sizeof(vector3*)*NUMENTITIES);


	// Define the number of blocks and threads to use
	int block_size = 256;										
	int num_blocks = (NUMENTITIES + block_size - 1) / block_size;

	// Calls parallel compute to do the calculations
	parallel_compute<<<num_blocks, block_size>>>(values, accels, d_hVel, d_hPos, d_mass);

	// Check to make sure all calculations are done
	cudaDeviceSynchronize();

	// Copu data from device memory to host memory
	cudaMemcpy(hVel, d_hVel, sizeof(vector3) * NUMENTITIES, cudaMemcpyDefault);
	cudaMemcpy(hPos, d_hPos, sizeof(vector3) * NUMENTITIES, cudaMemcpyDefault);
	cudaMemcpy(mass, d_mass, sizeof(double) * NUMENTITIES, cudaMemcpyDefault);

	// Free accels and values
	cudaFree(accels);
	cudaFree(values);
}

