#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <conio.h>
#include <curand.h>
#include <curand_kernel.h>
#include <time.h>
#define POPSIZE 4500	
#define DIMENSIONS 1000
#define OPERATORSNUMBER 2
#define ITERATIONS 500
#define BLOCKS 12
#define THREADSPERBLOCK 256


__global__ void setup_kernel(unsigned int seed, curandState *state, int totalThreads);

__global__ void initPop(curandState *my_curandstate, int *d_pop, int dimensions, int popSize, int totalThreads);

__global__ void initProbabilites(int operators_number, double *d_operators_probabilites, int popSize, int totalThreads);

__global__ void applyOperators(double *d_operators_probabilites, int *d_pop, int dimensions, int operators_number, curandState *my_curandstate, int popSize, int totalThreads);

__device__ void one_point_cross(int *newIndividuals, int *d_pop, int start, int end, curandState *my_curandstate, int dimensions, int pop_size);

__device__ void mutation(int *newIndividuals, int *d_pop, int start, int end, curandState *my_curandstate, int dimensions);

void printBest(int* h_pop, int dimensions)
{
	int best = 0, i, pos_best = 0;
	for (i = 0; i < POPSIZE; i++)
	{
		if (h_pop[(i + 1)*dimensions - 1] > best)
		{
			best = h_pop[(i + 1)*dimensions - 1];
			pos_best = i;
		}
	}
	int m;
	for (m = pos_best*dimensions; m < pos_best*dimensions + dimensions - 1; m++)
	{
		printf("%i", h_pop[m]);
	}
	printf(" %i\n", h_pop[m]);


}

int main()
{

	clock_t start, end;
	start = clock();

	//Declare host variables
	int *h_pop;
	double *h_operators_probababilites;

	//Declare device variables
	int *d_pop;
	double *d_operators_probababilites;
	curandState *d_state;

	cudaError_t err = cudaSuccess;

	int dimensions = DIMENSIONS + 1;
	int operators_number = OPERATORSNUMBER;
	int popSize = POPSIZE;
	int totalThreads = BLOCKS*THREADSPERBLOCK;

	//Declare auxiliar variables
	int m;
	h_pop = (int *)malloc(POPSIZE * (DIMENSIONS + 1) * sizeof(int));
	if (h_pop == NULL) {
		fprintf(stderr, "Failed to allocate host vectors!\n");
		exit(EXIT_FAILURE);
	}
	h_operators_probababilites = (double *)malloc(POPSIZE * OPERATORSNUMBER * sizeof(double));
	if (h_operators_probababilites == NULL) {
		fprintf(stderr, "Failed to allocate host vectors!\n");
		exit(EXIT_FAILURE);
	}

	err = cudaMalloc((void **)&d_pop, POPSIZE * (DIMENSIONS + 1) * sizeof(int));
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	err = cudaMalloc((void **)&d_operators_probababilites, POPSIZE * OPERATORSNUMBER * sizeof(double));
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	err = cudaMalloc(&d_state, sizeof(curandState));
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}



	setup_kernel << <BLOCKS, THREADSPERBLOCK >> > (time(NULL), d_state, totalThreads);
	cudaDeviceSynchronize();
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		fprintf(stderr, "SK: Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	initPop << <BLOCKS, THREADSPERBLOCK >> > (d_state, d_pop, dimensions, popSize, totalThreads);
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		fprintf(stderr, "IP: Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	initProbabilites << <BLOCKS, THREADSPERBLOCK >> > (operators_number, d_operators_probababilites, popSize, totalThreads);
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		fprintf(stderr, "IProb: Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	cudaDeviceSynchronize();

	for (int ex = 0; ex < ITERATIONS; ex++)
	{

		applyOperators << <BLOCKS, THREADSPERBLOCK >> > (d_operators_probababilites, d_pop, dimensions, operators_number, d_state, popSize, totalThreads);
		err = cudaGetLastError();
		if (err != cudaSuccess) {
			fprintf(stderr, "FOR %d: Failed to launch vectorAdd kernel (error code %s)!\n", ex, cudaGetErrorString(err));
			getchar();
			exit(EXIT_FAILURE);
		}
		cudaDeviceSynchronize();
	}
	end = clock();

	cudaMemcpy(h_pop, d_pop, POPSIZE * (DIMENSIONS + 1) * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_operators_probababilites, d_operators_probababilites, POPSIZE * OPERATORSNUMBER * sizeof(double), cudaMemcpyDeviceToHost);
	printBest(h_pop, dimensions);
	cudaFree(d_pop); cudaFree(d_operators_probababilites);
	free(h_pop); free(h_operators_probababilites);

	printf("Time: %.3fs", (double)(end - start) / CLOCKS_PER_SEC);



	getchar();

	return 0;
}

__global__ void setup_kernel(unsigned int seed, curandState *state, int totalThreads) {

	int idx = threadIdx.x + blockDim.x*blockIdx.x;
	if (idx < totalThreads)
	{
		curand_init(seed, idx, 0, &state[idx]);
	}

}

__global__ void initPop(curandState *my_curandstate, int *d_pop, int dimensions, int popSize, int totalThreads)
{
	int idx = threadIdx.x + blockDim.x*blockIdx.x;
	int row = idx * (popSize / totalThreads);
	if (popSize % totalThreads != 0)
		row++;
	
	if (idx < totalThreads)
	{
		for (int j = row; j < row + (popSize / totalThreads); j++)
		{
			int count = 0;
			int i;
			int initIterations = j * dimensions;
			int endIterations = initIterations + dimensions - 1;
			for (i = initIterations; i < endIterations; i++)
			{
				float myrandf = curand_uniform(my_curandstate + idx);
				if (myrandf < 0.5)
					d_pop[i] = 0;
				else
				{
					d_pop[i] = 1;
					count++;
				}

			}
			d_pop[i] = count;
		}
	}

}

__global__ void initProbabilites(int operators_number, double *d_operators_probabilites, int popSize, int totalThreads)
{
	int idx = threadIdx.x + blockDim.x*blockIdx.x;
	int row = idx * (popSize / totalThreads);
	if (popSize % totalThreads != 0)
		row++;
	if (idx < totalThreads)
	{
		for (int j = row; j < row + (popSize / totalThreads); j++)
		{
			int initIterations = j * operators_number;
			int endIterations = initIterations + operators_number;
			for (int i = initIterations; i < endIterations; i++)
			{
				d_operators_probabilites[i] = (1.0 / operators_number);
			}
		}
	}
}

__global__ void applyOperators(double *d_operators_probabilites, int *d_pop, int dimensions, int operators_number, curandState *my_curandstate, int pop_size, int totalThreads)
{
	int idx = threadIdx.x + blockDim.x*blockIdx.x;
	int row = idx * (pop_size / totalThreads);
	if (pop_size % totalThreads != 0)
		row++;
	if (idx < totalThreads)
	{
		for (int j = row; j < row + (pop_size / totalThreads); j++)
		{
			int i;
			int start, end;
			//Select Operator
			double sum = 0.0;
			int cont = 0;
			float rand_number = curand_uniform(my_curandstate + idx);
			for (int i = row * operators_number; i < row * operators_number + operators_number; i++)
			{
				sum += d_operators_probabilites[i];
				if (rand_number < sum)
					break;
				cont++;
			}

			start = row * dimensions;
			end = start + dimensions;
			float reward = -1.0;
			//Cross
			int newIndividuals[2 * 1000];
			if (cont == 0)
			{
				one_point_cross(newIndividuals, d_pop, start, end, my_curandstate + idx, dimensions, pop_size);
				int cont_new;
				if (newIndividuals[dimensions - 1] > d_pop[end - 1] && newIndividuals[dimensions - 1] > newIndividuals[2 * dimensions - 1])
				{
					cont_new = 0;
					for (int i = start; i < end; i++)
					{
						d_pop[i] = newIndividuals[cont_new];
						cont_new++;
					}
					reward = 1.0;
				}
				else if (newIndividuals[2 * dimensions - 1] > d_pop[end - 1] && newIndividuals[2 * dimensions - 1] > newIndividuals[dimensions - 1])
				{
					cont_new = dimensions;
					for (int i = start; i < end; i++)
					{
						d_pop[i] = newIndividuals[cont_new];
						cont_new++;
					}
					reward = 1.0;
				}

			}
			//Mutation
			else if (cont == 1)
			{
				mutation(newIndividuals, d_pop, start, end, my_curandstate + idx, dimensions);
				if (newIndividuals[dimensions - 1] > d_pop[end - 1])
				{
					int cont_new = 0;
					for (int i = start; i < end; i++)
					{
						d_pop[i] = newIndividuals[cont_new];
						cont_new++;
					}
					reward = 1.0;
				}
			}

			//Apply reward

			float plus = curand_uniform(my_curandstate + idx);
			plus = 1.0 + (plus * reward);
			d_operators_probabilites[row * operators_number + cont] *= plus;

			//Normalizes
			float sumP = 0.0;
			for (int i = row * operators_number; i < row * operators_number + operators_number; i++)
			{
				sumP += d_operators_probabilites[i];
			}

			for (int i = row * operators_number; i < row * operators_number + operators_number; i++)
			{
				d_operators_probabilites[i] /= sumP;
			}
		}
	}



}

__device__ void one_point_cross(int *newIndividuals, int *d_pop, int start, int end, curandState *my_curandstate, int dimensions, int pop_size)
{
	int pos = curand(my_curandstate) % (dimensions - 1);
	int partner = curand(my_curandstate) % pop_size;
	int cont = 0, sum = 0;
	for (int i = start; i < start + pos; i++)
	{
		newIndividuals[cont] = d_pop[i];
		sum += d_pop[i];
		cont++;
	}
	for (int i = dimensions*partner + pos; i < dimensions*partner + dimensions - 1; i++)
	{
		newIndividuals[cont] = d_pop[i];
		sum += d_pop[i];
		cont++;
	}
	newIndividuals[cont] = sum;
	cont++;
	sum = 0;
	for (int i = dimensions*partner; i < dimensions*partner + pos; i++)
	{
		newIndividuals[cont] = d_pop[i];
		sum += d_pop[i];
		cont++;
	}
	for (int i = start + pos; i < end - 1; i++)
	{
		newIndividuals[cont] = d_pop[i];
		sum += d_pop[i];
		cont++;
	}
	newIndividuals[cont] = sum;
}

__device__ void mutation(int *newIndividuals, int *d_pop, int start, int end, curandState *my_curandstate, int dimensions)
{
	int pos = curand(my_curandstate) % (dimensions - 1);
	int cont = 0, sum = 0;
	for (int i = start; i < end - 1; i++) {
		if (i == start + pos)
			newIndividuals[cont] = 1 - d_pop[i];
		else
			newIndividuals[cont] = d_pop[i];

		sum += newIndividuals[cont];
		cont++;
	}
	newIndividuals[cont] = sum;

}

