#include <string>
#include <vector>
#include <iostream>
#include <random>

#include <cuda_runtime.h>
#include "Common.h"
#include "function.h"
#include "reduction.h"

const unsigned int dataPow = 24; // data size is 2^dataPow
const size_t dataSize = 1 << dataPow; // data size
const size_t vmax = 1 << (sizeof(unsigned int)*8 - dataPow - 1);  // maximum value + 1 so that the sum fits in an uint

void initDevice()
{
	static_assert(dataPow < sizeof(unsigned int)*8, "Size of data cannot be higher than 2^32");
	static_assert(threadsPerBlock < dataSize, "There is no sense in doing reduction for such a small data");

	int device_count;
	checkCudaErrors(cudaGetDeviceCount(&device_count));

	if (device_count == 0)
	{
		std::cout << "Error: no CUDA capable device was found." << std::endl;
		exit(EXIT_FAILURE);
	}

	const int deviceID = 0; // Use the first GPU (ID == 0) found

	cudaDeviceProp deviceProp;
	checkCudaErrors(cudaGetDeviceProperties(&deviceProp, deviceID));

	// Check if we can use the GPU
	if (deviceProp.computeMode == cudaComputeModeProhibited)
	{
		fprintf(stderr, "Error: device is running in <Compute Mode Prohibited>, no threads can use ::cudaSetDevice().\n");
		exit(EXIT_FAILURE);
	}

	// Check for at least CUDA 1.0 support
	if (deviceProp.major < 1)
	{
		std::cout << "Error: the GPU device does not support CUDA" << std::endl;
		exit(EXIT_FAILURE);
	}

	std::cout << "CUDA device found: " << deviceProp.name << ". Compute capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;

	// This is not always needed, the first free GPU is made active at the first CUDA call
	checkCudaErrors(cudaSetDevice(deviceID));
}

// Fill the input with random data in [0,vmax)
void createInput(std::vector<unsigned int>& v)
{
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<> distribution(0, vmax - 1);

	v.clear();
	for (size_t index = 0; index < dataSize; ++index)
	{
		v.push_back(distribution(gen));
	}
}

// Sequential reduction on the CPU
unsigned int reduceCPU(const std::vector<unsigned int> &data)
{
	unsigned int sum = NullElement();

	for (auto d : data)
	{
		sum = assocFunc(sum, d);
	}
	return sum;
}

// Parallel sum on the CPU with OpenMP.
// Does not work with arbitrary associative functions, sum is hardwired now.
// (Higher OpenMP versions support custom reduction operator)
unsigned int reduceCPU_OMP(const std::vector<unsigned int> &data)
{
	unsigned int sum = 0;

#pragma omp parallel for reduction(+:sum)
	for (int i = 0; i < static_cast<int>(data.size()); ++i)
	{
		sum += data[i];
	}
	return sum;
}

/////////////////
// Main program
//////////////////
int main(int argc, char **argv)
{
	initDevice();

	// input buffer on host
	std::vector<unsigned int> h_idata;
	createInput(h_idata);
	/*
	// run reduction on the CPU
	int sumCpu = 0;
	std::cout << "Reduction (CPU): ";
	_PROFILE(sumCpu = reduceCPU(h_idata));
	std::cout << "CPU sum: " << sumCpu << std::endl;
	
	// run parallel reduction (sum) on the CPU
	sumCpu = 0;
	std::cout << "Sum (CPU-OMP): ";
	_PROFILE(sumCpu = reduceCPU_OMP(h_idata));
	std::cout << "CPU sum: " << sumCpu << std::endl;
	*/
	// allocate input buffer on device
	size_t dataSizeInBytes = h_idata.size() * sizeof(h_idata[0]);
	unsigned int* d_idata = nullptr;
	checkCudaErrors(cudaMalloc(&d_idata, dataSizeInBytes));
	// allocate output buffer on device
	unsigned int* d_odata = nullptr;
	checkCudaErrors(cudaMalloc(&d_odata, dataSizeInBytes));

	// run the different GPU versions
	const int nMethods = 7;
	std::vector<std::string> methodDescriptions = {
		"Base (global memory)",
		"Ping-pong",
		"Local memory",
		"Coalesced",
		"Coalesced (global)",
		"Unroll warp",
		"Full unroll",
	};
	for (int iMethod = 0; iMethod < nMethods; ++iMethod)
	{
		// copy data to the device
		cudaMemcpy(d_idata, h_idata.data(), dataSizeInBytes, cudaMemcpyHostToDevice);
		checkCudaErrors(cudaDeviceSynchronize());

		// clear the result buffer to make sure that we will the result of the next method, not something else from an earlier run
		cudaMemset(d_odata, 0, dataSizeInBytes);
		checkCudaErrors(cudaDeviceSynchronize());

		// optional: execute a reduction for warmup
		//reduceGPU(d_idata, d_odata, dataSize, iMethod, true);
		//cudaMemcpy(d_idata, h_idata.data(), dataSizeInBytes, cudaMemcpyHostToDevice);
		//checkCudaErrors(cudaDeviceSynchronize());

		// execute the reduction
		std::cout << "Reduction (GPU, v" << iMethod << " - " << methodDescriptions[iMethod] << "): ";
		_PROFILE(reduceGPU(d_idata, d_odata, dataSize, iMethod, true));

		// read back the result
		checkCudaErrors(cudaDeviceSynchronize()); //synchronize the device in case we forgot it in the reduction caller
		unsigned int sumGPU;
		cudaMemcpy(&sumGPU, d_odata, sizeof(h_idata[0]), cudaMemcpyDeviceToHost);

		unsigned int t;
		cudaMemcpy(&t, d_idata, sizeof(h_idata[0]), cudaMemcpyDeviceToHost);
		std::cout << "GPU sum: " << sumGPU  << std::endl;//<< " other: " << t
	}

	std::cout << std::endl << "Finished. Press any key (+Enter) to quit." << std::endl;
	char ch;
	std::cin >> ch;
}
