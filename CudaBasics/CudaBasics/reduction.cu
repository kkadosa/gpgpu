#include "reduction.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <iostream> // only to print "To be implemented"
#include "function.h"

// Reduction in the global memory
__global__ void reduceKernel_v0(unsigned int *g_idata, unsigned int *g_odata, size_t dataSize)
{
	unsigned int lid = threadIdx.x; // local id in the block
	unsigned int id = blockIdx.x*blockDim.x + threadIdx.x; // global id, base index of the thread block

	if (id >= dataSize) return;
	// do reduction in global memory
	for (unsigned int s = 1; s < blockDim.x; s *= 2) {
		// modulo arithmetic is slow!
		if ((lid % (2 * s)) == 0)
		{
			if (id + s < dataSize)	g_idata[id] = assocFunc(g_idata[id], g_idata[id + s]);
		}

		__syncthreads();
	}

	// write result for this block to global memory
	if (lid == 0) g_odata[blockIdx.x] = g_idata[id];
}

// Exactly the same as reduceKernel_v0
__global__ void reduceKernel_v1(unsigned int *g_idata, unsigned int *g_odata, size_t dataSize)
{
	unsigned int lid = threadIdx.x; // local id in the block
	unsigned int id = blockIdx.x*blockDim.x + threadIdx.x; // global id, base index of the thread block

	if (id >= dataSize) return;

	for (unsigned int s = 1; s < blockDim.x; s *= 2) {
		if ((lid % (2 * s)) == 0) {
			if (id + s < dataSize) {
				g_idata[id] = assocFunc(g_idata[id], g_idata[id + s]);
			}
		}

		__syncthreads();
	}
	
	// write result for this block to global memory
	if (lid == 0) g_odata[blockIdx.x] = g_idata[id];
}

// Copy the data to local (shared) memory and then do the reduction there
__global__ void reduceKernel_v2(unsigned int *g_idata, unsigned int *g_odata, size_t dataSize)
{
	// now we know the size in compile time, we need exactly one unsigned int per thread for the reduction
	// to set shared memory size in run-time, declare a pointer here and pass the required shared memory size to the kernel as <<<gridDim, blockDim, sharedMemorySizeInBytes>>>
	__shared__ unsigned int sdata[threadsPerBlock];

	unsigned int lid = threadIdx.x; // local id in the block
	unsigned int id = blockIdx.x*blockDim.x + threadIdx.x; // global id, index of the data

	if (id >= dataSize) return;

	sdata[id] = g_idata[id];
	__syncthreads();
	for (unsigned int s = 1; s < blockDim.x; s *= 2) {
		if ((lid % (2 * s)) == 0) {
			if (id + s < dataSize) {
				sdata[id] = assocFunc(sdata[id], sdata[id + s]);
			}
		}

		__syncthreads();
	}

	if (id == 0) {
		printf("%d %d\n", g_idata[0], sdata[0]);
	}
	// write result for this block to global memory
	if (lid == 0) g_odata[blockIdx.x] = sdata[0];
}

// Reduction in local (shared) memory using coalesced memory access. Same as reduceKernel_v2 with different memory read pattern.
__global__ void reduceKernel_v3(unsigned int *g_idata, unsigned int *g_odata, size_t dataSize)
{
	__shared__ unsigned int sdata[threadsPerBlock];

	unsigned int lid = threadIdx.x; // local id in the block
	unsigned int id = blockIdx.x*blockDim.x + threadIdx.x; // global id, index of the data

	// TODO
	// 1. Copy one element from g_idata to sdata
	// 2. Do reduction in shared memory, on sdata using coalesced memory access
	//    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) { ... }


	// write result for this block to global memory
	if (lid == 0) g_odata[blockIdx.x] = sdata[0];
}

// Perform the first level of the reduction from the global memory using coalesced read, then continue in the local memory.
__global__ void reduceKernel_v4(unsigned int *g_idata, unsigned int *g_odata, size_t dataSize)
{
	__shared__ unsigned int sdata[threadsPerBlock];

	unsigned int lid = threadIdx.x;
	unsigned int id = blockIdx.x*(blockDim.x * 2) + threadIdx.x;

	// TODO
	// 1. Copy one element from g_idata to sdata
	// 2. Fetch another element from g_idata using coalesced read and apply the associative function on this new element and the one read in the previous step.
	// 2. Do reduction in shared memory, on sdata using coalesced memory access, as in reduceKernel_v3


	// write result for this block to global memory
	if (lid == 0) g_odata[blockIdx.x] = sdata[0];
}

// Same as reduceKernel_v4, but this time we unroll the loop within a warp. Warp size is always 32 in CUDA.
template<unsigned int blockSize>
__global__ void reduceKernel_v5(unsigned int *g_idata, unsigned *g_odata, size_t dataSize)
{
	__shared__ unsigned int sdata[threadsPerBlock];

	unsigned int lid = threadIdx.x;
	unsigned int id = blockIdx.x*(blockDim.x * 2) + threadIdx.x;

	// TODO
	// 1. Start as reduceKernel_v4: 
	//   - Fetch two elements from g_idata and do the first level of reduction, save the result to sdata
	//   - Do the reduction in local memory (sdata), but this time down to 2x warp size (2 x 32 = 64) instead of a single element
	// 2. Do the reduction for the remaining 64 elements by unrolling the reduction loop, i.e. write replace the for loop by a sequence of IF statements:
	//    IF (blockSize > numberOfRemainingElements AND (lid < numberOfRemainingElements / 2) THEN
	//       sdata[lid] = assocFunc(sdata[lid], sdata[lid + numberOfRemainingElements / 2])
	//    ENDIF

	// write result for this block to global memory
	if (lid == 0) g_odata[blockIdx.x] = sdata[0];
}

// Same as reduceKernel_v5 but we unroll all the loops.
template<unsigned int blockSize>
__global__ void reduceKernel_v6(unsigned int *g_idata, unsigned *g_odata, size_t dataSize)
{
	__shared__ unsigned int sdata[threadsPerBlock];

	unsigned int lid = threadIdx.x;
	unsigned int id = blockIdx.x*(blockDim.x * 2) + threadIdx.x;

	// TODO: 
	// 1. Start as reduceKernel_v4: 
	//   - Fetch two elements from g_idata and do the first level of reduction, save the result to sdata
	// 2. Do the reduction by fully unrolling the loop of reduceKernel_v4.

	// write result for this block to global mem
	if (lid == 0) g_odata[blockIdx.x] = sdata[0];
}

void reduceGPU_v0(unsigned int* d_idata, unsigned int* d_odata, size_t dataSize)
{
	dim3 blockDim(threadsPerBlock, 1, 1);

#if 0
	// Number of blocks: ceil(dataSize / threadsPerBlock). Cast from size_t to uint to eliminate warning.
	dim3 gridDim(static_cast<unsigned int>((dataSize + threadsPerBlock - 1) / threadsPerBlock), 1, 1);

	// Compute partial reduction. Each block writes its partial sum to d_odata.
	reduceKernel_v0<<<gridDim, blockDim>>>(d_idata, d_odata, dataSize);
	
	// Continue the reduction on the partial sums
	dataSize = gridDim.x;
	while (dataSize > 1)
	{
		// Copy the output buffer to the input buffer (alternative: ping-pong)
		cudaMemcpy(d_idata, d_odata, dataSize * sizeof(unsigned int), cudaMemcpyDeviceToDevice);

		gridDim = dim3(static_cast<unsigned int>((dataSize + threadsPerBlock - 1) / threadsPerBlock), 1, 1);
		blockDim.x = static_cast<unsigned int>(std::min(static_cast<size_t>(blockDim.x), dataSize));
		reduceKernel_v0<<<gridDim, blockDim>>>(d_idata, d_odata, dataSize);
		dataSize = gridDim.x;
	}
#else
	// Same as above but everything merged in a single loop
	while (dataSize > 1)
	{
		// Number of blocks: ceil(dataSize / threadsPerBlock). Cast from size_t to uint to eliminate warning.
		dim3 gridDim(static_cast<unsigned int>((dataSize + threadsPerBlock - 1) / threadsPerBlock), 1, 1);
		blockDim.x = static_cast<unsigned int>(std::min(static_cast<size_t>(blockDim.x), dataSize));
		reduceKernel_v0<<<gridDim, blockDim>>>(d_idata, d_odata, dataSize);

		dataSize = gridDim.x;
		// Copy the output buffer to the input buffer (alternative: ping-pong)
		if (dataSize > 1)
		{
			cudaMemcpy(d_idata, d_odata, dataSize * sizeof(unsigned int), cudaMemcpyDeviceToDevice);
		}
	}
#endif
}

void reduceGPU_v1(unsigned int* d_idata, unsigned int* d_odata, size_t dataSize)
{
	dim3 blockDim(threadsPerBlock, 1, 1);

	bool odd = true;

	while (dataSize > 1) {
		// Number of blocks: ceil(dataSize / threadsPerBlock). Cast from size_t to uint to eliminate warning.
		dim3 gridDim(static_cast<unsigned int>((dataSize + threadsPerBlock - 1) / threadsPerBlock), 1, 1);
		blockDim.x = static_cast<unsigned int>(std::min(static_cast<size_t>(blockDim.x), dataSize));

		if (odd) {
			reduceKernel_v1 <<<gridDim, blockDim >>>(d_idata, d_odata, dataSize);
		}
		else {
			reduceKernel_v1 <<<gridDim, blockDim >>>(d_odata, d_idata, dataSize);
		}
		odd = !odd;

		dataSize = gridDim.x;
	}
	if (!odd) {
		cudaMemcpy(d_odata, d_idata, sizeof(unsigned int), cudaMemcpyDeviceToDevice);
	}
}

void reduceGPU_v2(unsigned int* d_idata, unsigned int* d_odata, size_t dataSize)
{
	dim3 blockDim(threadsPerBlock, 1, 1);

	bool odd = true;

	std::cout << "G" << dataSize;
	while (dataSize > 1) {
		// Number of blocks: ceil(dataSize / threadsPerBlock). Cast from size_t to uint to eliminate warning.
		dim3 gridDim(static_cast<unsigned int>((dataSize + threadsPerBlock - 1) / threadsPerBlock), 1, 1);
		blockDim.x = static_cast<unsigned int>(std::min(static_cast<size_t>(blockDim.x), dataSize));

		if (odd) {
			reduceKernel_v2 <<<gridDim, blockDim, dataSize * sizeof(unsigned int)>>>(d_idata, d_odata, dataSize);
		}
		else {
			reduceKernel_v2 <<<gridDim, blockDim, dataSize * sizeof(unsigned int)>>>(d_odata, d_idata, dataSize);
		}
		odd = !odd;
		std::cout << "G" << dataSize;

		dataSize = gridDim.x;
	}
	if (!odd) {
		cudaMemcpy(d_odata, d_idata, sizeof(unsigned int), cudaMemcpyDeviceToDevice);
	}
}

void reduceGPU_v3(unsigned int* d_idata, unsigned int* d_odata, size_t dataSize)
{
	dim3 blockDim(threadsPerBlock, 1, 1);

	bool odd = true;

	while (dataSize > 1) {
		// Number of blocks: ceil(dataSize / threadsPerBlock). Cast from size_t to uint to eliminate warning.
		dim3 gridDim(static_cast<unsigned int>((dataSize + threadsPerBlock - 1) / threadsPerBlock), 1, 1);
		blockDim.x = static_cast<unsigned int>(std::min(static_cast<size_t>(blockDim.x), dataSize));

		if (odd) {
			reduceKernel_v3 <<<gridDim, blockDim, dataSize * sizeof(unsigned int) >>> (d_idata, d_odata, dataSize);
		}
		else {
			reduceKernel_v3 <<<gridDim, blockDim, dataSize * sizeof(unsigned int) >>> (d_odata, d_idata, dataSize);
		}
		odd = !odd;

		dataSize = gridDim.x;
	}
	if (!odd) {
		cudaMemcpy(d_odata, d_idata, sizeof(unsigned int), cudaMemcpyDeviceToDevice);
	}
}

void reduceGPU_v4(unsigned int* d_idata, unsigned int* d_odata, size_t dataSize)
{
	// TODO: Same as reduceGPU_v3, but call reduceKernel_v4 (!!!)
	std::cout << "<TO BE IMPLEMENTED> "; // TODO: remove me
}

void reduceGPU_v5(unsigned int* d_idata, unsigned int* d_odata, size_t dataSize)
{
	dim3 blockDim(threadsPerBlock, 1, 1);

	std::cout << "<TO BE IMPLEMENTED> "; // TODO: remove me
	// TODO: 
	// - Do ping-pong as in reduceGPU_v1
	// - Call the proper template instant of reduceKernel_v5
	while (dataSize > 1)
	{
		// Number of blocks: ceil(dataSize / threadsPerBlock*2). Cast from size_t to uint to eliminate warning.
		dim3 gridDim(static_cast<unsigned int>((dataSize + threadsPerBlock*2 - 1) / (threadsPerBlock*2)), 1, 1);
		blockDim.x = static_cast<unsigned int>(std::min(static_cast<size_t>(blockDim.x), dataSize));

		// - Call the proper template instant of reduceKernel_v5, make sure pIn contains the input and pOut the output
		switch(blockDim.x)
		{
		case 1024:
			//reduceKernel_v5<1024><<<gridDim, blockDim>>>(pIn, pOut, dataSize);
			break;
		case 512:
			//reduceKernel_v5<512><<<gridDim, blockDim>>>(pIn, pOut, dataSize);
			break;
		case 256:
			//reduceKernel_v5<256><<<gridDim, blockDim>>>(pIn, pOut, dataSize);
			break;
		case 128:
			//reduceKernel_v5<128><<<gridDim, blockDim>>>(pIn, pOut, dataSize);
			break;
		case 64:
			//reduceKernel_v5<64><<<gridDim, blockDim>>>(pIn, pOut, dataSize);
			break;
		case 32:
			//reduceKernel_v5<32><<<gridDim, blockDim>>>(pIn, pOut, dataSize);
			break;
		case 16:
			//reduceKernel_v5<16><<<gridDim, blockDim>>>(pIn, pOut, dataSize);
			break;
		case 8:
			//reduceKernel_v5<8><<<gridDim, blockDim>>>(pIn, pOut, dataSize);
			break;
		case 4:
			//reduceKernel_v5<4><<<gridDim, blockDim>>>(pIn, pOut, dataSize);
			break;
		case 2:
			//reduceKernel_v5<2><<<gridDim, blockDim>>>(pIn, pOut, dataSize);
			break;
		case 1:
			//reduceKernel_v5<1><<<gridDim, blockDim>>>(pIn, pOut, dataSize);
			break;
		}

		dataSize = gridDim.x;
		// TODO: swap pointers
	}
}

void reduceGPU_v6(unsigned int* d_idata, unsigned int* d_odata, size_t dataSize)
{
	dim3 blockDim(threadsPerBlock, 1, 1);

	// TODO: Same as reduceGPU_v5, but call reduceKernel_v6 (!!!)
	std::cout << "<TO BE IMPLEMENTED> "; // TODO: remove me
}

void reduceGPU(unsigned int* d_idata, unsigned int* d_odata, size_t dataSize, int method, bool synchronizeDevice)
{
	switch (method)
	{
	case 0:
		reduceGPU_v0(d_idata, d_odata, dataSize);
		break;
	case 1:
		reduceGPU_v1(d_idata, d_odata, dataSize);
		break;
	case 2:
		reduceGPU_v2(d_idata, d_odata, dataSize);
		break;
	case 3:
		reduceGPU_v3(d_idata, d_odata, dataSize);
		break;
	case 4:
		reduceGPU_v4(d_idata, d_odata, dataSize);
		break;
	case 5:
		reduceGPU_v5(d_idata, d_odata, dataSize);
		break;
	case 6:
		reduceGPU_v6(d_idata, d_odata, dataSize);
		break;
	}

	if (synchronizeDevice) cudaDeviceSynchronize();
}