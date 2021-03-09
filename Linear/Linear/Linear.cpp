// Primitives.cpp : Defines the entry point for the console application.

#include <string>
#include <vector>
#include <chrono>
#include "Common.h"

// OpenCL C++ API
#include "cl.hpp"
#include "Linear.h"

// Gaussian elimination
const int GAn = 4;
const int GAm = 3;

float GA[] = { 2,  1, -1,   8,
			  -3, -1,  2, -11,
			  -2,  1,  2,  -3  };

int GBn = 6;
int GBm = 3;
float GB[] = {  2, -1,  0,  1, 0, 0,
			   -1,  2, -1,  0, 1, 0,
			    0, -1,  2,  0, 0, 1  };


void scalarMV(int n, int m, float* y, const float* A, const float* x, const float* b) {
	for (int i = 0; i<n; ++i) {
		float yi = b[i];
		for (int j = 0; j<m; ++j) {
			yi += A[i * m + j] * x[j];
		}
		y[i] = yi;
	}
}

// Jacobi iteration
const int Jn = 1024;
float* Ping = NULL;
float* Pong = NULL;
float* Matrix = NULL;
float* Offset = NULL;

void generateLinEq()
{
	Ping = new float[Jn];
	Pong = new float[Jn];
	for (int i = 0; i < Jn; ++i) {
		Ping[i] = 0.0f;
		Pong[i] = 0.0f;
	}

	Matrix = new float[Jn * Jn];
	for (int i = 0; i < Jn; ++i) {
		for (int j = 0; j < Jn; ++j) {
			float v = 0.0f;
			if (i == j) {
				v = 0.5f;
			}
			Matrix[i + j * Jn] = v;
		}
	}

	Offset = new float[Jn];
	for (int i = 0; i < Jn; ++i) {
		Offset[i] = 1.0f;
	}
}

void releaseLinEq()
{
	if (Ping == 0) delete[] Ping;
	if (Pong == 0) delete[] Pong;
	if (Matrix == 0) delete[] Matrix;
	if (Offset == 0) delete[] Offset;
}

void jacobi1(cl::Program& program, cl::Context& context, cl::CommandQueue& queue, cl::Device& device) {
	cl_int err = CL_SUCCESS;
	// Get the kernel handle
	cl::Kernel kernel(program, "simpleMV", &err);
	CheckCLError(err);
	cl::Event event;

	generateLinEq();
	auto start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < 10; ++i) {
		cl::Buffer pingBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * Jn, NULL, &err);
		queue.enqueueWriteBuffer(pingBuffer, true, 0, sizeof(float) * Jn, Ping);

		cl::Buffer pongBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * Jn, NULL, &err);
		queue.enqueueWriteBuffer(pongBuffer, true, 0, sizeof(float) * Jn, Pong);

		cl::Buffer matrixBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(float) * Jn *Jn, NULL, &err);
		queue.enqueueWriteBuffer(matrixBuffer, true, 0, sizeof(float) * Jn*Jn, Matrix);

		cl::Buffer offsetBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(float) * Jn, NULL, &err);
		queue.enqueueWriteBuffer(offsetBuffer, true, 0, sizeof(float) * Jn, Offset);

		kernel.setArg(0, Jn);
		kernel.setArg(1, Jn);
		kernel.setArg(2, pingBuffer);
		kernel.setArg(3, matrixBuffer);
		kernel.setArg(4, pongBuffer);
		kernel.setArg(5, offsetBuffer);

		queue.enqueueNDRangeKernel(kernel,
			cl::NullRange,
			cl::NDRange(Jn, 1),
			cl::NDRange(Jn, 1),
			NULL,
			&event);
		event.wait();

		queue.enqueueReadBuffer(pongBuffer, true, 0, sizeof(float) * Jn, Pong);

		kernel.setArg(2, pongBuffer);
		kernel.setArg(4, pingBuffer);

		queue.enqueueNDRangeKernel(kernel,
			cl::NullRange,
			cl::NDRange(Jn, 1),
			cl::NDRange(Jn, 1),
			NULL,
			&event);
		event.wait();

		queue.enqueueReadBuffer(pingBuffer, true, 0, sizeof(float) * Jn, Ping);
	}

	auto end = std::chrono::high_resolution_clock::now();
	std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;
	releaseLinEq();
}


void jacobi2(cl::Program& program, cl::Context& context, cl::CommandQueue& queue, cl::Device& device) {
	cl_int err = CL_SUCCESS;
	// Get the kernel handle
	cl::Kernel kernel(program, "reduceMV", &err);
	CheckCLError(err);
	cl::Event event;

	generateLinEq();
	auto start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < 10; ++i) {
		cl::Buffer pingBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * Jn, NULL, &err);
		queue.enqueueWriteBuffer(pingBuffer, true, 0, sizeof(float) * Jn, Ping);

		cl::Buffer pongBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * Jn, NULL, &err);
		queue.enqueueWriteBuffer(pongBuffer, true, 0, sizeof(float) * Jn, Pong);

		cl::Buffer matrixBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(float) * Jn * Jn, NULL, &err);
		queue.enqueueWriteBuffer(matrixBuffer, true, 0, sizeof(float) * Jn * Jn, Matrix);

		cl::Buffer offsetBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(float) * Jn, NULL, &err);
		queue.enqueueWriteBuffer(offsetBuffer, true, 0, sizeof(float) * Jn, Offset);

		kernel.setArg(0, Jn);
		kernel.setArg(1, Jn);
		kernel.setArg(2, pingBuffer);
		kernel.setArg(3, matrixBuffer);
		kernel.setArg(4, pongBuffer);
		kernel.setArg(5, offsetBuffer);
		kernel.setArg(6, sizeof(float) * Jn, nullptr);

		queue.enqueueNDRangeKernel(kernel,
			cl::NullRange,
			cl::NDRange(Jn * Jn, 1),
			cl::NDRange(Jn, 1),
			NULL,
			&event);
		event.wait();

		queue.enqueueReadBuffer(pongBuffer, true, 0, sizeof(float) * Jn, Pong);

		kernel.setArg(2, pongBuffer);
		kernel.setArg(4, pingBuffer);

		queue.enqueueNDRangeKernel(kernel,
			cl::NullRange,
			cl::NDRange(Jn*Jn, 1),
			cl::NDRange(Jn, 1),
			NULL,
			&event);
		event.wait();

		queue.enqueueReadBuffer(pingBuffer, true, 0, sizeof(float) * Jn, Ping);
	}
	auto end = std::chrono::high_resolution_clock::now();
	std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;
	releaseLinEq();
}


void jacobi3(cl::Program& program, cl::Context& context, cl::CommandQueue& queue, cl::Device& device) {
	cl_int err = CL_SUCCESS;
	// Get the kernel handle
	cl::Kernel kernel(program, "largeMV", &err);
	CheckCLError(err);
	cl::Event event;

	generateLinEq();
	auto start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < 10; ++i) {
		cl::Buffer pingBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * Jn, NULL, &err);
		queue.enqueueWriteBuffer(pingBuffer, true, 0, sizeof(float) * Jn, Ping);

		cl::Buffer pongBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * Jn, NULL, &err);
		queue.enqueueWriteBuffer(pongBuffer, true, 0, sizeof(float) * Jn, Pong);

		cl::Buffer matrixBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(float) * Jn * Jn, NULL, &err);
		queue.enqueueWriteBuffer(matrixBuffer, true, 0, sizeof(float) * Jn * Jn, Matrix);

		cl::Buffer offsetBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(float) * Jn, NULL, &err);
		queue.enqueueWriteBuffer(offsetBuffer, true, 0, sizeof(float) * Jn, Offset);

		const int T = 32;
		const int Z = 32;
		kernel.setArg(0, Jn);
		kernel.setArg(1, Jn);
		kernel.setArg(2, pingBuffer);
		kernel.setArg(3, matrixBuffer);
		kernel.setArg(4, pongBuffer);
		kernel.setArg(5, offsetBuffer);
		kernel.setArg(6, T);
		kernel.setArg(7, Z);
		kernel.setArg(8, sizeof(float) * T * Z , nullptr);

		queue.enqueueNDRangeKernel(kernel,
			cl::NullRange,
			cl::NDRange(T*Z, 1),
			cl::NDRange(T * Z, 1),
			NULL,
			&event);
		event.wait();

		queue.enqueueReadBuffer(pongBuffer, true, 0, sizeof(float) * Jn, Pong);

		kernel.setArg(2, pongBuffer);
		kernel.setArg(4, pingBuffer);

		queue.enqueueNDRangeKernel(kernel,
			cl::NullRange,
			cl::NDRange(T * Z, 1),
			cl::NDRange(T * Z, 1),
			NULL,
			&event);
		event.wait();

		queue.enqueueReadBuffer(pingBuffer, true, 0, sizeof(float) * Jn, Ping);
	}
	auto end = std::chrono::high_resolution_clock::now();
	std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;
	releaseLinEq();
}

void gauss(cl::Program& program, cl::Context& context, cl::CommandQueue& queue, cl::Device& device) {
	cl_int err = CL_SUCCESS;
	// Get the kernel handle
	cl::Kernel kernel(program, "gaussian", &err);
	CheckCLError(err);
	cl::Event event;

	// Allocate and upload the input data

	cl::Buffer clInputBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(float) * GAn * GAm, NULL, &err);
	queue.enqueueWriteBuffer(clInputBuffer, true, 0, sizeof(float) * GAn * GAm, GA);


	// Set the kernel parameters	
	kernel.setArg(0, GAn);
	kernel.setArg(1, GAm);
	kernel.setArg(2, clInputBuffer);

	// Enqueue the kernel
	queue.enqueueNDRangeKernel(kernel,
		cl::NullRange,
		cl::NDRange(GAm, 1),
		cl::NDRange(GAm, 1),
		NULL,
		&event);
	event.wait();

	// Copy result back to host
	queue.enqueueReadBuffer(clInputBuffer, true, 0, sizeof(float) * GAn * GAm, GA);

	// Validate the result
	for (int i = 0; i < GAm; ++i) {
		for (int j = 0; j < GAn; ++j) {
			std::cout << GA[j + i * GAn];
			if (j < GAn - 1) std::cout << ", ";
		}
		std::cout << std::endl;
	}

	//Same, but B matrix

	cl::Buffer clInputBuffer2 = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(float) * GBn * GBm, NULL, &err);
	queue.enqueueWriteBuffer(clInputBuffer2, true, 0, sizeof(float) * GBn * GBm, GB);


	kernel.setArg(0, GBn);
	kernel.setArg(1, GBm);
	kernel.setArg(2, clInputBuffer2);

	
	queue.enqueueNDRangeKernel(kernel,
		cl::NullRange,
		cl::NDRange(GBm, 1),
		cl::NDRange(GBm, 1),
		NULL,
		&event);
	event.wait();

	queue.enqueueReadBuffer(clInputBuffer2, true, 0, sizeof(float) * GBn * GBm, GB);

	for (int i = 0; i < GBm; ++i) {
		for (int j = 0; j < GBn; ++j) {
			std::cout << GB[j + i * GBn];
			if (j < GBn - 1) std::cout << ", ";
		}
		std::cout << std::endl;
	}
}

void cppapi()
{
	cl_int err = CL_SUCCESS;
	// Get a platform ID
	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);
	if (platforms.size() == 0)
	{
		std::cout << "Unable to find suitable platform." << std::endl;
		exit(-1);
	}

	// Create a context
	cl_context_properties properties[] =
	{ CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[0])(), 0 };
	cl::Context context(CL_DEVICE_TYPE_GPU, properties);

	// Enumerate the devices
	std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();

	// Create the command queue
	cl::Event event;
	cl::CommandQueue queue(context, devices[0], 0, &err);

	// Create the OpenCL program
	std::string programSource = FileToString("../kernels/programs.cl");
	cl::Program program = cl::Program(context, programSource);
	program.build(devices);
	std::cout << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]) << std::endl;
	
	gauss(program, context, queue, devices[0]);
	jacobi1(program, context, queue, devices[0]);
	jacobi2(program, context, queue, devices[0]);
	jacobi3(program, context, queue, devices[0]);
	
	std::cout << "Finished" << std::endl;
}

int main()
{
	cppapi();
	std::string F;
	std::getline(std::cin, F);
    return 0;
}

