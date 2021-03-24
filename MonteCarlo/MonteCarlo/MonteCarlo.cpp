// MonteCarlo.cpp : Defines the entry point for the console application.

#include <string>
#include <vector>
#include "Common.h"

// OpenCL C++ API
#include "cl.hpp"

const bool writeOutRandoms = true;
const size_t randomNumbers = 1000;
const size_t threadCount = 512;

void sina(cl::Program& program, cl::Context& context, cl::CommandQueue& queue, cl::Device& device) {
	cl_int err = CL_SUCCESS;
	cl::Event event1;
	// Get the kernel handle
	cl::Kernel kernel1(program, "randomLCG", &err);
	CheckCLError(err);

	std::vector<float> random(threadCount * randomNumbers);
	std::vector<float> seeds(threadCount);
	for (int i = 0; i < threadCount; ++i){
		seeds[i] = rand();
	}

	cl::Buffer seedBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(float) * threadCount, NULL, &err);
	queue.enqueueWriteBuffer(seedBuffer, true, 0, sizeof(float) * threadCount, seeds.data());

	cl::Buffer randomBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * threadCount * randomNumbers, NULL, &err);

	// Set the kernel parameters	
	kernel1.setArg(0, randomNumbers);
	kernel1.setArg(1, seedBuffer);
	kernel1.setArg(2, randomBuffer);

	// Enqueue the kernel: threadCount threads in total, each generating random numbers in [0,1] randomNumbers times
	queue.enqueueNDRangeKernel(kernel1,
		cl::NullRange,
		cl::NDRange(threadCount, 1),
		cl::NullRange,
		NULL,
		&event1);
	event1.wait();
	std::vector<float> output(threadCount);
	cl::Buffer outputBuffer = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * threadCount, NULL, &err);
	queue.enqueueReadBuffer(randomBuffer, true, 0, sizeof(float) * threadCount * randomNumbers, random.data());
	queue.enqueueWriteBuffer(randomBuffer, true, 0, sizeof(float) * threadCount * randomNumbers, random.data());
	cl::Kernel kernel2(program, "mcInt1D", &err);
	cl::Event event2;
	kernel2.setArg(0, randomNumbers);
	kernel2.setArg(1, randomBuffer);
	kernel2.setArg(2, outputBuffer);
	queue.enqueueNDRangeKernel(kernel2,
		cl::NullRange,
		cl::NDRange(threadCount, 1),
		cl::NullRange,
		NULL,
		&event2);
	event2.wait();

	queue.enqueueReadBuffer(outputBuffer, true, 0, sizeof(float) * threadCount, output.data());

	float sum = 0;
	for (auto t : output) {
		//std::cout << t << std::endl;
		sum += t;
	}
	std::cout << "Integral: " << sum/threadCount << std::endl;
}

void sinb(cl::Program& program, cl::Context& context, cl::CommandQueue& queue, cl::Device& device) {
	cl_int err = CL_SUCCESS;
	cl::Event event1;
	// Get the kernel handle
	cl::Kernel kernel1(program, "haltonSequence", &err);
	CheckCLError(err);

	std::vector<float> random(threadCount * randomNumbers);

	cl::Buffer randomBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * threadCount * randomNumbers, NULL, &err);

	// Set the kernel parameters	
	kernel1.setArg(0, randomNumbers);
	kernel1.setArg(1, 2);
	kernel1.setArg(2, randomBuffer);

	// Enqueue the kernel: threadCount threads in total, each generating random numbers in [0,1] randomNumbers times
	queue.enqueueNDRangeKernel(kernel1,
		cl::NullRange,
		cl::NDRange(threadCount, 1),
		cl::NullRange,
		NULL,
		&event1);
	event1.wait();
	std::vector<float> output(threadCount);
	cl::Buffer outputBuffer = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * threadCount, NULL, &err);
	queue.enqueueReadBuffer(randomBuffer, true, 0, sizeof(float) * threadCount * randomNumbers, random.data());
	queue.enqueueWriteBuffer(randomBuffer, true, 0, sizeof(float) * threadCount * randomNumbers, random.data());
	cl::Kernel kernel2(program, "mcInt1D", &err);
	cl::Event event2;
	kernel2.setArg(0, randomNumbers);
	kernel2.setArg(1, randomBuffer);
	kernel2.setArg(2, outputBuffer);
	queue.enqueueNDRangeKernel(kernel2,
		cl::NullRange,
		cl::NDRange(threadCount, 1),
		cl::NullRange,
		NULL,
		&event2);
	event2.wait();

	queue.enqueueReadBuffer(outputBuffer, true, 0, sizeof(float) * threadCount, output.data());

	float sum = 0;
	for (auto t : output) {
		//std::cout << t << std::endl;
		sum += t;
	}
	std::cout << "Integral: " << sum / threadCount << std::endl;
}

void sphere(cl::Program& program, cl::Context& context, cl::CommandQueue& queue, cl::Device& device) {
	cl_int err = CL_SUCCESS;
	cl::Event event1;
	// Get the kernel handle
	cl::Kernel kernel1(program, "haltonSequence", &err);
	CheckCLError(err);

	std::vector<float> random1(threadCount * randomNumbers);
	std::vector<float> random2(threadCount * randomNumbers);
	std::vector<float> random3(threadCount * randomNumbers);

	cl::Buffer randomBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * threadCount * randomNumbers, NULL, &err);

	// Set the kernel parameters	
	kernel1.setArg(0, randomNumbers);
	kernel1.setArg(1, 2);
	kernel1.setArg(2, randomBuffer);

	// Enqueue the kernel: threadCount threads in total, each generating random numbers in [0,1] randomNumbers times
	queue.enqueueNDRangeKernel(kernel1,
		cl::NullRange,
		cl::NDRange(threadCount, 1),
		cl::NullRange,
		NULL,
		&event1);
	event1.wait();
	queue.enqueueReadBuffer(randomBuffer, true, 0, sizeof(float) * threadCount * randomNumbers, random1.data());
	std::cout << "Sphere: " << std::endl;
	kernel1.setArg(1, 3);
	queue.enqueueNDRangeKernel(kernel1,
		cl::NullRange,
		cl::NDRange(threadCount, 1),
		cl::NullRange,
		NULL,
		&event1);
	event1.wait();
	queue.enqueueReadBuffer(randomBuffer, true, 0, sizeof(float) * threadCount * randomNumbers, random2.data());
	kernel1.setArg(1, 5);
	std::cout << "Sphere: " << std::endl;
	queue.enqueueNDRangeKernel(kernel1,
		cl::NullRange,
		cl::NDRange(threadCount, 1),
		cl::NullRange,
		NULL,
		&event1);
	event1.wait();
	queue.enqueueReadBuffer(randomBuffer, true, 0, sizeof(float) * threadCount * randomNumbers, random3.data());

	std::vector<float> output(threadCount);
	cl::Buffer outputBuffer = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * threadCount, NULL, &err);

	cl::Buffer randomBuffer1 = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(float) * threadCount * randomNumbers, NULL, &err);
	queue.enqueueWriteBuffer(randomBuffer1, true, 0, sizeof(float) * threadCount * randomNumbers, random1.data());
	cl::Buffer randomBuffer2 = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(float) * threadCount * randomNumbers, NULL, &err);
	queue.enqueueWriteBuffer(randomBuffer2, true, 0, sizeof(float) * threadCount * randomNumbers, random2.data());
	cl::Buffer randomBuffer3 = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(float) * threadCount * randomNumbers, NULL, &err);
	queue.enqueueWriteBuffer(randomBuffer3, true, 0, sizeof(float) * threadCount * randomNumbers, random3.data());

	cl::Kernel kernel2(program, "sphere", &err);
	cl::Event event2;
	kernel2.setArg(0, randomNumbers);
	kernel2.setArg(1, randomBuffer1);
	kernel2.setArg(2, randomBuffer2);
	kernel2.setArg(3, randomBuffer3);
	kernel2.setArg(4, outputBuffer);
	queue.enqueueNDRangeKernel(kernel2,
		cl::NullRange,
		cl::NDRange(threadCount, 1),
		cl::NullRange,
		NULL,
		&event2);
	event2.wait();

	queue.enqueueReadBuffer(outputBuffer, true, 0, sizeof(float) * threadCount, output.data());

	float sum = 0;
	for (auto t : output) {
		sum += t;
	}
	std::cout << "Sphere: " << sum/threadCount << std::endl;
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

	sina(program, context, queue, devices[0]);
	sinb(program, context, queue, devices[0]);
	sphere(program, context, queue, devices[0]);

	std::cout << "Finished" << std::endl;
}

int main()
{
	cppapi();
	std::string F;
	std::getline(std::cin, F);
	return 0;
}

