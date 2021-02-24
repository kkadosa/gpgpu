// Primitives.cpp : Defines the entry point for the console application.

#include "Common.h"
#include "cl.hpp"

#include <random>
#include <string>
#include <vector>
#include <chrono>
#include <functional>
#include <numeric>
#include <iomanip>

class Timer
{

private:
	static std::chrono::time_point<std::chrono::high_resolution_clock> t_start;

public:
	static void start()
	{
		t_start = std::chrono::high_resolution_clock::now();
	}

	static void end(unsigned int nRuns = 1)
	{
		auto t_end = std::chrono::high_resolution_clock::now();
		std::cout << "CPU [time] " <<
			std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start).count() / nRuns << " ns" << std::endl;
	}

	static void measure(const std::function<void(void)>& program, unsigned int nRuns = 10000)
	{
		start();
		for (unsigned int i = 0; i < nRuns; ++i)
		{
			program();
		}
		end(nRuns);
	}
};
std::chrono::time_point<std::chrono::high_resolution_clock> Timer::t_start;
void printTimeStats(const cl::Event& event)
{
	cl_int err = CL_SUCCESS;
	event.wait();
	cl_ulong execStart, execEnd;
	execStart = event.getProfilingInfo<CL_PROFILING_COMMAND_START>(&err);
	if (err != CL_SUCCESS)
	{
		std::cerr << "Error during profile query: CL_PROFILING_COMMAND_START ["
			<< err << "]." << std::endl;
	}
	execEnd = event.getProfilingInfo<CL_PROFILING_COMMAND_END>(&err);
	if (err != CL_SUCCESS)
	{
		std::cerr << "Error during profile query: CL_PROFILING_COMMAND_END ["
			<< err << "]." << std::endl;
	}
	//std::cout << "[start] " << execStart << " [end] " << execEnd
	// << " [time] " << (execEnd - execStart) / 1e+06 << "ms." << std::endl;
	std::cout << "GPU [time] " << (execEnd - execStart) << " ns" <<
		std::endl;
}

void histogram_global(cl::Program& program, cl::Context& context, cl::CommandQueue& queue)
{
	const size_t dataSize = 4096;
	cl_int err = CL_SUCCESS;

	const int vmax = 41;
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<int> distr(0, vmax);
	std::vector<int> hostBuffer;
	for (size_t index = 0; index < dataSize; ++index)
	{
		hostBuffer.push_back(distr(gen));
	}
	std::vector<int> control(43, 0);
	Timer::measure([&]() {
		for (int i : hostBuffer) {
			control[i] += 1;
		}
		}, 1);

	cl::Kernel kernel(program, "histogram_global", &err);
	CheckCLError(err);

	cl::Buffer clInputBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(int) * dataSize, NULL, &err);
	queue.enqueueWriteBuffer(clInputBuffer, true, 0, sizeof(int) * dataSize, hostBuffer.data());

	// Allocate the output data
	cl::Buffer clResultBuffer = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(int) * dataSize, NULL, &err);

	// Set the kernel parameters
	kernel.setArg(0, clInputBuffer);
	kernel.setArg(1, clResultBuffer);

	// Enqueue the kernel
	cl::Event event;
	queue.enqueueNDRangeKernel(kernel,
		cl::NullRange,
		cl::NDRange(dataSize, 1),
		cl::NullRange,
		NULL,
		&event);
	event.wait();
	printTimeStats(event);

	// Copy result back to host
	queue.enqueueReadBuffer(clResultBuffer, true, 0, sizeof(int) * dataSize, hostBuffer.data());

	//Validate
	for (size_t index = 0; index <= vmax; ++index) {
		std::cout << index << " " << hostBuffer[index] << ", should be " << control[index];
		if (hostBuffer[index] == control[index]) {
			std::cout << " correct" << std::endl;
		} else {
			std::cout << " wrong" << std::endl;
		}
	}
}

void histogram_local(cl::Program& program, cl::Context& context, cl::CommandQueue& queue)
{
	const size_t dataSize = 4096;
	cl_int err = CL_SUCCESS;

	const int vmax = 41;
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<int> distr(0, vmax);
	std::vector<int> hostBuffer;
	for (size_t index = 0; index < dataSize; ++index)
	{
		hostBuffer.push_back(distr(gen));
	}
	std::vector<int> control(43, 0);
	Timer::measure([&]() {
		for (int i : hostBuffer) {
			control[i] += 1;
		}
		}, 1);

	cl::Kernel kernel(program, "histogram_local", &err);
	CheckCLError(err);

	cl::Buffer clInputBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(int) * dataSize, NULL, &err);
	queue.enqueueWriteBuffer(clInputBuffer, true, 0, sizeof(int) * dataSize, hostBuffer.data());

	// Allocate the output data
	cl::Buffer clResultBuffer = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(int) * dataSize, NULL, &err);

	// Set the kernel parameters
	kernel.setArg(0, clInputBuffer);
	kernel.setArg(1, clResultBuffer);
	kernel.setArg(2, sizeof(int) * (vmax+1), NULL);
	kernel.setArg(3, vmax + 1);

	// Enqueue the kernel
	cl::Event event;
	queue.enqueueNDRangeKernel(kernel,
		cl::NullRange,
		cl::NDRange(dataSize, 1),
		cl::NullRange,
		NULL,
		&event);
	event.wait();
	printTimeStats(event);

	// Copy result back to host
	queue.enqueueReadBuffer(clResultBuffer, true, 0, sizeof(int) * dataSize, hostBuffer.data());

	//Validate
	for (size_t index = 0; index <= vmax; ++index) {
		std::cout << index << " " << hostBuffer[index] << ", should be " << control[index];
		if (hostBuffer[index] == control[index]) {
			std::cout << " correct" << std::endl;
		}
		else {
			std::cout << " wrong" << std::endl;
		}
	}
}

void reduce(cl::Program& program, cl::Context& context, cl::CommandQueue& queue, cl::Device& device) {
	auto dataSize = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
	std::cout << dataSize << std::endl;
	cl_int err = CL_SUCCESS;

	const float vmax = 256.0;
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<float> distr(0.0, vmax);
	std::vector<float> hostBuffer;
	for (size_t index = 0; index < dataSize; ++index)
	{
		hostBuffer.push_back(distr(gen));
	}

	float control = 0.0;
	Timer::measure([&]() {
		for (int i = 0; i < dataSize; ++i) {
			control += hostBuffer[i];
		}
	}, 1);

	cl::Kernel kernel(program, "reduce_global", &err);
	CheckCLError(err);

	cl::Buffer clInputBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(float) * dataSize, NULL, &err);
	queue.enqueueWriteBuffer(clInputBuffer, true, 0, sizeof(float) * dataSize, hostBuffer.data());
	kernel.setArg(0, clInputBuffer);

	cl::Event event;
	queue.enqueueNDRangeKernel(kernel,
		cl::NullRange,
		cl::NDRange(dataSize, 1),
		cl::NDRange(dataSize, 1),
		NULL,
		&event);
	event.wait();
	printTimeStats(event);

	queue.enqueueReadBuffer(clInputBuffer, true, 0, sizeof(int) * dataSize, hostBuffer.data());

	std::cout << std::setprecision(10) << hostBuffer[0] << ", should be " << control << std::endl; //fload adders seem to have different precision
}

void exscan(cl::Program& program, cl::Context& context, cl::CommandQueue& queue, cl::Device& device)
{
	auto dataSize = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
	cl_int err = CL_SUCCESS;

	const int vmax = 25;
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<int> distr(0, vmax);
	std::vector<int> hostBuffer;
	for (size_t index = 0; index < dataSize; ++index)
	{
		hostBuffer.push_back(distr(gen));
	}

	std::vector<int> control(dataSize, 0);
	Timer::measure([&]() {
		for (int i = 1; i < dataSize; ++i) {
			control[i] = control[i-1] + hostBuffer[i-1];
		}
		}, 1);

	cl::Kernel kernel(program, "exscan_global", &err);
	CheckCLError(err);

	cl::Buffer clInputBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(int) * dataSize, NULL, &err);
	queue.enqueueWriteBuffer(clInputBuffer, true, 0, sizeof(int) * dataSize, hostBuffer.data());

	// Allocate the output data
	cl::Buffer clResultBuffer = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(int) * dataSize, NULL, &err);

	// Set the kernel parameters
	kernel.setArg(0, clInputBuffer);
	kernel.setArg(1, clResultBuffer);

	// Enqueue the kernel
	cl::Event event;
	queue.enqueueNDRangeKernel(kernel,
		cl::NullRange,
		cl::NDRange(dataSize, 1),
		cl::NDRange(dataSize, 1),
		NULL,
		&event);
	event.wait();
	printTimeStats(event);
	// Copy result back to host
	queue.enqueueReadBuffer(clResultBuffer, true, 0, sizeof(int) * dataSize, hostBuffer.data());
	//Validate

	for (size_t index = 0; index < dataSize; ++index) {
		if (hostBuffer[index] != control[index]) {
			std::cout << index << " Wrong!" << std::endl;
			break;
		}
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
	cl::CommandQueue queue(context, devices[0], CL_QUEUE_PROFILING_ENABLE, &err);

	// Create the OpenCL program
	std::string programSource = FileToString("../kernels/programs.cl");
	cl::Program program = cl::Program(context, programSource);
	program.build(devices);
	std::cout << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]) << std::endl;

	histogram_global(program, context, queue);
	histogram_local(program, context, queue);
	reduce(program, context, queue, devices[0]);
	exscan(program, context, queue, devices[0]);

	std::cout << "Finished" << std::endl;
}

int main()
{
	cppapi();
	std::string F;
	std::getline(std::cin, F);
	return 0;
}

