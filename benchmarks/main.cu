#include <algorithm>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <device_launch_parameters.h>

extern "C" __global__ void saxpy(int n, float a, float* x, float* y) {
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) {
		y[i] += a*x[i];
	}
}

extern "C" __global__ void saxpy2(int n, int a, float* x, float* y) {
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) {
		float input = x[i];
		float tmp = 0;

		#pragma unroll 1
		for(int cnt = 0; cnt < a; ++cnt) {
			tmp += input;
		}

		y[i] += tmp;
	}
}

__constant__ bool dummyFlag = false;

template <int Granularity>
__global__ void saxpy3(int n, int a, float* y) {
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) {
		float input = 1;
		float tmp = 0;

		for(int cnt = 0; cnt < a; ++cnt) {
			#pragma unroll
			for(int k = 0; k < Granularity; ++k) {
				tmp += input;
			}
		}

		if( dummyFlag ) {
			y[i] += tmp;
		}
	}
}

template <int Granularity>
__global__ void saxpy3_clock_cycles(int n, int a, float* x, long long int* y) {
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) {
		long long int start = clock64();
		float input = 1;
		float tmp = 0;

		for(int cnt=0; cnt < a; ++cnt) {
			#pragma unroll
			for(int k = 0; k < Granularity; ++k) {
				tmp += input;
			}
		}

		y[i] = clock64() - start;

		if( dummyFlag ) {
			x[i] += tmp;
		}
	}
}

__global__ void saxpy4(int n, float a, float* x, float* y)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i > 0 && i < n-1) {
		float t1 = x[i];
		float t2 = x[i-1];
		float t3 = x[i+1];

		float tmp = t1 + t2 + t3;

		y[i] += a*tmp;
	}
}

__global__ void empty_kernel(){};

__global__ void empty_kernel2(int a, int b, float* c, float* d, int e, int f, float* g, float* h){};

#include <nvToolsExt.h>
class SimpleTimer {
public:
	explicit SimpleTimer(std::string name = "") :
		name(name) {
		cudaEventCreate(&start);
		cudaEventCreate(&end);
		nvtxEventAttributes_t nvtxEvent = createNvtxEvent();
		nvtxRangePushEx(&nvtxEvent);
		cudaEventRecord(start);
	}

	~SimpleTimer() {
		cudaEventRecord(end);
		cudaEventSynchronize(end);
		nvtxRangePop();
		float time_ms;
		cudaEventElapsedTime(&time_ms, start, end);
		std::cout << std::fixed << std::setprecision(6) << name << " " << time_ms;
	}

private:
	std::string name;
	cudaEvent_t start;
	cudaEvent_t end;

	nvtxEventAttributes_t createNvtxEvent() {
		const uint32_t colors[] = { 0x0000ff00, 0x000000ff, 0x00ffff00, 0x00ff00ff, 
									0x0000ffff, 0x00ff0000, 0x00ffffff };
		const int num_colors = sizeof(colors)/sizeof(uint32_t);
		static int colorCounter = 0;

		nvtxEventAttributes_t eventAttrib = {0};
		eventAttrib.version = NVTX_VERSION;
		eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
		eventAttrib.colorType = NVTX_COLOR_ARGB;
		eventAttrib.color = colors[colorCounter++ % num_colors];
		eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
		eventAttrib.message.ascii = name.c_str();

		return eventAttrib;
	}
};


void arithmeticInstructionsBenchmark(int arithmeticIntensity) {
	int N = 10 * 1024 * 1024;
	int STEPS = 1000;

	{
		SimpleTimer timer;
		for (int step = 0; step < STEPS; ++step) 
			saxpy3<1><<<(N+255)/256, 256>>>(N, arithmeticIntensity, 0);
	}
	{
		SimpleTimer timer;
		for (int step = 0; step < STEPS; ++step) 
			saxpy3<2><<<(N+255)/256, 256>>>(N, arithmeticIntensity, 0);
	}
	{
		SimpleTimer timer;
		for (int step = 0; step < STEPS; ++step) 
			saxpy3<4><<<(N+255)/256, 256>>>(N, arithmeticIntensity, 0);
	}
	{
		SimpleTimer timer;
		for (int step = 0; step < STEPS; ++step) 
			saxpy3<8><<<(N+255)/256, 256>>>(N, arithmeticIntensity, 0);
	}
	{
		SimpleTimer timer;
		for (int step = 0; step < STEPS; ++step) 
			saxpy3<16><<<(N+255)/256, 256>>>(N, arithmeticIntensity, 0);
	}
	{
		SimpleTimer timer;
		for (int step = 0; step < STEPS; ++step) 
			saxpy3<32><<<(N+255)/256, 256>>>(N, arithmeticIntensity, 0);
	}
}

void printClockCycleStats(const std::vector<long long int>& y) {
//	auto minmax = std::minmax_element(y.begin(), y.end());
//	std::cout << *minmax.first << " " << *minmax.second << " ";
	auto average = std::accumulate(y.begin(), y.end(), 0ll) / y.size();
	std::cout << average << " ";
}

void arithmeticInstructionsBenchmark_clock_cycles(int arithmeticIntensity) {
	int N = 10 * 1024 * 1024;
	std::vector<long long int> y(N);
	long long int *d_y;
	cudaMalloc(&d_y, N*sizeof(long long int));

	saxpy3_clock_cycles<1><<<(N+255)/256, 256>>>(N, arithmeticIntensity, 0, d_y);
	cudaMemcpy(y.data(), d_y, N*sizeof(long long int), cudaMemcpyDeviceToHost);
	printClockCycleStats(y);

	saxpy3_clock_cycles<2><<<(N+255)/256, 256>>>(N, arithmeticIntensity, 0, d_y);
	cudaMemcpy(y.data(), d_y, N*sizeof(long long int), cudaMemcpyDeviceToHost);
	printClockCycleStats(y);

	saxpy3_clock_cycles<4><<<(N+255)/256, 256>>>(N, arithmeticIntensity, 0, d_y);
	cudaMemcpy(y.data(), d_y, N*sizeof(long long int), cudaMemcpyDeviceToHost);
	printClockCycleStats(y);

	saxpy3_clock_cycles<8><<<(N+255)/256, 256>>>(N, arithmeticIntensity, 0, d_y);
	cudaMemcpy(y.data(), d_y, N*sizeof(long long int), cudaMemcpyDeviceToHost);
	printClockCycleStats(y);

	saxpy3_clock_cycles<16><<<(N+255)/256, 256>>>(N, arithmeticIntensity, 0, d_y);
	cudaMemcpy(y.data(), d_y, N*sizeof(long long int), cudaMemcpyDeviceToHost);
	printClockCycleStats(y);

	saxpy3_clock_cycles<32><<<(N+255)/256, 256>>>(N, arithmeticIntensity, 0, d_y);
	cudaMemcpy(y.data(), d_y, N*sizeof(long long int), cudaMemcpyDeviceToHost);
	printClockCycleStats(y);

	cudaFree(d_y);
}

void kernelLaunchBenchmark(int gridSize, int blockSize) {
	int STEPS = 100;
	{
		SimpleTimer timer;
		for (int step = 0; step < STEPS; ++step) {
			empty_kernel<<<gridSize,blockSize>>>();
		}
	}

	{
		SimpleTimer timer;
		for (int step = 0; step < STEPS; ++step) {
			empty_kernel<<<gridSize,blockSize>>>();
			cudaDeviceSynchronize();
		}
	}
}

void memcpyBenchmark(int N) {
	int STEPS = 10;
	std::vector<float> x(N);
	float* d_x;
	cudaMalloc(&d_x, N*sizeof(float));

	{
		SimpleTimer timer;
		for (int step = 0; step < STEPS; ++step) cudaMemcpy(d_x, x.data(), N*sizeof(float), cudaMemcpyHostToDevice);
	}

	{
		SimpleTimer timer;
		for (int step = 0; step < STEPS; ++step) cudaMemcpy(x.data(), d_x, N*sizeof(float), cudaMemcpyDeviceToHost);
	}

	cudaFree(d_x);
}

void simpleCUDAApplication(int arithmeticIntensity, int blockSize, int sharedMemAllocSize = 0, int N = 500000000) {
	std::vector<float> x(N);
	std::vector<float> y(N);
	float *d_x, *d_y;
	cudaMalloc(&d_x, N*sizeof(float));
	cudaMalloc(&d_y, N*sizeof(float));

	for (int i = 0; i < N; i++) {
		x[i] = 1.0f;
		y[i] = 2.0f;
	}

	// https://devblogs.nvidia.com/parallelforall/cuda-pro-tip-occupancy-api-simplifies-launch-configuration/
	int occupancyBlocksPerSM;
	cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancyBlocksPerSM, saxpy2, blockSize, sharedMemAllocSize);

//	int device;
//	cudaDeviceProp p;
//	cudaGetDevice(&device);
//	cudaGetDeviceProperties(&p, device);
//	float maxWarpsPerSM = p.maxThreadsPerMultiProcessor / p.warpSize;
//	float occupancy = std::max(occupancyBlocksPerSM * blockSize / p.warpSize, 1) / maxWarpsPerSM;
//	std::cout << std::fixed << std::setprecision(3) << occupancy << " ";

	{
		SimpleTimer timer("Entire app");
		{
			SimpleTimer timer("memcpy HtD");
			cudaMemcpy(d_x, x.data(), N*sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_y, y.data(), N*sizeof(float), cudaMemcpyHostToDevice);
		}

		{
			SimpleTimer timer("kernel");
			saxpy2<<<(N+blockSize-1)/blockSize, blockSize, sharedMemAllocSize>>>(N, arithmeticIntensity, d_x, d_y);
		}

		{
			SimpleTimer timer("memcpy DtH");
			cudaMemcpy(y.data(), d_y, N*sizeof(float), cudaMemcpyDeviceToHost);
		}
	}

	cudaFree(d_x);
	cudaFree(d_y);
}

int main(int argc, char** argv)
{
	memcpyBenchmark(std::atoi(argv[1]));
//	arithmeticInstructionsBenchmark(std::atoi(argv[1]));
//	arithmeticInstructionsBenchmark_clock_cycles(std::atoi(argv[1]));
//	kernelLaunchBenchmark(std::atoi(argv[1]), std::atoi(argv[2]));
//	simpleCUDAApplication(std::atoi(argv[1]), std::atoi(argv[2]), std::atoi(argv[3]), std::atoi(argv[4]));
	cudaDeviceReset();
}
