#include "CakeSort.cuh"

__global__ void BBox::CUDA::binSearchTidGroupKernel(const int *start, int *subset, const int length,
	const int paddedLength, const int globalWS)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id > globalWS)
		return;

	//int blockSize = get_local_size(0);

	int lower = 0;
	for (int size = paddedLength; size > 0; size >>= 1) {
		int upper = lower + size - 1;
		bool smaller = upper < length && (start[upper] >> 1) <= id;
		lower += smaller ? size : 0;
		lower = MIN(lower, length);
	}

	subset[id] = lower - 1;

	/*
	int subset = lower - 1;
	int nextStart = start[lower] >> 1;

	for (int i = 0; i < blockSize; ++i) {
	if (id * blockSize + i >= globalWS)
	return;

	if (id * blockSize + i == nextStart) {
	subset++;
	nextStart = start[++lower] >> 1;
	}

	subsetNumber[id * blockSize + i] = subset;
	}
	*/
}

__global__ void BBox::CUDA::setGidKernel(int* data)
{
	int gid = blockDim.x * blockIdx.x + threadIdx.x;
	data[gid] = gid;
}
