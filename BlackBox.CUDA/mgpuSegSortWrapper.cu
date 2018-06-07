#include <cuda.h>
#include "cuda_runtime.h"

// #include <device_functions.h>
// #include <device_launch_parameters.h>

#include "mgpuSegSortWrapper.cuh"

#include <cassert>

using namespace std;
using namespace BBox::Core;
using namespace BBox::CUDA;

using namespace mgpu;

bool BBox::CUDA::mgpuSegSortWrapper::Process()
{	
	segmented_sort(data.data(), segmentActiveLength, segs.data(), segmentCount, less_t<int>(), context);

	return true;
}

bool BBox::CUDA::mgpuSegSortWrapper::Write(const SegmentedArray<int>& input)
{
	vector<int> offset(input.Counts.size());

	_segments = input.Counts;

	if (input.Counts.size() > 0) {
		offset[0] = input.Counts[0];
	}
	for (int i = 1; i < offset.size(); ++i) {
		offset[i] = offset[i - 1] + input.Counts[i];
	}
	segmentActiveLength = offset[offset.size() - 1];

	segs = to_mem(offset, context);
	data = to_mem(input.Data, context);
	_length = input.Data.size();
	segmentCount = input.Counts.size();

	return true;
}

bool BBox::CUDA::mgpuSegSortWrapper::Read(SegmentedArray<int>& result)
{
	result.Data.resize(_length);
	result.Counts.resize(segmentCount);
	assert(_length == result.Data.size());

	result.Data = from_mem(data);
	// result.Counts = from_mem(segs);
	result.Counts = _segments;

	return true;
}
