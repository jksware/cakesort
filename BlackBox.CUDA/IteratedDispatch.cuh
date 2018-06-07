#pragma once

#include <algorithm>
#include <cassert>
#include <string>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_functions.h>
#include <device_launch_parameters.h>

#include "BlackBox.Core/SegmentedArray.h"
#include "BlackBox.Core/ExtendedStd.h"
#include "CudaBlackBox.h"


namespace BBox
{
	namespace CUDA
	{
		template <typename T>
		__global__ void bitonicSort(T *data, const int globalThreads, const int offset, const int stage, const int pass)
		{
			int localId = blockDim.x * blockIdx.x + threadIdx.x;
			if (localId >= globalThreads)
				return;

			data += offset;
			int pairDistance = 1 << (stage - pass);

			//uint leftId = (localId % pairDistance) + ((localId / pairDistance) * blockWidth);
			int leftId = (localId & (pairDistance - 1)) | ((localId & ~(pairDistance - 1)) << 1);
			int rightId = leftId + pairDistance;

		#if DEBUG
			printf("threadIdx=%d\tstage=%d\tpass=%d\tlocalId=%d\toffset=%d\tleftId=%d\trightId=%d\n", 
				threadIdx.x, stage, pass, localId, offset, leftId, rightId);

			++data[leftId];
			++data[rightId];
			__threadfence();
			//barrier(CLK_LOCAL_MEM_FENCE);
		#else

			T leftKey = data[leftId];
			T rightKey = data[rightId];

			T greaterKey;
			T lesserKey;
			if (leftKey > rightKey) {
				greaterKey = leftKey;
				lesserKey = rightKey;
			}
			else {
				greaterKey = rightKey;
				lesserKey = leftKey;
			}
			
			// sameDirectionBlockWidth = 1 << stage;
			// (localId / sameDirectionBlockWidth) % 2 == 1
			int direction = -((localId >> stage) & 1);
			data[leftId] = (greaterKey & direction) | (lesserKey & ~direction);
			data[rightId] = (lesserKey & direction) | (greaterKey & ~direction);
		#endif
		}

		using namespace BBox::Core;

		template <typename T>
		class IteratedDispatch : public CudaBlackBox<SegmentedArray<T>, SegmentedArray<T>>
		{
			//typedef T myType;
			//typedef CudaBlackBox<SegmentedArray<T>, std::vector<T>> myBase;

			T *_buffer;

			std::vector<int> segmentLengths;

		public:
			std::string Name() override { return "Iterated Parallel Sort CUDA"; }
			std::string ShortName() override { return "cuIterDispatch"; }

			bool Process() override
			{
				for (int i = 0, start = 0; i < segmentLengths.size(); ++i) {
					assert(start + segmentLengths[i] < this->_length);

					int stagesNumber = 0;
					for (int temp = segmentLengths[i]; temp > 1; temp >>= 1)
						++stagesNumber;

					int globalThreads = segmentLengths[i] >> 1;
					if (globalThreads != 0) {
						int localWorkSize = min(this->BlockSize, globalThreads);

						for (int stage = 0; stage < stagesNumber; ++stage) {
							for (int pass = 0; pass < stage + 1; ++pass) {
								bitonicSort<T><<<static_cast<int>(ceil(static_cast<float>(globalThreads) / localWorkSize)), localWorkSize>>> (_buffer, globalThreads, start, stage, pass);
							}
							//bitonicSort<<<localWorkSize, ceil(globalThreads / localWorkSize)>>> (_buffer, globalThreads, start, stage);
							//bitonicSort <<<1, globalThreads >>> (_buffer, globalThreads, start, stage);
						}
					}
					start += segmentLengths[i];
				}

				return true;
			}			

			bool Write(const SegmentedArray<T>& input) override
			{
				cudaError_t err;
				this->_length = input.Data.size();
				err = cudaFree(_buffer);
				if (err != cudaSuccess)
					throw err;
				_buffer = 0;
				err = cudaMalloc(&_buffer, this->_length * sizeof(T));
				if (err != cudaSuccess)
					throw err;
				segmentLengths.resize(input.Counts.size());
				Estd::copy(input.Counts, segmentLengths);

			#if DEBUG
				vector<int> tmp;
				tmp.assign(input.Data.size(), 0);
				cudaMemcpy(_buffer, &tmp.front(), _length * sizeof(T), cudaMemcpyHostToDevice);
			#else
				cudaMemcpy(_buffer, &input.Data.front(), this->_length * sizeof(T), cudaMemcpyHostToDevice);
			#endif

				return true;
			}			

			bool Read(SegmentedArray<T>& result) override
			{
				result.Data.resize(this->_length);
				result.Counts.resize(segmentLengths.size());
				assert(this->_length == result.Data.size());
				cudaMemcpy(&result.Data.front(), _buffer, this->_length * sizeof(T), cudaMemcpyDeviceToHost);
				Estd::copy(segmentLengths, result.Counts);	

			#if DEBUG
				for (int i = 0, offset = 0; i < segmentLengths.size(); offset += segmentLengths[i++]) {
					int logS = static_cast<int>(std::log2(segmentLengths[i]));
					int binomLogS = logS * (logS + 1) / 2;
					for (int j = 0; j < segmentLengths[i]; ++j) {
						if (result[offset + j] != binomLogS) {
							cout << endl << "=== here ===" << endl;
						}
					}		
				}
			#endif

				return true;
			}
			
		};
	}
}
