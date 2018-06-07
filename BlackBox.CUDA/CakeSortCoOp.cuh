/******************************************************************************
*
*	File:			CakeSortGlobalBarrier.cuh
*	Author:			Juan Carlos Pujol Mainegra
*	Description:	Layered bitonic sort, CakeSort.
*
*	Copyright 2015-2018 Juan Carlos Pujol Mainegra
*
******************************************************************************/

#pragma once

#include <memory>
#include <vector>
#include <algorithm>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_functions.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>

#include "BlackBox.Core/SegmentedArray.h"
#include "BlackBox.Core/ExtendedStd.h"
#include "BlackBox.Core/Extensions.h"
#include "CudaBlackBox.h"
#include "BlellochPlus.cuh"

#define GetKey(x)			(x)
#define MAX(a,b)            (((a) > (b)) ? (a) : (b))
#define MIN(a,b)            (((a) < (b)) ? (a) : (b))


typedef unsigned int uint;

namespace BBox
{
	namespace CUDA
	{
		using namespace cooperative_groups; 
		namespace cg = cooperative_groups;

		__global__ void binSearchTidGroupKernelCoOp(const int *start, int *subset, const int length,
			const int paddedLength, const int globalWS);

		__global__ void setGidKernelCoOp(int* data);		
		
		template <typename T>
		__global__ void bitonicSortKernelCoOp(T *key, int *index, const int stage, int pass)
		{
			int localId = blockDim.x * blockIdx.x + threadIdx.x;
			int pairDistance = 1 << (stage - pass);

			//int leftId = (localId % pairDistance) + ((localId / pairDistance) * blockWidth);
			int leftId = (localId & (pairDistance - 1)) | ((localId & ~(pairDistance - 1)) << 1);
			int rightId = leftId + pairDistance;

			int leftIndex = index[leftId];
			int rightIndex = index[rightId];
			T leftKey = key[leftId];
			T rightKey = key[rightId];

			T greaterKey, greaterIndex;
			T lesserKey, lesserIndex;
			if (leftKey > rightKey) {
				greaterKey = leftKey;
				greaterIndex = leftIndex;
				lesserKey = rightKey;
				lesserIndex = rightIndex;
			}
			else {
				greaterKey = rightKey;
				greaterIndex = rightIndex;
				lesserKey = leftKey;
				lesserIndex = leftIndex;
			}

			// sameDirectionBlockWidth = 1 << stage;
			// (localId / sameDirectionBlockWidth) % 2 == 1
			int direction = -((localId >> stage) & 1);
			index[leftId] = (greaterIndex & ~direction) | (lesserIndex & direction);
			key[leftId] = (greaterKey & ~direction) | (lesserKey & direction);
			index[rightId] = (lesserIndex & ~direction) | (greaterIndex & direction);
			key[rightId] = (lesserKey & ~direction) | (greaterKey & direction);
		}

		template <typename Data>
		__global__ void bitonicSortLayeredKernelCoOp(Data *data, const int stage, const int *subset, const int *start,
			const int *sortedStart, const int *permutation, const int assumeSorted, const int globalWS)
		{
			grid_group grid = this_grid();

			int tid = blockDim.x * blockIdx.x + threadIdx.x;

			if (tid >= globalWS) {
				return;
			}

			// subsetIdx index (in [0..k-1] where K is the partition cardinality) is the mapping of each thread to the corresponding subset subsetNumber
			int subsetIdx = subset[tid];

			// localId is the offset of the current thread from each start of the current thread subset
			int localId = tid - (sortedStart[subsetIdx] >> 1);

			// permutedIdx (in [0..k-1]) is a permutation of the subset index where the partition is sorted by cardinality of each subset
			int permutedIdx = assumeSorted == 0 ? permutation[subsetIdx] : subsetIdx;

			// start index (in [0..N-1] where N is the fragment set cardinality) is the index of the half of the start of each subset on the global memory
			int startIdx = start[permutedIdx];

			for (int pass = 0; pass < stage + 1; ++pass) {
				int pairDistance = 1 << (stage - pass);

				//int leftId = startIdx + (localId % pairDistance) + ((localId / pairDistance) * blockWidth);
				int leftId = startIdx + ((localId & (pairDistance - 1)) | ((localId & ~(pairDistance - 1)) << 1));
				int rightId = leftId + pairDistance;

		#if DEBUG
				if (stage == 0 && pass == 0)
					printf("tid=%d\tlocalid=%d\tstartIdx=%d\tleftId=%d\n", tid, localId, startIdx, leftId);
		#endif

				Data leftData = data[leftId];
				Data rightData = data[rightId];

				Data greater;
				Data lesser;
				if (GetKey(leftData) > GetKey(rightData)) {
					greater = leftData;
					lesser = rightData;
				}
				else {
					greater = rightData;
					lesser = leftData;
				}

				// sameDirectionBlockWidth = 1 << stage;
				// (localId / sameDirectionBlockWidth) % 2 == 1
				int direction = -((localId >> stage) & 1);
				data[leftId] = (greater & direction) | (lesser & ~direction);
				data[rightId] = (lesser & direction) | (greater & ~direction);

		#if DEBUG
					if (stage == 0 && pass == 0)
						printf("tid=%d\tdirection=%d\tgreater=%d\tlesser=%d\tdata[leftId]=%d\tdata[rightId]=%d\n",
							tid, direction, greater, lesser, data[leftId], data[rightId]);
		#endif

				grid.sync();
			}
		}

		template <typename Data, typename Key>
		__global__ void bitonicSortForSortedCoOp(Data *data, const int stage, const int pass, const int globalWS)
		{
			int localId = blockDim.x * blockIdx.x + threadIdx.x;
			if (localId >= globalWS)
				return;

			int pairDistance = 1 << (stage - pass);

			//int leftId = (localId % pairDistance) + ((localId / pairDistance) * blockWidth);
			int leftId = ((localId & (pairDistance - 1)) | ((localId & ~(pairDistance - 1)) << 1));
			int rightId = leftId + pairDistance;

		#if DEBUG
			if (stage == 0 && pass == 0)
				printf("tid=%d\tlocalid=%d\tstartIdx=%d\tleftId=%d\n", tid, localId, startIdx, leftId);
		#endif

			Data leftData = data[leftId];
			Data rightData = data[rightId];
			Key leftKey = GetKey(leftData);
			Key rightKey = GetKey(rightData);

			Data greater;
			Data lesser;
			if (leftKey > rightKey)
			{
				greater = leftData;
				lesser = rightData;
			}
			else
			{
				greater = rightData;
				lesser = leftData;
			}

			// sameDirectionBlockWidth = 1 << stage;
			// (localId / sameDirectionBlockWidth) % 2 == 1
			int direction = (localId >> stage) & 1;

			if (direction) {
				data[leftId] = greater;
				data[rightId] = lesser;
			}
			else {
				data[leftId] = lesser;
				data[rightId] = greater;
			}
		}

		template <typename Data>
		__global__ void swapKernelCoOp(Data *data, const int from, int width, const int to)
		{
			int gid = blockDim.x * blockIdx.x + threadIdx.x;
			data += from + width;

			int leftId = (gid % (width / 2)) + (gid / (width / 2) * width * 2);
			int rightId = ((leftId / width) + 1) * width - (leftId % width) - 1;

			if ((from + width + leftId >= to) || (from + width + rightId >= to))
				return;

			Data leftData = data[leftId];
			Data rightData = data[rightId];

			data[leftId] = rightData;
			data[rightId] = leftData;

			/*
			data[leftId] = 1;
			data[rightId] = 2;
			*/
		}

		template <typename Data, typename Key>
		__global__ void bitonicSortIndexLayeredKernelCoOp(const Data *data, const int stage, const int pass,
				const int *subset, const int *start, const int *sortedStart, const int *permutation, 
				const int assumeSorted, const int globalWS, int *index)
		{
			int tid = blockDim.x * blockIdx.x + threadIdx.x;
			if (tid >= globalWS)
				return;

			// subsetIdx index (in [0..k-1] where K is the partition cardinality) is the mapping of each thread to the corresponding subset subsetNumber
			int subsetIdx = subset[tid];

			// localId is the offset of the current thread from each start of the current thread subset
			int localId = tid - (sortedStart[subsetIdx] >> 1);

			// permutedIdx (in [0..k-1]) is a permutation of the subset index where the partition is sorted by cardinality of each subset
			int permutedIdx = assumeSorted == 0 ? permutation[subsetIdx] : subsetIdx;

			// start index (in [0..N-1] where N is the fragment set cardinality) is the index of the half of the start of each subset on the global memory
			int startIdx = start[permutedIdx];

			int pairDistance = 1 << (stage - pass);

			//int leftId = startIdx + (localId % pairDistance) + ((localId / pairDistance) * blockWidth);
			int leftId = startIdx + ((localId & (pairDistance - 1)) | ((localId & ~(pairDistance - 1)) << 1));
			int rightId = leftId + pairDistance;

		#if DEBUG
			if (stage == 0 && pass == 0)
				printf("tid=%d\tlocalid=%d\tstartIdx=%d\tleftId=%d\n", tid, localId, startIdx, leftId);
		#endif

			int leftIndex = index[leftId];
			int rightIndex = index[rightId];
			Key leftKey = GetKey(data[leftIndex]);
			Key rightKey = GetKey(data[rightIndex]);

			int greater;
			int lesser;
			if (leftKey > rightKey)
			{
				greater = leftIndex;
				lesser = rightIndex;
			}
			else
			{
				greater = rightIndex;
				lesser = leftIndex;
			}

			// sameDirectionBlockWidth = 1 << stage;
			// (localId / sameDirectionBlockWidth) % 2 == 1
			int direction = (localId >> stage) & 1;

			direction = -direction;
			index[leftId] = (greater & direction) | (lesser & ~direction);
			index[rightId] = (lesser & direction) | (greater & ~direction);
		}

		template <typename Data>
		__global__ void mergeSortKernelCoOp(Data *data, int *index)
		{
			extern __shared__ Data auxData[];
			extern __shared__ int auxIndex[];

			int lid = threadIdx.x;
			int wg = blockDim.x; // workgroup size = block size, power of 2
			int bid = blockIdx.x;

			// Move IN, OUT to block start
			int offset = blockIdx.x * wg;
			data += offset; index += offset;

			// Load block in AUX[WG]
			auxData[lid] = data[lid];
			auxIndex[lid] = index[lid];

			__syncthreads(); // make sure AUX is entirely up to date

			// Now we will merge sub-sequences of length 1,2,...,WG/2
			for (int length = 1; length < wg; length <<= 1)
			{
				Data iKey = auxData[lid];
				int iIndex = auxIndex[lid];
				int ii = lid & (length - 1);  // index in our sequence in 0..length-1
				int sibling = (lid - ii) ^ length; // beginning of the sibling sequence
				int lower = 0;
				for (int size = length; size > 0; size >>= 1) // increment for dichotomic search
				{
					int upper = sibling + lower + size - 1;
					Data upperKey = auxData[upper];
					bool smaller = (upperKey < iKey) || (upperKey == iKey && upper < lid);
					lower += (smaller) ? size : 0;
					lower = MIN(lower, length);
				}
				int bits = (length << 1) - 1; // mask for destination
				int dest = ((ii + lower) & bits) | (lid & ~bits); // destination index in merged sequence
				__syncthreads();
				auxData[dest] = iKey;
				auxIndex[dest] = iIndex;
				__syncthreads();
			}

			int direction = -((bid & 1) == 1);
			int outpos = (lid & direction) | ((wg - lid - 1) & ~direction);
			// Write output
			data[lid] = auxData[outpos];
			index[lid] = auxIndex[outpos];
		}

		template <typename Data, typename Key>
		__global__ void mergeSortLayeredKernelCoOp(Data *data, const int *start, const int *permutation, const int assumeSorted,
			const int lowGid, const int count, const int globalWS)
		{
			extern __shared__ Data aux[];

			int lid = threadIdx.x;
			int wg = blockIdx.x;
			int gid = blockIdx.x;

			int idx = lowGid + gid * wg / count;
			int permutedIdx = assumeSorted ? idx : permutation[idx];

			int offset = start[permutedIdx] + wg * (gid % (count / wg));
			data += offset;

		#if DEBUG	
			if (lid < 10)
				printf("lid=%d\tassumeSorted=%d\tlowGid=%d\tgid=%d\tpermutedIdx=%d\tstart[permutedIdx]=%d\tcount=%d\toffset=%d\n",
					lid, assumeSorted, lowGid, gid, permutedIdx, start[permutedIdx], count, offset);
		#endif

			aux[lid] = data[lid];

			__syncthreads(); // make sure AUX is entirely up to date

			// Now we will merge sub-sequences of length 1,2,...,WG/2
			for (int length = 1; length < wg; length <<= 1)
			{
				Data iData = aux[lid];
				Key iKey = GetKey(iData);
				int ii = lid & (length - 1);  // index in our sequence in 0..length-1
				int sibling = (lid - ii) ^ length; // beginning of the sibling sequence
				int lower = 0;
				for (int size = length; size > 0; size >>= 1) // increment for dichotomic search
				{
					int upper = sibling + lower + size - 1;
					Key upperKey = GetKey(aux[upper]);
					bool smaller = (upperKey < iKey) || (upperKey == iKey && upper < lid);
					lower += (smaller) ? size : 0;
					lower = MIN(lower, length);
				}
				int bits = (length << 1) - 1; // mask for destination
				int dest = ((ii + lower) & bits) | (lid & ~bits); // destination index in merged sequence
				__syncthreads();
				aux[dest] = iData;
				__syncthreads();
			}

			if (blockDim.x * blockIdx.x + threadIdx.x >= globalWS)
				return;

			int outpos = (count <= wg || (gid % (count / wg)) & 1) ? lid : wg - lid - 1;
			data[lid] = aux[outpos];
		}

		using namespace BBox::Core;

		template <typename T>
		class CakeSortCoOp : public CudaBlackBox<SegmentedArray<T>, SegmentedArray<T>>
		{
			typedef int Data;
			typedef int Key;

			//typedef T myType;
			//typedef CakeSort<T> myOwn;
			//typedef OpenCLBlackBox<SegmentedArray<T>, std::vector<T>> myBase;

			thrust::device_vector<T> _dataBuffer;
			thrust::device_vector<int> _permutationBuffer;
			thrust::device_vector<int> _offsetBuffer;
			thrust::device_vector<int> _rsCountsBuffer;
			thrust::device_vector<int> _sortedStartBuffer;
			thrust::device_vector<int> _startBuffer;
			thrust::device_vector<int> _subsetBuffer;

			// Reverse sorted counts
			std::vector<int> _rsCounts;

			// Reverse permutations
			std::vector<int> _rPermutation;

			// Sorted start
			std::vector<int> _sStart;

			std::vector<int> _Counts;

			// number of segments
			unsigned int _segments;

			// next power of two of number of segments
			unsigned int _segmentsNpot;

			typedef void(*cakeFunc) (const Data*, const int, const int, const int*,
				const int *, const int *, const int *, const int, const int, int*);

			// useIndex ? _bitonicSortIndexLayeredKernel : _bitonicSortLayeredKernel
			cakeFunc layeredBitonic;

			//assumeSorted ? _bitonicSortForSorted : layeredBitonic;
			cakeFunc finalKernel;
			
		public:
			// Minimum cardinality of the larger list to consider sorting by an hybrid algorithm (if Hybrid is given).
			const int hybridMinimumSize = 64;

			// Whether or not to assume the list are already (as input) sorted by cardinalities.
			bool assumeSorted = false;

			// Suggest whether or not to use an hybrid device (it is only used if larger list bigger than hybridMinimumSize)
			bool hybrid = true;

			// Sets whether the list are sorted by cardinalities at the OpenCL device or at the host.
			bool useHost = false;

			bool reuseStartBuffer = false;

			bool useIndex = false;

			CakeSortCoOp() 
				: assumeSorted(false), hybrid(true), useHost(true), reuseStartBuffer(false), useIndex(false)
			{	
			}

			CakeSortCoOp(bool assumeSorted, bool hybrid, bool useHost, bool reuseStartBuffer, bool useIndex)
				: assumeSorted(assumeSorted), hybrid(hybrid), useHost(useHost), reuseStartBuffer(reuseStartBuffer), useIndex(useIndex)
			{
			}

			std::string Name() override { return "Cake Sort Co-Op CUDA"; }
			std::string ShortName() override { return "cuCakeSortCoOp"; }

			bool Write(const SegmentedArray<T>& input) override
			{
				for (auto& p : input.Counts) {
					if (!IsPowerOfTwo(p)) {
						throw std::length_error("Segmented array's segments must be a power of two.");
					}
				}

				_Counts.resize(input.Counts.size());
				Estd::copy(input.Counts, _Counts);

				std::vector<int> startBuffer;
				//_data = input;

				this->_length = input.Data.size();
				_segments = input.Counts.size();
				_segmentsNpot = NextGreatestPowerOfTwo(_segments);

				if (assumeSorted)
					reuseStartBuffer = true;

				_dataBuffer.resize(this->_length);
				//_indexBuffer.resize(input.Index.size());

				if (reuseStartBuffer) {
					startBuffer.resize(_segmentsNpot);
					startBuffer[0] = 0;
					/*
					for (int i = 1; i < startBuffer.size(); ++i) {
					if (i <= _segments)
					startBuffer[i] = startBuffer[i - 1] + input.Counts[i - 1];
					else
					startBuffer[i] = startBuffer[i - 1];
					}
					*/
					for (int i = 1; i <= _segments; ++i)
						startBuffer[i] = startBuffer[i - 1] + input.Counts[i - 1];

					for (int i = _segments + 1; i < startBuffer.size(); ++i)
						startBuffer[i] = startBuffer[i - 1];
				}

				_startBuffer.resize(reuseStartBuffer ? startBuffer.size() : _segmentsNpot);
				_sortedStartBuffer.resize(assumeSorted ? startBuffer.size() : _segmentsNpot);

				_rsCounts.resize(_segmentsNpot);
				_sStart.resize(_segmentsNpot);
				_rPermutation.resize(_segments);

				_rsCountsBuffer.resize(_rsCounts.size());

				_offsetBuffer.resize(this->_length >> 1);
				_subsetBuffer.resize(this->_length >> 1);
				_permutationBuffer.resize(useHost ? _segments : _segmentsNpot);

				//_sStart = startBuffer;
				//Estd::copy(input.Counts, _rsCounts);
				thrust::copy(input.Counts.begin(), input.Counts.end(), _rsCounts.begin());

				thrust::copy(input.Data.begin(), input.Data.end(), _dataBuffer.begin());
				//if (input.Index.size() != 0) {
				//	thrust::copy(input.Index.begin(), input.Index.end(), _indexBuffer.begin());
				//}
				thrust::copy(_rsCounts.begin(), _rsCounts.end(), _rsCountsBuffer.begin());

				if (reuseStartBuffer) {
					thrust::copy(startBuffer.begin(), startBuffer.end(), _startBuffer.begin());
				}
				if (assumeSorted) {
					thrust::copy(startBuffer.begin(), startBuffer.end(), _sortedStartBuffer.begin());
				}
				if (reuseStartBuffer) {
					_sStart.resize(_segmentsNpot);
					//Estd::copy(startBuffer, _sStart);
					thrust::copy(startBuffer.begin(), startBuffer.end(), _sStart.begin());
				}

				//layeredBitonic = useIndex ? bitonicSortIndexLayeredKernel : bitonicSortLayeredKernel;
				//finalKernel = assumeSorted ? bitonicSortForSorted : layeredBitonic;

				return true;
			}

			bool Read(SegmentedArray<T>& result) override
			{
				result.Data.resize(this->_length);
				thrust::copy(_dataBuffer.begin(), _dataBuffer.end(), result.Data.begin());
				result.Counts.resize(_Counts.size());
				Estd::copy(_Counts, result.Counts);
				return true;
			}

			void BinarySearch(int inputLength, int outputLength, thrust::device_vector<int> const& input, 
				thrust::device_vector<int> & output)
			{
				int globalWorkSize = outputLength;
				int localWorkSize = this->BlockSize;
				int npotLength = NextGreatestPowerOfTwo(inputLength);

				binSearchTidGroupKernelCoOp<<<static_cast<int>(ceil(static_cast<float>(globalWorkSize)/localWorkSize)), localWorkSize, sizeof(T) * localWorkSize>>>(
					thrust::raw_pointer_cast(input.data()), 
					thrust::raw_pointer_cast(output.data()),
					inputLength, npotLength, outputLength);
			}

			bool Process() override
			{
				int globalWorkSize = 0;

				/* With ORIGINAL counts */

				if (!reuseStartBuffer) {
					/* write count buffer with unsorted (original) counts in order to make start buffer */
					thrust::copy(_rsCounts.begin(), _rsCounts.end(), _rsCountsBuffer.begin());
				}

				if (!assumeSorted) {
					if (!reuseStartBuffer) {
						/* time complexity: S(k) */
						thrust::exclusive_scan(_rsCountsBuffer.begin(), _rsCountsBuffer.end(), _startBuffer.begin());
					}
					/* With SORTED counts */

					/* time complexity: k log k */

					if (useHost) {
						std::vector<std::pair<int, int>> _pairs{ static_cast<uint>(_segments)};

						for (int i = 0; i < _pairs.size(); ++i) {
							_pairs[i].first = _rsCounts[i];
							_pairs[i].second = i;
						}
						auto pairComparer = [](std::pair<int, int> x, std::pair<int, int> y) { return x.first > y.first; };

						sort(begin(_pairs), end(_pairs), pairComparer);

						for (int i = 0; i < _pairs.size(); ++i) {
							_rsCounts[i] = _pairs[i].first;
							_rPermutation[i] = _pairs[i].second;
						}

						/* Writting indexes */
						thrust::copy(_rPermutation.begin(), _rPermutation.end(), _permutationBuffer.begin());

						/* Writing countBuffer with sorted counts */
						thrust::copy(_rsCounts.begin(), _rsCounts.end(), _rsCountsBuffer.begin());
					}
					else {
						int npotPartitionStagesNumber = 0;
						for (int temp = _segmentsNpot; temp > 1; temp >>= 1)
							++npotPartitionStagesNumber;

						globalWorkSize = _segmentsNpot;
						setGidKernelCoOp<<<static_cast<int>(ceil(static_cast<float>(globalWorkSize)/this->BlockSize)), this->BlockSize>>>(
							thrust::raw_pointer_cast(_permutationBuffer.data()));

						int beginPartitionsStageNumber = 0;
						if (hybrid) {
							mergeSortKernelCoOp<<<static_cast<int>(ceil(static_cast<float>(globalWorkSize)/this->BlockSize)),
								this->BlockSize, 2 * sizeof(int) * this->BlockSize>>>(
								thrust::raw_pointer_cast(_rsCountsBuffer.data()),
								thrust::raw_pointer_cast(_permutationBuffer.data()));

							for (int temp = this->BlockSize; temp > 1; temp >>= 1)
								++beginPartitionsStageNumber;
						}

						globalWorkSize = _segmentsNpot >> 1;

						for (int stage = beginPartitionsStageNumber; stage < npotPartitionStagesNumber; stage++) {
							for (int pass = 0; pass < stage + 1; ++pass) {
								bitonicSortKernelCoOp<<<static_cast<int>(ceil(static_cast<float>(globalWorkSize)/this->BlockSize)), this->BlockSize>>>(
									thrust::raw_pointer_cast(_rsCountsBuffer.data()),
									thrust::raw_pointer_cast(_permutationBuffer.data()), stage, pass);
							}
						}
						thrust::copy(_rsCountsBuffer.begin(), _rsCountsBuffer.end(), _rsCounts.begin());
					}
				}

				//int onePos = Extensions.BinarySearchDecreasing(_rsCounts, 0);
				//int npotOnePos = NextGreatestPowerOfTwo(onePos);

				if (!(assumeSorted & reuseStartBuffer)) {
					/* time complexity: S(k) */
					thrust::exclusive_scan(_rsCountsBuffer.begin(), _rsCountsBuffer.end(), _sortedStartBuffer.begin());
					thrust::copy(_sortedStartBuffer.begin(), _sortedStartBuffer.end(), _sStart.begin());
				}

				int MAX = _sStart[_segmentsNpot - 1] + _rsCounts[_segmentsNpot - 1];
				if (MAX < 2) {
					throw std::runtime_error("don't know what is happening");
				}

				if (!assumeSorted) {
					/* Mapping each thread id to the Data fragment */
					/* time complexity: n log k */
					BinarySearch(_segments, MAX >> 1, _sortedStartBuffer, _subsetBuffer);
				}

				int endStage = 0;
				for (int temp = _rsCounts[0]; temp > 1; temp >>= 1)
					++endStage;

				if (endStage == 0) {
					throw std::runtime_error("don't know what is happening");
				}

				/* MAX{ log |c_i| } log k */
				std::vector<int> threadsPerStage(endStage);
				for (int stage = 0; stage < endStage; ++stage) {
					uint lowIndex = _rsCounts.rend() - std::upper_bound(_rsCounts.rbegin(), _rsCounts.rend(), 1 << stage);
					if (lowIndex >= _segmentsNpot) {
						threadsPerStage[stage] = (_sStart[_segmentsNpot - 1] + _rsCounts[_segmentsNpot - 1]) >> 1;
					}
					else {
						threadsPerStage[stage] = _sStart[lowIndex] >> 1;
					}

					if (threadsPerStage[stage] == 0) {
						throw std::runtime_error("don't know what is happening");
					}
				}

				/* Merge sort */

				int beginStage = 0;

				if (hybrid && !useIndex && _rsCounts[0] >= hybridMinimumSize) {
					for (int count = 0; ; ) {
						int topIndex = _rsCounts.rend() - std::upper_bound(_rsCounts.rbegin(), _rsCounts.rend(), count + 1);
						if (topIndex == 0) {
							break;
						}

						count = _rsCounts[topIndex - 1];
						int lowIndex = _rsCounts.rend() - std::upper_bound(_rsCounts.rbegin(), _rsCounts.rend(), count + 1);

						if (count == 0) {
							continue;
						}

						int ws = (topIndex - lowIndex) * count;

						if (ws <= 1) {
							continue;
						}

						int globalMergeWS = ws;
						int localMergeWS = MIN(count, this->BlockSize);

						mergeSortLayeredKernelCoOp<T, T><<<static_cast<int>(ceil(globalMergeWS/localMergeWS)),
							localMergeWS, sizeof(T) * localMergeWS>>>(
							thrust::raw_pointer_cast(_dataBuffer.data()), 
							thrust::raw_pointer_cast(assumeSorted ? _sortedStartBuffer.data() : _startBuffer.data()),
							thrust::raw_pointer_cast(_permutationBuffer.data()), 
							assumeSorted, lowIndex, count, ws);
					}

					for (int temp = this->BlockSize; temp > 1; temp >>= 1)
						++beginStage;
				}

				/* time complexity: \sum |c_i| (log |c_i|) (log |c_i| - 1)  */

				if (!useIndex) {
					if (!assumeSorted) {
						int tc = 1;
						int tf = 0;

						auto dataBufferData = thrust::raw_pointer_cast(_dataBuffer.data());
						auto subsetBufferData = thrust::raw_pointer_cast(_subsetBuffer.data());
						auto assumeSortedBuffer = thrust::raw_pointer_cast(assumeSorted ? _sortedStartBuffer.data() : _startBuffer.data());
						auto sortedStartData = thrust::raw_pointer_cast(_sortedStartBuffer.data());
						auto permutationBufferData = thrust::raw_pointer_cast(_permutationBuffer.data());

						int stage = 0/*, pass = 0*/;

						void *bitSortArgs[8] = {
							&dataBufferData,
							&stage,
							&subsetBufferData,
							&assumeSortedBuffer,
							&sortedStartData,
							&permutationBufferData,
							assumeSorted ? &tc : &tf,
							&threadsPerStage[stage]
						};

						for (stage = beginStage; stage < endStage; ++stage) {
							globalWorkSize = threadsPerStage[stage];
							bitSortArgs[7] = &threadsPerStage[stage];

							// for (pass = 0; pass < stage + 1; ++pass) {
								auto error = cudaLaunchCooperativeKernel(
									(void*)bitonicSortLayeredKernelCoOp<T>,
									dim3{ static_cast<uint>(ceil(static_cast<float>(globalWorkSize) / this->BlockSize)) },
									dim3{ static_cast<uint>(this->BlockSize) },
									bitSortArgs
								);

								//bitonicSortLayeredKernelCoOp<<<static_cast<int>(ceil(static_cast<float>(globalWorkSize) / BlockSize)), BlockSize >>> (
								//	thrust::raw_pointer_cast(_dataBuffer.data()), stage, pass,
								//	thrust::raw_pointer_cast(_subsetBuffer.data()),
								//	thrust::raw_pointer_cast(assumeSorted ? _sortedStartBuffer.data() : _startBuffer.data()),
								//	thrust::raw_pointer_cast(_sortedStartBuffer.data()),
								//	thrust::raw_pointer_cast(_permutationBuffer.data()),
								//	assumeSorted ? 1 : 0, threadsPerStage[stage]);
							// }
						}
					}
					else {
						throw std::runtime_error("_bitonicSortForSorted not implemented");
					}
				}
				else {
					throw std::runtime_error("_bitonicSortIndexLayeredKernel not implemented");
				}

				if (assumeSorted) {
					for (int width = 1 << endStage;; width >>= 1) {
						int topIndex = _rsCounts.rend() - std::upper_bound(_rsCounts.rbegin(), _rsCounts.rend(), width);
						if (topIndex > threadsPerStage[0]) {
							break;
						}

						width = _rsCounts[topIndex];
						int lowIndex = _rsCounts.rend() - std::upper_bound(_rsCounts.rbegin(), _rsCounts.rend(), width - 1);

						if (lowIndex - topIndex == 1) {
							continue;
						}

						if (width <= 1 << beginStage) {
							break;
						}

						int from = _sStart[topIndex];
						int to = _sStart[lowIndex];
						int globalWs = (lowIndex - topIndex) * width >> 2;

						swapKernelCoOp<<<static_cast<int>(ceil(globalWs/this->BlockSize)), this->BlockSize>>>(
							thrust::raw_pointer_cast(_dataBuffer.data()), from, width, to);
					}
				}

				return true;
			}
		};
	}
}