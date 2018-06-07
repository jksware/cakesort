#pragma once

#include <vector>
#include <algorithm>
#include <memory>

#include "BlackBox.Core/SegmentedArray.h"
#include "BlackBox.Core/ExtendedStd.h"
#include "BlackBox.Core/Extensions.h"
#include "OpenCLBlackBox.h"
#include "BlellochPlus.h"

namespace BBox {
	namespace OpenCL {
		using namespace BBox::Core;
	

		template <typename T>
		class CakeSort : public OpenCLBlackBox<SegmentedArray<T>, SegmentedArray<T>>
		{
			typedef T myType;
			typedef CakeSort<T> myOwn;
			typedef OpenCLBlackBox<SegmentedArray<T>, SegmentedArray<T>> myBase;

			cl::Kernel _binSearchTidGroupKernel;
			cl::Kernel _bitonicSortLayeredKernel;
			cl::Kernel _bitonicSortIndexLayeredKernel;
			cl::Kernel _bitonicSortKernel;
			cl::Kernel _mergeSortLayeredKernel;
			cl::Kernel _mergeSortKernel;
			cl::Kernel _setGidKernel;
			cl::Kernel _bitonicSortForSorted;
			cl::Kernel _swapKernel;

			cl::Buffer _permutationBuffer;
			cl::Buffer _dataBuffer;
			cl::Buffer _offsetBuffer;
			cl::Buffer _rsCountsBuffer;
			cl::Buffer _sortedStartBuffer;
			cl::Buffer _startBuffer;
			cl::Buffer _subsetBuffer;

			const std::string BinSearchTidGroupKernelName = "binSearchTidGroup";
			const std::string LayeredBitonicSortKernelName = "bitonicSortLayered";
			const std::string BitonicSortForSorted = "bitonicSortForSorted";
			const std::string LayeredBitonicSortIndexKernelName = "bitonicSortIndexLayered";
			const std::string BitonicSortKernelName = "bitonicSort";
			const std::string LayeredMergeSortKernelName = "mergeSortLayered";
			const std::string MergeSortKernelName = "mergeSort";
			const std::string SetGidKernelName = "setGid";
			const std::string SwapKernelName = "swap";

			const std::string CakeSortKernelPath = std::string("Kernels") + SEP + "CakeSortLayered_kernel.cl";

			std::unique_ptr<BlellochPlus<int>> _prefixSum;

			// Reverse sorted counts
			std::vector<int> _rsCounts;

			// Reverse permutations
			std::vector<int> _rPermutation;

			// Sorted start
			std::vector<int> _sStart;
			
			std::vector<int> _Counts;

			// number of segments
			int _segments;

			// next power of two of number of segments
			int _segmentsNpot;

			// useIndex ? _bitonicSortIndexLayeredKernel : _bitonicSortLayeredKernel
			cl::Kernel layeredBitonic;

			//assumeSorted ? _bitonicSortForSorted : layeredBitonic;
			cl::Kernel finalKernel;

			typename myBase::kernelInfo mergeSortKernelInfo;
			typename myBase::kernelInfo finalKernelInfo;
			typename myBase::kernelInfo binSearchKernelInfo;

			size_t finalKernelLocalWorkSize = 0;

			using myBase::_length;

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

			CakeSort() = delete;

			CakeSort(cl::Context context, cl::Device device, cl::CommandQueue queue, bool assumeSorted = false,
				bool hybrid = true, bool useHost = true, bool reuseStartBuffer = false, bool useIndex = false)
					: myBase{ context, device, queue },
					_prefixSum{ new BlellochPlus<int>{ context, device, queue } },
					assumeSorted(assumeSorted), hybrid(hybrid), useHost(useHost),
					reuseStartBuffer(reuseStartBuffer), useIndex(useIndex)
			{
				this->extraDefines = "#define GetKey(x) ((x))";
				this->declareTypes["__DATA__"] = typeid(myType).hash_code();
				this->declareTypes["__KEY__"] = typeid(myType).hash_code();

				_binSearchTidGroupKernel = this->LoadKernel(CakeSortKernelPath, BinSearchTidGroupKernelName);
				_bitonicSortLayeredKernel = this->LoadKernel(CakeSortKernelPath, LayeredBitonicSortKernelName);
				_bitonicSortIndexLayeredKernel = this->LoadKernel(CakeSortKernelPath, LayeredBitonicSortIndexKernelName);
				_bitonicSortKernel = this->LoadKernel(CakeSortKernelPath, BitonicSortKernelName);
				_mergeSortLayeredKernel = this->LoadKernel(CakeSortKernelPath, LayeredMergeSortKernelName);
				_mergeSortKernel = this->LoadKernel(CakeSortKernelPath, MergeSortKernelName);
				_setGidKernel = this->LoadKernel(CakeSortKernelPath, SetGidKernelName);
				_bitonicSortForSorted = this->LoadKernel(CakeSortKernelPath, BitonicSortForSorted);
				_swapKernel = this->LoadKernel(CakeSortKernelPath, SwapKernelName);
			}				

			std::string Name() override { return "Cake Sort OpenCL"; }
			std::string ShortName() override { return "oclCakeSort"; }

			bool Write(const SegmentedArray<T>& input) override
			{
				for (auto& p : input.Counts)
					if (!IsPowerOfTwo(p))
						throw std::length_error("Segmented array's segments must be a power of two.");

				std::vector<int> startBuffer;
				//this->_data = input;

				_length = input.Data.size();
				this->_segments = input.Counts.size();
				this->_segmentsNpot = NextGreatestPowerOfTwo(this->_segments);

				if (assumeSorted)
					reuseStartBuffer = true;

				_dataBuffer = cl::Buffer(this->context, CL_MEM_READ_WRITE, _length * sizeof(T), nullptr);

				if (reuseStartBuffer) {
					startBuffer.resize(_segmentsNpot);
					startBuffer[0] = 0;
					/*
					for (int i = 1; i < startBuffer.size(); ++i) {
						if (i <= this->segments)
							startBuffer[i] = startBuffer[i - 1] + input.Counts[i - 1];
						else
							startBuffer[i] = startBuffer[i - 1];
					}
					*/
					for (int i = 1; i <= this->_segments; ++i)
						startBuffer[i] = startBuffer[i - 1] + input.Counts[i - 1];

					for (int i = this->_segments + 1; i < startBuffer.size(); ++i)
						startBuffer[i] = startBuffer[i - 1];
				}

				_startBuffer = cl::Buffer(this->context, CL_MEM_READ_WRITE, 
					(reuseStartBuffer ? startBuffer.size() : _segmentsNpot) * sizeof(int), nullptr);

				_sortedStartBuffer = cl::Buffer(this->context, CL_MEM_READ_WRITE,
					(assumeSorted ? startBuffer.size() : _segmentsNpot) * sizeof(int), nullptr);

				_rsCounts.resize(_segmentsNpot);
				_sStart.resize(_segmentsNpot);
				_rPermutation.resize(_segments);
				_prefixSum->Preprocess(_segmentsNpot);

				_rsCountsBuffer = cl::Buffer(this->context, CL_MEM_READ_WRITE, _rsCounts.size() * sizeof(int), nullptr);

				_offsetBuffer = cl::Buffer(this->context, CL_MEM_READ_WRITE, (_length >> 1) * sizeof(int), nullptr);
				_subsetBuffer = cl::Buffer(this->context, CL_MEM_READ_WRITE, (_length >> 1) * sizeof(int), nullptr);
				_permutationBuffer = cl::Buffer(this->context, CL_MEM_READ_WRITE, (useHost ? _segments : _segmentsNpot) * sizeof(int), nullptr);

				_Counts.resize(input.Counts.size());
				Estd::copy(input.Counts, _Counts);

				//_sStart = startBuffer;
				Estd::copy(input.Counts, this->_rsCounts);
				cl::copy(this->queue, begin(input.Data), end(input.Data), _dataBuffer);
				cl::copy(this->queue, begin(_rsCounts), end(_rsCounts), _rsCountsBuffer);

				if (reuseStartBuffer)
					cl::copy(this->queue, begin(startBuffer), end(startBuffer), _startBuffer);

				if (assumeSorted)
					cl::copy(this->queue, begin(startBuffer), end(startBuffer), _sortedStartBuffer);

				if (reuseStartBuffer) {
					_sStart.resize(_segmentsNpot);
					Estd::copy(startBuffer, _sStart);
				}

				layeredBitonic = useIndex ? _bitonicSortIndexLayeredKernel : _bitonicSortLayeredKernel;
				finalKernel = assumeSorted ? _bitonicSortForSorted : layeredBitonic;
				finalKernelInfo = myBase::getKernelInfo(finalKernel, this->_device);
				finalKernelLocalWorkSize = finalKernelInfo.kernelWorkGroupSize;

				mergeSortKernelInfo = myBase::getKernelInfo(_mergeSortKernel, this->_device);

				binSearchKernelInfo = myBase::getKernelInfo(_binSearchTidGroupKernel, this->_device);

				return true;
			}

			bool Read(SegmentedArray<T>& result) override
			{
				result.Data.resize(_length);
				cl::copy(this->queue, _dataBuffer, begin(result.Data), end(result.Data));

				result.Counts.resize(_Counts.size());
				Estd::copy(_Counts, result.Counts);

				return true;
			}
			
			void BinarySearch(int inputLength, int outputLength, const cl::Buffer& input, const cl::Buffer& output)
			{
				size_t globalWorkSize = outputLength;
				size_t localWorkSize = binSearchKernelInfo.kernelWorkGroupSize;
				//size_t localWorkSize = 8;
				getAligned(globalWorkSize, localWorkSize);
				//globalWorkSize /= localWorkSize;

				int npotLength = NextGreatestPowerOfTwo(inputLength);

				auto binSearchTidGroupKernelWrapper = cl::KernelFunctor<cl::Buffer, cl::Buffer, /*cl::LocalSpaceArg,*/ int, int, int>{ _binSearchTidGroupKernel };

				binSearchTidGroupKernelWrapper(cl::EnqueueArgs{ this->queue, cl::NDRange { globalWorkSize }, cl::NDRange{ localWorkSize } },
					input, output, /*cl::Local(sizeof(int) * localWorkSize),*/ inputLength, npotLength, outputLength);
			}


			bool Process() override
			{
				size_t globalWorkSize = 0;

				/* With ORIGINAL counts */

				if (!reuseStartBuffer) {
					/* write count buffer with unsorted (original) counts in order to make start buffer */
					cl::copy(this->queue, begin(_rsCounts), end(_rsCounts), _rsCountsBuffer);
				}

				if (!assumeSorted) {
					if (!reuseStartBuffer) {
						/* time complexity: S(k) */
						_prefixSum->length(_segmentsNpot);
						_prefixSum->inputBuffer(_rsCountsBuffer);
						_prefixSum->outputBuffer(_startBuffer);
						_prefixSum->Process();
					}
					/* With SORTED counts */

					/* time complexity: k log k */

					if (useHost) {
						std::vector<std::pair<int, int>> _pairs(_segments);

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

						//reverse(&_rsCounts[0], &_rsCounts[segments]);
						//reverse(_rPermutation);

						/* Writting indexes */
						cl::copy(this->queue, begin(_rPermutation), end(_rPermutation), _permutationBuffer);

						/* Writing countBuffer with sorted counts */
						cl::copy(this->queue, begin(_rsCounts), end(_rsCounts), _rsCountsBuffer);
					}
					else {
						int npotPartitionStagesNumber = 0;
						for (int temp = _segmentsNpot; temp > 1; temp >>= 1)
							++npotPartitionStagesNumber;

						globalWorkSize = _segmentsNpot;

						_setGidKernel.setArg(0, _permutationBuffer);

						// todo change finalKernelLocalWorkSize for the right value
						this->queue.enqueueNDRangeKernel(_setGidKernel,
							cl::NDRange{}, cl::NDRange{ globalWorkSize }, cl::NDRange{ finalKernelLocalWorkSize });

						_bitonicSortKernel.setArg(0, _rsCountsBuffer);
						_bitonicSortKernel.setArg(1, _permutationBuffer);

						int beginPartitionsStageNumber = 0;

						if (hybrid) {
							_mergeSortKernel.setArg(0, _rsCountsBuffer);
							_mergeSortKernel.setArg(1, _permutationBuffer);
							_mergeSortKernel.setArg(2, cl::Local(sizeof(T) * finalKernelLocalWorkSize));
							_mergeSortKernel.setArg(3, cl::Local(sizeof(int) * finalKernelLocalWorkSize));

							this->queue.enqueueNDRangeKernel(_mergeSortKernel,
								cl::NDRange{}, cl::NDRange{ globalWorkSize }, cl::NDRange{ finalKernelLocalWorkSize });

							for (int temp = finalKernelLocalWorkSize; temp > 1; temp >>= 1)
								++beginPartitionsStageNumber;
						}

						globalWorkSize = _segmentsNpot >> 1;

						for (int stage = beginPartitionsStageNumber; stage < npotPartitionStagesNumber; stage++) {
							_bitonicSortKernel.setArg(2, stage);
							_bitonicSortKernel.setArg(3, 0);
							this->queue.enqueueNDRangeKernel(_bitonicSortKernel,
								cl::NDRange{}, cl::NDRange{ globalWorkSize }, cl::NDRange{ finalKernelLocalWorkSize });
						}

						cl::copy(this->queue, _rsCountsBuffer, begin(_rsCounts), end(_rsCounts));			
					}
				}

				//int onePos = Extensions.BinarySearchDecreasing(_rsCounts, 0);
				//int npotOnePos = NextGreatestPowerOfTwo(onePos);

				if (!(assumeSorted & reuseStartBuffer)) {	
					_prefixSum->length(_segmentsNpot);
					_prefixSum->inputBuffer(_rsCountsBuffer);
					_prefixSum->outputBuffer(_sortedStartBuffer);
					/* time complexity: S(k) */
					_prefixSum->Process();

					cl::copy(this->queue, _prefixSum->outputBuffer(), begin(_sStart), end(_sStart));
				}

				int max = _sStart[_segmentsNpot - 1] + _rsCounts[_segmentsNpot - 1];
				if (max < 2)
					throw std::runtime_error("don't know what is happening");

				if (!assumeSorted) {
					/* Mapping each thread id to the Data fragment */
					/* time complexity: n log k */		
					BinarySearch(_segments, max >> 1, _sortedStartBuffer, _subsetBuffer);
				}

				int endStage = 0;
				for (int temp = _rsCounts[0]; temp > 1; temp >>= 1) {
					++endStage;
				}

				if (endStage == 0) {
					throw std::runtime_error("don't know what is happening");
				}

				/* max{ log |c_i| } log k */
				std::vector<int> threadsPerStage(endStage);
				for (int stage = 0; stage < endStage; ++stage) {
					int lowIndex = _rsCounts.rend() - std::upper_bound(_rsCounts.rbegin(), _rsCounts.rend(), 1 << stage);
					//int lowIndex = std::lower_bound(_rsCounts.begin(), _rsCounts.end(), 1 << stage) - _rsCounts.begin();
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
					auto mergeSortLayeredKernelWrapper = cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer,
							int, cl::LocalSpaceArg, int, int, int>{ _mergeSortLayeredKernel};

					for (size_t count = 0; ; ) {
						int topIndex = _rsCounts.rend() - std::upper_bound(_rsCounts.rbegin(), _rsCounts.rend(), count + 1);
						if (topIndex == 0)
							break;

						count = _rsCounts[topIndex - 1];
						int lowIndex = _rsCounts.rend() - std::upper_bound(_rsCounts.rbegin(), _rsCounts.rend(), count + 1);

						if (count == 0)
							continue;

						size_t ws = (topIndex - lowIndex) * count;

						if (ws <= 1)
							continue;

						size_t globalMergeWS = ws;
						size_t localMergeWS = min(count, mergeSortKernelInfo.kernelWorkGroupSize);
						getAligned(globalMergeWS, localMergeWS);

						mergeSortLayeredKernelWrapper(
							cl::EnqueueArgs{ this->queue, cl::NDRange{ globalMergeWS }, cl::NDRange{ localMergeWS } },
							_dataBuffer, assumeSorted ? _sortedStartBuffer : _startBuffer, _permutationBuffer, assumeSorted,
							cl::Local(sizeof(T) * localMergeWS), lowIndex, count, ws);
					}

					for (int temp = finalKernelLocalWorkSize; temp > 1; temp >>= 1)
						++beginStage;
				}

				cl::KernelFunctor<cl::Buffer, int, int, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, int, int> 
					bitonicSortLayeredKernelWrapper{ _bitonicSortLayeredKernel };
				
				/* time complexity: \sum |c_i| (log |c_i|) (log |c_i| - 1)  */	
				if (!useIndex) {
					if (!assumeSorted) {
						for (int stage = beginStage; stage < endStage; ++stage) {
							globalWorkSize = threadsPerStage[stage];
							getAligned(globalWorkSize, finalKernelLocalWorkSize);
							for (int pass = 0; pass < stage + 1; ++pass) {
								bitonicSortLayeredKernelWrapper(
									cl::EnqueueArgs{ this->queue, cl::NDRange{ globalWorkSize }, cl::NDRange{ finalKernelLocalWorkSize } },
									_dataBuffer, stage, pass, _subsetBuffer, assumeSorted ? _sortedStartBuffer : _startBuffer,
									_sortedStartBuffer, _permutationBuffer, assumeSorted ? 1 : 0, threadsPerStage[stage]);
							}
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
					auto swapKernelWrapper = cl::KernelFunctor<cl::Buffer, int, int, int>{ _swapKernel };

					for (int width = 1 << endStage;; width >>= 1) {
						int topIndex = _rsCounts.rend() - std::upper_bound(_rsCounts.rbegin(), _rsCounts.rend(), width);
						if (topIndex > threadsPerStage[0])
							break;

						width = _rsCounts[topIndex];
						int lowIndex = _rsCounts.rend() - std::upper_bound(_rsCounts.rbegin(), _rsCounts.rend(), width - 1);

						if (lowIndex - topIndex == 1)
							continue;

						if (width <= 1 << beginStage)
							break;

						int from = _sStart[topIndex];
						int to = _sStart[lowIndex];
						size_t globalWs = (lowIndex - topIndex) * width >> 2;

						swapKernelWrapper(cl::EnqueueArgs{ this->queue, cl::NDRange { globalWs}, cl::NDRange{ finalKernelLocalWorkSize } },
							_dataBuffer, from, width, to);
					}
				}

				return true;
			}
		};
	}
}
