#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <vector>
#include <functional>
#include <cmath>

#include "Logger.h"
#include "BlackBox.h"
#include "SegmentedArray.h"
#include "Stopwatch.h"
#include "Extensions.h"
#include "ExtendedStd.h"

namespace BBox 
{
	namespace Core
	{
		using namespace std;
		using namespace Estd;

		template <typename T>
		using matrix1 = vector<T>;
		
		template <typename T>
		using matrix2 = vector<matrix1<T>>;
		
		template <typename T>
		using matrix3 = vector<matrix2<T>>;
		
		template <typename T>
		using matrix4 = vector<matrix3<T>>;
		
		template <typename T>
		using segmentedArrayBBoxPtr = shared_ptr<BlackBox<SegmentedArray<T>, SegmentedArray<T>>>;

		class Runner
		{
			Logger _logger;
			Stopwatch _stopwatch;

		public:
			Runner(const Logger& logger) {
				_logger = logger;
			}

			bool showPerformance;
			bool showCheck;
			bool showRatios;

			template <typename T, typename R>
			void runMultipleTests(
				const vector<shared_ptr<BlackBox<vector<T>, vector<R>>>>& blackBoxes,
				const vector<int>& sizes,
				const function<void(vector<T>&)>& feedData,
				const int iterations,
				const bool showPerformance,
				const bool showOk);

			template <typename T, typename R>
			void runSingleTest(
				const vector<shared_ptr<BlackBox<T, R>>>& blackBoxes,
				const T& original,
				R& result,
				R& reference,
				const bool dryTest,
				vector<double>& elapsed,
				vector<double>& counter,
				vector<int>& refOk);

			template <typename T>
			void segmentedArrayTest(
				const vector<segmentedArrayBBoxPtr<T>>& blackBoxes,
				const vector<int>& sizes,
				const int iterations,
				function<void(vector<T>&)> dataRandom,
				const bool presort,
				function<double(void)> partitionRandom, //Random partitionRandom, 
				const vector<int>& segmentSizes,
				matrix4<int>& counts,
				const std::string& postCaption);
		};
	}
}
