#include "Runner.h"

template <typename T, typename R>
void BBox::Core::Runner::runMultipleTests(
	const vector<shared_ptr<BlackBox<vector<T>, vector<R>>>>& blackBoxes,
	const vector<int>& sizes,
	const function<void(vector<T>&)>& feedData, 
	const int iterations, 
	const bool showPerformance, 
	const bool showOk)
{
	if (blackBoxes.size() * sizes.size() == 0)
		return;

	vector<double> elapsed(blackBoxes.size());
	vector<double> perf(blackBoxes.size());
	vector<double> refOk(blackBoxes.size());

	_logger.LogLine("Computing for sizes up to %s for %s algorithms and 1+%s iterations each.",
		sizes[sizes.size() - 1], blackBoxes.size(), iterations);
	_logger.Log("Size");

	for (auto& blackBox : blackBoxes) {
		_logger.Log(blackBox->ShortName());
	}

	_logger.LogLine();
	_logger.Line();

	for (int i = 0; i < sizes.size(); ++i) {
		const int& size = sizes[i];

		vector<T> original(size);
		vector<R> result(size);
		vector<R> reference(size);

		vector<double> partialElapsed(blackBoxes.size());
		vector<double> partialPerf(blackBoxes.size());
		vector<int> partialRefOk(blackBoxes.size());

		for (int j = i == 0 ? -1 : 0; j < iterations; ++j) {
			feedData(original);

			runSingleTest<vector<T>, vector<R>>(
				blackBoxes, original, result, reference, j == -1,
				partialElapsed, partialPerf, partialRefOk);
		}
		_logger.Log(size);

		for (int j = 0; j < blackBoxes.size(); ++j) {
			double partialAvgElapsed = partialElapsed[j] / iterations;
			double partialAvgPerf = partialPerf[j] / iterations;
			double partialAvgRefOk = static_cast<double>(partialRefOk[j]) / iterations;

			elapsed[j] += partialAvgElapsed;
			perf[j] += partialAvgPerf;
			refOk[j] += partialAvgRefOk;

			_logger.Log(showPerformance ? partialAvgPerf : partialAvgElapsed);

			if (showOk) {
				_logger.Log(partialAvgRefOk);
			}
			else {
				_logger.Log();
		}
		}

		_logger.LogLine();
	}

	_logger.Line();
	_logger.Log("Average");
	for (int i = 0; i < blackBoxes.size(); ++i) {
		double averageElapsed = elapsed[i] / sizes.size();
		double averagePerf = perf[i] / sizes.size();
		double averageRefOk = refOk[i] / sizes.size();

		_logger.Log(showPerformance ? averagePerf : averageElapsed);

		if (showOk)
			_logger.Log(averageRefOk);
		else
			_logger.Log();
	}
	_logger.LogLine();

	_logger.Log("Ratio");
	for (int j = 0; j < blackBoxes.size(); ++j) {
		for (int i = 0; i < blackBoxes.size(); ++i) {
			double ratio = showPerformance ? perf[i] / perf[j] : elapsed[i] / elapsed[j];
			
			if (j != 0 && i == 0)
				_logger.Log();
			_logger.Log(ratio);
			_logger.Log();
		}
		_logger.LogLine();
	}

	_logger.LogLine();
}

template <typename T, typename R>
void BBox::Core::Runner::runSingleTest(
	const vector<shared_ptr<BlackBox<T, R>>>& blackBoxes,
	const T& original, 
	R& result,
	R& reference,
	const bool dryTest,
	vector<double>& elapsed,
	vector<double>& perf,
	vector<int>& refOk)
{
	for (int i = 0; i < blackBoxes.size(); ++i) {
		auto& blackbox = blackBoxes[i];
		_stopwatch.Reset();
#ifndef _DEBUG
		try {
#endif			
			blackbox->Write(original);
			blackbox->Finish();
			_stopwatch.Start();
			blackbox->Process();
			blackbox->Finish();
			_stopwatch.Stop();
			clear(result);
			blackbox->Read(i == 0 ? reference : result);
			blackbox->Finish();
#ifndef _DEBUG
		}
		catch (const std::exception& exc) {
			_stopwatch.Stop();
			cerr << exc.what() << endl;
			_logger.Log("X");
		}
#endif
		_stopwatch.Stop();

		if (dryTest)
			continue;

		bool isResultOk = (i == 0) || compare(reference, result);

		int size = elementCount<T>::count(original);
		perf[i] += size / _stopwatch.ElapsedSeconds<double>();

		elapsed[i] += _stopwatch.ElapsedSeconds<double>();
		refOk[i] += isResultOk ? 1 : 0;
	}
}

template<typename T>
void BBox::Core::Runner::segmentedArrayTest(
	const vector<segmentedArrayBBoxPtr<T>>& blackBoxes,
	const vector<int>& dataSizes,
	const int iterationCount,
	function<void(vector<T>&)> dataRandom,
	const bool presort,
	function<double(void)> partitionRandom,
	const vector<int>& segmentCounts,
	matrix4<int>& counts,
	const std::string& postCaption)
{
	counts = matrix4<int>(dataSizes.size(), matrix3<int>(segmentCounts.size(),
		matrix2<int>(iterationCount, matrix1<int>(32))));
	if (blackBoxes.size() * dataSizes.size() == 0)
		return;

	_logger.LogLine("Computing for sizes up to %s, segments up to %s, "
		"for %s algorithms and 1+%s iterations each.",
		dataSizes[dataSizes.size() - 1],
		segmentCounts[segmentCounts.size() - 1], blackBoxes.size(),
		iterationCount);

	vector<double> elapsed(blackBoxes.size());
	vector<double> perf(blackBoxes.size());
	vector<double> refOk(blackBoxes.size());

	_logger.BeginTable();
	_logger.BeginTabular();
	_logger.Log("Size");
	_logger.Log("Segm.");
	for (auto& blackBox : blackBoxes)
		_logger.ColumnHeader(blackBox->ShortName());
	_logger.LogLine();
	_logger.Line();

	SegmentedArray<T> original;
	SegmentedArray<T> result;
	SegmentedArray<T> reference;

	vector<double> partialElapsed(blackBoxes.size());
	vector<double> partialPerf(blackBoxes.size());
	vector<int> partialRefOk(blackBoxes.size());

	int iterationRunsPerBlackbox = 0;

	for (int iDataSize = 0; iDataSize < dataSizes.size(); ++iDataSize) {
		int size = dataSizes[iDataSize];

		original.Data.resize(size);
		result.Data.resize(size);
		reference.Data.resize(size);

		for (int iSegCount = 0; iSegCount < segmentCounts.size(); ++iSegCount) {
			if (segmentCounts[iSegCount] >= size)
				continue;

			_logger.Log(size);
			_logger.Log(segmentCounts[iSegCount]);

			Estd::fill(partialElapsed, 0);
			Estd::fill(partialPerf, 0);
			Estd::fill(partialRefOk, 0);

			for (int i = iDataSize == 0 ? -1 : 0; i < iterationCount; ++i) {
				dataRandom(original.Data);

				Estd::fill(result, 0);
				original.Counts = RandomPartitionCover(partitionRandom, segmentCounts[iSegCount], size);

				int upTo = 0;
				for (int j = 0; j < original.Counts.size(); ++j) {
					int npot = NextGreatestPowerOfTwo(original.Counts[j]) >> 1;
					if (upTo + npot < size) {
						upTo += npot;
						original.Counts[j] = npot;
						if (i != -1) {
							int log2_index = static_cast<int>(log2(original.Counts[j] + 1));
							++counts[iDataSize][iSegCount][i][log2_index];
						}
					}
					else {
						original.Counts[j] = 0;
					}
				}

				if (presort) {
					sort(original.Counts);
					reverse(original.Counts);
				}

				runSingleTest<SegmentedArray<T>, SegmentedArray<T>>(blackBoxes, original, result, reference, i == -1,
					partialElapsed, partialPerf, partialRefOk);
			}

			++iterationRunsPerBlackbox;
			for (int algo = 0; algo < blackBoxes.size(); algo++) {
				double partialAvgElapsed = partialElapsed[algo] / iterationCount;
				double partialAvgPerf = partialPerf[algo] / iterationCount;
				double partialAvgRefOk = static_cast<double>(partialRefOk[algo]) / iterationCount;

				elapsed[algo] += partialAvgElapsed;
				perf[algo] += partialAvgPerf;
				refOk[algo] += partialAvgRefOk;

				_logger.Log(showPerformance ? partialAvgPerf : partialAvgElapsed);
				if (showCheck)
					_logger.Log(partialAvgRefOk);
			}

			_logger.LogLine();
			_logger.Flush();
		}

		_logger.LogLine();
		_logger.Flush();
	}

	_logger.Line();

	_logger.Log("Avg");
	_logger.Log();
	for (int i = 0; i < blackBoxes.size(); ++i) {
		double averageElapsed = elapsed[i] / iterationRunsPerBlackbox;
		double averagePerf = perf[i] / iterationRunsPerBlackbox;
		double averageRefOk = refOk[i] / iterationRunsPerBlackbox;

		_logger.Log(showPerformance ? averagePerf : averageElapsed);
		if (showCheck)
			_logger.Log(averageRefOk);
	}
	_logger.LogLine();

	if (showRatios) {
		_logger.Log("Ratio");

		for (int j = 0; j < blackBoxes.size(); ++j) {
			_logger.Log();
			for (int i = 0; i < blackBoxes.size(); ++i) {
				double perfRatio = showPerformance ? perf[i] / perf[j] : elapsed[i] / elapsed[j];
				if (j != 0 && i == 0)
					_logger.Log();

				_logger.Log(perfRatio);
				if (showCheck)
					_logger.Log();
			}
			_logger.LogLine();
		}
	}

	_logger.EndTabular();

	long nanosecPerTick = (1000L * 1000L * 1000L) / _stopwatch.Frequency<long>();

	_logger.EndTable();
	_logger.Flush();
}
