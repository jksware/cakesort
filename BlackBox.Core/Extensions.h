#pragma once

#include "SegmentedArray.h"

#include <string>
#include <vector>
#include <map>
#include <functional>

namespace BBox {
	namespace Core {

		template <typename T, typename Compare = std::less<T>> void SwapIfFirstIsGreater(T& i, T& j);
		std::vector<int> RandomPartitionCover(std::function<double(void)> rnd, int partitions, int setCardinality);
		std::vector<int> RandomPartitionCoverPowerOfTwo(std::function<double(void)> rnd, int partitions, int setCardinality);
		int BinarySearch(std::vector<int> array, int value);
		int BinarySearchDecreasing(std::vector<int> array, int value);

		inline bool IsPowerOfTwo(int n) 
		{
			return (n & (n - 1)) == 0;
		}

		// computes and returns the next greatest power of two from the argument value
		inline int NextGreatestPowerOfTwo(int n)
		{
			n--;
			n |= n >> 1;
			n |= n >> 2;
			n |= n >> 4;
			n |= n >> 8;
			n |= n >> 16;
			n++;
			return n;
		}

		inline void getAligned(size_t& global, size_t local)
		{
			if (/*local < global &&*/ global % local != 0)
				global = (global / local + 1) * local;
		}

		std::string MakeGenericCode(const std::map<std::string, std::string>& declareTypes, const std::string& extraDefines, std::string source);

		bool compareSums(const std::vector<float>& refData, const std::vector<float>& data, const float epsilon = 1e-6f);

		template <typename T>
		void CopySegmentedArray(const SegmentedArray<T>& x, SegmentedArray<T>& y)
		{
			y.Data.resize(x.Data.size());
			copy(x.Data.begin(), x.Data.end(), y.Data.begin());
			y.Counts.resize(x.Counts.size());
			copy(x.Counts.begin(), x.Counts.end(), y.Counts.begin());
			y.Index.resize(x.Index.size());
			copy(x.Index.begin(), x.Index.end(), y.Index.begin());
		}

		template <typename T>
		void ReleaseSegmentedArray(SegmentedArray<T>& y)
		{
			y.Data.empty();
		}

		template <typename T>
		void clear(SegmentedArray<T>& x)
		{
			x.Counts.resize(0);
			x.Index.resize(0);
			x.Data.resize(0);
		}

		template <typename T>
		void clear(std::vector<T>& x) 
		{
			x.resize(0);
		}

		template <typename T>
		bool compare(const SegmentedArray<T>& x, const SegmentedArray<T>& y)
		{
			return x == y;
		}

		template <typename T>
		bool compare(const std::vector<T>& x, const std::vector<T>& y)
		{
			return x == y;
		}
	}
}
