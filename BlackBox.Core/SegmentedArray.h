#pragma once

#include <vector>
#include <algorithm>

namespace BBox {
	namespace Core {

		template <typename T>
		struct SegmentedArray
		{
			std::vector<T> Data;
			std::vector<int> Index;
			std::vector<int> Counts;

			friend inline bool operator== (const SegmentedArray& a, const SegmentedArray& b) {
				if (a.Data.size() != b.Data.size() || a.Counts != b.Counts) {
					return false;
				}
				int total = 0;
				for (auto& c: a.Counts) {
					total += c;
				}
				int mismatch = std::mismatch(a.Data.begin(), a.Data.end(), b.Data.begin()).first - a.Data.begin();
				return mismatch >= total;					
			}

			friend inline bool operator!= (const SegmentedArray& a, const SegmentedArray& b) {
				return !(a == b);
			}
		};
	}
}
