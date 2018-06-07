#pragma once

#include <vector>
#include <cassert>
#include <algorithm>

#include "SegmentedArray.h"
#include "BlackBox.h"
#include "ExtendedStd.h"

namespace BBox {
	namespace Core {

		template <typename T>
		class CPUReferenceSegmentedSort : public BlackBox<SegmentedArray<T>, SegmentedArray<T>>
		{
			SegmentedArray<T> _data;

		public:
			std::string Name() override { return "CPU Segmented Sort Reference "; }
			std::string ShortName() override { return "CPU Sort Ref"; }

			bool Process() override
			{
				for (int i = 0, start = 0; i < _data.Counts.size(); ++i) {
					assert(start + _data.Counts[i] < _data.Data.size());

					//sort(_data.Data, _data.Index, start, _data.Counts[i]);
					std::sort(&_data.Data[start], &_data.Data[start + _data.Counts[i]]);

					start += _data.Counts[i];
				}

				return true;
			}

			bool Read(SegmentedArray<T>& result) override
			{
				result = _data;
				return true;
			}			

			bool Write(const SegmentedArray<T>& input) override
			{
				_data = input;
				return true;
			}

			bool Finish() override
			{
				return true;
			}
		};
	}
}