#pragma once

#include "BlackBox.h"
#include "ExtendedStd.h"

#include <vector>
#include <cassert>

namespace BBox
{
	namespace Core
	{
		template <typename T>
		class CPUReferencePrefixSum : public BlackBox<std::vector<T>, std::vector<T>>
		{
			std::vector<T> _data;
			std::vector<T> _output;

		public:
			CPUReferencePrefixSum(bool inclusive = true) : inclusive(inclusive)
			{
			}

			std::string Name() override { return "CPU Prefix Sum Reference"; }
			std::string ShortName() override { return "CPU Scan Ref."; }

			bool Read(std::vector<T>& result) override
			{
				assert(_output.size() == result.size());
				Estd::copy(_output, result);
				return true;
			}

			bool Process() override
			{
				if (inclusive) {
					_output[0] = _data[0];
					for (unsigned int i = 1; i < _data.size(); ++i)
						_output[i] = _output[i - 1] + _data[i];
				}
				else {
					_output[0] = 0;
					for (unsigned int i = 1; i < _data.size(); ++i)
						_output[i] = _output[i - 1] + _data[i - 1];
				}

				return true;
			}

			bool Finish() override
			{
				return true;
			}
			
			bool Write(const std::vector<T>& input) override
			{
				_data.resize(input.size());
				_output.resize(input.size());
				Estd::copy(input, _data);
				return true;
			}			

			bool inclusive;
		};
	}
}