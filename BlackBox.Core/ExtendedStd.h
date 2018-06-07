#pragma once

#include <algorithm>

#include "SegmentedArray.h"

namespace Estd
{
	using namespace std;
	using namespace BBox::Core;

	template <typename C>
	void sort(C& c)
	{
		sort(c.begin(), c.end());
	}

	template<typename C, typename Pred>
	void sort(C& c, Pred p)
	{
		sort(c.begin(), c.end(), p);
	}

	template <typename C>
	void reverse(C& c)
	{
		reverse(c.begin(), c.end());
	}

	template<typename C>
	void copy(const C& in, C& out)
	{
		copy(in.begin(), in.end(), out.begin());
	}

	template<typename T>
	void fill(SegmentedArray<T>& in, const T& val)
	{
		fill(in.Data.begin(), in.Data.end(), val);
		fill(in.Counts.begin(), in.Counts.end(), val);
	}

	template<typename C, typename T>
	void fill(C& in, const T& val)
	{
		fill(in.begin(), in.end(), val);
	}

	template <typename T>
	struct elementCount
	{
		static inline size_t count(const T& s)
		{
			return 1;
		}
	};

	template <typename T>
	struct elementCount<vector<T>>
	{
		static inline size_t count(const vector<T>& v)
		{
			return v.size();
		}
	};

	template <typename T>
	struct elementCount<SegmentedArray<T>>
	{
		static inline size_t count(const SegmentedArray<T>& v)
		{
			return v.Data.size();
		}
	};
}