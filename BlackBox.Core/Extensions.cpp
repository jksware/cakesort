#include "Extensions.h"

#include <math.h>
#include <vector>
#include <functional>
#include <map>
#include <algorithm>

using namespace std;
using namespace BBox::Core;

bool BBox::Core::compareSums(const vector<float>& refData, const vector<float>& data, const float epsilon)
{
	int length = refData.size();
	float error = 0.0f;
	float ref = 0.0f;
	for (int i = 0; i < length; ++i) {
		float diff = refData[i] - data[i];
		error += diff * diff;
		ref += refData[i] * refData[i];
	}
	float normRef = ::sqrtf(static_cast<float>(ref));
	if (::fabs(static_cast<float>(ref)) < 1e-7f) {
		return false;
	}
	float normError = ::sqrtf(static_cast<float>(error));
	error = normError / normRef;
	return error < epsilon;
}

//template <typename T, typename Compare=less<T>>
template <typename T, typename Compare>
void BBox::Core::SwapIfFirstIsGreater(T& i, T& j)
{
	if (i > j)
		swap(i, j);
}

vector<int> BBox::Core::RandomPartitionCover(function<double(void)> rnd, int partitions, int setCardinality)
{
	double sum = 0;
	vector<double> r(partitions);

	for (int i = 0; i < partitions; i++)
		sum += r[i] = ::fabs(rnd());

	double unit = setCardinality / sum;

	double eps = 0;

#ifdef _DEBUG
	double checkN = 0;
	int intCheckN = 0;
#endif

	vector<int> partitionCounts(partitions);

	for (int i = 0; i < partitions; i++) {
		double s = r[i] * unit;
		double epsStep = s - static_cast<int>(s);
		eps += epsStep;
		int rounded = static_cast<int>(s) + static_cast<int>(eps);
		eps -= static_cast<int>(eps);
		partitionCounts[i] = rounded;
#ifdef _DEBUG
		checkN += s;
		intCheckN += rounded;
#endif
	}

	return partitionCounts;
}

vector<int> BBox::Core::RandomPartitionCoverPowerOfTwo(function<double(void)> rnd, int partitions, int setCardinality)
{
	double normalizedSum = 0;
	vector<double> normalizedCounts(partitions);

	for (int i = 0; i < partitions; ++i)
		normalizedSum += normalizedCounts[i] = rnd();
	double unit = setCardinality / normalizedSum;
	double cumulativeError = 0;
	vector<int> partitionCounts(partitions);

#if _DEBUG
	double sum = 0;
	unsigned int intSum = 0;
#endif

	for (int i = 0; i < partitions; i++) {
		double count = normalizedCounts[i] * unit;
		int rounded = NextGreatestPowerOfTwo(static_cast<unsigned int>(count)) >> 1;
		cumulativeError += count - rounded;
		if (cumulativeError >= rounded) {
			cumulativeError -= rounded;
			rounded <<= 1;
		}
		partitionCounts[i] = rounded;
#if _DEBUG
		sum += count;
		intSum += static_cast<unsigned int>(rounded);
#endif
	}

	return partitionCounts;
}

// todo: these two are no longer needed, since the STL provides a better one (namely lower_bound, upper_bound, with begin..end or rbegin..rend)
int BBox::Core::BinarySearch(vector<int> array, int value)
{
	int low = 0;
	int high = array.size() - 1;
	while (low <= high) {
		int midst = (high + low) >> 1;
		if (array[midst] >= value)
			high = midst - 1;
		else
			low = midst + 1;
	}
	return low;
}

int BBox::Core::BinarySearchDecreasing(vector<int> array, int value)
{
	int low = 0;
	int high = array.size() - 1;
	while (low <= high) {
		int midst = (high + low) >> 1;
		if (array[midst] <= value)
			high = midst - 1;
		else
			low = midst + 1;
	}
	return low;
}
