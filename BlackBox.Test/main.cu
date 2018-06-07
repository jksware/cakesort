#include <random>
#include <algorithm>
#include <boost/program_options.hpp>

#include "BlackBox.Core/BlackBox.h"
#include "BlackBox.Core/AcmeBlackBox.h"
#include "BlackBox.Core/Logger.h"
#include "BlackBox.Core/Runner.h"
#include "BlackBox.Core/Runner.cpp"
#include "BlackBox.Core/Stopwatch.h"
#include "BlackBox.Core/Extensions.h"
#include "BlackBox.Core/CPUReferencePrefixSum.h"
#include "BlackBox.Core/CPUReferenceSegmentedSort.h"

#include "BlackBox.OpenCL/OpenCLBlackBox.h"
#include "BlackBox.OpenCL/BlellochSimple.h"
#include "BlackBox.OpenCL/BlellochPlus.h"
#include "BlackBox.OpenCL/CakeSort.h"
#include "BlackBox.OpenCL/IteratedDispatch.h"
#include "BlackBox.OpenCL/ParallelDispatch.h"

#ifdef __CUDA_BLACKBOX__
#include "BlackBox.CUDA/CudaBlackBox.h"
#include "BlackBox.CUDA/IteratedDispatch.cuh"
#include "BlackBox.CUDA/BlellochPlus.cuh"
#include "BlackBox.CUDA/ParallelDispatch.cuh"
#include "BlackBox.CUDA/CakeSort.cuh"
#include "BlackBox.CUDA/CakeSortCoOp.cuh"
#include "BlackBox.CUDA/mgpuSegSortWrapper.cuh"
#endif

using namespace std;
using namespace BBox::Core;

namespace po = boost::program_options;

/* defaults for program arguments */
const int default_precision = 2;
const int default_iterations = 10;
const vector<int>& default_sizes { 16, 32, 64, 128, 256, 512, 1024, 2048, 10000, 100000, 1000000, 10000000 };
const string default_sizes_rep { "16 32 64 128 256 512 1024 2048 10000 100000" };

const vector<int> default_segments { 1, 2, 4, 8, 16, 32, 64, 128, 6144, 10240, 98304 };
const string default_segments_rep { "1 2 4 8 16 32 64 128 6144 10240 98304" };

/* program arguments */
struct options
{
	bool verbose;

	int precision;
	bool showPerformance;
	bool showCheck;
	bool showRatios;
	int iterations;
	bool useCoop;

	bool runOpenCL;
	bool runCuda;
	bool doScanInt;
	bool doScanFloat;
	bool doSort;
	bool doSegmentedSort;

	vector<int> sizes;
	vector<int> segments;
};

options getProgramOptions(int argc, char* argv[])
{
	po::options_description generalOptDesc("Output Options");	
	generalOptDesc.add_options()
		("help", 
			"produces this help message")		
		("verbose,v", 
			"to be set if a lot of output is desired")
	;

	po::options_description testResultsOptDesc("Results configurations");
	testResultsOptDesc.add_options()
		("precision", po::value<int>()->default_value(default_precision), 
			"sets output decimal precision")
		("show-performance", po::value<bool>()->default_value(true), 
			"show performance numbers?")
		("show-check", po::value<bool>()->default_value(true), 
			"show ratio [0-1] of test success?")
		("show-ratios", po::value<bool>()->default_value(true), 
			"show table for ratios of performances?")
		("iterations", po::value<int>()->default_value(default_iterations), 
			"number of total iterations, minus a warm-up one")
		("sizes", po::value<vector<int>>()->multitoken()->default_value(default_sizes, default_sizes_rep), 
			"the length of the array, per test")
		("chunks", po::value<vector<int>>()->multitoken()->default_value(default_segments, default_segments_rep), 
			"the number of segments, per test")
		("coop",
			"uses cooperative kernel launch (cuda 9 or above)")
	;

	po::options_description platformOptDesc("Platforms");
	platformOptDesc.add_options()
		("opencl,o", 
			"performs tests on OpenCL platform")
		("cuda,c", 
			"performs tests on CUDA platform")
	;

	po::options_description availTestDesc("Available Tests");
	availTestDesc.add_options()
		("scan-int,a", 
			"executes prefix sum on integers tests")
		("scan-float,f",
			"executes prefix sum on floating points tests")
		("sort,s", 
			"executes regular sorting tests")
		("segmented,e", 
			"executes segmented sorting tests")	
	;

	po::options_description allOptDesc("Allowed Options");
	allOptDesc.add(generalOptDesc).add(testResultsOptDesc).add(platformOptDesc).add(availTestDesc);

	po::variables_map vm;
	try {
		po::store(po::parse_command_line(argc, argv, allOptDesc), vm);
		po::notify(vm);    
	}
	catch (const po::error& e) {
		cout << e.what() << endl;
		exit(-1);
	}

	if (vm.count("help")) {
		cout << allOptDesc << endl;
		exit(1);
	}
	
	options result;

	result.verbose = vm.count("verbose");
	if (result.verbose) {
		cout << "Built on " <<  __DATE__ << endl;
	}

	result.precision = vm["precision"].as<int>();
	result.showPerformance = vm["show-performance"].as<bool>();
	result.showCheck = vm["show-check"].as<bool>();
	result.showRatios = vm["show-ratios"].as<bool>();
	result.iterations = vm["iterations"].as<int>();
	result.sizes = vm["sizes"].as<vector<int>>();
	result.segments = vm["chunks"].as<vector<int>>();
	result.useCoop = vm.count("coop");

	result.runOpenCL = vm.count("opencl");
	result.runCuda = vm.count("cuda");
	result.doScanInt = vm.count("scan-int");
	result.doScanFloat = vm.count("scan-float");
	result.doSort = vm.count("sort");
	result.doSegmentedSort = vm.count("segmented");

#ifndef __CUDA_BLACKBOX__
	if (result.runCuda) {
		cerr << "While --cuda command line argument was specified, this binary has been compiled without CUDA." << endl;
		exit(-1);
	}
#endif

	if (!(result.doScanInt || result.doScanInt || result.doSort || result.doSegmentedSort)) {
		cout << "No test has been specified." << endl;
	}

	return result;
}

const std::string getOpenCLErrorString(cl_int error)
{
	switch(error) {
		// run-time and JIT compiler errors
		case 0: return "CL_SUCCESS";
		case -1: return "CL_DEVICE_NOT_FOUND";
		case -2: return "CL_DEVICE_NOT_AVAILABLE";
		case -3: return "CL_COMPILER_NOT_AVAILABLE";
		case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
		case -5: return "CL_OUT_OF_RESOURCES";
		case -6: return "CL_OUT_OF_HOST_MEMORY";
		case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
		case -8: return "CL_MEM_COPY_OVERLAP";
		case -9: return "CL_IMAGE_FORMAT_MISMATCH";
		case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
		case -11: return "CL_BUILD_PROGRAM_FAILURE";
		case -12: return "CL_MAP_FAILURE";
		case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
		case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
		case -15: return "CL_COMPILE_PROGRAM_FAILURE";
		case -16: return "CL_LINKER_NOT_AVAILABLE";
		case -17: return "CL_LINK_PROGRAM_FAILURE";
		case -18: return "CL_DEVICE_PARTITION_FAILED";
		case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

		// compile-time errors
		case -30: return "CL_INVALID_VALUE";
		case -31: return "CL_INVALID_DEVICE_TYPE";
		case -32: return "CL_INVALID_PLATFORM";
		case -33: return "CL_INVALID_DEVICE";
		case -34: return "CL_INVALID_CONTEXT";
		case -35: return "CL_INVALID_QUEUE_PROPERTIES";
		case -36: return "CL_INVALID_COMMAND_QUEUE";
		case -37: return "CL_INVALID_HOST_PTR";
		case -38: return "CL_INVALID_MEM_OBJECT";
		case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
		case -40: return "CL_INVALID_IMAGE_SIZE";
		case -41: return "CL_INVALID_SAMPLER";
		case -42: return "CL_INVALID_BINARY";
		case -43: return "CL_INVALID_BUILD_OPTIONS";
		case -44: return "CL_INVALID_PROGRAM";
		case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
		case -46: return "CL_INVALID_KERNEL_NAME";
		case -47: return "CL_INVALID_KERNEL_DEFINITION";
		case -48: return "CL_INVALID_KERNEL";
		case -49: return "CL_INVALID_ARG_INDEX";
		case -50: return "CL_INVALID_ARG_VALUE";
		case -51: return "CL_INVALID_ARG_SIZE";
		case -52: return "CL_INVALID_KERNEL_ARGS";
		case -53: return "CL_INVALID_WORK_DIMENSION";
		case -54: return "CL_INVALID_WORK_GROUP_SIZE";
		case -55: return "CL_INVALID_WORK_ITEM_SIZE";
		case -56: return "CL_INVALID_GLOBAL_OFFSET";
		case -57: return "CL_INVALID_EVENT_WAIT_LIST";
		case -58: return "CL_INVALID_EVENT";
		case -59: return "CL_INVALID_OPERATION";
		case -60: return "CL_INVALID_GL_OBJECT";
		case -61: return "CL_INVALID_BUFFER_SIZE";
		case -62: return "CL_INVALID_MIP_LEVEL";
		case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
		case -64: return "CL_INVALID_PROPERTY";
		case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
		case -66: return "CL_INVALID_COMPILER_OPTIONS";
		case -67: return "CL_INVALID_LINKER_OPTIONS";
		case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

		// extension errors
		case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
		case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
		case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
		case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
		case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
		case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
		default: return "Unknown OpenCL error";
	}
}

int main(int argc, char* argv[])
{
	options args = getProgramOptions(argc, argv);

	Stopwatch sw;
	sw.Start();

	cout.precision(args.precision);

	auto freq = Stopwatch::Frequency<long>();

	if (args.verbose) {
		cout << "Clock frequency is " << freq << " Hz" << endl;
		cout << "Resolution is " << 1.0e9 / freq << " nanoseconds per tick." << endl;
	}

	cl::Device device;
	cl::Context context;
	cl::CommandQueue queue;

	try {		
		if (args.runOpenCL) {
			vector<cl::Platform> platforms;
			cl::Platform::get(&platforms);

			if (platforms.size() != 0) {
				auto vector_iterator = find_if(begin(platforms), end(platforms),
					[](const cl::Platform& p) { 
						return p.getInfo<CL_PLATFORM_VENDOR>().find("Advanced Micro Devices") != -1;
					});
				cl::Platform platform;

				if (vector_iterator == platforms.end()) {
					if (args.verbose) {
						cout << "No AMD OpenCL platforms available." << endl;
					}
					platform = platforms[0];
				}
				else {
					platform = *vector_iterator;
				}

				vector<cl::Device> devices;
				platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

				device = devices[0];
				context = cl::Context(device);
				queue = cl::CommandQueue(context, device);

				if (args.verbose) {
					cout << "There are " << platforms.size() << " OpenCL platforms." << endl;
				}
				
				for (int i = 0; i < platforms.size(); ++i) {
					vector<cl::Device> platDevices;
					platforms[i].getDevices(CL_DEVICE_TYPE_ALL, &platDevices);

					if (args.verbose) {
						string platName = platforms[i].getInfo<CL_PLATFORM_NAME>();
						string platVendor = platforms[i].getInfo<CL_PLATFORM_VENDOR>();
						cout << "OpenCL platform " << i << " is " << platName << " from vendor " << platVendor << endl;
						cout << "There is(are) " << platDevices.size() << " device(s) in this platform." << endl;

						for (int j = 0; j < platDevices.size(); ++j) {
							string deviceName = platDevices[j].getInfo<CL_DEVICE_NAME>();
							cout << "\tDevice " << j << " is " << deviceName << endl;
						}
					}
				}

				if (args.verbose) {
					cout << endl;
					cout << "The platform and device selected for running OpenCL are "
						<< platform.getInfo<CL_PLATFORM_NAME>() << " and " << device.getInfo<CL_DEVICE_NAME>()
						<< endl << endl;
				}
			}
			else {
				if (args.verbose) {		
					cout << "There are no OpenCL platforms available." << endl;
				}
			}
		}
	}
	catch (const cl::Error& exc) {
		auto errorCode = exc.err();
		cerr << "Exception while initializing OpenCL. Error code " << 
			errorCode << " " << getOpenCLErrorString(errorCode) <<
			" while calling " << exc.what() << "." << endl;
		return -1;
	}
	catch (const std::exception& exc) {
		cerr << "General exception while initializing OpenCL. " << exc.what() << endl;
		return -1;
	}

#ifdef __CUDA_BLACKBOX__
	if (args.runCuda) {
		cudaFree(0);
		if (args.verbose) {
			cout << "CUDA initialized." << endl;
		}
	}
#endif

	if (args.verbose) {
		cout << "Initialization is over." << endl;
	}

	WriterOptions wo;
	wo.DoLinePad = true;
	wo.LinePad = '-';
	wo.PadLength = 80;
	wo.Separator = "\t";

	Logger logger;
	logger.AddWriter(&cout, wo);

	try {
		Runner runner{ logger };
		runner.showPerformance = args.showPerformance;
		runner.showCheck = args.showCheck;
		runner.showRatios = args.showRatios;

		default_random_engine engine;
		uniform_int_distribution<int> uniformInt{0, 100};
		auto feedIntDataLambda = [&uniformInt, &engine](vector<int> &data)->void {
			for (auto& x : data)
				x = uniformInt(engine);
		};

		uniform_real_distribution<float> uniformFloat{ 0, 100 };
		auto feedFloatDataLambda = [&uniformFloat, &engine](vector<float> &data)->void {
			for (auto& x : data)
				x = uniformFloat(engine);
		};

		//normal_distribution<double> normalFloat{ 1, 10 };
		auto segmentLengthRandom = [&uniformFloat, &engine]()->double {
			return uniformFloat(engine);
		};

		float epsilon = 1e-7f;

		for (auto& s : args.sizes) {
			s = NextGreatestPowerOfTwo(s);
		}

		auto vector_clear_float = [](vector<float>& x) {for (auto& xi : x) xi = 0; };
		auto vector_compare_float = [epsilon](const vector<float>& x, const vector<float>& y)->bool {
			if (x.size() != y.size())
				return false;
			return compareSums(x, y, epsilon * x.size());
		};

		if (args.doScanFloat) {
			vector<shared_ptr<BlackBox<vector<float>, vector<float>>>> bbPrefixSumFloat;

			auto scanRefFloat = make_unique<CPUReferencePrefixSum<float>>();
			scanRefFloat->inclusive = true;
			bbPrefixSumFloat.push_back(move(scanRefFloat));

			auto acmeFloat = make_unique<AcmeBlackBox<vector<float>, vector<float>>>();
			bbPrefixSumFloat.push_back(move(acmeFloat));

			if (args.runOpenCL) {
				auto oclBlellochSimpleFloat = make_unique<BBox::OpenCL::BlellochSimple<float>>(context, device, queue);
				oclBlellochSimpleFloat->inclusive = true;
				bbPrefixSumFloat.push_back(move(oclBlellochSimpleFloat));
			}

			cout << "Running prefix sum tests for floats [inclusive] ..." << endl;

			runner.runMultipleTests<float, float>(bbPrefixSumFloat, args.sizes, feedFloatDataLambda,
				args.iterations, true, true);
		}

		if (args.doScanInt) {
			vector<shared_ptr<BlackBox<vector<int>, vector<int>>>> bbPrefixSumInt;

			auto scanRefInt = make_shared<CPUReferencePrefixSum<int>>();
			bbPrefixSumInt.push_back(scanRefInt);

			auto acmeInt = make_unique<AcmeBlackBox<vector<int>, vector<int>>>();
			bbPrefixSumInt.push_back(move(acmeInt));			
			
			shared_ptr<BBox::OpenCL::BlellochSimple<int>> oclBlellochSimpleInt;
			if (args.runOpenCL) {
				oclBlellochSimpleInt = make_shared<BBox::OpenCL::BlellochSimple<int>>(context, device, queue);
				bbPrefixSumInt.push_back(oclBlellochSimpleInt);

				auto oclBlellochPlusInt = make_unique<BBox::OpenCL::BlellochPlus<int>>(context, device, queue);
				bbPrefixSumInt.push_back(move(oclBlellochPlusInt));
			}

#ifdef __CUDA_BLACKBOX__
			if (args.runCuda) {
				auto cuBlellochPlus = make_unique<BBox::CUDA::BlellochPlus<int>>();
				bbPrefixSumInt.push_back(move(cuBlellochPlus));
			}
#endif			

			cout << "Running prefix sum tests for integers [inclusive]..." << endl;

			runner.runMultipleTests<int, int>(bbPrefixSumInt, args.sizes, feedIntDataLambda,
				args.iterations, true, true);

			cout << "Running prefix sum tests for integers [exclusive]..." << endl;

			scanRefInt->inclusive = false;
			if (args.runOpenCL) {
				oclBlellochSimpleInt->inclusive = false;
			}
			runner.runMultipleTests<int, int>(bbPrefixSumInt, args.sizes, feedIntDataLambda,
				args.iterations, true, true);
		}
		
		if (args.doSegmentedSort) {
			vector<BBox::Core::segmentedArrayBBoxPtr<int>> bbSegmentedSortInt;
			
			auto cpurefSegmentedInt = make_unique<CPUReferenceSegmentedSort<int>>();
			bbSegmentedSortInt.push_back(move(cpurefSegmentedInt));
			
			auto acmeSegmented = make_unique<AcmeBlackBox<SegmentedArray<int>, SegmentedArray<int>>>();
			bbSegmentedSortInt.push_back(move(acmeSegmented));

			if (args.runOpenCL) {
				auto oclCakeSort = make_unique<BBox::OpenCL::CakeSort<int>>(context, device, queue);
				oclCakeSort->hybrid = false;
				auto oclIteratedDispatch = make_unique<BBox::OpenCL::IteratedDispatch<int>>(context, device, queue);
				auto oclParallelDispatch = make_unique<BBox::OpenCL::ParallelDispatch<int>>(context, device, queue);

				bbSegmentedSortInt.push_back(move(oclCakeSort));
				bbSegmentedSortInt.push_back(move(oclIteratedDispatch));
				bbSegmentedSortInt.push_back(move(oclParallelDispatch));
			}		

#ifdef __CUDA_BLACKBOX__
			if (args.runCuda) {
				// auto cuIteratedDispatch = make_unique<BBox::CUDA::IteratedDispatch<int>>();
				// bbSegmentedSortInt.push_back(move(cuIteratedDispatch));

				// auto cuParallelDispatch = make_unique<BBox::CUDA::ParallelDispatch<int>>();
				// bbSegmentedSortInt.push_back(move(cuParallelDispatch));

				auto cuCakeSort = make_unique<BBox::CUDA::CakeSort<int>>();
				cuCakeSort->hybrid = false;
				cuCakeSort->reuseStartBuffer = true;
				bbSegmentedSortInt.push_back(move(cuCakeSort));

				auto mgpuSegSortWrapper = make_unique<BBox::CUDA::mgpuSegSortWrapper>();
				bbSegmentedSortInt.push_back(move(mgpuSegSortWrapper));

				if (args.useCoop) {
					auto cuCakeSortCoOp = make_unique<BBox::CUDA::CakeSortCoOp<int>>();
					cuCakeSortCoOp->hybrid = false;
					bbSegmentedSortInt.push_back(move(cuCakeSortCoOp));
				}
			}
#endif

			cout << "Segmented Array tests." << endl;

			matrix4<int> counts;
			runner.segmentedArrayTest<int>(bbSegmentedSortInt, args.sizes, args.iterations, feedIntDataLambda, false,
				segmentLengthRandom, args.segments, counts, "");
		}

	}	
	catch (const cl::Error& exc) {
		auto errorCode = exc.err();
		cerr << "Exception while running OpenCL. Error code " << 
			errorCode << " " << getOpenCLErrorString(errorCode) <<
			" while calling " << exc.what() << "." << endl;
		return -1;
	}
	catch (const std::exception& exc) {
		cerr << "Error: " << exc.what() << endl;
		return -1;
	}

	sw.Stop();
	if (args.verbose) {
		cout << sw.ElapsedSeconds<float>() << " seconds elapsed since started." << endl;
	}
	
	return 0;
}
