#pragma once

#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_CL_1_2_DEFAULT_BUILD

#include "BlackBox.Core/BlackBox.h"

#include <string>
#include <map>
#include <unordered_map>
#include <list>
#include <iostream>
#include <fstream>
#include <istream>
#include <ostream>
#include <regex>
#include <CL/cl2.hpp>

#ifdef _WIN32
#define SEP "\\"
#else
#define SEP "/"
#endif

namespace BBox
{
	namespace OpenCL
    {
		using namespace BBox::Core;    

		template <typename T, typename R>
		class OpenCLBlackBox : public BBox::Core::BlackBox <T, R>
		{
			std::map<std::string, cl::Program> _programs;
			std::list<cl::Kernel> _loadedKernel;

		protected:
			// Holds current device
			cl::Device _device;

			// Holds the length (given the context) of the main data types.
			size_t _length;

			// Holds name types to substitute by a valid .net structure when loading a program source.
			std::unordered_map<std::string, size_t> declareTypes;

			const std::unordered_map<size_t, std::string> validTypes = {
					{typeid(float).hash_code(), "float"},
					{typeid(int).hash_code(), "int"},
					{typeid(double).hash_code(), "double"},
					{typeid(char).hash_code(), "char"}
			};

			// Holds information to substitute in the programs sources loaded.
			std::string extraDefines;

			// Holds the OpenCL context
			cl::Context context;

			// Holds the OpenCL command queue.
			cl::CommandQueue queue;

			// Holds the last error by an OpenCL API call.
			cl_int lastError;

			// from kernel information
			typedef struct {
				size_t kernelWorkGroupSize;
				cl_ulong localMemoryUsed;
				cl_ulong localMemSize;
				size_t compileWorkGroupSize[3];
			} kernelInfo;

			// gets kernel info
			static kernelInfo getKernelInfo(cl::Kernel kernel, cl::Device device)
            {
                kernelInfo result;

                device.getInfo(CL_DEVICE_LOCAL_MEM_SIZE, &result.localMemSize);

                kernel.getWorkGroupInfo(device, CL_KERNEL_WORK_GROUP_SIZE, &result.kernelWorkGroupSize);
                kernel.getWorkGroupInfo(device, CL_KERNEL_LOCAL_MEM_SIZE, &result.localMemoryUsed);
                kernel.getWorkGroupInfo(device, CL_KERNEL_COMPILE_WORK_GROUP_SIZE, &result.compileWorkGroupSize);

                return result;
            }            

		public:

			OpenCLBlackBox(const cl::Device& device = cl::Device::getDefault())
				: _device(device)
            {
				context = cl::Context(_device);
				queue = cl::CommandQueue(this->context, _device);
			}

			OpenCLBlackBox(const cl::Context& context, const cl::Device& device, const cl::CommandQueue& commandQueue)
				: _device(device), context(context), queue(commandQueue)
            {
			}

			cl::Kernel LoadKernel(const std::string& path, const std::string& kernelName)            
            {
                auto programPair = _programs.find(path);
                cl::Program program;

                if (programPair == _programs.end()) {
                    std::string source;
                    std::ifstream sourceFile{ path };
                    if (!sourceFile.is_open()) {
                        throw std::runtime_error(std::string("Couldn't open file ") + path);
                    }

                    std::getline(sourceFile, source, static_cast<char>(0));

                    for (auto& pair : declareTypes) {
                        auto subsStringPair = validTypes.find(pair.second);
                        if (subsStringPair == validTypes.end())
                            throw std::runtime_error("No valid substitution was found for a type.");

                        source = std::regex_replace(source, std::regex{pair.first}, subsStringPair->second);
                    }

                    std::regex extraDefKeyword{ "____EXTRA_DEFINES____" };
                    source = std::regex_replace(source, extraDefKeyword, extraDefines);

                    try {
                        std::vector<cl::Device> devices{ this->_device };
                        program = cl::Program(this->context, source, false);
                        program.build(devices);
                    }
                    catch (const std::exception& exc) {
                        std::cerr << "Error building program. The exception is on: " << exc.what() << std::endl;

                        cl_int buildErr = CL_SUCCESS;
                        auto buildInfo = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(this->_device, &buildErr);
                        std::cerr << "Build log is: " << buildInfo << std::endl;
                    }

                    _programs[path] = program;
                }
                else {
                    program = programPair->second;
                }

                cl::Kernel result{ program, kernelName.data() };
                _loadedKernel.push_back(cl::Kernel{ result });

                return result;
            }

			virtual bool Finish() override
            {
                queue.flush();
                queue.finish();
                return true;
            }


			inline void length(int length)
            {
                _length = length;
            }
            
			inline int length() const
            {
            	return _length;
            }

			virtual ~OpenCLBlackBox() {};
		};
	}
}
