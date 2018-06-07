################################################################################
#   Copyright (C) 2018 Juan Carlos Pujol Mainegra
#
#   This program is free software; you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation; version 2 dated June, 1991.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program; if not, write to the Free Software
#   Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
################################################################################

# general compiler options

CUDAPATH         := /usr/local/cuda
# CC               := clang++
CC               := g++
NVCC             := $(CUDAPATH)/bin/nvcc

# -Xptxas="-v"
MGPUFLAGS        := --expt-extended-lambda -use_fast_math 
DEBUGMACRO       := -D"_DEBUG"
CUMACRO          := -D"__CUDA_BLACKBOX__"

CXXINCLUDES      := -Iinclude/CL2 -I.
CXXCOMMON        := -std=c++14 $(CXXINCLUDES)

CUINCLUDES       := -I$(CUDAPATH)/include -Iinclude/moderngpu/src
CUCOMMON         := -arch=sm_61 -rdc=true $(CUINCLUDES) -ccbin $(CC) $(MGPUFLAGS)

CUFLAGS          := $(CXXCOMMON) $(CUCOMMON) -g -O3
CUDFLAGS         := $(CXXCOMMON) $(CUCOMMON) $(DEBUGMACRO) -g -G
CXXFLAGS         := $(CXXCOMMON) -g -O3
CXXDFLAGS        := $(CXXCOMMON) $(DEBUGMACRO) -g3 -Og

# for future use -L/opt/intel/opencl-1.2-6.4.0.37/lib64 -L/usr/local/cuda/targets/x86_64-linux/lib/
LDDIRS           := -L/usr/lib/x86_64-linux-gnu 
LDFLAGS          := $(LDDIRS) -lOpenCL -lc
LDTESTFLAGS      := -lboost_program_options

LDCUDIRS         := -L/usr/local/cuda/targets/x86_64-linux/lib/
LDCUFLAGS        := $(LDCUDIRS) -lOpenCL -lc -lcudart

# directory structure

APPNAME          := BlackBox

COREDIR          := $(APPNAME).Core
CORESRC          := $(wildcard $(COREDIR)/*.cpp)
COREHEADER       := $(wildcard $(COREDIR)/*.h)
COREOBJ          := $(CORESRC:.cpp=.o)
COREDOBJ         := $(CORESRC:.cpp=.debug.o)

OCLDIR           := $(APPNAME).OpenCL
OCLSRC           := $(wildcard $(OCLDIR)/*.cpp) 
OCLHEADER        := $(wildcard $(OCLDIR)/*.h) 
OCLOBJ           := $(OCLSRC:.cpp=.o)
OCLDOBJ          := $(OCLSRC:.cpp=.debug.o)

CUDADIR          := $(APPNAME).CUDA
CUDASRC          := $(wildcard $(CUDADIR)/*.cu)
CUDAHEADER       := $(wildcard $(CUDADIR)/*.cuh) 
CUDAOBJ          := $(CUDASRC:.cu=.o)
CUDADOBJ         := $(CUDASRC:.cu=.debug.o)

TESTDIR          := $(APPNAME).Test
TESTSRC          := $(wildcard $(TESTDIR)/*.cu) 
TESTHEADER       := $(wildcard $(TESTDIR)/*.h) 
TESTOBJ          := $(TESTSRC:.cu=.o)
TESTDOBJ         := $(TESTSRC:.cu=.debug.o)

TESTOBJNOCUDA    := $(TESTSRC:.cu=.nocuda.o)
TESTDOBJNOCUDA   := $(TESTSRC:.cu=.debug.nocuda.o)

FINALALL         := $(APPNAME).a
FINALNOCUDA      := $(APPNAME).nocuda.a
FINALDEBUG       := $(APPNAME).debug.a
FINALDEBUGNOCUDA := $(APPNAME).debug.nocuda.a

# main targets

all: \
	$(FINALALL)

nocuda: \
	$(FINALNOCUDA)

debug: \
	$(FINALDEBUG)

debugnocuda: \
	$(FINALDEBUGNOCUDA)

clean:
	rm -f \
		$(COREOBJ) $(OCLOBJ) $(CUDAOBJ) $(TESTOBJ) $(TESTOBJNOCUDA) \
		$(COREDOBJ) $(OCLDOBJ) $(CUDADOBJ) $(TESTDOBJ) $(TESTDOBJNOCUDA) \
		$(FINALALL) $(FINALNOCUDA) $(FINALDEBUG) $(FINALDEBUGNOCUDA)

# implicit pattern rules

%.debug.nocuda.o: %.cpp
	$(CC) $(CXXDFLAGS) -c $< -o $@

# in order to compile .cu files with no kernels that could include cuda headers
%.debug.nocuda.o: %.cu 
	$(CC) $(CXXDFLAGS) -x c++ -c $< -o $@

%.nocuda.o: %.cpp
	$(CC) $(CXXFLAGS) -c $< -o $@

# in order to compile .cu files with no kernels that could include cuda headers
%.nocuda.o: %.cu
	$(CC) $(CXXFLAGS) -x c++ -c $< -o $@

%.debug.o: %.cpp
	$(CC) $(CXXDFLAGS) $(CUINCLUDES) $(CUMACRO) -c $< -o $@

%.debug.o: %.cu
	$(NVCC) $(CUDFLAGS) $(CUMACRO) -c $< -o $@

%.o: %.cpp
	$(CC) $(CXXFLAGS) $(CUINCLUDES) $(CUMACRO) -c $< -o $@

%.o: %.cu
	$(NVCC) $(CUFLAGS) $(CUMACRO) -c $< -o $@

# build DAG

$(COREOBJ): \
	$(COREHEADER)

$(OCLOBJ): \
	$(COREOBJ) \
	$(OCLHEADER)

$(CUDAOBJ): \
	$(COREH) \
	$(CUDAHEADER)

$(TESTOBJ): \
	$(OCLOBJ) $(CUDAOBJ) \
	$(TESTHEADER)

$(TESTOBJNOCUDA): \
	$(OCLOBJ) \
	$(TESTHEADER)

$(COREDOBJ): \
	$(COREHEADER)

$(OCLDOBJ): \
	$(COREDOBJ) \
	$(OCLHEADER)

$(CUDADOBJ): \
	$(COREDOBJ) \
	$(CUDAHEADER)

$(TESTDOBJ): \
	$(OCLDOBJ) $(CUDADOBJ) \
	$(TESTHEADER)

$(TESTDOBJNOCUDA): \
	$(OCLDOBJ) \
	$(TESTHEADER)

$(FINALALL): $(COREOBJ) $(OCLOBJ) $(CUDAOBJ) $(TESTOBJ)
	$(NVCC) $(LDCUFLAGS) $(LDTESTFLAGS) $(CUFLAGS) $^ -o $@

$(FINALNOCUDA): $(COREOBJ) $(OCLOBJ) $(TESTOBJNOCUDA) 
	$(CC) $(LDFLAGS) $(LDTESTFLAGS) $(CXXFLAGS) $^ -o $@

$(FINALDEBUG): $(COREDOBJ) $(OCLDOBJ) $(CUDADOBJ) $(TESTDOBJ)
	$(NVCC) $(LDCUFLAGS) $(LDTESTFLAGS) $(CUDFLAGS) $^ -o $@

$(FINALDEBUGNOCUDA): $(COREDOBJ) $(OCLDOBJ) $(TESTDOBJNOCUDA)
	$(CC) $(LDFLAGS) $(LDTESTFLAGS) $(CXXDFLAGS) $^ -o $@
