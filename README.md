BlackBox: CakeSort's comparative benchmark test tool
====================================================

This Program is a comparative benchmark of a Cake Sort implementation versus
other segmented-sort algorithms.

However, BlackBox can be used to fit your C++-implemented algorithm-testing
needs too.


Abstract
---------

The goal of this project is to provide a flexible C++ testing bed and
benchmark opportunity for Cake Sort and competing algorithms. Each of the
algorithms, or competing modules, is referred to as a **black box**.

There are technical requisites to the implementation of each black box, but
for the main part, the basics are that in order to compare black boxes, each
should be expose a `Write`, a `Process` and a `Read` callable interface,
enabling the feeding of the *input*, the *processing* and the gathering of the
data of the *output*, respectively.

The code is comprised of four different projects, three of which are
library/header projects and the fourth is a Test program, designed to make use
and to pose as an sample of each of the features of the ones before.

The first one, `BlackBox.Core`, has all the headers needed in order to test
two or more deterministic black boxes side by side. All black boxes are run
several times, one after the other, by batch, and then those batches repeated,
in a collated form, as explained in [below](#benchmark-methodology).

In second and third place, `BlackBox.OpenCL` and `BlackBox.CUDA` provide
algorithmic implementations of black boxes that are to be run in OpenCL and
CUDA, respectively. Both of these projects depend, code-wise, on
`BlackBox.Core`.

Finally, `BlackBox.Test` provides an example of a benchmark program for CPU,
OpenCL or CUDA algorithms, to pose and directly compare each and everyone,
side by side, with multiple options and testing features available at runtime.
In case a Nvidia graphics card is not available, or a user does not need to
run a fully-fledged example, a way is given (`nocuda`), for Linux and Windows
environments, both to compile and execute this program that does not depend on
CUDA, hardware, SDK or headers, and thus does not use CUDA at all. Further
explanation on how to do this is given [below](#how-to-build).


Cake Sort
---------

Cake Sort is an algorithm designed to reduce the number of synchronizations
required to sort a segmented sequence, that is, a sequence to be sorted by
segments of variable length of regular items.

Applying Cake Sort, the number of synchronizations is reduced when compared to
regular parallel sorting algorithms applied to segmented sequences in several
ways, i.e. consecutive applications of a sorting algorithm to each segment,
synchronous, one after the other, or applications in which each thread runs
the sorting of a whole segment, asynchronously, applied in parallel --
henceforth given the names of *iterated dispatch* and *parallel dispatch* in
both this documentation and code -- amongst others.

In parallel programming, a synchronization is the operation that awaits for
the finalization of instructions, orders or commands in groups of concurrent
or semi-concurrent threads. In a Graphics Processor Unit (GPU),
synchronization typically refers to the set of all the threads that
concurrently execute the same program in the whole device. The term may also
refer to the synchronization of a single execution block, i.e. constrained to
the threads in a Single Instruction Multiple Data (SIMD) processor --or
Streaming Multiprocessor (SM) in Nvidia nomenclature.

This discussion of Cake Sort, however, refers to the to whole grid
synchronization, that is, synchronizations to sort a whole sequence of
segments.


Benchmark methodology
---------------------

Say, for instance, that black boxes `a`, `b` and `c` are to be compared with
`n` iterations each, with `n` a large number, in order to better provide
significant statistics. We characterize two main methods of execution order in
a deterministic fashion each of the iterations of each of the black boxes.

One way is to run each black box `n` times, i.e. `a`, ..., `a`, then `b`,
..., `b` and `c`, ..., `c`. We call this first approach **non-collated**.
Another way would be to run `n` batches of `a`, `b`, `c`, in **collated**
order.

The **collated** approach minimizes two risks:

1. Of any black box from being disproportionately affected by any sudden,
   short-lived and frequent activity popping of a third party process that
   competes for the system's resources, that is, CPU-bound, I/O-bound or any
   other short-spanned demanding activity, which would influence the results
   in a disadvantageous way to a single black box if run in batches of the
   same black box at a time.

2. Of any black box of taking advantage of cached data, cache-locality, and
   other processing enhancements or optimizations that a hardware architecture
   may provide to take advantage of iterated pieces of code that are run in a
   quick, uninterrupted sequence. Is our belief that although many algorithms
   do make appropriate use of such hardware enhancements, a majority of black
   boxes would be executed in a production environment in real life in such a
   manner that those enhancements could not be applied outside of the
   algorithm-- a lower rate of cache hits and such-- hence, this methodology
   would overestimate the performance of such black box. This approach does
   not affect any kind of optimization to be applied intra-black box, that is,
   in an isolated execution of the algorithm, and therefore, does nothing to
   underestimate such.

As stated [above](#abstract), each black box exposes a `Write`,
`Process` and `Read` callable. In each batch of the **collated** method, a set
of data for input purposes is generated and then fed (via `Write`) to the
input each black box. Once the feeding is done, a stopwatch starts ticking and
`Process` is called synchronously. Once the control is returned from the
callee, the clock is halted and the elapsed time of the iteration is recorded.
The output of a black box is retrieved (via `Read`) and compared to a sample
--ground truth-- result, previously generated by a known implementation.


How to build
============

The project can be built -- and has been tested both in Ubuntu 16.04 and
Windows 10.


Instructions for Linux
----------------------

If you are cloning from github:

```bash
git clone https://github.com/jksware/cakesort.git
cd cakesort
git submodule update --init
make
```

If you are downloading the ZIP file, it will also make sense to download
[moderngpu](https://github.com/jksware/moderngpu)'s zip file. After you have
downloaded both files to the same (final) directory:

```bash
unzip cakesort-master.zip
unzip moderngpu-master.zip
cp -r moderngpu cakesort/include/
cd cakesort
make
```
If executing `make` returns an error, be sure to check [dependencies](#build-
dependencies-for-linux) and [troubleshooting](#build-troubleshooting-for-
linux) below.

Instructions for Windows
------------------------

If you are cloning from github:

1. Execute the following on Windows' console from the desired directory:
   ```bash
   git clone https://github.com/jksware/cakesort.git
   cd cakesort
   git submodule update --init
   ```
2. Open the appropriate Solution File.

If you are download the ZIP file, as for Linux, it will also make sense to
download [moderngpu](https://github.com/jksware/moderngpu)'s zip file. After
you have downloaded both files to the same (final) directory:

1. Extract the contents of both files to the same folder the ZIP files are in.
   Be careful not to select "extract to moderngpu" with the unzipping program
   program, instead choose "extract here".
2. Copy moderngpu's uncompressed folder --resulting from the decompression--
   to cakesort/include.
3. Open the appropriate Solution File. 


Build dependencies for Linux
---------------------------

On GNU/Linux environments all you need is:

Mandatory:
* One of:
    * g++ 5.x or latter
    * clang 4.x or latter
* GNU Make or equivalent
* boost::program_options 1.65 or latter
* OpenCL Installable Client Driver (OpenCL ICD)

Recommended:
* CUDA 9.x SDK or latter (to be able to compile tests written in CUDA, thus
  running all possible tests).

Other C++ compiler supported versions include:
 * g++ 7.x
 * clang 6.x

There are five main `make` targets, namely:
1. `all`: produces a non-debug optimized binary with CUDA support. This is the
   **default target for production**.
2. `nocuda`: produces a non-debug optimized binary with no CUDA support.
3. `debug`: produces a debug non-optimized binary with CUDA support. This is
   the **default target for debug purposes**.
4. `debugnocuda`: produces a debug non-optimized target with no CUDA support.
5. `clean`: Cleans the clutter. Removes all of intermediary and final binary
   objects produced by previous runs of *this version* of 
   [Makefile](./Makefile). Produces nothing.

As the previous are in the same order as listed here in `Makefile`, if no
argument is given to `make`, the first target will run, thus creating a
finished production-ready build, as if `make all` was given.

All targets require OpenCL for C++, whose headers are included in the project
for greater simplicity and version compatibility.

CUDA targets, namely `all` and `debug`, require having installed the CUDA SDK
--known as cuda toolkit--, as mentioned above. Non-CUDA targets, are only CPU-
and-OpenCL dependant, so those targets require only a proper OpenCL ICD
installation. Usually ICDs provided by hardware vendors should be fine.

Furthermore, all debug targets do not capture Exceptions, while all non-debug
do at several levels.

Binary objects' usual extension that are produced by a non-cuda build target
(OpenCL only) is prefixed by `.nocuda`. In a similar fashion, all generated
debug binary objects' usual extension is prefixed by `.debug`. All other
objects, not affected by neither of these remain with the usual extensions,
i.e., `.o` and `.a`.

The following table summarizes the behavior of Makefile for each main build
target.

Target        | OpenCL   | CUDA     | Debug Info | Optimized? | Exceptions?  | Object ext        | Executable ext
--------------|----------|----------|------------|------------|--------------|-------------------|-----------------
`all`         | Enabled  | Enabled  | minimal    | Yes        | Handled      | `.o`              | `.a`
`nocuda`      | Enabled  | Disabled | minimal    | Yes        | Handled      | `.nocuda.o`       | `.nocuda.a`
`debug`       | Enabled  | Enabled  | full       | No         | Unhandled    | `.debug.o`        | `.debug.a`
`debugnocuda` | Enabled  | Disabled | full       | No         | Unhandled    | `.nocuda.debug.o` | `.nocuda.debug`


Build Troubleshooting for Linux
-------------------------------

The most common build error, if any, you might have when first downloading
this should be a broken dependencies issue.

So, first, make sure to be using the invoking the right tools in `make` time.
Step by step:

1. Make sure you are calling `make` on a device with CUDA toolkit and/or CUDA-
   enabled device.

   There is a point to be made calling `nvcc` (the CUDA compiler) from a non-
   CUDA-enabled device to build, and then just copying the resulting binary to
   another machine to execute, since this two processes are unrelated. But
   this might not be everyone's case. If this is your situation, you might
   want to
   a. Head over to CUDA's download 
      [page](https://developer.nvidia.com/cuda-downloads) and download the
      appropriate file.
   b. Proceed through the installation instructions as listed for the file,
      but instead of installing `cuda`, just install `cuda-toolkit-9-2`, which
      does not include a client driver for the GPU, as follows:

   If you do not want to compile nor execute this project's CUDA code, just
   compile with `nocuda` or `debugnocuda`, passing one of those as arguments to
   make, i.e. `make nocuda`.
 
   For Ubuntu, Debian:
   ```bash   
   sudo apt install cuda-toolkit-9-2
   ```

   For Fedora:
   ```bash
   sudo dnf install cuda-toolkit-9-2
   ```

   For OpenSuse, SLES:
   ```bash
   sudo zypper install cuda-toolkit-9-2
   ```

   For RHEL, CentOS :
   ```bash
   sudo yum install cuda-toolkit-9-2
   ```
 
2. Make sure you are using a correct version of `g++` or `clang++`

   ```bash
   g++ --version
   clang++ --version
   ```

   Should return a string with version information, or bash saying there is no
   such thing as that binary.

   If you have a `g++` or `clang++` installed, but it is a non-supported
   version, you might have to install a younger or older one, depending on the
   issue. Usually a younger, stable version is better.

   If, for any reason, no `g++`, or alternatively, no `clang++` compiler can
   be installed on the system, head over to [Makefile](./Makefile) and change the
   corresponding `CC` variable to the other one, which just might have have
   compatible version on the system's package manager. Instructions to do so 
   are [below](#Direct-`make`-to-a-specific-compiler-binary).


3. Make sure you have installed a correct **boost** library.
   
   **Install the whole boost lib**

   From the system's package manager:

   For Ubuntu, Debian:
   ```bash
   sudo apt update
   sudo apt install libboost-dev
   ```
   For Fedora:
   ```bash
   sudo dnf clean all
   sudo dnf install <package-name>
   ```

   For OpenSuse, SLES:
   ```bash
   sudo zypper refresh
   sudo zypper install <package-name>
   ```

   For RHEL, CentOS :
   ```bash
   sudo yum clean
   sudo yum install <package-name>
   ```

   **Just install just the boost's program-options library**
   1. Substitute `libboost-dev` of the above commands with 
      `libboost-program-options-dev`.

   **Clone from github**

   N.B. The whole download of repositories at time of writing this might be a
   little to much, since all-in the folder reaches 1.6 GB.

   ```bash
   git clone https://github.com/boostorg/boost.git
   cd boost
   git submodule update --init
   make
   ```

   **Download from official website**

   Head over to boost's official [download
   page](https://www.boost.org/users/download/) and download the corresponding
   `.tar.gz` or `.tar.bz2` compressed file. At time of writing, it is a bit
   over 83MB compressed. Move the file to where you want to uncompress it. For
   instance, say you have just downloaded `boost_1_67_0.tar.bz2`:

   ```bash
   tar -xf boost_1_67_0.tar.bz2
   cd boost_1_67_0
   ```
   The uncompressed unbuilt folder is about 690MB.

   If you are cloning or downloading, then read the Getting Started guide for
   unix-variants, included in the downloaded file or on the web
   [here](https://www.boost.org/doc/libs/release/more/getting_started/).


How to install guides: Step by Step
===================================

Manually locate a missing compiler binary
-----------------------------------------

Although your common use binaries might be placed on `/usr/bin` by default,
some installations modes do not do comply with this behavior, and might put
binaries in non-standard or more obscure places.

To manually locate the placement of any binary on the system, for instance
`g++`:

Using `find`:
```bash
find / -name \*g++\* 2>/dev/null | xargs file | grep -i elf | awk -F':' '{print $1}'
```
Using `locate` (faster):

First, install `mlocate`:

For Ubuntu, Debian:
```bash
sudo apt update
sudo apt install mlocate
```

Then update and search
```bash
sudo updatedb
locate g++ | xargs file | grep -i elf | awk -F':' '{print $1}'
```
The update part can be skipped if no changes are presumed to have being made
to the filesystem.


Set a default compiler version
------------------------------

If this is the case, then at least two versions of the same compiler are
installed on the system, but the path on the shell is only configured to call
one by default (the one that lives on /usr/bin, usually).

Since you might not have "sudo powers", or access to root in any way, you
might not have any way to alter the symbolic link of the compiler on the
default binary folder.

There are other reasons why you might not want to change it: to keep the
things the way they are, i.e. not to alter the default g++ or clang++ compiler
version being called user-wide or system-wide, that could brake other projects
whose code depends on a specific version. That said, if you do have "sudo
powers", and are aware of the latter, you might want to set a default
recommended version:

For Ubuntu, Debian, for g++ 7.x:
```bash
sudo apt update
sudo apt install g++-7
sudo rm /usr/bin/g++
sudo ln -s /usr/bin/g++-7 /usr/bin/g++
```
Just change the package name for any other version or compiler, i.e.
`clang-<version>`. For any other system, do as **Install the whole boost
lib**, with the correct compiler's package name.

N.B. Although `clang++` is the compiler binary, `clang-X.Y` is the Ubuntu package.


Direct `make` to a specific compiler binary
--------------------------------------------


Open `Makefile` on the project's folder with your preferred text editor.
Let's call that `vi`.

In order to amend the `nvcc`'s binary path:

1. Open the file. To edit with `vi`, it is as simple as:
   ```bash
   vi Makefile   
   ```
2. Locate a line that begins with `CUDAPATH`. On `vi`, that task is
   accomplished by typing:
   ```vi
   /CUDAPATH
   ```
    and then hitting return.
3. Change the string to the right side of `:=` that finishes with the line
   --what's after the `:=` until line end-- to the main cuda installation
   folder (which is the one that contains `bin/nvcc`). On `vi`, position the
   caret after the assignment symbol `:=` with the arrow keys. Then hit:
   ```
   i
   ```
   which puts the editor on insert mode, delete the lines with backspace or
   delete keys, and type the correct path.
4. Save the work. On `vi`, that's done hitting escape, then typing:
   ```
   :x
   ```
   and
   hitting enter. If you have just messed up the file, and do not want to save
   what's left, hit escape, but instead type:
   ```
   :q!
   ```
   and begin from step 1.


Build dependencies for Windows
------------------------------

Windows **is not** the recommended environment to build the targets. Thus, it
is barely supported by the author.

That said, if you are feeling brave enough you can try building the provided
solution files for Visual Studio in Windows. That being the case, the
following is required:

Mandatory:
* Any Visual Studio, community or otherwise, that supports C++ 14 or enough
  extensions of it, as required by the project, with corresponding Windows SDK
  version being compatible to that of the Visual Studio version installed.
* boost::program_options 1.65 or latter
* OpenCL Installable Client Driver (OpenCL ICD)
* OpenCL.lib from some SDK provider.

Recommended:
* CUDA 9.x SDK or better.

Although recent Visual Studios, beginning at the 2015 edition, provide a way
of managing a project through Makefile, is our feeling that it is not the
correct approach for portability, and a more native method should be used
instead. Hence, two native and platform appropriate Visual Studio Solutions
(`.sln`) are provided.

Solution one, `BlackBox-vs<version>.sln` reaches four different projects, of
which three are configured as libraries and the fourth is compiled to produce
an executable. Those are `BlackBox.Core`, `BlackBox.OpenCL`, `BlackBox.CUDA`
and `BlackBox.Test`, the same four projects described in [Abstract
](#abstract). A second solution, `BlackBox-vs<version>-nocuda.sln`, provides
support for the `nocuda` version of the project, which is for those who would
like to compile and run the project's `BlackBox.Test` benchmark program
without CUDA-enabled hardware or CUDA SDK. It comprises the projects
`BlackBox.Core`, `BlackBox.OpenCL` and `BlackBox.Test`. Each solution is
provided to a number of current Visual Studio versions, i.e., "vs2015" and
"vs2017".

Licensing
=========

BlackBox (hereinafter the "Program") and CakeSort are copyrighted software
distributed under the terms of the GNU General Public License (hereinafter the
"GPL"). All rights remain with its author.

```
   BlackBox is Copyright (C) 2018 Juan Carlos Pujol Mainegra
   CakeSort is Copyright (C) 2015-2018 Juan Carlos Pujol Mainegra

   This program is free software; you can redistribute it and/or modify it
   under the terms of the GNU General Public License as published by the Free
   Software Foundation; version 2 dated June, 1991.

   This program is distributed in the hope that it will be useful, but WITHOUT
   ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
   FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
   more details.

   You should have received a copy of the GNU General Public License along
   with this program (see the file COPYING); if not, write to the Free
   Software Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA
   02110-1301  USA
```   

Some third-party source code used is available from its respective authors
under a more permissive license.

See full license details at [LICENSE.md](./LICENSE.md) and
[COPYING](./COPYING) files.
