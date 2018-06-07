/**********************************************************************
Copyright ©2014 Advanced Micro Devices, Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

*   Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
*   Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
 other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
********************************************************************/

__kernel 
void group_prefixSum(__global __T__ * output,
                     __global __T__ * input,
                     __local  __T__ * block,
                     const uint length,
                     const uint idxStride) {
    int localId = get_local_id(0);
    int localSize = get_local_size(0);
    int globalIdx = get_group_id(0) * localSize + localId;

    // Cache the computational window in shared memory
    globalIdx = (idxStride * (2 * globalIdx + 1)) - 1;
    if(globalIdx < length)             { block[2*localId]     = input[globalIdx];               }
    if(globalIdx + idxStride < length) { block[2*localId + 1] = input[globalIdx + idxStride];   }

    // Build up tree 
    int stride = 1;
    for(int l = length>>1; l > 0; l >>= 1)
    {
      barrier(CLK_LOCAL_MEM_FENCE);
      if(localId < l) {
            int ai = stride*(2*localId + 1) - 1;
            int bi = stride*(2*localId + 2) - 1;
            block[bi] += block[ai];
         }
         stride <<= 1;
    }
         
    if (length > 2)
    {
        if(stride < length) { stride <<= 1; }

        // Build down tree
        int maxThread = stride>>1;
        for(int d = 0; d < maxThread; d<<=1)
        {
            d += 1;
            stride >>=1;
            barrier(CLK_LOCAL_MEM_FENCE);

            if(localId < d) {
                int ai = stride*(localId + 1) - 1;
                int bi = ai + (stride>>1);
                block[bi] += block[ai];
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // write the results back to global memory
    if(globalIdx < length)           { output[globalIdx]             = block[2*localId];        }
    if(globalIdx+idxStride < length) { output[globalIdx + idxStride] = block[2*localId + 1];    }
}

__kernel 
void group_prefixSumExclusive(__global __T__ * output,
                              __global __T__ * input,
                              __local  __T__ * block,
                              const uint length,
                              const uint idxStride) {
    int localId = get_local_id(0);
    int localSize = get_local_size(0);
    int globalIdx = get_group_id(0) * localSize + localId;

    // Cache the computational window in shared memory
    globalIdx = (idxStride * (2 * globalIdx + 1)) - 1;
    if(globalIdx < length)             { block[2*localId]     = input[globalIdx];               }
    if(globalIdx + idxStride < length) { block[2*localId + 1] = input[globalIdx + idxStride];   }

    // Build up tree 
    int stride = 1;
    for (int d = length >> 1; d > 0; d >>= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (localId < d) {
            int ai = stride * (2 * localId + 1) - 1;
            int bi = stride * (2 * localId + 2) - 1;
            block[bi] += block[ai];
        }
        stride <<= 1;
    }

    if (length > 2)
    {
        barrier(CLK_LOCAL_MEM_FENCE);
        
        if (localId == 0) {
            block[length] = block[length - 1];
            block[length - 1] = 0;
        }
        
        for (int d = length >> 1; d > 0; d >>= 1) {
            barrier(CLK_LOCAL_MEM_FENCE);
            
            if (localId % d == 0) {
            //if (localId < d) {
                int ai = 2 * localId + d - 1;
                int bi = 2 * localId + (d << 1) - 1;

                __T__ temp = block[ai];
                block[ai] = block[bi];
                block[bi] += temp;          
                
                //block[bi] += block[ai];
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // write the results back to global memory
    if(globalIdx < length)           { output[globalIdx]             = block[2 * localId];      }
    if(globalIdx+idxStride < length) { output[globalIdx + idxStride] = block[2 * localId + 1];  }
}

__kernel
void global_prefixSum(__global __T__ * buffer,
                      const uint stride,
                      const uint length) {
    int localSize = get_local_size(0);
    int groupIdx  = get_group_id(0);

    int blocks = stride / localSize;        // sorted groups per block
    // Map the gids to unsorted local blocks.
    int gidToUnsortedBlocks = groupIdx + (groupIdx / ((stride<<1) - blocks) + 1) * blocks;

    // Get the corresponding global index
    int globalIdx = gidToUnsortedBlocks * localSize + get_local_id(0);
    if ( ((globalIdx + 1) % stride != 0) && (globalIdx < length)) {
        buffer[globalIdx] += buffer[globalIdx - (globalIdx % stride + 1)];
    }
}

__kernel
void global_prefixSumExclusive(__global __T__ * buffer,
                               const uint stride,
                               const uint length) {
    int localSize = get_local_size(0);
    int groupIdx  = get_group_id(0);

    int blocks = stride / localSize;        // sorted groups per block
    // Map the gids to unsorted local blocks.
    int gidToUnsortedBlocks = groupIdx + (groupIdx / ((stride << 1) - blocks) + 1) * blocks;

    // Get the corresponding global index
    int globalIdx = gidToUnsortedBlocks * localSize + get_local_id(0);
    if ( ((globalIdx + 1) % stride != 0) &&  (globalIdx < length)) {
        buffer[globalIdx] = buffer[globalIdx - (globalIdx % stride + 1)];
    }
}

