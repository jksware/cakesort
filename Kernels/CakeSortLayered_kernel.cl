/******************************************************************************
*
*   File:           CakeSortLayered_kernel.cl
*   Author:         Juan Carlos Pujol Mainegra
*   Description:    Layered bitonic sort, device part of CakeSort.
*
*   Copyright 2015 Juan Carlos Pujol Mainegra
*
******************************************************************************/

#define DEBUG 0
#define COUNT_ACCESSES 0

typedef __DATA__ Data;
typedef __KEY__ Key;

____EXTRA_DEFINES____

__kernel
void binSearchTidGroup(__global const int *start,
                       __global int *subset,
                       const int length,
                       const int paddedLength,
                       const int globalWS)
{
    int id = get_global_id(0);
    if (id > globalWS)
        return;

    //int blockSize = get_local_size(0);

    int lower = 0;
    for (int size = paddedLength; size > 0; size >>= 1) {
        int upper = lower + size - 1;
        bool smaller = upper < length && (start[upper] >> 1) <= id;
        lower += smaller ? size : 0;
        lower = min(lower, length);
    }   

    subset[id] = lower - 1;

    /*
    int subset = lower - 1;
    int nextStart = start[lower] >> 1;

    for (int i = 0; i < blockSize; ++i) {
        if (id * blockSize + i >= globalWS)
            return;

        if (id * blockSize + i == nextStart) {          
            subset++;
            nextStart = start[++lower] >> 1;
        }
                        
        subsetNumber[id * blockSize + i] = subset;
    }
    */
}

__kernel
void setGid(__global int* data)
{ 
    int gid = get_global_id(0);
    data[gid] = gid;
}


__kernel 
void bitonicSort(   __global uint *key,
                    __global int *index,
                    const uint stage, 
                    uint pass)
{
    uint localId = get_global_id(0);

    for (pass = 0; pass < stage + 1; ++pass) {  
        uint pairDistance = 1 << (stage - pass);
    
        //uint leftId = (localId % pairDistance) + ((localId / pairDistance) * blockWidth);
        uint leftId = (localId & (pairDistance - 1)) | ((localId & ~(pairDistance - 1)) << 1);
        uint rightId = leftId + pairDistance;

    #if DEBUG
        if (stage == 0 && pass == 0)
            printf("tid=%d\tlocalid=%d\tstartIdx=%d\tleftId=%d\n", tid, localId, startIdx, leftId);
    #endif

        int leftIndex = index[leftId];
        int rightIndex = index[rightId];
        uint leftKey = key[leftId];
        uint rightKey = key[rightId];
 
        uint greaterKey, greaterIndex;
        uint lesserKey, lesserIndex;
        if (leftKey > rightKey) {
            greaterKey      = leftKey;
            greaterIndex    = leftIndex;
            lesserKey       = rightKey;
            lesserIndex     = rightIndex;
        }
        else {
            greaterKey      = rightKey;
            greaterIndex    = rightIndex;
            lesserKey       = leftKey;
            lesserIndex     = leftIndex;
        }
    
        // sameDirectionBlockWidth = 1 << stage;
        // (localId / sameDirectionBlockWidth) % 2 == 1
        uint direction = -((localId >> stage) & 1);
        index[leftId] = (greaterIndex & ~direction) | (lesserIndex & direction);
        key[leftId] = (greaterKey & ~direction) | (lesserKey & direction);
        index[rightId] = (lesserIndex & ~direction) | (greaterIndex & direction);
        key[rightId] = (lesserKey & ~direction) | (greaterKey & direction);

        barrier(CLK_LOCAL_MEM_FENCE);
    }
}


__kernel 
void bitonicSortLayered(__global Data *data,
                        const uint stage, 
                        const uint pass,
                        __global const uint *subset,
                        __global const uint *start,
                        __global const uint *sortedStart,
                        __global const uint *permutation,
                        const uint assumeSorted,
                        const int globalWS)
{
    int tid = get_global_id(0);

    if (tid >= globalWS)
        return;
    
    // subsetIdx index (in [0..k-1] where K is the partition cardinality) is the mapping of each thread to the corresponding subset subsetNumber
    uint subsetIdx = subset[tid];

    // localId is the offset of the current thread from each start of the current thread subset
    uint localId = tid - (sortedStart[subsetIdx] >> 1);

    // permutedIdx (in [0..k-1]) is a permutation of the subset index where the partition is sorted by cardinality of each subset
    uint permutedIdx = assumeSorted == 0 ? permutation[subsetIdx] : subsetIdx;

    // start index (in [0..N-1] where N is the fragment set cardinality) is the index of the half of the start of each subset on the global memory
    uint startIdx = start[permutedIdx];

    uint pairDistance = 1 << (stage - pass);
    
    //uint leftId = startIdx + (localId % pairDistance) + ((localId / pairDistance) * blockWidth);
    uint leftId = startIdx + ((localId & (pairDistance - 1)) | ((localId & ~(pairDistance - 1)) << 1));
    uint rightId = leftId + pairDistance;

#if DEBUG
    if (stage == 0 && pass == 0)
        printf("tid=%d\tlocalid=%d\tstartIdx=%d\tleftId=%d\n", tid, localId, startIdx, leftId);
#endif

    Data leftData = data[leftId];
    Data rightData = data[rightId];
 
    Data greater;
    Data lesser;
    if (GetKey(leftData) > GetKey(rightData)) {
        greater = leftData;
        lesser = rightData;
    }
    else {
        greater = rightData;
        lesser = leftData;
    }

    // sameDirectionBlockWidth = 1 << stage;
    // (localId / sameDirectionBlockWidth) % 2 == 1
    uint direction = -((localId >> stage) & 1);
    data[leftId] = (greater & direction) | (lesser & ~direction);
    data[rightId] = (lesser & direction) | (greater & ~direction);


#if DEBUG
    if (stage == 0 && pass == 0)
        printf("tid=%d\tdirection=%d\tgreater=%d\tlesser=%d\tdata[leftId]=%d\tdata[rightId]=%d\n", 
            tid, direction, greater, lesser, data[leftId], data[rightId]);
#endif
}


__kernel 
void bitonicSortForSorted(__global Data *data,
                        const uint stage, 
                        const uint pass,
                        const int globalWS
                        )
{
    int localId = get_global_id(0);

    if (localId >= globalWS)
        return;

    uint pairDistance = 1 << (stage - pass);
    
    //uint leftId = (localId % pairDistance) + ((localId / pairDistance) * blockWidth);
    uint leftId = ((localId & (pairDistance - 1)) | ((localId & ~(pairDistance - 1)) << 1));
    uint rightId = leftId + pairDistance;

#if DEBUG
    if (stage == 0 && pass == 0)
        printf("tid=%d\tlocalid=%d\tstartIdx=%d\tleftId=%d\n", tid, localId, startIdx, leftId);
#endif

    Data leftData = data[leftId];
    Data rightData = data[rightId];
    Key leftKey = GetKey(leftData);
    Key rightKey = GetKey(rightData);
 
    Data greater;
    Data lesser;
    if (leftKey > rightKey)
    {
        greater = leftData;
        lesser = rightData;
    }
    else
    {
        greater = rightData;
        lesser = leftData;
    }

    // sameDirectionBlockWidth = 1 << stage;
    // (localId / sameDirectionBlockWidth) % 2 == 1
    uint direction = (localId >> stage) & 1;

    if (direction)
    { 
        data[leftId] = greater;
        data[rightId] = lesser;
    }
    else
    { 
        data[leftId] = lesser;
        data[rightId] = greater;
    }
}

__kernel
void swap(__global Data *data,
            const int from,
            int width,
            const int to)
{
    int gid = get_global_id(0);
    data += from + width;

    uint leftId = (gid % (width / 2)) + (gid / (width / 2) * width * 2);
    uint rightId = ((leftId / width) + 1) * width - (leftId % width) - 1;

    if ((from + width + leftId >= to) || (from + width + rightId >= to))
        return;

    Data leftData = data[leftId];
    Data rightData = data[rightId];

    data[leftId] = rightData;
    data[rightId] = leftData;

    /*
    data[leftId] = 1;
    data[rightId] = 2;
    */
}

__kernel 
void bitonicSortIndexLayered(__global const Data *data,
                             const uint stage, 
                             const uint pass,
                             __global const uint *subset,
                             __global const uint *start,
                             __global const uint *sortedStart,
                             __global const uint *permutation,
                             const uint assumeSorted,
                             const int globalWS,
                             __global int *index)
{
    int tid = get_global_id(0);

    if (tid >= globalWS)
        return;
    
    // subsetIdx index (in [0..k-1] where K is the partition cardinality) is the mapping of each thread to the corresponding subset subsetNumber
    uint subsetIdx = subset[tid];

    // localId is the offset of the current thread from each start of the current thread subset
    uint localId = tid - (sortedStart[subsetIdx] >> 1);

    // permutedIdx (in [0..k-1]) is a permutation of the subset index where the partition is sorted by cardinality of each subset
    uint permutedIdx = assumeSorted == 0 ? permutation[subsetIdx] : subsetIdx;

    // start index (in [0..N-1] where N is the fragment set cardinality) is the index of the half of the start of each subset on the global memory
    uint startIdx = start[permutedIdx];

    uint pairDistance = 1 << (stage - pass);
    
    //uint leftId = startIdx + (localId % pairDistance) + ((localId / pairDistance) * blockWidth);
    uint leftId = startIdx + ((localId & (pairDistance - 1)) | ((localId & ~(pairDistance - 1)) << 1));
    uint rightId = leftId + pairDistance;

#if DEBUG
    if (stage == 0 && pass == 0)
        printf("tid=%d\tlocalid=%d\tstartIdx=%d\tleftId=%d\n", tid, localId, startIdx, leftId);
#endif

    int leftIndex = index[leftId];
    int rightIndex = index[rightId];
    Key leftKey = GetKey(data[leftIndex]);
    Key rightKey = GetKey(data[rightIndex]);
 
    int greater;
    int lesser;
    if (leftKey > rightKey)
    {
        greater = leftIndex;
        lesser = rightIndex;
    }
    else
    {
        greater = rightIndex;
        lesser = leftIndex;
    }

    // sameDirectionBlockWidth = 1 << stage;
    // (localId / sameDirectionBlockWidth) % 2 == 1
    uint direction = (localId >> stage) & 1;

    direction = -direction;
    index[leftId] = (greater & direction) | (lesser & ~direction);
    index[rightId] = (lesser & direction) | (greater & ~direction);
}


__kernel void mergeSort(__global uint *data,
                        __global uint *index,
                        __local uint *auxData,
                        __local uint *auxIndex)
{
    int lid = get_local_id(0); // index in workgroup
    int wg = get_local_size(0); // workgroup size = block size, power of 2

    // Move IN, OUT to block start
    int offset = get_group_id(0) * wg;
    data += offset; index += offset;

    // Load block in AUX[WG]
    auxData[lid] = data[lid];
    auxIndex[lid] = index[lid];

    barrier(CLK_LOCAL_MEM_FENCE); // make sure AUX is entirely up to date

    // Now we will merge sub-sequences of length 1,2,...,WG/2
    for (int length = 1; length < wg; length <<= 1)
    {
        uint iKey = auxData[lid];
        uint iIndex = auxIndex[lid];
        int ii = lid & (length - 1);  // index in our sequence in 0..length-1
        int sibling = (lid - ii) ^ length; // beginning of the sibling sequence
        int lower = 0;
        for (int size = length; size > 0; size >>= 1) // increment for dichotomic search
        {
            int upper = sibling + lower + size - 1;
            uint upperKey = auxData[upper];
            bool smaller = (upperKey < iKey) || ( upperKey == iKey && upper < lid );
            lower += (smaller) ? size : 0;
            lower = min(lower, length);
        }
        int bits = (length << 1) - 1; // mask for destination
        int dest = ((ii + lower) & bits) | (lid & ~bits); // destination index in merged sequence
        barrier(CLK_LOCAL_MEM_FENCE);
        auxData[dest] = iKey;
        auxIndex[dest] = iIndex;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    int direction = -((get_group_id(0) & 1) == 1);
    int outpos = (lid & direction) | ((wg - lid - 1) & ~direction);
    // Write output
    data[lid] = auxData[outpos];
    index[lid] = auxIndex[outpos];
}


__kernel 
void mergeSortLayered(  __global Data *data,
                        __global const uint *start,
                        __global const uint *permutation,
                        const uint assumeSorted,
                        __local Data *aux,
                        const uint lowGid,
                        const uint count,
                        const int globalWS)
{
    int lid = get_local_id(0);
    int wg = get_local_size(0);
    int gid = get_group_id(0);

    uint idx = lowGid + gid * wg / count;
    uint permutedIdx = assumeSorted ? idx : permutation[idx];

    uint offset = start[permutedIdx] + wg * (gid % (count / wg));
    data += offset;

#if DEBUG   
    if (lid < 10)
        printf("lid=%d\tassumeSorted=%d\tlowGid=%d\tgid=%d\tpermutedIdx=%d\tstart[permutedIdx]=%d\tcount=%d\toffset=%d\n",
            lid, assumeSorted, lowGid, gid, permutedIdx, start[permutedIdx], count, offset);
#endif

    aux[lid] = data[lid];

    barrier(CLK_LOCAL_MEM_FENCE); // make sure AUX is entirely up to date

    // Now we will merge sub-sequences of length 1,2,...,WG/2
    for (int length = 1; length < wg; length <<= 1)
    {
        Data iData = aux[lid];
        Key iKey = GetKey(iData);
        int ii = lid & (length - 1);  // index in our sequence in 0..length-1
        int sibling = (lid - ii) ^ length; // beginning of the sibling sequence
        int lower = 0;
        for (int size = length; size > 0; size >>= 1) // increment for dichotomic search
        {
            int upper = sibling + lower + size - 1;
            Key upperKey = GetKey(aux[upper]);
            bool smaller = (upperKey < iKey) || ( upperKey == iKey && upper < lid );
            lower += (smaller) ? size : 0;
            lower = min(lower, length);
        }
        int bits = (length << 1) - 1; // mask for destination
        int dest = ((ii + lower) & bits) | (lid & ~bits); // destination index in merged sequence
        barrier(CLK_LOCAL_MEM_FENCE);
        aux[dest] = iData;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (get_global_id(0) >= globalWS)
        return;

    int outpos = (count <= wg || (gid % (count / wg)) & 1) ? lid : wg - lid - 1;
    data[lid] = aux[outpos];
}