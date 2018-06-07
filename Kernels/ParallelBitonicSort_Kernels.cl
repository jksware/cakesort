__kernel 
void bitonicSort(__global uint *data,
                 const uint dataLength,
                 __global uint *rank,
                 __global uint *offset,
                 const uint rankLength)
{
    uint localId = get_global_id(0);

    if (localId >= rankLength)
        return;

    int length = rank[localId];
    data += offset[localId];

    /*
    int localIndex = atomic_inc(index);
    int localOffset = atomic_add(offset, length );
    */

    int stagesNumber = 0;
    for (int temp = length; temp > 1; temp >>= 1)
        ++stagesNumber;

    for (int stage = 0; stage < stagesNumber; ++stage) {
        for (int pass = 0; pass < stage + 1; ++pass) {
            uint pairDistance = 1 << (stage - pass);

            for (int i = 0; i < (length >> 1); ++i) {
    
                //uint leftId = (i % pairDistance) + ((i / pairDistance) * blockWidth);
                uint leftId = (i & (pairDistance - 1)) | ((i & ~(pairDistance - 1)) << 1);
                uint rightId = leftId + pairDistance;

            #if DEBUG
                if (stage == 0 && pass == 0)
                    printf("tid=%d\tlocalid=%d\tstartIdx=%d\tleftId=%d\n", tid, localId, startIdx, leftId);
            #endif

                uint leftKey = data[leftId];
                uint rightKey = data[rightId];

                uint greaterKey;
                uint lesserKey;
                if (leftKey > rightKey) {
                    greaterKey      = leftKey;
                    lesserKey       = rightKey;
                }
                else {
                    greaterKey      = rightKey;
                    lesserKey       = leftKey;
                }
    
                // sameDirectionBlockWidth = 1 << stage;
                // (i / sameDirectionBlockWidth) % 2 == 1
                uint direction = -((i >> stage) & 1);
                data[leftId] = (greaterKey & direction) | (lesserKey & ~direction);
                data[rightId] = (lesserKey & direction) | (greaterKey & ~direction);
            }
        }
    }
}