__kernel 
void bitonicSort(__global uint *data, const uint offset, const uint stage, const uint pass)
{
    data += offset;
    uint localId = get_global_id(0);

    uint pairDistance = 1 << (stage - pass);
    
    //uint leftId = (localId % pairDistance) + ((localId / pairDistance) * blockWidth);
    uint leftId = (localId & (pairDistance - 1)) | ((localId & ~(pairDistance - 1)) << 1);
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
    // (localId / sameDirectionBlockWidth) % 2 == 1
    uint direction = -((localId >> stage) & 1);
    data[leftId] = (greaterKey & direction) | (lesserKey & ~direction);
    data[rightId] = (lesserKey & direction) | (greaterKey & ~direction);
}