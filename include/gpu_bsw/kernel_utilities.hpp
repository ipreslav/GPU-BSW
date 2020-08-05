#pragma once

#include <cassert>
#include <cstdint>
#include <cstdlib>

#define NUM_OF_AA      21
#define ENCOD_MAT_SIZE 91
#define SCORE_MAT_SIZE 576

enum class DataType {
  DNA,
  RNA
};

enum class Direction {
  FORWARD,
  REVERSE
};

struct Cell {
  short H = 0;
  short F = 0;
  short E = 0;
};

namespace gpu_bsw {

///@brief Within a warp, find the maximum value and its associated indices
///@param val        Value associated with this thread
///@param myIndex    Index of that value along the i-axis
///@param myIndex2   Index of that value along the j-axis
///@param lengthSeqB Length of the sequence being aligned (<=1024)
///@return Lane 0 of the warp returns the maximum value found in the warp,
///        and Lane 0's myIndex and myIndex2 are set to correspond to the
//         indices of this value. Other lanes' values and returns are garbage.
template<Direction DIR>
static __inline__ __device__ short
warpReduceMax_with_index(const short my_initial_val, short& myIndex, short& myIndex2, const unsigned lengthSeqB)
{
    assert(lengthSeqB<=1024);

    constexpr int warpSize = 32;
    //We set the initial maximum value to INT16_MIN for threads which are out of
    //range. Since all other allowed values are greater than this, we'll always
    //get the right answer unless all of the threads have INT16_MIN as their
    //value and we're moving in the reverse direction.
    short maxval = (threadIdx.x < lengthSeqB) ? my_initial_val : INT16_MIN;
    short maxi   = myIndex;
    short maxi2  = myIndex2;

    const auto mask = __ballot_sync(0xffffffff, threadIdx.x < lengthSeqB);

    for(int offset = warpSize / 2; offset > 0; offset /= 2)
    {
        //Get the value and indices corresponding to our shuffle buddy
        const auto buddy_val = __shfl_down_sync(mask, maxval, offset);
        const auto buddy_i   = __shfl_down_sync(mask, maxi,   offset);
        const auto buddy_i2  = __shfl_down_sync(mask, maxi2,  offset);

        //If our buddy's value is greater than our value, we take our buddy's
        //information
        if(buddy_val>maxval)
        {
            maxi   = buddy_i;
            maxi2  = buddy_i2;
            maxval = buddy_val;
        }
        else if(buddy_val == maxval)
        {
            // We want to keep the first maximum value we find in the direction
            // of travel. This is kind of redundant and has been done purely to
            // match the results with SSW to get the smallest alignment with
            // highest score. Theoreticaly all the alignmnts with same score are
            // same.
            if((DIR==Direction::REVERSE && buddy_i2 > maxi2) || (DIR==Direction::FORWARD && buddy_i < maxi)){
              maxi  = buddy_i;
              maxi2 = buddy_i2;
            }
        }
    }
    myIndex  = maxi;
    myIndex2 = maxi2;
    return maxval;
}



template<Direction DIR>
static __device__ short
blockShuffleReduce_with_index(short myVal, short& myIndex, short& myIndex2, unsigned lengthSeqB)
{
    assert(lengthSeqB<=1024);

    constexpr int WS = 32; //Warp size
    const auto laneId = threadIdx.x % WS;
    const auto warpId = threadIdx.x / WS;

    __shared__ short warp_lead_max  [WS]; //Maximum value of each warp's lane 0
    __shared__ short warp_lead_inds [WS]; //Index1 associated with that max
    __shared__ short warp_lead_inds2[WS]; //Index2 associated with that max

    //Get the maximum value information for each of the warps
    short myInd  = myIndex;
    short myInd2 = myIndex2;
    myVal = warpReduceMax_with_index<DIR>(myVal, myInd, myInd2, lengthSeqB);

    //Each warp calculates and writes its answer independently and all the
    //threads in the warp move together, so we don't need a sync point here
    if(laneId == 0){
        warp_lead_max  [warpId] = myVal;
        warp_lead_inds [warpId] = myInd;
        warp_lead_inds2[warpId] = myInd2;
    }

    //We need to gather each warp's maximum info into a single warp, so we need
    //to sync here.
    __syncthreads();

    //We'll now set up the 0th warp of the block to finish the reduction. Each
    //thread in that warp will take the maximum information retrieved from each
    //of the block's warps above.

    //A standard block reduction would use `blockDim.x/WS` to determine which
    //warps were active, but this assumes all the threads in a warp were active.
    //In our case, only some of the threads might be activate, so we need to
    //round up. Therefore, we use the integer ceiling function below.
    const unsigned warp_was_active = (blockDim.x + (WS-1)) / WS;

    if(threadIdx.x < warp_was_active)
    {
        myVal  = warp_lead_max  [laneId];
        myInd  = warp_lead_inds [laneId];
        myInd2 = warp_lead_inds2[laneId];
    }
    else
    {
        myVal  = INT16_MIN;
        myInd  = INT16_MIN;
        myInd2 = INT16_MIN;
    }

    //The zeroth warp now holds the maximums of each of the other warps. We now
    //do a reduction on the zeroth warp to find the overall max.
    if(warpId == 0)
    {
        myVal    = warpReduceMax_with_index<DIR>(myVal, myInd, myInd2, lengthSeqB);
        myIndex  = myInd;
        myIndex2 = myInd2;
    }

    return myVal;
}



__inline__ __device__ __host__ short
findMaxFour(const short first, const short second, const short third, const short fourth)
{
    short maxScore = 0;

    maxScore = max(first,    second);
    maxScore = max(maxScore, third );
    maxScore = max(maxScore, fourth);

    return maxScore;
}

}
