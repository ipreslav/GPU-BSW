#pragma once

#include <gpu_bsw/kernel_utilities.hpp>

#include <thrust/swap.h>

#include <cassert>
#include <cstdint>
#include <cstdlib>

namespace gpu_bsw{

template<DataType DT, Direction DIR>
inline __device__ void
sequence_process(
  const char     *const seqA_array,
  const char     *const seqB_array,
  const size_t   *const startsA,
  const size_t   *const startsB,
  short          *      seqA_align_begin,
  short          *      seqA_align_end,
  short          *      seqB_align_begin,
  short          *      seqB_align_end,
  short          *const top_scores,
  const short           startGap,
  const short           extendGap,
  const short *const    scoring_matrix,
  const short *const    encoding_matrix
){
  const int   block_Id  = blockIdx.x;
  const int   thread_Id = threadIdx.x;
  const short laneId    = threadIdx.x%32;
  const short warpId    = threadIdx.x/32;

  //Only used by DNA sequencing
  const short matchScore    = scoring_matrix[0];
  const short misMatchScore = scoring_matrix[1];

  //Determine sequence lengths
  unsigned lengthSeqA;
  unsigned lengthSeqB;
  if(DIR==Direction::FORWARD){
    lengthSeqA = startsA[block_Id+1] - startsA[block_Id];
    lengthSeqB = startsB[block_Id+1] - startsB[block_Id];
  } else {
    lengthSeqA = seqA_align_end[block_Id];
    lengthSeqB = seqB_align_end[block_Id];
  }

  if(lengthSeqA==0 || lengthSeqB==0){
    if(DIR==Direction::FORWARD){
      seqB_align_end[block_Id] = -1;
      seqA_align_end[block_Id] = -1;
      top_scores[block_Id]     = -1;
    } else {
      seqB_align_begin[block_Id] = -1; //newlengthSeqB
      seqA_align_begin[block_Id] = -1; //newlengthSeqA
    }
    return;
  }

  const char *seqA = &seqA_array[startsA[block_Id]-startsA[0]];
  const char *seqB = &seqB_array[startsB[block_Id]-startsB[0]];

  // We arbitrarily decide that Sequence A will always be shorter (or equal to)
  // Sequence B. If this isn't the case, we swap things around to make the code
  // below simpler.
  if(lengthSeqA>lengthSeqB){
    thrust::swap(lengthSeqA,       lengthSeqB      );
    thrust::swap(seqA_align_begin, seqB_align_begin);
    thrust::swap(seqA_align_end,   seqB_align_end  );
    thrust::swap(seqA,             seqB            );
  }

  extern __shared__ char is_valid_array[];
  char*                  is_valid = &is_valid_array[0];

  // shared memory space for storing longer of the two strings
  memset(is_valid, 0, lengthSeqA);
  is_valid += lengthSeqA;
  memset(is_valid, 1, lengthSeqA);
  is_valid += lengthSeqA;
  memset(is_valid, 0, lengthSeqA);

  const auto pos = (DIR==Direction::FORWARD) ? thread_Id : (lengthSeqA - 1) - thread_Id;
  const char myColumnChar = (thread_Id < lengthSeqA) ? seqA[pos] : -1;   // read only once

  //Used only by RNA. Has a length of 1 for DNA because length 0 is not allowed.
  __shared__ short sh_aa_encoding[(DT==DataType::RNA)?ENCOD_MAT_SIZE:1];// length = 91
  __shared__ short sh_aa_scoring [(DT==DataType::RNA)?SCORE_MAT_SIZE:1];

  if(DT==DataType::RNA){
    const auto max_threads = blockDim.x;
    for(int p = thread_Id; p < SCORE_MAT_SIZE; p+=max_threads){
      sh_aa_scoring[p] = scoring_matrix[p];
    }
    for(int p = thread_Id; p < ENCOD_MAT_SIZE; p+=max_threads){
      sh_aa_encoding[p] = encoding_matrix[p];
    }
  }

  __shared__ short sh_prev_E[32]; // one such element is required per warp
  __shared__ short sh_prev_H[32];
  __shared__ short sh_prev_prev_H[32];

  __shared__ short local_spill_prev_E[1024]; // each threads local spill,
  __shared__ short local_spill_prev_H[1024];
  __shared__ short local_spill_prev_prev_H[1024];

  Cell curr;
  Cell prev;
  Cell pprev;

  int   i            = 1;
  short thread_max   = 0; // to maintain the thread max score
  short thread_max_i = 0; // to maintain the DP coordinate i for the longer string
  short thread_max_j = 0;// to maintain the DP cooirdinate j for the shorter string

  //We don't need to sync before the loop since none of the shared memory that
  //is written above is used before the first sync point within the loop.

  // iterate for the number of anti-diagonals
  for(int diag = 0; diag < lengthSeqA + lengthSeqB - 1; diag++)
  {
    // Move the pointer to left by 1 if the condition is true
    is_valid -= (diag < lengthSeqA || diag >= lengthSeqB);

    // Value exchange happens here to setup registers for next iteration
    pprev = prev;
    prev  = curr;
    curr  = {0,0,0};

    if(laneId == 31)
    { // if you are the last thread in your warp then spill your values to shmem
      sh_prev_E[warpId] = prev.E;
      sh_prev_H[warpId] = prev.H;
      sh_prev_prev_H[warpId] = pprev.H;
    }

    if(diag >= lengthSeqB)
    { // if you are invalid in this iteration, spill your values to shmem
      local_spill_prev_E[thread_Id] = prev.E;
      local_spill_prev_H[thread_Id] = prev.H;
      local_spill_prev_prev_H[thread_Id] = pprev.H;
    }

    __syncthreads(); // this is needed so that all the shmem writes are completed.

    if(thread_Id < lengthSeqA && is_valid[thread_Id])
    {
      const unsigned mask = __ballot_sync(__activemask(), (thread_Id < lengthSeqA) && is_valid[thread_Id]);
      const short fVal  = prev.F + extendGap;
      const short hfVal = prev.H + startGap;
      const short valeShfl  = __shfl_sync(mask, prev.E, laneId - 1, 32);
      const short valheShfl = __shfl_sync(mask, prev.H, laneId - 1, 32);
      short eVal  = 0;
      short heVal = 0;

      if(diag >= lengthSeqB) // when the previous thread has phased out, get value from shmem
      {
        eVal  = local_spill_prev_E[thread_Id - 1] + extendGap;
        heVal = local_spill_prev_H[thread_Id - 1] + startGap;
      }
      else
      {
        eVal  = ((warpId !=0 && laneId == 0)?sh_prev_E[warpId-1]: valeShfl) + extendGap;
        heVal = ((warpId !=0 && laneId == 0)?sh_prev_H[warpId-1]:valheShfl) + startGap;
      }

      if(warpId == 0 && laneId == 0) // make sure that values for lane 0 in warp 0 is not undefined
      {
        eVal = 0;
        heVal = 0;
      }
      curr.F = max(fVal, hfVal);
      curr.E = max(eVal, heVal);

      const short testShufll = __shfl_sync(mask, pprev.H, laneId - 1, 32);
      short final_prev_prev_H = 0;

      if(diag >= lengthSeqB)
      {
        final_prev_prev_H = local_spill_prev_prev_H[thread_Id - 1];
      }
      else
      {
        final_prev_prev_H =(warpId !=0 && laneId == 0)?sh_prev_prev_H[warpId-1]:testShufll;
      }

      if(warpId == 0 && laneId == 0) final_prev_prev_H = 0;

      short diag_score;
      const int diag_pos = (DIR==Direction::FORWARD) ? i-1 : lengthSeqB-i;
      if(DT==DataType::DNA){
        diag_score = final_prev_prev_H + ((seqB[diag_pos] == myColumnChar) ? matchScore : misMatchScore);
      } else {
        const short mat_index_q = sh_aa_encoding[seqB[diag_pos]]; //encoding_matrix
        const short mat_index_r = sh_aa_encoding[myColumnChar];
        const short add_score = sh_aa_scoring[mat_index_q*24 + mat_index_r]; // doesnt really matter in what order these indices are used, since the scoring table is symmetrical

        diag_score = final_prev_prev_H + add_score;
      }

      curr.H = findMaxFour(diag_score, curr.F, curr.E, 0);

      if(DIR==Direction::FORWARD){
        thread_max_i = (thread_max >= curr.H) ? thread_max_i : i;
        thread_max_j = (thread_max >= curr.H) ? thread_max_j : thread_Id + 1;
      } else {
        thread_max_i = (thread_max >= curr.H) ? thread_max_i : lengthSeqB - i;            // begin_A (longer string)
        thread_max_j = (thread_max >= curr.H) ? thread_max_j : lengthSeqA - thread_Id -1; // begin_B (shorter string)
      }
      thread_max   = (thread_max >= curr.H) ? thread_max : curr.H;

      i++;
    }
    __syncthreads(); // why do I need this? commenting it out breaks it
  }

  thread_max = blockShuffleReduce_with_index<DIR>(thread_max, thread_max_i, thread_max_j, lengthSeqA);  // thread 0 will have the correct values

  if(thread_Id == 0)
  {
    if(DIR==Direction::FORWARD){
      seqB_align_end[block_Id] = thread_max_i;
      seqA_align_end[block_Id] = thread_max_j;
      top_scores[block_Id]     = thread_max;
    } else {
      seqB_align_begin[block_Id] = thread_max_i; //newlengthSeqB
      seqA_align_begin[block_Id] = thread_max_j; //newlengthSeqA
    }
  }
}



template<DataType DT>
inline __global__ void
sequence_process_forward_and_reverse(
  const char     *const seqA_array,
  const char     *const seqB_array,
  const size_t   *const startsA,
  const size_t   *const startsB,
  short          *      seqA_align_begin,
  short          *      seqA_align_end,
  short          *      seqB_align_begin,
  short          *      seqB_align_end,
  short          *const top_scores,
  const short           startGap,
  const short           extendGap,
  const short *const    scoring_matrix,
  const short *const    encoding_matrix
){
  sequence_process<DT, Direction::FORWARD>(seqA_array,seqB_array,startsA,startsB,seqA_align_begin,seqA_align_end,seqB_align_begin,seqB_align_end,top_scores,startGap,extendGap,scoring_matrix,encoding_matrix);
  sequence_process<DT, Direction::REVERSE>(seqA_array,seqB_array,startsA,startsB,seqA_align_begin,seqA_align_end,seqB_align_begin,seqB_align_end,top_scores,startGap,extendGap,scoring_matrix,encoding_matrix);
}

}
