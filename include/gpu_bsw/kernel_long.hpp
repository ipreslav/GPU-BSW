#pragma once

#include <gpu_bsw/kernel_utilities.hpp>

#include <thrust/swap.h>

#include <cassert>
#include <cstdint>
#include <cstdlib>

namespace gpu_bsw {

enum class DiagType {
  PREMAIN,
  MAIN2,
  POSTMAIN
};



///Return the length of the requested diagonal
///
///@param diagonal_i Diagonal to get the length of
///@param m          Number of rows in the matrix (must be <=n)
///@param n          Number of columns in the matrix (must be >=m)
///
///@return Length of the requested diagonal
template<DiagType diag_type>
inline __host__ __device__ int diagonal_length(const int diagonal_i, const int m, const int n){
  assert(m<=n);
  assert(0<=diagonal_i && diagonal_i<m+n-1);
  assert(diag_type!=DiagType::PREMAIN  || diagonal_i<m);
  assert(diag_type!=DiagType::MAIN2    || diagonal_i==m);
  assert(diag_type!=DiagType::POSTMAIN || diagonal_i>m);

  if(diag_type==DiagType::PREMAIN){
    return diagonal_i+1;
  } else {
    return min(m, m+n-(diagonal_i+1));
  }
}



//We have the following rules, where i and j are the row,column of the matrix
//and D is the diagonal index.
//PREMAIN            MAIN2             POSTMAIN
//E[i  ][j-1] = D-1  E[i  ][j-1] = D   E[i  ][j-1] = D
//F[i-1][j  ] = D    F[i-1][j  ] = D+1 F[i-1][j  ] = D+1
//H[i-1][j-1] = D-1  H[i-1][j-1] = D   H[i-1][j-1] = D+1
//H[i-1][j  ] = D    H[i-1][j  ] = D+1 H[i-1][j  ] = D+1
//H[i  ][j-1] = D-1  H[i  ][j-1] = D   H[i  ][j-1] = D

inline __device__ void
walk_pre_main_diagonal(
  const int          diagonal_i,
  const char         seq_a[],
  const char         seq_b[],
        short        pE[],
        short        pF[],
        short        ppH[],
        short        pH[],
  const int          len_seq_a,
  const int          len_seq_b,
  const short        gap_init,
  const short        gap_extend,
  const short scoring_matrix[],
  const short encoding_matrix[]
){
  const auto  tid     = threadIdx.x;
  const short lane_id = threadIdx.x%32;
  const auto  dlen    = diagonal_i+1;
  //We're about to do a grid stride loop along an anti-diagonal of the matrix.
  //Thread zero is on the bottom left of the diagonal.

  //Storage for the threads is shifted one to the right so that there's always a
  //-1 location, eliminating if statements. Storage is always one longer than
  //necessary to avoid edge effects.

  short Hup = 0;

  for(int d=tid;d<dlen;d+=blockDim.x){
    const auto seq_a_char = seq_a[dlen-1-d];
    const auto seq_b_char = seq_b[d];

    //Calculate new scores
    const auto E = (d>0)      ? max(pE[d-1+1]+gap_extend, pH[d-1+1]+gap_init) : 0;
    const auto F = (d<dlen-1) ? max(pF[d  +1]+gap_extend, pH[d  +1]+gap_init) : 0;
    const auto char_match = (seq_a_char==seq_b_char)?scoring_matrix[0]:scoring_matrix[1];
    const auto H = findMaxFour(0, E, F, ppH[d-1+1] + char_match);

    printf("A[%d] = %c\n", d, seq_a_char);
    printf("B[%d] = %c\n", d, seq_b_char);
    printf("fMF %d %d %d %d\n", 0, E, F, char_match);
    printf("E[%d]  = %d\n", d, E);
    printf("F[%d]  = %d\n", d, F);
    printf("H[%d]  = %d\n", d, H);
    printf("pH[%d] = %d\n", d, pH[d-1+1]);
    printf("ppH[%d] = %d\n", d, ppH[d-1+1]);
    printf("lt[%d]  = %d\n", d, ppH[d-1+1] + char_match);
    printf("cm[%d]  = %d\n", d, char_match);

    //Hup for the last thread is the same as Hleft for the first thread. Since
    //we've now calculated scores we need to move
    if(dlen>32){
      if(lane_id==0){
        pH[d-1] = Hup;
      }
      Hup = __shfl_down_sync(0x80000001, pH[d], 31);
    }

    ppH[d+1] = pH[d+1];
    pH [d+1] = H;
    pE [d+1] = E;
    pF [d+1] = F;

    __syncwarp();
  }
}

/*
template<DiagType diag_type>
inline __device__ void
diagonal_walk(
  const int          diagonal_i,
  const char  *const seq_a,
  const char  *const seq_b,
  const short *const pE,
  const short *const pF,
  const short *const ppH,
  const short *const pH,
  const int          len_seq_a,
  const int          len_seq_b,
  const short        gap_init,
  const short        gap_extend,
  const short *const scoring_matrix,
  const short *const encoding_matrix
){
  const auto  tid        = threadIdx.x;
  const short lane_id     = threadIdx.x%32;
  const auto dlen = diagonal_length<diag_type>(diagonal_i, len_seq_a, len_seq_b);

  //We have the following rules, where i and j are the row,column of the matrix
  //and D is the diagonal index.
  //PREMAIN            MAIN2             POSTMAIN
  //E[i  ][j-1] = D-1  E[i  ][j-1] = D   E[i  ][j-1] = D
  //F[i-1][j  ] = D    F[i-1][j  ] = D+1 F[i-1][j  ] = D+1
  //H[i-1][j-1] = D-1  H[i-1][j-1] = D   H[i-1][j-1] = D+1
  //H[i-1][j  ] = D    H[i-1][j  ] = D+1 H[i-1][j  ] = D+1
  //H[i  ][j-1] = D-1  H[i  ][j-1] = D   H[i  ][j-1] = D

  //We're about to do a grid stride loop along an anti-diagonal of the matrix.
  //Thread zero is on the bottom left of the diagonal.
  for(int d=tid;d<dlen;d+=blockDim.x){
    const auto seq_a_char = seq_a[d];
    const auto seq_b_char = seq_b[d]; //TODO

    short E;
    short F;
    short H;
    if(diag_type==DiagType::PREMAIN){
      E = std::max(pE[d-1]+gap_extend, ppH[d-1]+gap_init);
      F = std::max(pF[d  ]+gap_extend, ppH[d  ]+gap_init);
    } else {
      E = std::max(pE[d  ]+gap_extend, ppH[d  ]+gap_init);
      F = std::max(pF[d+1]+gap_extend, ppH[d+1]+gap_init);
    }

    switch(diag_type){
      case DiagType::PREMAIN:  H = findMaxFour(0, E, F, pH[d-1]+char_cmp_score); break;
      case DiagType::MAIN2:    H = findMaxFour(0, E, F, pH[d  ]+char_cmp_score); break;
      case DiagType::POSTMAIN: H = findMaxFour(0, E, F, pH[d+1]+char_cmp_score); break;
    }

    if(lane_id==31){
      Hup = pH[d];
      Fup = pF[d];

      const auto Hup = __shfl_down_sync(0x80000001, H[-1], 31);
      const auto Fup = __shfl_down_sync(0x80000001, F[-1], 31);
    }

    ppH[d] = pH[d];
    pH[d]  = H;
    pE[d]  = E;
    pF[d]  = F;

    __syncthreads();
  }
}
*/


template<DataType DT, Direction DIR>
inline __device__ void
sequence_process_long(
  const char     *const seqA_array,
  const char     *const seqB_array,
  const size_t   *const startsA,
  const size_t   *const startsB,
  short          *      seqA_align_begin,
  short          *      seqA_align_end,
  short          *      seqB_align_begin,
  short          *      seqB_align_end,
  short          *const top_scores,
  const short           gap_init,
  const short           gap_extend,
  const short *const    scoring_matrix,
  const short *const    encoding_matrix
){
  const auto  block_id   = blockIdx.x;
  const auto  block_size = blockDim.x;
  const auto  tid        = threadIdx.x;
  const short lane_id     = threadIdx.x%32;
  const short warpId     = threadIdx.x/32;

  //Only used by DNA sequencing
  const short matchScore    = scoring_matrix[0];
  const short misMatchScore = scoring_matrix[1];

  //Determine sequence lengths
  unsigned len_seq_a;
  unsigned len_seq_b;
  if(DIR==Direction::FORWARD){
    len_seq_a = startsA[block_id+1] - startsA[block_id];
    len_seq_b = startsB[block_id+1] - startsB[block_id];
  } else {
    len_seq_a = seqA_align_end[block_id];
    len_seq_b = seqB_align_end[block_id];
  }

  if(len_seq_a==0 || len_seq_b==0){
    if(DIR==Direction::FORWARD){
      seqB_align_end[block_id] = -1;
      seqA_align_end[block_id] = -1;
      top_scores[block_id]     = -1;
    } else {
      seqB_align_begin[block_id] = -1; //newlengthSeqB
      seqA_align_begin[block_id] = -1; //newlengthSeqA
    }
    return;
  }

  const char *seqA_ptr = &seqA_array[startsA[block_id]-startsA[0]];
  const char *seqB_ptr = &seqB_array[startsB[block_id]-startsB[0]];

  // We arbitrarily decide that Sequence A will always be shorter (or equal to)
  // Sequence B. If this isn't the case, we swap things around to make the code
  // below simpler.
  if(len_seq_a>len_seq_b){
    thrust::swap(len_seq_a,        len_seq_b       );
    thrust::swap(seqA_align_begin, seqB_align_begin);
    thrust::swap(seqA_align_end,   seqB_align_end  );
    thrust::swap(seqA_ptr,         seqB_ptr        );
  }

  //Copy shorter sequence into memory with grid-strided loop
  __shared__ char seqA[1024];
  __shared__ char seqB[1024];
  for(size_t i=tid;i<len_seq_a;i+=block_size){
    seqA[i] = seqA_ptr[i];
    seqB[i] = seqB_ptr[i];
  }

  //Storage for antidiagonals
  __shared__ short ppH[1024];
  __shared__ short pH [1024];
  __shared__ short pE [1024];
  __shared__ short pF [1024];

  const auto pos = (DIR==Direction::FORWARD) ? tid : (len_seq_a - 1) - tid;
  const char myColumnChar = (tid < len_seq_a) ? seqA[pos] : -1;   // read only once

  //Used only by RNA. Has a length of 1 for DNA because length 0 is not allowed.
  __shared__ short sh_aa_encoding[(DT==DataType::RNA)?ENCOD_MAT_SIZE:1];
  __shared__ short sh_aa_scoring [(DT==DataType::RNA)?SCORE_MAT_SIZE:1];

  if(DT==DataType::RNA){
    for(auto p = tid; p < SCORE_MAT_SIZE; p+=block_size){
      sh_aa_scoring[p] = scoring_matrix[p];
    }
    for(auto p = tid; p < ENCOD_MAT_SIZE; p+=block_size){
      sh_aa_encoding[p] = encoding_matrix[p];
    }
  }

  int   i            = 1;
  short thread_max   = 0; // to maintain the thread max score
  short thread_max_i = 0; // to maintain the DP coordinate i for the longer string
  short thread_max_j = 0; // to maintain the DP cooirdinate j for the shorter string

  //We don't need to sync before the loop since none of the shared memory that
  //is written above is used before the first sync point within the loop.

  const auto num_diagonals = len_seq_a + len_seq_b - 1;

  for(int diagonal_i=0;diagonal_i<len_seq_a;diagonal_i++){
    walk_pre_main_diagonal(diagonal_i, seqA, seqB, pE, pF, ppH, pH, len_seq_a, len_seq_b, gap_init, gap_extend, scoring_matrix, encoding_matrix);
  }

  // diagonal_walk<DiagType::MAIN2>();

  // for(int diag=len_seq_a+1;diag<num_diagonals;diag++){
  //   diagonal_walk<DiagType::POSTMAIN>();
  // }


  // // iterate for the number of anti-diagonals
  // for(int diag = 0; diag < num_diagonals; diag++)
  // {
  //   __syncthreads(); // this is needed so that all the shmem writes are completed.

  //   if(tid < len_seq_a && is_valid[tid])
  //   {
  //     const unsigned mask = __ballot_sync(__activemask(), (tid < len_seq_a) && is_valid[tid]);
  //     const short fVal  = prev.F + gap_extend;
  //     const short hfVal = prev.H + gap_init;
  //     const short valeShfl  = __shfl_sync(mask, prev.E, lane_id - 1, 32);
  //     const short valheShfl = __shfl_sync(mask, prev.H, lane_id - 1, 32);
  //     short eVal  = 0;
  //     short heVal = 0;

  //     if(diag >= len_seq_b) // when the previous thread has phased out, get value from shmem
  //     {
  //       eVal  = local_spill_prev_E[tid - 1] + gap_extend;
  //       heVal = local_spill_prev_H[tid - 1] + gap_init;
  //     }
  //     else
  //     {
  //       eVal  = ((warpId !=0 && lane_id == 0)?sh_prev_E[warpId-1]: valeShfl) + gap_extend;
  //       heVal = ((warpId !=0 && lane_id == 0)?sh_prev_H[warpId-1]:valheShfl) + gap_init;
  //     }

  //     if(warpId == 0 && lane_id == 0) // make sure that values for lane 0 in warp 0 is not undefined
  //     {
  //       eVal = 0;
  //       heVal = 0;
  //     }
  //     curr.F = max(fVal, hfVal);
  //     curr.E = max(eVal, heVal);

  //     const short testShufll = __shfl_sync(mask, pprev.H, lane_id - 1, 32);
  //     short final_prev_prev_H = 0;

  //     if(diag >= len_seq_b)
  //     {
  //       final_prev_prev_H = local_spill_prev_prev_H[tid - 1];
  //     }
  //     else
  //     {
  //       final_prev_prev_H =(warpId !=0 && lane_id == 0)?sh_prev_prev_H[warpId-1]:testShufll;
  //     }

  //     if(warpId == 0 && lane_id == 0) final_prev_prev_H = 0;

  //     short diag_score;
  //     const int diag_pos = (DIR==Direction::FORWARD) ? i-1 : len_seq_b-i;
  //     if(DT==DataType::DNA){
  //       diag_score = final_prev_prev_H + ((seqB[diag_pos] == myColumnChar) ? matchScore : misMatchScore);
  //     } else {
  //       const short mat_index_q = sh_aa_encoding[seqB[diag_pos]]; //encoding_matrix
  //       const short mat_index_r = sh_aa_encoding[myColumnChar];
  //       const short add_score = sh_aa_scoring[mat_index_q*24 + mat_index_r]; // doesnt really matter in what order these indices are used, since the scoring table is symmetrical

  //       diag_score = final_prev_prev_H + add_score;
  //     }

  //     curr.H = findMaxFour(diag_score, curr.F, curr.E, 0);

  //     if(DIR==Direction::FORWARD){
  //       thread_max_i = (thread_max >= curr.H) ? thread_max_i : i;
  //       thread_max_j = (thread_max >= curr.H) ? thread_max_j : tid + 1;
  //     } else {
  //       thread_max_i = (thread_max >= curr.H) ? thread_max_i : len_seq_b - i;            // begin_A (longer string)
  //       thread_max_j = (thread_max >= curr.H) ? thread_max_j : len_seq_a - tid -1; // begin_B (shorter string)
  //     }
  //     thread_max   = (thread_max >= curr.H) ? thread_max : curr.H;

  //     i++;
  //   }
  //   __syncthreads(); // why do I need this? commenting it out breaks it
  // }

  // thread_max = blockShuffleReduce_with_index<DIR>(thread_max, thread_max_i, thread_max_j, len_seq_a);  // thread 0 will have the correct values

  // if(tid == 0)
  // {
  //   if(DIR==Direction::FORWARD){
  //     seqB_align_end[block_id] = thread_max_i;
  //     seqA_align_end[block_id] = thread_max_j;
  //     top_scores[block_id]     = thread_max;
  //   } else {
  //     seqB_align_begin[block_id] = thread_max_i; //newlengthSeqB
  //     seqA_align_begin[block_id] = thread_max_j; //newlengthSeqA
  //   }
  // }
}



// template<DataType DT>
// inline __global__ void
// sequence_process_forward_and_reverse(
//   const char     *const seqA_array,
//   const char     *const seqB_array,
//   const size_t   *const startsA,
//   const size_t   *const startsB,
//   short          *      seqA_align_begin,
//   short          *      seqA_align_end,
//   short          *      seqB_align_begin,
//   short          *      seqB_align_end,
//   short          *const top_scores,
//   const short           gap_init,
//   const short           gap_extend,
//   const short *const    scoring_matrix,
//   const short *const    encoding_matrix
// ){
//   sequence_process<DT, Direction::FORWARD>(seqA_array,seqB_array,startsA,startsB,seqA_align_begin,seqA_align_end,seqB_align_begin,seqB_align_end,top_scores,gap_init,gap_extend,scoring_matrix,encoding_matrix);
//   sequence_process<DT, Direction::REVERSE>(seqA_array,seqB_array,startsA,startsB,seqA_align_begin,seqA_align_end,seqB_align_begin,seqB_align_end,top_scores,gap_init,gap_extend,scoring_matrix,encoding_matrix);
// }

}
