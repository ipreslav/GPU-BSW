#include <albp/doctest.hpp>
#include <albp/doctest_vectors.hpp>
#include <albp/error_handling.hpp>
#include <albp/random.hpp>
#include <albp/simple_sw.hpp>

#include <gpu_bsw/kernel_long.hpp>

#include <thrust/device_vector.h>

#include <random>
#include <sstream>

using namespace albp;
using namespace gpu_bsw;

__global__ void test_diagonal_walk(
  const int   diagonal_i,
  const char  *seq_a,
  const char  *seq_b,
  short       *in_pE,
  short       *in_pF,
  short       *in_pH,
  short       *in_ppH,
  const int   len_seq_a,
  const int   len_seq_b,
  const int   gap_init,
  const int   gap_extend,
  const short *scoring_matrix,
  const short *encoding_matrix
){
  __shared__ short pE [64+2];
  __shared__ short pF [64+2];
  __shared__ short pH [64+2];
  __shared__ short ppH[64+2];

  //Copy into shared memory
  pE [0] = 0; pE [65] = 0;
  pF [0] = 0; pF [65] = 0;
  pH [0] = 0; pH [65] = 0;
  ppH[0] = 0; ppH[65] = 0;
  for(auto i=threadIdx.x;i<64;i+=blockDim.x){
    pE [i+1] = in_pE [i];
    pF [i+1] = in_pF [i];
    pH [i+1] = in_pH [i];
    ppH[i+1] = in_ppH[i];
  }
  __syncthreads();

  walk_pre_main_diagonal(diagonal_i, seq_a, seq_b, pE, pF, ppH, pH, len_seq_a, len_seq_b, gap_init, gap_extend, scoring_matrix, encoding_matrix);

  //Copy out of shared memory
  __syncthreads();
  for(auto i=threadIdx.x;i<64;i+=blockDim.x){
    in_pE[i]  = pE [i+1];
    in_pF[i]  = pF [i+1];
    in_pH[i]  = pH [i+1];
    in_ppH[i] = ppH[i+1];
  }
  __syncthreads();
}



TEST_CASE("[diagonal] Pre main diagonal walk"){
  const int seq_a_len = 40;
  const int seq_b_len = 60;

  std::mt19937 gen;

  const auto seq_a_str = random_dna_sequence(seq_a_len, gen);
  const auto seq_b_str = random_dna_sequence(seq_b_len, gen);

  std::cerr<<"seq_a = "<<seq_a_str<<std::endl;
  std::cerr<<"seq_b = "<<seq_b_str<<std::endl;

  const int gap_init = -6;
  const int gap_extend = -1;
  thrust::device_vector<short> scoring_matrix  = std::vector<short>{1,-4};
  thrust::device_vector<short> encoding_matrix = std::vector<short>{0,0};

  const auto ssw = simple_smith_waterman(seq_a_str, seq_b_str, gap_init, gap_extend, scoring_matrix[0], scoring_matrix[1]);

  std::cerr<<ssw.E.width<<std::endl;
  std::cerr<<ssw.E.height<<std::endl;
  print_matrix("E", ssw.seqa, ssw.seqb, ssw.E);
  print_matrix("F", ssw.seqa, ssw.seqb, ssw.F);
  print_matrix("H", ssw.seqa, ssw.seqb, ssw.H);

  const auto vec_set_size = [&](std::vector<int> vec) { vec.resize(seq_a_len, 0); return vec; };

  thrust::device_vector<char> seq_a(seq_a_str.begin(), seq_a_str.end());
  thrust::device_vector<char> seq_b(seq_b_str.begin(), seq_b_str.end());

  for(int diag=0;diag<(int)seq_a.size();diag++){
    std::cerr<<"\n\ndiag ="<< diag<<std::endl;

    thrust::device_vector<short> pE  = vec_set_size(extract_diagonal(ssw.E, diag-1));
    thrust::device_vector<short> pF  = vec_set_size(extract_diagonal(ssw.F, diag-1));
    thrust::device_vector<short> pH  = vec_set_size(extract_diagonal(ssw.H, diag-1));
    thrust::device_vector<short> ppH = vec_set_size(extract_diagonal(ssw.H, diag-2));

    test_diagonal_walk<<<1,32>>>(
      diag,
      thrust::raw_pointer_cast(seq_a.data()),
      thrust::raw_pointer_cast(seq_b.data()),
      thrust::raw_pointer_cast(pE.data()),
      thrust::raw_pointer_cast(pF.data()),
      thrust::raw_pointer_cast(pH.data()),
      thrust::raw_pointer_cast(ppH.data()),
      seq_a.size(),
      seq_b.size(),
      gap_init,
      gap_extend,
      thrust::raw_pointer_cast(scoring_matrix.data()),
      thrust::raw_pointer_cast(encoding_matrix.data())
    );
    ALBP_CUDA_ERROR_CHECK(cudaGetLastError());

    ALBP_CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    CHECK(pE ==vec_set_size(extract_diagonal(ssw.E, diag  )));
    CHECK(pF ==vec_set_size(extract_diagonal(ssw.F, diag  )));
    CHECK(pH ==vec_set_size(extract_diagonal(ssw.H, diag  )));
    CHECK(ppH==vec_set_size(extract_diagonal(ssw.H, diag-1)));
  }

  print_matrix("E", ssw.seqa, ssw.seqb, ssw.E);
  print_matrix("F", ssw.seqa, ssw.seqb, ssw.F);
  print_matrix("H", ssw.seqa, ssw.seqb, ssw.H);
}














TEST_CASE("diagonal_length"){
  SUBCASE("3x6"){
    // Lengths   DiagonalIndex
    // 123333    012345
    // 233332    123456
    // 333321    234567

    CHECK(1==diagonal_length<DiagType::PREMAIN> (0, 3, 6));
    CHECK(2==diagonal_length<DiagType::PREMAIN> (1, 3, 6));
    CHECK(3==diagonal_length<DiagType::PREMAIN> (2, 3, 6));
    CHECK(3==diagonal_length<DiagType::MAIN2>   (3, 3, 6));
    CHECK(3==diagonal_length<DiagType::POSTMAIN>(4, 3, 6));
    CHECK(3==diagonal_length<DiagType::POSTMAIN>(5, 3, 6));
    CHECK(2==diagonal_length<DiagType::POSTMAIN>(6, 3, 6));
    CHECK(1==diagonal_length<DiagType::POSTMAIN>(7, 3, 6));
  }

  SUBCASE("4x4"){
    // Lengths DiagonalIndex
    // 1234    0123
    // 2343    1234
    // 3432    2345
    // 4321    3456

    CHECK(1==diagonal_length<DiagType::PREMAIN> (0, 4, 4));
    CHECK(2==diagonal_length<DiagType::PREMAIN> (1, 4, 4));
    CHECK(3==diagonal_length<DiagType::PREMAIN> (2, 4, 4));
    CHECK(4==diagonal_length<DiagType::MAIN2>   (3, 4, 4));
    CHECK(3==diagonal_length<DiagType::POSTMAIN>(4, 4, 4));
    CHECK(2==diagonal_length<DiagType::POSTMAIN>(5, 4, 4));
    CHECK(1==diagonal_length<DiagType::POSTMAIN>(6, 4, 4));
  }
}
