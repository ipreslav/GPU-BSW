#include <albp/error_handling.hpp>
#include <albp/simple_sw.hpp>

#include <gpu_bsw/kernel_long.hpp>

#include <thrust/device_vector.h>

#include <iostream>
#include <sstream>

using namespace albp;
using namespace gpu_bsw;

template<class T>
std::ostream& operator<<(std::ostream &out, const std::vector<T> &vec){
  out<<"[";
  for(const auto &x: vec) out<<x<<" ";
  out<<"]";
  return out;
}

template<class T>
std::ostream& operator<<(std::ostream &out, const thrust::device_vector<T> &vec){
  out<<"[";
  for(const auto &x: vec) out<<x<<" ";
  out<<"]";
  return out;
}

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
  for(auto i=threadIdx.x;i<64;i+=blockDim.x){
    pE [i+1] = in_pE[i];
    pF [i+1] = in_pF[i];
    pH [i+1] = in_pH[i];
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


int main(){
  const std::string seq_a_str = "CGGAGGAGACCTTGCTGTAATTCTGTGCTCTGGAATAACTTTCCTCAATACTATCATGGCTGG";
  const std::string seq_b_str = "CATGTTTGGTACTATGGCTGGCCCGAACGTACCTAAATTTGACTTCAGCACATATAACCCCCGTGTTT";

  const int gap_init = -6;
  const int gap_extend = -1;
  thrust::device_vector<short> scoring_matrix  = std::vector<short>{1,-4};
  thrust::device_vector<short> encoding_matrix = std::vector<short>{0,0};

  const auto ssw = simple_smith_waterman(seq_a_str, seq_b_str, gap_init, gap_extend, scoring_matrix[0], scoring_matrix[1]);
  const auto vec_set_size = [&](std::vector<int> vec) { vec.resize(64, 0); return vec; };

  print_matrix("E", ssw.seqa, ssw.seqb, ssw.E);

  thrust::device_vector<char> seq_a(seq_a_str.begin(), seq_a_str.end());
  thrust::device_vector<char> seq_b(seq_b_str.begin(), seq_b_str.end());

  for(int diag=0;diag<(int)seq_a.size();diag++){
    thrust::device_vector<short> pE  = vec_set_size(extract_diagonal(ssw.E, diag-1));
    thrust::device_vector<short> pF  = vec_set_size(extract_diagonal(ssw.F, diag-1));
    thrust::device_vector<short> pH  = vec_set_size(extract_diagonal(ssw.H, diag-1));
    thrust::device_vector<short> ppH = vec_set_size(extract_diagonal(ssw.H, diag-2));

    test_diagonal_walk<<<1,32>>>(
      0,
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

    std::cout<<"\n###"<<diag<<std::endl;
    std::cout<<pE<<std::endl;
    std::cout<<pF<<std::endl;
    std::cout<<pH<<std::endl;
    std::cout<<ppH<<std::endl;

    std::cout<<(pE ==vec_set_size(extract_diagonal(ssw.E, diag)))<<std::endl;
    std::cout<<(pF ==vec_set_size(extract_diagonal(ssw.F, diag)))<<std::endl;
    std::cout<<(pH ==vec_set_size(extract_diagonal(ssw.H, diag)))<<std::endl;
    std::cout<<(ppH==vec_set_size(extract_diagonal(ssw.H, diag)))<<std::endl;
  }
}
