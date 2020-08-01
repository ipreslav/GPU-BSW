#include <gpu_bsw/driver.hpp>

void initialize_alignments(gpu_bsw_driver::alignment_results *alignments, int max_alignments){
    alignments->ref_begin   = albp::PageLockedMalloc<short>(max_alignments);
    alignments->ref_end     = albp::PageLockedMalloc<short>(max_alignments);
    alignments->query_begin = albp::PageLockedMalloc<short>(max_alignments);
    alignments->query_end   = albp::PageLockedMalloc<short>(max_alignments);
    alignments->top_scores  = albp::PageLockedMalloc<short>(max_alignments);
}

void free_alignments(gpu_bsw_driver::alignment_results *alignments){
       ALBP_CUDA_ERROR_CHECK(cudaFreeHost(alignments->ref_begin));
       ALBP_CUDA_ERROR_CHECK(cudaFreeHost(alignments->ref_end));
       ALBP_CUDA_ERROR_CHECK(cudaFreeHost(alignments->query_begin));
       ALBP_CUDA_ERROR_CHECK(cudaFreeHost(alignments->query_end));
       ALBP_CUDA_ERROR_CHECK(cudaFreeHost(alignments->top_scores));
}

void asynch_mem_copies_htd(gpu_alignments* gpu_data, unsigned* offsetA_h, unsigned* offsetB_h, char* strA, char* strA_d, char* strB, char* strB_d, unsigned half_length_A,
unsigned half_length_B, unsigned totalLengthA, unsigned totalLengthB, int sequences_per_stream, int sequences_stream_leftover, const std::array<cudaStream_t,2> &streams_cuda){

        ALBP_CUDA_ERROR_CHECK(cudaMemcpyAsync(gpu_data->offset_ref_gpu, offsetA_h, (sequences_per_stream) * sizeof(int),
        cudaMemcpyHostToDevice,streams_cuda[0]));
        ALBP_CUDA_ERROR_CHECK(cudaMemcpyAsync(gpu_data->offset_ref_gpu + sequences_per_stream, offsetA_h + sequences_per_stream,
        (sequences_per_stream + sequences_stream_leftover) * sizeof(int), cudaMemcpyHostToDevice,streams_cuda[1]));

        ALBP_CUDA_ERROR_CHECK(cudaMemcpyAsync(gpu_data->offset_query_gpu, offsetB_h, (sequences_per_stream) * sizeof(int),
        cudaMemcpyHostToDevice,streams_cuda[0]));
        ALBP_CUDA_ERROR_CHECK(cudaMemcpyAsync(gpu_data->offset_query_gpu + sequences_per_stream, offsetB_h + sequences_per_stream,
        (sequences_per_stream + sequences_stream_leftover) * sizeof(int), cudaMemcpyHostToDevice,streams_cuda[1]));


        ALBP_CUDA_ERROR_CHECK(cudaMemcpyAsync(strA_d, strA, half_length_A * sizeof(char),
                              cudaMemcpyHostToDevice,streams_cuda[0]));
        ALBP_CUDA_ERROR_CHECK(cudaMemcpyAsync(strA_d + half_length_A, strA + half_length_A, (totalLengthA - half_length_A) * sizeof(char),
                              cudaMemcpyHostToDevice,streams_cuda[1]));

        ALBP_CUDA_ERROR_CHECK(cudaMemcpyAsync(strB_d, strB, half_length_B * sizeof(char),
                              cudaMemcpyHostToDevice,streams_cuda[0]));
        ALBP_CUDA_ERROR_CHECK(cudaMemcpyAsync(strB_d + half_length_B, strB + half_length_B, (totalLengthB - half_length_B) * sizeof(char),
                              cudaMemcpyHostToDevice,streams_cuda[1]));

}

void asynch_mem_copies_dth_mid(gpu_alignments* gpu_data, short* alAend, short* alBend, int sequences_per_stream, int sequences_stream_leftover, const std::array<cudaStream_t,2> &streams_cuda){
            ALBP_CUDA_ERROR_CHECK(cudaMemcpyAsync(alAend, gpu_data->ref_end_gpu, sequences_per_stream * sizeof(short),
                cudaMemcpyDeviceToHost, streams_cuda[0]));
            ALBP_CUDA_ERROR_CHECK(cudaMemcpyAsync(alAend + sequences_per_stream, gpu_data->ref_end_gpu + sequences_per_stream,
                (sequences_per_stream + sequences_stream_leftover) * sizeof(short), cudaMemcpyDeviceToHost, streams_cuda[1]));

            ALBP_CUDA_ERROR_CHECK(cudaMemcpyAsync(alBend, gpu_data->query_end_gpu, sequences_per_stream * sizeof(short), cudaMemcpyDeviceToHost, streams_cuda[0]));
            ALBP_CUDA_ERROR_CHECK(cudaMemcpyAsync(alBend + sequences_per_stream, gpu_data->query_end_gpu + sequences_per_stream, (sequences_per_stream + sequences_stream_leftover) * sizeof(short),
                cudaMemcpyDeviceToHost, streams_cuda[1]));
}

void asynch_mem_copies_dth(gpu_alignments* gpu_data, short* alAbeg, short* alBbeg, short* top_scores_cpu, int sequences_per_stream, int sequences_stream_leftover, const std::array<cudaStream_t,2> &streams_cuda){
           ALBP_CUDA_ERROR_CHECK(cudaMemcpyAsync(alAbeg, gpu_data->ref_start_gpu, sequences_per_stream * sizeof(short),
                                  cudaMemcpyDeviceToHost, streams_cuda[0]));
          ALBP_CUDA_ERROR_CHECK(cudaMemcpyAsync(alAbeg + sequences_per_stream, gpu_data->ref_start_gpu + sequences_per_stream, (sequences_per_stream + sequences_stream_leftover) * sizeof(short),
                                  cudaMemcpyDeviceToHost, streams_cuda[1]));

          ALBP_CUDA_ERROR_CHECK(cudaMemcpyAsync(alBbeg, gpu_data->query_start_gpu, sequences_per_stream * sizeof(short),
                          cudaMemcpyDeviceToHost, streams_cuda[0]));
          ALBP_CUDA_ERROR_CHECK(cudaMemcpyAsync(alBbeg + sequences_per_stream, gpu_data->query_start_gpu + sequences_per_stream, (sequences_per_stream + sequences_stream_leftover) * sizeof(short),
                          cudaMemcpyDeviceToHost, streams_cuda[1]));

          ALBP_CUDA_ERROR_CHECK(cudaMemcpyAsync(top_scores_cpu, gpu_data->scores_gpu, sequences_per_stream * sizeof(short),
                          cudaMemcpyDeviceToHost, streams_cuda[0]));
          ALBP_CUDA_ERROR_CHECK(cudaMemcpyAsync(top_scores_cpu + sequences_per_stream, gpu_data->scores_gpu + sequences_per_stream,
          (sequences_per_stream + sequences_stream_leftover) * sizeof(short), cudaMemcpyDeviceToHost, streams_cuda[1]));

}
