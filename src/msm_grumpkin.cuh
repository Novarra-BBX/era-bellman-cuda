#pragma once
#include "msm_grumpkin_kernels.cuh"

// Grumpkin G1 MSM — top-level API mirroring msm::execute_async.
// Scalars are in fd_p (BN254 Fq = Grumpkin Fr).
// Bases are Grumpkin affine points (coordinates in fd_q = BN254 Fr = Grumpkin Fq).
namespace msm_grumpkin {

cudaError_t set_up();

struct execution_configuration {
  cudaMemPool_t mem_pool;
  cudaStream_t stream;
  point_affine *bases;
  fd_p::storage *scalars;   // Grumpkin Fr = BN254 Fq = fd_p
  point_jacobian *results;
  unsigned log_scalars_count;
  cudaEvent_t h2d_copy_finished;
  cudaHostFn_t h2d_copy_finished_callback;
  void *h2d_copy_finished_callback_data;
  cudaEvent_t d2h_copy_finished;
  cudaHostFn_t d2h_copy_finished_callback;
  void *d2h_copy_finished_callback_data;
  bool force_min_chunk_size;
  unsigned log_min_chunk_size;
  bool force_max_chunk_size;
  unsigned log_max_chunk_size;
};

cudaError_t execute_async(const execution_configuration &configuration);

cudaError_t tear_down();

} // namespace msm_grumpkin
