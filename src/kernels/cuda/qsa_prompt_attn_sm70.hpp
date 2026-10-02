// src/kernels/cuda/qsa_prompt_attn_sm70.hpp - the Volta (sm_70) entry of the prompt attention (qsa_prompt_attn.hpp).
#pragma once

#include "strata/kernels/qsa_prompt_attn.hpp"

#include <cuda_runtime.h>

namespace strata::kernels {

/// Same contract as `qsa_prompt_attn_batch`: the Volta kernel (nvcuda::wmma m16n16k16) for sm_70.  Returns false
/// (nothing launched) on the pools or geometry the kernel does not take, so the caller uses the old kernel.
bool qsa_prompt_attn_sm70_batch(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps,
                                int64_t cap, const QsaShapes& s, float* attn, int64_t n_q, cudaStream_t stream);

}  // namespace strata::kernels
