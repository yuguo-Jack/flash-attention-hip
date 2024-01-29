/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#include "flash_runner.hpp"
#include "random_utils.h"

#include <cmath>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <cstring>
#include <exception>
#include <string>

#define ASSERT_CHECK(__cond)                             \
      do {                                               \
        const bool __cond_var = (__cond);                \
        if (!__cond_var) {                               \
          ::std::string __err_msg = ::std::string("`") + \
                #__cond + "` check failed at " +         \
                __FILE__ + ":" +                         \
                ::std::to_string(__LINE__);              \
          throw std::runtime_error(__err_msg);           \
        }                                                \
      } while (0)

#ifdef __cplusplus
extern "C" {
#endif

static thread_local std::unique_ptr<char[]> flash_attn_err_msg;

void flash_attn_set_error(const char *msg) {
  if (msg == nullptr || *msg == '\0') {
    msg = "unknown error";
  }

  auto n = strlen(msg);
  std::unique_ptr<char[]> new_err_msg(new char[n+1]);
  std::strcpy(new_err_msg.get(), msg);
  flash_attn_err_msg = std::move(new_err_msg);
}

const char *flash_attn_error() {
  return flash_attn_err_msg.get();
}

#ifdef __cplusplus
}
#endif

#define FLASHATTNLIB_BEGIN_FUNC try {
#define FLASHATTNLIB_END_FUNC } catch (::std::exception &__e) { flash_attn_set_error(__e.what()); return false; } catch (...) { flash_attn_set_error(nullptr); return false; }

#define CHECK_FWD_EXECTUABLE(__seqlen_q, __seqlen_k)                     \
      ASSERT_CHECK(batch_size > 0);                                      \
      ASSERT_CHECK(head_size % 8 == 0);                                  \
      ASSERT_CHECK(head_size <= 128);                                    \
      ASSERT_CHECK(num_heads % num_heads_k == 0);                        \
      if (attn_mask) {                                                   \
          ASSERT_CHECK(mask_dims[0] == batch_size);                      \
          ASSERT_CHECK(mask_dims[1] == 1 || mask_dims[1] == num_heads);  \
          ASSERT_CHECK(mask_dims[2] == 1 || mask_dims[2] == __seqlen_q); \
          ASSERT_CHECK(mask_dims[3] == __seqlen_k);                      \
      }

#define CHECK_BWD_EXECTUABLE(__seqlen_q, __seqlen_k)                                       \
      CHECK_FWD_EXECTUABLE(__seqlen_q, __seqlen_k)                                         

void run_mha_fwd(FlashFwdBatchedParams &params, hipStream_t stream) {
    FlashRunner flash_runner;
    flash_runner.Run(params, stream);
}

void run_mha_varlen_fwd(FlashFwdGroupedParams &params, hipStream_t stream) {
    FlashRunner flash_runner;
    flash_runner.Run(params, stream);
}

#ifdef __cplusplus
extern "C" {
#endif

bool flash_attn_fwd(const void * const q,
                    const void * const k,
                    const void * const v,
                    void * const rng_state,
                    void * const out,
                    void * const softmax_ptr,
                    void * const softmax_lse_ptr,
                    const int batch_size,
                    const int seqlen_q,
                    const int seqlen_k,
                    const int seqlen_q_rounded,
                    const int seqlen_k_rounded,
                    const int num_heads,
                    const int num_heads_k,
                    const int head_size,
                    const int head_size_rounded,
                    const float p_dropout,
                    const float softmax_scale,
                    const float softmax_unscale,
                    const bool is_causal,
                    const bool return_softmax,
                    const bool is_bf16,
                    hipStream_t stream,
                    uint64_t seed,
                    uint64_t offset,
                    const void * const attn_mask,
                    const int64_t * const mask_dims) {
    FLASHATTNLIB_BEGIN_FUNC
    const bool is_dropout = p_dropout > 0.0;
    CHECK_FWD_EXECTUABLE(seqlen_q, seqlen_k)
    ASSERT_CHECK(is_bf16 == false);
    ASSERT_CHECK(return_softmax == false);

    FlashFwdBatchedParams params(batch_size, seqlen_q, seqlen_k, num_heads,
                                num_heads_k, head_size, const_cast<void *>(q), const_cast<void *>(k),
                                const_cast<void *>(v), out, nullptr, softmax_lse_ptr, 
                                p_dropout, softmax_scale, is_causal, return_softmax);

    if (is_dropout) {
        auto philox_args = at::PhiloxCudaState(seed, offset);
        params.seeds = at::cuda::philox::unpack(philox_args);
    }

    run_mha_fwd(params, stream);
    
    return true;

    FLASHATTNLIB_END_FUNC
}

bool flash_attn_varlen_fwd(const void * const q,
                           const void * const k,
                           const void * const v,
                           const int32_t * const cu_seqlens_q,
                           const int32_t * const cu_seqlens_k,
                           void * const rng_state,
                           void * const out,
                           void * const softmax_ptr,
                           void * const softmax_lse_ptr,
                           const int batch_size,
                           const int max_seqlen_q,
                           const int max_seqlen_k,
                           const int seqlen_q_rounded,
                           const int seqlen_k_rounded,
                           const int num_heads,
                           const int num_heads_k,
                           const int head_size,
                           const int head_size_rounded,
                           const float p_dropout,
                           const float softmax_scale,
                           const float softmax_unscale,
                           const bool is_causal,
                           const bool return_softmax,
                           const bool is_bf16,
                           hipStream_t stream,
                           uint64_t seed,
                           uint64_t offset,
                           const void * const attn_mask,
                           const int64_t * const mask_dims) {
    FLASHATTNLIB_BEGIN_FUNC
    const bool is_dropout = p_dropout > 0.0;
    CHECK_FWD_EXECTUABLE(max_seqlen_q, max_seqlen_k)
    ASSERT_CHECK(is_bf16 == false);
    ASSERT_CHECK(return_softmax == false);
    
    FlashFwdGroupedParams params(batch_size, max_seqlen_q, max_seqlen_k, num_heads, num_heads_k,
                                head_size, const_cast<void *>(q), const_cast<void *>(k), const_cast<void *>(v), out, cu_seqlens_q,
                                cu_seqlens_k, std::vector<void*>(), softmax_lse_ptr, p_dropout, softmax_scale,
                                is_causal, return_softmax);

    if (is_dropout) {
        auto philox_args = at::PhiloxCudaState(seed, offset);
        params.seeds = at::cuda::philox::unpack(philox_args);
    }

    run_mha_varlen_fwd(params, stream);

    return true;
    
    FLASHATTNLIB_END_FUNC
}

bool flash_attn_bwd(const void * const dout,
                    const void * const q,
                    const void * const k,
                    const void * const v,
                    const void * const out,
                    const void * const softmax_d,
                    const void * const softmax_lse,
                    void * const rng_state,
                    void * const dq,
                    void * const dk,
                    void * const dv,
                    void * const dq_accum,
                    const int batch_size,
                    const int seqlen_q,
                    const int seqlen_k,
                    const int seqlen_q_rounded,
                    const int seqlen_k_rounded,
                    const int num_heads,
                    const int num_heads_k,
                    const int head_size,
                    const int head_size_rounded,
                    const float p_dropout,
                    const float softmax_scale,
                    const float softmax_unscale,
                    const bool is_causal,
                    const bool is_bf16,
                    const int num_splits,
                    hipStream_t stream,
                    uint64_t seed,
                    uint64_t offset,
                    const void * const attn_mask,
                    const int64_t * const mask_dims) {
    FLASHATTNLIB_BEGIN_FUNC
    const bool is_dropout = p_dropout > 0.0;
    const int mask_head_mod_size = attn_mask ? mask_dims[1] : 0;
    const int mask_seq_q_mod_size = attn_mask ? mask_dims[2] : 0;

    CHECK_BWD_EXECTUABLE(seqlen_q, seqlen_k)

    // bool loop = seqlen_k > blocksize_c;
    // TODO: change later, for now set to true for simplicity
    const bool loop = true;

    std::cout << "!!!!!!!!!!!!!!!!!!!! flashattn2 bwd is not surpport on zifang dcu for now !!!!!!!!!!!!!!!!!!!!" << std::endl;
    
    return true;
    
    FLASHATTNLIB_END_FUNC

}

bool flash_attn_varlen_bwd(const void * const dout,
                           const void * const q,
                           const void * const k,
                           const void * const v,
                           const void * const out,
                           const void * const softmax_d,
                           const void * const softmax_lse,
                           const int32_t * const cu_seqlens_q,
                           const int32_t * const cu_seqlens_k,
                           void * const rng_state,
                           void * const dq,
                           void * const dk,
                           void * const dv,
                           void * const dq_accum,
                           const int batch_size,
                           const int max_seqlen_q,
                           const int max_seqlen_k,
                           const int seqlen_q_rounded,
                           const int seqlen_k_rounded,
                           const int num_heads,
                           const int num_heads_k,
                           const int head_size,
                           const int head_size_rounded,
                           const float p_dropout,
                           const float softmax_scale,
                           const float softmax_unscale,
                           const bool is_causal,
                           const bool is_bf16,
                           const int num_splits,
                           hipStream_t stream,
                           uint64_t seed,
                           uint64_t offset,
                           const void * const attn_mask,
                           const int64_t * const mask_dims) {
    FLASHATTNLIB_BEGIN_FUNC
    const bool is_dropout = p_dropout > 0.0;
    const int mask_head_mod_size = attn_mask ? mask_dims[1] : 0;
    const int mask_seq_q_mod_size = attn_mask ? mask_dims[2] : 0;

    const bool loop = true;

    CHECK_BWD_EXECTUABLE(max_seqlen_q, max_seqlen_k)

    std::cout << "!!!!!!!!!!!!!!!!!!!! flashattn2 bwd is not surpport on zifang dcu for now !!!!!!!!!!!!!!!!!!!!" << std::endl;
    
    return true;
    
    FLASHATTNLIB_END_FUNC

}

#ifdef __cplusplus
}
#endif

