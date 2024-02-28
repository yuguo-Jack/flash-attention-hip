// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <sstream>
#include <cstring>

#include "ck/utility/common_header.hpp"
#include "ck/utility/philox_rand.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_grouped_gemm_softmax_gemm_permute.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/matrix_padder.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_batched_mha_fwd_xdl_cshuffle_v2.hpp"
#include "ck/tensor_operation/operator_transform/transform_contraction_to_gemm.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename GridwiseGemm,
          typename D0DataType,
          typename GemmAccDataType,
          typename GroupKernelArg,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename AccElementwiseOperation,
          typename B1ElementwiseOperation,
          typename CElementwiseOperation,
          bool HasMainKBlockLoop,
          bool IsDropout,
          bool IsLseStoring>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
    __launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
        kernel_grouped_gemm_softmax_gemm_xdl_cshuffle_v2(
            const void CK_CONSTANT_ADDRESS_SPACE* group_kernel_args,
            const index_t group_count,
            const index_t h_ratio,
            const AElementwiseOperation a_element_op,
            const BElementwiseOperation b_element_op,
            const AccElementwiseOperation acc_element_op,
            const B1ElementwiseOperation b1_element_op,
            const CElementwiseOperation c_element_op,
            const uint8_t p_dropout_in_uint8_t,
            const GemmAccDataType p_dropout_rescale,
            const unsigned long long seed,
            const unsigned long long offset)
{
#if(!defined(__HIP_DEVICE_COMPILE__) || defined(__gfx908__) || defined(__gfx90a__) || \
    defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__))
    __shared__ char p_shared[GridwiseGemm::GetSharedMemoryNumberOfByte()];

    const index_t block_id         = get_block_1d_id();
    const index_t global_thread_id = get_thread_global_1d_id();

    ck::philox ph(seed, global_thread_id, offset);

    const auto arg_ptr = reinterpret_cast<const GroupKernelArg*>(
        cast_pointer_to_generic_address_space(group_kernel_args));

    index_t left     = 0;
    index_t right    = group_count;
    index_t group_id = index_t((left + right) / 2);

    while(
        (!(block_id >= arg_ptr[group_id].block_start_ && block_id < arg_ptr[group_id].block_end_)))
    {
        if(block_id < arg_ptr[group_id].block_start_)
        {
            right = group_id;
        }
        else
        {
            left = group_id;
        }
        group_id = index_t((left + right) / 2);
    }

    // per-group batch offset
    const index_t num_blocks_per_batch = arg_ptr[group_id].num_blocks_per_batch_;
    const index_t g_idx                = __builtin_amdgcn_readfirstlane(
        (block_id - arg_ptr[group_id].block_start_) / num_blocks_per_batch);
    const index_t gkv_idx = __builtin_amdgcn_readfirstlane(g_idx / h_ratio);

    const long_index_t a_batch_offset = __builtin_amdgcn_readfirstlane(
        static_cast<long_index_t>(arg_ptr[group_id].compute_base_ptr_of_batch_.GetABasePtr(g_idx)));
    const long_index_t b_batch_offset  = __builtin_amdgcn_readfirstlane(static_cast<long_index_t>(
        arg_ptr[group_id].compute_base_ptr_of_batch_.GetBBasePtr(gkv_idx)));
    const long_index_t b1_batch_offset = __builtin_amdgcn_readfirstlane(static_cast<long_index_t>(
        arg_ptr[group_id].compute_base_ptr_of_batch_.GetB1BasePtr(gkv_idx)));
    const long_index_t c_batch_offset  = __builtin_amdgcn_readfirstlane(
        static_cast<long_index_t>(arg_ptr[group_id].compute_base_ptr_of_batch_.GetCBasePtr(g_idx)));
    const long_index_t z_batch_offset = __builtin_amdgcn_readfirstlane(
        static_cast<long_index_t>(arg_ptr[group_id].compute_base_ptr_of_batch_.GetZBasePtr(g_idx)));
    const long_index_t lse_batch_offset = __builtin_amdgcn_readfirstlane(static_cast<long_index_t>(
        arg_ptr[group_id].compute_base_ptr_of_batch_.GetLSEBasePtr(g_idx)));

    const D0DataType* tmp_p_d0_grid = nullptr;

    if constexpr(!is_same<D0DataType, void>::value)
    {
        const long_index_t d0_batch_offset =
            __builtin_amdgcn_readfirstlane(static_cast<long_index_t>(
                arg_ptr[group_id].compute_base_ptr_of_batch_.GetD0BasePtr(g_idx)));

        tmp_p_d0_grid = arg_ptr[group_id].p_d0_grid_ + d0_batch_offset;
    }

    GridwiseGemm::template Run<HasMainKBlockLoop, IsDropout, IsLseStoring>(
        arg_ptr[group_id].p_a_grid_ + a_batch_offset,
        arg_ptr[group_id].p_b_grid_ + b_batch_offset,
        tmp_p_d0_grid,
        arg_ptr[group_id].p_b1_grid_ + b1_batch_offset,
        arg_ptr[group_id].p_c_grid_ + c_batch_offset,
        arg_ptr[group_id].p_z_grid_ == nullptr ? nullptr
                                               : arg_ptr[group_id].p_z_grid_ + z_batch_offset,
        arg_ptr[group_id].p_lse_grid_ == nullptr ? nullptr
                                                 : arg_ptr[group_id].p_lse_grid_ + lse_batch_offset,
        // arg_ptr[group_id].p_lse_grid_ + lse_batch_offset,
        p_shared,
        a_element_op,
        b_element_op,
        acc_element_op,
        b1_element_op,
        c_element_op,
        arg_ptr[group_id].a_grid_desc_ak0_m_ak1_,
        arg_ptr[group_id].b_grid_desc_bk0_n_bk1_,
        arg_ptr[group_id].d0_grid_desc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5_,
        arg_ptr[group_id].b1_grid_desc_bk0_n_bk1_,
        arg_ptr[group_id].c_grid_desc_mblock_mperblock_nblock_nperblock_,
        arg_ptr[group_id].z_grid_desc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5_,
        arg_ptr[group_id].lse_grid_desc_m_,
        arg_ptr[group_id].block_2_ctile_map_,
        arg_ptr[group_id].c0_matrix_mask_,
        p_dropout_in_uint8_t,
        p_dropout_rescale,
        ph,
        arg_ptr[group_id].z_random_matrix_offset_ +
            g_idx * arg_ptr[group_id].raw_m_padded_ * arg_ptr[group_id].raw_n_padded_,
        arg_ptr[group_id].raw_n_padded_);
#else
    ignore = group_kernel_args;
    ignore = group_count;
    ignore = h_ratio;
    ignore = a_element_op;
    ignore = b_element_op;
    ignore = acc_element_op;
    ignore = b1_element_op;
    ignore = c_element_op;
    ignore = p_dropout_in_uint8_t;
    ignore = p_dropout_rescale;
    ignore = seed;
    ignore = offset;
#endif // end of if (defined(__gfx908__) || defined(__gfx90a__))
}

// Computes C = A * B0 * B1
//              ^^^^^^ (Acc0)
//              ^^^^^^^^^^^ (Acc1)
template <index_t NumDimG,
          index_t NumDimM,
          index_t NumDimN,
          index_t NumDimK,
          index_t NumDimO, // NumDimGemm1N
          typename ADataType,
          typename BDataType,
          typename B1DataType,
          typename CDataType,
          typename GemmDataType,
          typename ZDataType,
          typename LSEDataType,
          typename Acc0BiasDataType,
          typename Acc1BiasDataType,
          typename GemmAccDataType,
          typename CShuffleDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename AccElementwiseOperation,
          typename B1ElementwiseOperation,
          typename CElementwiseOperation,
          GemmSpecialization GemmSpec,
          TensorSpecialization ASpec,
          TensorSpecialization BSpec,
          TensorSpecialization B1Spec,
          TensorSpecialization CSpec,
          index_t NumGemmKPrefetchStage,
          index_t BlockSize,
          index_t MPerBlock,
          index_t NPerBlock, // Gemm0NPerBlock
          index_t KPerBlock, // Gemm0KPerBlock
          index_t Gemm1NPerBlock,
          index_t Gemm1KPerBlock,
          index_t AK1,
          index_t BK1,
          index_t B1K1,
          index_t MPerXDL,
          index_t NPerXDL,
          index_t MXdlPerWave,
          index_t NXdlPerWave,
          index_t Gemm1NXdlPerWave,
          index_t DropoutStep,
          typename ABlockTransferThreadClusterLengths_AK0_M_AK1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          index_t ABlockTransferSrcVectorDim,
          index_t ABlockTransferSrcScalarPerVector,
          index_t ABlockTransferDstScalarPerVector_AK1,
          bool ABlockLdsExtraM,
          typename BBlockTransferThreadClusterLengths_BK0_N_BK1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          index_t BBlockTransferSrcVectorDim,
          index_t BBlockTransferSrcScalarPerVector,
          index_t BBlockTransferDstScalarPerVector_BK1,
          bool BBlockLdsExtraN,
          index_t Acc0BiasTransferSrcScalarPerVector,
          typename B1BlockTransferThreadClusterLengths_BK0_N_BK1,
          typename B1BlockTransferThreadClusterArrangeOrder,
          typename B1BlockTransferSrcAccessOrder,
          index_t B1BlockTransferSrcVectorDim,
          index_t B1BlockTransferSrcScalarPerVector,
          index_t B1BlockTransferDstScalarPerVector_BK1,
          bool B1BlockLdsExtraN,
          index_t CShuffleMXdlPerWavePerShuffle,
          index_t CShuffleNXdlPerWavePerShuffle,
          typename CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          index_t CShuffleBlockTransferScalarPerVector_NPerBlock,
          index_t Acc1BiasTransferSrcScalarPerVector,
          MaskingSpecialization MaskingSpec,
          LoopScheduler LoopSched = LoopScheduler::Default>
struct DeviceGroupedMultiheadAttentionForward_Xdl_CShuffle_V2
    : public DeviceGroupedMultiheadAttentionForward<NumDimG,
                                                    NumDimM,
                                                    NumDimN,
                                                    NumDimK,
                                                    NumDimO,
                                                    ADataType,
                                                    BDataType,
                                                    B1DataType,
                                                    CDataType,
                                                    ZDataType,
                                                    LSEDataType,
                                                    Acc0BiasDataType,
                                                    Acc1BiasDataType,
                                                    AElementwiseOperation,
                                                    BElementwiseOperation,
                                                    AccElementwiseOperation,
                                                    B1ElementwiseOperation,
                                                    CElementwiseOperation,
                                                    MaskingSpec>
{
    static_assert(NumDimG > 0 && NumDimM > 0 && NumDimN > 0 && NumDimK > 0 && NumDimO > 0,
                  "Number of dimension must be greater than 0");

    using D0DataType = Acc0BiasDataType;
    using D1DataType = Acc1BiasDataType;
    // TODO ANT: implement bias combination
    static_assert(std::is_void<Acc1BiasDataType>::value, "Acc1 Bias addition is unimplemented");

#if 0
    // TODO ANT: use alias
    static constexpr index_t NumDimGemm0M = NumDimM;
    static constexpr index_t NumDimGemm0N = NumDimN;
    static constexpr index_t NumDimGemm0K = NumDimK;
    static constexpr index_t NumDimGemm1M = NumDimM;
    static constexpr index_t NumDimGemm1N = NumDimO;
    static constexpr index_t NumDimGemm1K = NumDimN;
#endif

    using DeviceOp    = DeviceGroupedMultiheadAttentionForward_Xdl_CShuffle_V2;
    using ProblemDesc = typename DeviceGroupedMultiheadAttentionForward<NumDimG,
                                                                        NumDimM,
                                                                        NumDimN,
                                                                        NumDimK,
                                                                        NumDimO,
                                                                        ADataType,
                                                                        BDataType,
                                                                        B1DataType,
                                                                        CDataType,
                                                                        ZDataType,
                                                                        LSEDataType,
                                                                        Acc0BiasDataType,
                                                                        Acc1BiasDataType,
                                                                        AElementwiseOperation,
                                                                        BElementwiseOperation,
                                                                        AccElementwiseOperation,
                                                                        B1ElementwiseOperation,
                                                                        CElementwiseOperation,
                                                                        MaskingSpec>::ProblemDesc;

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};

    using Transform = TransformBatchedContractionContractionToBatchedGemmGemm<
        Sequence<NumDimG, NumDimM, NumDimN, NumDimK, NumDimO>,
        Sequence<MPerBlock, NPerBlock, KPerBlock, Gemm1NPerBlock>,
        GemmSpec,
        ASpec,
        BSpec,
        B1Spec,
        CSpec>;

    static auto MakeAGridDescriptor_AK0_M_AK1(const std::vector<index_t>& a_gs_ms_ks_lengths_vec,
                                              const std::vector<index_t>& a_gs_ms_ks_strides_vec)
    {
        return Transform::MakeAGridDescriptor_AK0_M_AK1(
            Transform::MakeAGridDescriptor_M_K(a_gs_ms_ks_lengths_vec, a_gs_ms_ks_strides_vec),
            Number<AK1>{});
    }

    static auto MakeBGridDescriptor_BK0_N_BK1(const std::vector<index_t>& b_gs_ns_ks_lengths_vec,
                                              const std::vector<index_t>& b_gs_ns_ks_strides_vec)
    {
        return Transform::MakeB0GridDescriptor_BK0_N_BK1(
            Transform::MakeB0GridDescriptor_N_K(b_gs_ns_ks_lengths_vec, b_gs_ns_ks_strides_vec),
            Number<BK1>{});
    }

    static auto
    MakeB1GridDescriptor_BK0_N_BK1(const std::vector<index_t>& b1_gs_gemm1ns_gemm1ks_lengths_vec,
                                   const std::vector<index_t>& b1_gs_gemm1ns_gemm1ks_strides_vec)
    {
        return Transform::MakeB1GridDescriptor_BK0_N_BK1(
            Transform::MakeB1GridDescriptor_N_K(b1_gs_gemm1ns_gemm1ks_lengths_vec,
                                                b1_gs_gemm1ns_gemm1ks_strides_vec),
            Number<B1K1>{});
    }

    static auto MakeZGridDescriptor_M_N(const std::vector<index_t>& z_gs_ms_ns_lengths_vec,
                                        const std::vector<index_t>& z_gs_ms_ns_strides_vec)
    {
        return Transform::MakeCGridDescriptor_M_N(z_gs_ms_ns_lengths_vec, z_gs_ms_ns_strides_vec);
    }

    static auto MakeLSEGridDescriptor_M(index_t MRaw)
    {
        const auto lse_grid_desc_mraw = make_naive_tensor_descriptor_packed(make_tuple(MRaw));

        const auto M    = math::integer_divide_ceil(MRaw, MPerBlock) * MPerBlock;
        const auto MPad = M - MRaw;

        if constexpr(GemmSpec == GemmSpecialization::MPadding ||
                     GemmSpec == GemmSpecialization::MNPadding ||
                     GemmSpec == GemmSpecialization::MKPadding ||
                     GemmSpec == GemmSpecialization::MNKPadding)
        {
            // pad M
            return transform_tensor_descriptor(lse_grid_desc_mraw,
                                               make_tuple(make_right_pad_transform(MRaw, MPad)),
                                               make_tuple(Sequence<0>{}),
                                               make_tuple(Sequence<0>{}));
        }
        else
        {
            // not pad M
            return lse_grid_desc_mraw;
        }
    }

    static auto
    MakeD0GridDescriptor_M_N(const std::vector<ck::index_t>& acc0_biases_gs_ms_ns_lengths,
                             const std::vector<ck::index_t>& acc0_biases_gs_ms_ns_strides)
    {
        return Transform::MakeC0GridDescriptor_M_N(acc0_biases_gs_ms_ns_lengths,
                                                   acc0_biases_gs_ms_ns_strides);
    }

    static auto
    MakeD0GridDescriptor_G_M_N(const std::vector<ck::index_t>& acc0_biases_gs_ms_ns_lengths,
                               const std::vector<ck::index_t>& acc0_biases_gs_ms_ns_strides)
    {
        return Transform::MakeC0GridDescriptor_G_M_N(acc0_biases_gs_ms_ns_lengths,
                                                     acc0_biases_gs_ms_ns_strides);
    }

    using AGridDesc_AK0_M_AK1  = decltype(MakeAGridDescriptor_AK0_M_AK1({}, {}));
    using BGridDesc_BK0_N_BK1  = decltype(MakeBGridDescriptor_BK0_N_BK1({}, {}));
    using D0GridDesc_M_N       = decltype(MakeD0GridDescriptor_M_N({}, {}));
    using B1GridDesc_BK0_N_BK1 = decltype(MakeB1GridDescriptor_BK0_N_BK1({}, {}));
    using CGridDesc_M_N        = decltype(Transform::MakeCGridDescriptor_M_N({}, {}));
    using LSEGridDesc_M        = decltype(MakeLSEGridDescriptor_M(1));
    using ZGridDesc_M_N        = decltype(MakeZGridDescriptor_M_N({}, {}));

    using AGridDesc_G_M_K  = decltype(Transform::MakeAGridDescriptor_G_M_K({}, {}));
    using BGridDesc_G_N_K  = decltype(Transform::MakeB0GridDescriptor_G_N_K({}, {}));
    using D0GridDesc_G_M_N = decltype(MakeD0GridDescriptor_G_M_N({}, {}));
    using B1GridDesc_G_N_K = decltype(Transform::MakeB1GridDescriptor_G_N_K({}, {}));
    using CGridDesc_G_M_N  = decltype(Transform::MakeCGridDescriptor_G_M_N({}, {}));
    using ZGridDesc_G_M_N  = decltype(Transform::MakeCGridDescriptor_G_M_N({}, {}));

    constexpr static auto make_MaskOutPredicate()
    {
        if constexpr(MaskingSpec == MaskingSpecialization::MaskDisabled)
        {
            return MaskDisabledPredicate{};
        }
        else if constexpr(MaskingSpec == MaskingSpecialization::MaskUpperTriangleFromTopLeft)
        {
            return MaskUpperTriangleFromTopLeftPredicate{};
        }
        else if constexpr(MaskingSpec == MaskingSpecialization::MaskUpperTriangleFromBottomRight)
        {
            return MaskUpperTriangleFromBottomRightPredicate{};
        }
    }
    using C0MatrixMask = C0MatrixMask_impl<decltype(make_MaskOutPredicate())>;

    struct ComputeBasePtrOfStridedBatch
    {
        ComputeBasePtrOfStridedBatch(const AGridDesc_G_M_K& a_grid_desc_g_m_k,
                                     const BGridDesc_G_N_K& b_grid_desc_g_n_k,
                                     const D0GridDesc_G_M_N& d0_grid_desc_g_m_n,
                                     const B1GridDesc_G_N_K& b1_grid_desc_g_n_k,
                                     const CGridDesc_G_M_N& c_grid_desc_g_m_n,
                                     const ZGridDesc_G_M_N& z_grid_desc_g_m_n,
                                     index_t BatchStrideLSE)
            : a_grid_desc_g_m_k_(a_grid_desc_g_m_k),
              b_grid_desc_g_n_k_(b_grid_desc_g_n_k),
              d0_grid_desc_g_m_n_(d0_grid_desc_g_m_n),
              b1_grid_desc_g_n_k_(b1_grid_desc_g_n_k),
              c_grid_desc_g_m_n_(c_grid_desc_g_m_n),
              z_grid_desc_g_m_n_(z_grid_desc_g_m_n),
              BatchStrideLSE_(BatchStrideLSE)
        {
        }

        __host__ __device__ constexpr long_index_t GetABasePtr(index_t g_idx) const
        {
            return a_grid_desc_g_m_k_.CalculateOffset(make_multi_index(g_idx, 0, 0));
        }

        __host__ __device__ constexpr long_index_t GetBBasePtr(index_t g_idx) const
        {
            return b_grid_desc_g_n_k_.CalculateOffset(make_multi_index(g_idx, 0, 0));
        }

        __host__ __device__ constexpr long_index_t GetD0BasePtr(index_t g_idx) const
        {
            return d0_grid_desc_g_m_n_.CalculateOffset(make_multi_index(g_idx, 0, 0));
        }

        __host__ __device__ constexpr long_index_t GetB1BasePtr(index_t g_idx) const
        {
            return b1_grid_desc_g_n_k_.CalculateOffset(make_multi_index(g_idx, 0, 0));
        }

        __host__ __device__ constexpr long_index_t GetCBasePtr(index_t g_idx) const
        {
            return c_grid_desc_g_m_n_.CalculateOffset(make_multi_index(g_idx, 0, 0));
        }

        __host__ __device__ constexpr long_index_t GetZBasePtr(index_t g_idx) const
        {
            return z_grid_desc_g_m_n_.CalculateOffset(make_multi_index(g_idx, 0, 0));
        }

        __host__ __device__ constexpr long_index_t GetLSEBasePtr(index_t g_idx) const
        {
            return g_idx * static_cast<long_index_t>(BatchStrideLSE_);
        }

        private:
        AGridDesc_G_M_K a_grid_desc_g_m_k_;
        BGridDesc_G_N_K b_grid_desc_g_n_k_;
        D0GridDesc_G_M_N d0_grid_desc_g_m_n_;
        B1GridDesc_G_N_K b1_grid_desc_g_n_k_;
        CGridDesc_G_M_N c_grid_desc_g_m_n_;
        ZGridDesc_G_M_N z_grid_desc_g_m_n_;

        index_t BatchStrideLSE_;
    };

    // GridwiseGemm
    using GridwiseGemm = GridwiseBatchedMultiheadAttentionForward_Xdl_CShuffle_V2<
        ADataType, // TODO: distinguish A/B datatype
        Acc0BiasDataType,
        ZDataType,
        GemmDataType,
        GemmAccDataType,
        CShuffleDataType,
        CDataType,
        LSEDataType,
        AElementwiseOperation,
        BElementwiseOperation,
        AccElementwiseOperation,
        B1ElementwiseOperation,
        CElementwiseOperation,
        InMemoryDataOperationEnum::Set,
        AGridDesc_AK0_M_AK1,
        BGridDesc_BK0_N_BK1,
        D0GridDesc_M_N,
        B1GridDesc_BK0_N_BK1,
        CGridDesc_M_N,
        ZGridDesc_M_N,
        LSEGridDesc_M,
        NumGemmKPrefetchStage,
        BlockSize,
        MPerBlock,
        NPerBlock,
        KPerBlock,
        Gemm1NPerBlock,
        Gemm1KPerBlock,
        AK1,
        BK1,
        B1K1,
        MPerXDL,
        NPerXDL,
        MXdlPerWave,
        NXdlPerWave,
        Gemm1NXdlPerWave,
        DropoutStep,
        ABlockTransferThreadClusterLengths_AK0_M_AK1,
        ABlockTransferThreadClusterArrangeOrder,
        ABlockTransferSrcAccessOrder,
        ABlockTransferSrcVectorDim,
        ABlockTransferSrcScalarPerVector,
        ABlockTransferDstScalarPerVector_AK1,
        true,
        ABlockLdsExtraM,
        BBlockTransferThreadClusterLengths_BK0_N_BK1,
        BBlockTransferThreadClusterArrangeOrder,
        BBlockTransferSrcAccessOrder,
        BBlockTransferSrcVectorDim,
        BBlockTransferSrcScalarPerVector,
        BBlockTransferDstScalarPerVector_BK1,
        true,
        BBlockLdsExtraN,
        Acc0BiasTransferSrcScalarPerVector,
        B1BlockTransferThreadClusterLengths_BK0_N_BK1,
        B1BlockTransferThreadClusterArrangeOrder,
        B1BlockTransferSrcAccessOrder,
        B1BlockTransferSrcVectorDim,
        B1BlockTransferSrcScalarPerVector,
        B1BlockTransferDstScalarPerVector_BK1,
        false,
        B1BlockLdsExtraN,
        CShuffleMXdlPerWavePerShuffle,
        CShuffleNXdlPerWavePerShuffle,
        CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
        CShuffleBlockTransferScalarPerVector_NPerBlock,
        Acc1BiasTransferSrcScalarPerVector,
        LoopSched,
        Transform::matrix_padder.PadN,
        MaskingSpec != MaskingSpecialization::MaskDisabled>;

    using Block2CTileMap = OffsettedBlockToCTileMap<typename GridwiseGemm::DefaultBlock2CTileMap>;

    struct GroupKernelArg
    {
        // pointers
        const ADataType* p_a_grid_;
        const BDataType* p_b_grid_;
        const D0DataType* p_d0_grid_;
        const B1DataType* p_b1_grid_;
        CDataType* p_c_grid_;
        ZDataType* p_z_grid_;
        LSEDataType* p_lse_grid_;

        // tensor descriptors for block/thread-wise copy
        AGridDesc_AK0_M_AK1 a_grid_desc_ak0_m_ak1_;
        BGridDesc_BK0_N_BK1 b_grid_desc_bk0_n_bk1_;
        typename GridwiseGemm::D0GridDescriptor_M0_N0_M1_N1_M2_N2_M3_N3_N4_N5
            d0_grid_desc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5_;
        B1GridDesc_BK0_N_BK1 b1_grid_desc_bk0_n_bk1_;
        typename GridwiseGemm::CGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock
            c_grid_desc_mblock_mperblock_nblock_nperblock_;
        typename GridwiseGemm::ZGridDescriptor_M0_N0_M1_N1_M2_N2_M3_N3_N4_N5
            z_grid_desc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5_;
        ZGridDesc_M_N z_grid_desc_m_n_;
        LSEGridDesc_M lse_grid_desc_m_;

        // batch & stride
        index_t num_blocks_per_batch_;
        ComputeBasePtrOfStridedBatch compute_base_ptr_of_batch_;

        // check C0 masking and padding
        C0MatrixMask c0_matrix_mask_;

        // block-to-c-tile map
        Block2CTileMap block_2_ctile_map_;

        index_t block_start_, block_end_;

        index_t z_random_matrix_offset_;
        index_t raw_m_padded_, raw_n_padded_;
    };

    struct GroupDeviceArg
    {
        // lengths for the last dimensions of overall problem for sanity check of vector load/store
        std::vector<index_t> raw_lengths_mz_nz_kz_gemm1nz_;

        // strides for the last dimensions of each tensor for sanity check of vector load/store
        std::vector<index_t> a_mz_kz_strides_;
        std::vector<index_t> b_nz_kz_strides_;
        std::vector<index_t> b1_nz_kz_strides_;
        std::vector<index_t> c_mz_gemm1nz_strides_;

        // for gridwise gemm check
        CGridDesc_M_N c_grid_desc_m_n_;
        BGridDesc_G_N_K b_grid_desc_g_n_k_;
        CGridDesc_G_M_N c_grid_desc_g_m_n_;

        // raw data
        std::vector<ck::index_t> d0_n_length_stride_;
    };

    // Argument
    // FIXME: constness
    struct Argument : public BaseArgument
    {
        Argument(std::vector<const void*> p_a_vec,
                 std::vector<const void*> p_b_vec,
                 std::vector<const void*> p_b1_vec,
                 std::vector<void*> p_c_vec,
                 std::vector<void*> p_z_vec,
                 std::vector<void*> p_lse_vec,
                 std::vector<const void*> p_acc0_biases_vec,
                 std::vector<const void*> p_acc1_biases_vec,
                 std::vector<ProblemDesc>& problem_desc_vec,
                 AElementwiseOperation a_element_op,
                 BElementwiseOperation b_element_op,
                 AccElementwiseOperation acc_element_op,
                 B1ElementwiseOperation b1_element_op,
                 CElementwiseOperation c_element_op,
                 float p_dropout,
                 std::tuple<unsigned long long, unsigned long long> seeds)
            : a_element_op_{a_element_op},
              b_element_op_{b_element_op},
              acc_element_op_{acc_element_op},
              b1_element_op_{b1_element_op},
              c_element_op_{c_element_op}
        {
            ignore = p_acc1_biases_vec;
            // TODO ANT: implement bias addition
            group_count_ = problem_desc_vec.size();

            if(!(group_count_ == p_a_vec.size() && group_count_ == p_b_vec.size() &&
                 group_count_ == p_b1_vec.size() && group_count_ == p_c_vec.size() &&
                 (group_count_ == p_acc0_biases_vec.size() || p_acc0_biases_vec.size() == 0) &&
                 (group_count_ == p_z_vec.size() || p_z_vec.size() == 0) &&
                 (group_count_ == p_lse_vec.size() || p_lse_vec.size() == 0)))
            {
                throw std::runtime_error("wrong! group_count_ != a/b/b1/c_vec.size");
            }

            grid_size_ = 0;

            index_t z_random_matrix_offset = 0;

            h_ratio_ = problem_desc_vec[0].a_gs_ms_ks_lengths[NumDimG - 1] /
                       problem_desc_vec[0].b0_gs_ns_ks_lengths[NumDimG - 1];

            for(std::size_t i = 0; i < group_count_; i++)
            {
                const auto p_a_grid  = static_cast<const ADataType*>(p_a_vec[i]);
                const auto p_b_grid  = static_cast<const BDataType*>(p_b_vec[i]);
                const auto p_d0_grid = (p_acc0_biases_vec.size() == group_count_)
                                           ? static_cast<const D0DataType*>(p_acc0_biases_vec[i])
                                           : nullptr;
                const auto p_b1_grid = static_cast<const B1DataType*>(p_b1_vec[i]);
                const auto p_c_grid  = static_cast<CDataType*>(p_c_vec[i]);
                const auto p_z_grid  = (p_z_vec.size() == group_count_)
                                          ? static_cast<ZDataType*>(p_z_vec[i])
                                          : nullptr;
                const auto p_lse_grid = (p_lse_vec.size() == group_count_)
                                            ? static_cast<LSEDataType*>(p_lse_vec[i])
                                            : nullptr;

                if(p_lse_grid == nullptr)
                {
                    is_lse_storing_ = false;
                }

                const auto& problem_desc         = problem_desc_vec[i];
                const auto a_grid_desc_ak0_m_ak1 = MakeAGridDescriptor_AK0_M_AK1(
                    problem_desc.a_gs_ms_ks_lengths, problem_desc.a_gs_ms_ks_strides);
                const auto b_grid_desc_bk0_n_bk1 = MakeBGridDescriptor_BK0_N_BK1(
                    problem_desc.b0_gs_ns_ks_lengths, problem_desc.b0_gs_ns_ks_strides);

                std::vector<index_t> tmp_d0_gs_ms_ns_lengths;
                std::vector<index_t> tmp_d0_gs_ms_ns_strides;

                if constexpr(!is_same<D0DataType, void>::value)
                {
                    tmp_d0_gs_ms_ns_lengths = problem_desc.acc0_biases_gs_ms_ns_lengths;
                    tmp_d0_gs_ms_ns_strides = problem_desc.acc0_biases_gs_ms_ns_strides;
                }
                else
                {
                    tmp_d0_gs_ms_ns_lengths = {1, 1, 1, 1};
                    tmp_d0_gs_ms_ns_strides = {0, 0, 0, 0};
                }

                const D0GridDesc_M_N d0_grid_desc_m_n{DeviceOp::MakeD0GridDescriptor_M_N(
                    tmp_d0_gs_ms_ns_lengths, tmp_d0_gs_ms_ns_strides)};
                const auto d0_grid_desc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5 =
                    GridwiseGemm::MakeD0GridDescriptor_M0_N0_M1_N1_M2_N2_M3_N3_N4_N5(
                        d0_grid_desc_m_n);
                const auto b1_grid_desc_bk0_n_bk1 = MakeB1GridDescriptor_BK0_N_BK1(
                    problem_desc.b1_gs_os_ns_lengths, problem_desc.b1_gs_os_ns_strides);
                const auto c_grid_desc_m_n = Transform::MakeCGridDescriptor_M_N(
                    problem_desc.c_gs_ms_os_lengths, problem_desc.c_gs_ms_os_strides);
                const auto z_grid_desc_m_n = MakeZGridDescriptor_M_N(
                    problem_desc.z_gs_ms_ns_lengths, problem_desc.z_gs_ms_ns_strides);
                const auto lse_grid_desc_m =
                    DeviceOp::MakeLSEGridDescriptor_M(problem_desc.lse_gs_ms_lengths[NumDimG]);

                const auto a_grid_desc_g_m_k = Transform::MakeAGridDescriptor_G_M_K(
                    problem_desc.a_gs_ms_ks_lengths, problem_desc.a_gs_ms_ks_strides);
                const auto b_grid_desc_g_n_k = Transform::MakeB0GridDescriptor_G_N_K(
                    problem_desc.b0_gs_ns_ks_lengths, problem_desc.b0_gs_ns_ks_strides);
                const auto d0_grid_desc_g_m_n = DeviceOp::MakeD0GridDescriptor_G_M_N(
                    tmp_d0_gs_ms_ns_lengths, tmp_d0_gs_ms_ns_strides);
                const auto b1_grid_desc_g_n_k = Transform::MakeB1GridDescriptor_G_N_K(
                    problem_desc.b1_gs_os_ns_lengths, problem_desc.b1_gs_os_ns_strides);
                const auto c_grid_desc_g_m_n = Transform::MakeCGridDescriptor_G_M_N(
                    problem_desc.c_gs_ms_os_lengths, problem_desc.c_gs_ms_os_strides);
                const auto z_grid_desc_g_m_n = Transform::MakeCGridDescriptor_G_M_N(
                    problem_desc.z_gs_ms_ns_lengths, problem_desc.z_gs_ms_ns_strides);

                const auto c_grid_desc_mblock_mperblock_nblock_nperblock =
                    GridwiseGemm::MakeCGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
                        c_grid_desc_m_n);

                const auto z_grid_desc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5 =
                    GridwiseGemm::MakeCGridDescriptor_M0_N0_M1_N1_M2_N2_M3_N3_N4_N5(
                        z_grid_desc_m_n);

                const index_t BlockStart     = grid_size_;
                const auto block_2_ctile_map = Block2CTileMap(c_grid_desc_m_n, BlockStart);
                const index_t batch_count    = c_grid_desc_g_m_n.GetLength(I0);
                const index_t grid_size_grp =
                    block_2_ctile_map.CalculateGridSize(c_grid_desc_m_n) * batch_count;
                const index_t BlockEnd = grid_size_ + grid_size_grp;

                // batch stride
                const auto compute_base_ptr_of_batch = ComputeBasePtrOfStridedBatch(
                    a_grid_desc_g_m_k,
                    b_grid_desc_g_n_k,
                    d0_grid_desc_g_m_n,
                    b1_grid_desc_g_n_k,
                    c_grid_desc_g_m_n,
                    z_grid_desc_g_m_n,
                    type_convert<index_t>(problem_desc.lse_gs_ms_strides[NumDimG - 1]));

                // C0 mask
                const auto c0_matrix_mask =
                    C0MatrixMask(a_grid_desc_g_m_k.GetLength(I1), b_grid_desc_g_n_k.GetLength(I1));

                grid_size_ += grid_size_grp;

                const auto raw_m_padded = GridwiseGemm::GetPaddedSize(
                    problem_desc.a_gs_ms_ks_lengths[NumDimG + NumDimM - 1]);
                const auto raw_n_padded = GridwiseGemm::GetPaddedSize(
                    problem_desc.b0_gs_ns_ks_lengths[NumDimG + NumDimN - 1]);

                group_kernel_args_.push_back({p_a_grid,
                                              p_b_grid,
                                              p_d0_grid,
                                              p_b1_grid,
                                              p_c_grid,
                                              p_z_grid,
                                              p_lse_grid,
                                              a_grid_desc_ak0_m_ak1,
                                              b_grid_desc_bk0_n_bk1,
                                              d0_grid_desc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5,
                                              b1_grid_desc_bk0_n_bk1,
                                              c_grid_desc_mblock_mperblock_nblock_nperblock,
                                              z_grid_desc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5,
                                              z_grid_desc_m_n,
                                              lse_grid_desc_m,
                                              block_2_ctile_map.CalculateGridSize(c_grid_desc_m_n),
                                              compute_base_ptr_of_batch,
                                              c0_matrix_mask,
                                              block_2_ctile_map,
                                              BlockStart,
                                              BlockEnd,
                                              z_random_matrix_offset,
                                              raw_m_padded,
                                              raw_n_padded});

                z_random_matrix_offset =
                    z_random_matrix_offset + raw_m_padded * raw_n_padded * batch_count;

                // for  check
                std::vector<ck::index_t> d0_n_length_stride;
                d0_n_length_stride.push_back(tmp_d0_gs_ms_ns_lengths[NumDimG + NumDimM]);
                d0_n_length_stride.push_back(tmp_d0_gs_ms_ns_strides[NumDimG + NumDimM]);

                group_device_args_.push_back(
                    {{problem_desc.a_gs_ms_ks_lengths[NumDimG + NumDimM - 1],
                      problem_desc.b0_gs_ns_ks_lengths[NumDimG + NumDimN - 1],
                      problem_desc.b0_gs_ns_ks_lengths[NumDimG + NumDimN + NumDimK - 1],
                      problem_desc.b1_gs_os_ns_lengths[NumDimG + NumDimO - 1]},
                     {problem_desc.a_gs_ms_ks_strides[NumDimG + NumDimM - 1],
                      problem_desc.a_gs_ms_ks_strides[NumDimG + NumDimM + NumDimK - 1]},
                     {problem_desc.b0_gs_ns_ks_strides[NumDimG + NumDimN - 1],
                      problem_desc.b0_gs_ns_ks_strides[NumDimG + NumDimN + NumDimK - 1]},
                     {problem_desc.b1_gs_os_ns_strides[NumDimG + NumDimO - 1],
                      problem_desc.b1_gs_os_ns_strides[NumDimG + NumDimO + NumDimN - 1]},
                     {problem_desc.c_gs_ms_os_strides[NumDimG + NumDimM - 1],
                      problem_desc.c_gs_ms_os_strides[NumDimG + NumDimM + NumDimO - 1]},
                     c_grid_desc_m_n,
                     b_grid_desc_g_n_k,
                     c_grid_desc_g_m_n,
                     d0_n_length_stride});
            }

            use_dropout_          = p_dropout > 0.0; //
            p_dropout_            = 1.f - p_dropout;
            p_dropout_in_uint8_t_ = uint8_t(std::floor(p_dropout_ * 255.0));
            p_dropout_            = 1.f / p_dropout_;
            p_dropout_rescale_    = type_convert<GemmAccDataType>(p_dropout_);

            seed_   = std::get<0>(seeds);
            offset_ = std::get<1>(seeds);
        }

        std::vector<GroupKernelArg> group_kernel_args_;
        std::vector<GroupDeviceArg> group_device_args_;

        std::size_t group_count_;
        index_t grid_size_;

        AElementwiseOperation a_element_op_;
        BElementwiseOperation b_element_op_;
        AccElementwiseOperation acc_element_op_;
        B1ElementwiseOperation b1_element_op_;
        CElementwiseOperation c_element_op_;

        index_t h_ratio_;
        float p_dropout_;
        uint8_t p_dropout_in_uint8_t_;
        unsigned long long seed_;
        unsigned long long offset_;
        GemmAccDataType p_dropout_rescale_;
        bool use_dropout_;

        bool is_lse_storing_ = true;
    };

    // Invoker
    struct Invoker : public BaseInvoker
    {
        using Argument = DeviceOp::Argument;

        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            if(!DeviceOp::IsSupportedArgument(arg))
            {
                throw std::runtime_error("wrong! unsupported argument");
            }

            bool all_has_main_k_block_loop  = true;
            bool some_has_main_k_block_loop = false;
            for(std::size_t i = 0; i < arg.group_count_; i++)
            {
                const auto K = arg.group_kernel_args_[i].a_grid_desc_ak0_m_ak1_.GetLength(I0) *
                               arg.group_kernel_args_[i].a_grid_desc_ak0_m_ak1_.GetLength(I2);
                const bool y = GridwiseGemm::CalculateHasMainKBlockLoop(K);
                all_has_main_k_block_loop &= y;
                some_has_main_k_block_loop |= y;
            }

            hipStreamCaptureStatus status = hipStreamCaptureStatusNone;

            HIP_CHECK_ERROR(hipStreamIsCapturing(stream_config.stream_id_, &status));

            if(status == hipStreamCaptureStatusActive)
            {
                size_t copy_size = arg.group_kernel_args_.size() * sizeof(GroupKernelArg);

                // ToDO: when to release this memory buffer?
                char* persistent_ptr = new char[copy_size];

                (void)std::memcpy(persistent_ptr, arg.group_kernel_args_.data(), copy_size);

                HIP_CHECK_ERROR(hipMemcpyAsync(arg.p_workspace_,
                                               persistent_ptr,
                                               copy_size,
                                               hipMemcpyHostToDevice,
                                               stream_config.stream_id_));
            }
            else
            {
                HIP_CHECK_ERROR(
                    hipMemcpyAsync(arg.p_workspace_,
                                   arg.group_kernel_args_.data(),
                                   arg.group_kernel_args_.size() * sizeof(GroupKernelArg),
                                   hipMemcpyHostToDevice,
                                   stream_config.stream_id_));
            }

            float ave_time = 0;

            auto launch_kernel =
                [&](auto has_main_k_block_loop_, auto use_dropout_, auto is_lse_storing_) {
                    const auto kernel =
                        kernel_grouped_gemm_softmax_gemm_xdl_cshuffle_v2<GridwiseGemm,
                                                                         D0DataType,
                                                                         GemmAccDataType,
                                                                         GroupKernelArg,
                                                                         AElementwiseOperation,
                                                                         BElementwiseOperation,
                                                                         AccElementwiseOperation,
                                                                         B1ElementwiseOperation,
                                                                         CElementwiseOperation,
                                                                         has_main_k_block_loop_,
                                                                         use_dropout_,
                                                                         is_lse_storing_>;

                    return launch_and_time_kernel(
                        stream_config,
                        kernel,
                        dim3(arg.grid_size_),
                        dim3(BlockSize),
                        0,
                        cast_pointer_to_constant_address_space(arg.p_workspace_),
                        arg.group_count_,
                        arg.h_ratio_,
                        arg.a_element_op_,
                        arg.b_element_op_,
                        arg.acc_element_op_,
                        arg.b1_element_op_,
                        arg.c_element_op_,
                        arg.p_dropout_in_uint8_t_,
                        arg.p_dropout_rescale_,
                        arg.seed_,
                        arg.offset_);
                };

            // Gemm1_K is split into Gemm1_K0/K1 where K1 is known at compile time, so we only need
            // to concern Gemm0's loop
            if(all_has_main_k_block_loop)
            {
                if(arg.use_dropout_)
                {
                    if(arg.is_lse_storing_)
                    {
                        ave_time = launch_kernel(integral_constant<bool, true>{},
                                                 integral_constant<bool, true>{},
                                                 integral_constant<bool, true>{});
                    }
                    else
                    {
                        ave_time = launch_kernel(integral_constant<bool, true>{},
                                                 integral_constant<bool, true>{},
                                                 integral_constant<bool, false>{});
                    }
                }
                else
                {
                    if(arg.is_lse_storing_)
                    {
                        ave_time = launch_kernel(integral_constant<bool, true>{},
                                                 integral_constant<bool, false>{},
                                                 integral_constant<bool, true>{});
                    }
                    else
                    {
                        ave_time = launch_kernel(integral_constant<bool, true>{},
                                                 integral_constant<bool, false>{},
                                                 integral_constant<bool, false>{});
                    }
                }
            }
            else if(!some_has_main_k_block_loop)
            {
                if(arg.use_dropout_)
                {
                    if(arg.is_lse_storing_)
                    {
                        ave_time = launch_kernel(integral_constant<bool, false>{},
                                                 integral_constant<bool, true>{},
                                                 integral_constant<bool, true>{});
                    }
                    else
                    {
                        ave_time = launch_kernel(integral_constant<bool, false>{},
                                                 integral_constant<bool, true>{},
                                                 integral_constant<bool, false>{});
                    }
                }
                else
                {
                    if(arg.is_lse_storing_)
                    {
                        ave_time = launch_kernel(integral_constant<bool, false>{},
                                                 integral_constant<bool, false>{},
                                                 integral_constant<bool, true>{});
                    }
                    else
                    {
                        ave_time = launch_kernel(integral_constant<bool, false>{},
                                                 integral_constant<bool, false>{},
                                                 integral_constant<bool, false>{});
                    }
                }
            }
            else
            {
                throw std::runtime_error("wrong! all gemm problems have to simultaneously meet "
                                         "has_main_k_block_loop or no_main_k_block_loop");
            }

            return ave_time;
        }

        // polymorphic
        float Run(const BaseArgument* p_arg,
                  const StreamConfig& stream_config = StreamConfig{}) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg), stream_config);
        }
    };

    static constexpr bool IsValidCompilationParameter()
    {
        // TODO: properly implement this check
        return true;
    }

    static bool IsSupportedArgument(const Argument& arg)
    {
        if(!(ck::get_device_name() == "gfx908" || ck::get_device_name() == "gfx90a" ||
             ck::get_device_name() == "gfx940" || ck::get_device_name() == "gfx941" ||
             ck::get_device_name() == "gfx942"))
        {
            return false;
        }

        // TODO ANT: Check if tensor specialization & strides mismatch

        bool all_has_main_k_block_loop  = true;
        bool some_has_main_k_block_loop = false;

        for(std::size_t i = 0; i < arg.group_count_; i++)
        {
            const auto& kernel_arg = arg.group_kernel_args_[i];
            const auto& device_arg = arg.group_device_args_[i];

            // Check if C permute dimension matches GEMM + GEMM shape
            const index_t c_g       = device_arg.c_grid_desc_g_m_n_.GetLength(I0); // unpadded
            const index_t b_g       = device_arg.b_grid_desc_g_n_k_.GetLength(I0);
            const index_t c_m       = device_arg.c_grid_desc_m_n_.GetLength(I0);
            const index_t c_gemm1n  = device_arg.c_grid_desc_m_n_.GetLength(I1);
            const index_t a_m       = kernel_arg.a_grid_desc_ak0_m_ak1_.GetLength(I1);
            const index_t b1_gemm1n = kernel_arg.b1_grid_desc_bk0_n_bk1_.GetLength(I1);
            if(!(c_m == a_m && c_gemm1n == b1_gemm1n && c_g % b_g == 0 &&
                 c_g / b_g == arg.h_ratio_))
            {
                return false;
            }

            if constexpr(!is_same<D0DataType, void>::value)
            {
                if(device_arg.d0_n_length_stride_[1] == 1)
                {
                    if(device_arg.d0_n_length_stride_[0] % Acc0BiasTransferSrcScalarPerVector != 0)
                        return false;
                }
                else if(Acc0BiasTransferSrcScalarPerVector != 1)
                {
                    return false;
                }
            }

            // Check if having main loop
            const auto K = kernel_arg.a_grid_desc_ak0_m_ak1_.GetLength(I0) *
                           kernel_arg.a_grid_desc_ak0_m_ak1_.GetLength(I2);
            const bool y = GridwiseGemm::CalculateHasMainKBlockLoop(K);
            all_has_main_k_block_loop &= y;
            some_has_main_k_block_loop |= y;

            // Note: we need raw lengths since threadwise copy can not handle vector load when
            // part of vector is out of bounds
            const auto MzRaw      = device_arg.raw_lengths_mz_nz_kz_gemm1nz_[0];
            const auto NzRaw      = device_arg.raw_lengths_mz_nz_kz_gemm1nz_[1];
            const auto KzRaw      = device_arg.raw_lengths_mz_nz_kz_gemm1nz_[2];
            const auto Gemm1NzRaw = device_arg.raw_lengths_mz_nz_kz_gemm1nz_[3];

            // Check scalar per vector requirement
            const auto a_extent_lowest  = ABlockTransferSrcVectorDim == 2 ? KzRaw : MzRaw;
            const auto b_extent_lowest  = BBlockTransferSrcVectorDim == 2 ? KzRaw : NzRaw;
            const auto b1_extent_lowest = B1BlockTransferSrcVectorDim == 2 ? NzRaw : Gemm1NzRaw;
            const auto c_extent_lowest  = Gemm1NzRaw;

            if(!(a_extent_lowest % ABlockTransferSrcScalarPerVector == 0 &&
                 b_extent_lowest % BBlockTransferSrcScalarPerVector == 0 &&
                 b1_extent_lowest % B1BlockTransferSrcScalarPerVector == 0 &&
                 c_extent_lowest % CShuffleBlockTransferScalarPerVector_NPerBlock == 0))
            {
                return false;
            }

            // Check vector load/store requirement
            const auto a_stride_lowest = ABlockTransferSrcVectorDim == 2
                                             ? device_arg.a_mz_kz_strides_[1]
                                             : device_arg.a_mz_kz_strides_[0];
            const auto b_stride_lowest = BBlockTransferSrcVectorDim == 2
                                             ? device_arg.b_nz_kz_strides_[1]
                                             : device_arg.b_nz_kz_strides_[0];
            const auto b1_stride_lowest = B1BlockTransferSrcVectorDim == 2
                                              ? device_arg.b1_nz_kz_strides_[1]
                                              : device_arg.b1_nz_kz_strides_[0];
            const auto c_stride_lowest =
                device_arg.c_mz_gemm1nz_strides_[1]; // cshuffle assumes lowest dim in Gemm1Ns to be
                                                     // contiguous

            if(!(a_stride_lowest == 1 || b_stride_lowest == 1 || b1_stride_lowest == 1 ||
                 c_stride_lowest == 1))
            {
                return false;
            }

            if(!GridwiseGemm::CheckValidity(kernel_arg.a_grid_desc_ak0_m_ak1_,
                                            kernel_arg.b_grid_desc_bk0_n_bk1_,
                                            kernel_arg.b1_grid_desc_bk0_n_bk1_,
                                            device_arg.c_grid_desc_m_n_,
                                            kernel_arg.block_2_ctile_map_))
            {
                return false;
            }
        }

        // all gemm problems have to simultaneously meet has_main_k_block_loop or
        // no_main_k_block_loop
        if(!(all_has_main_k_block_loop || !some_has_main_k_block_loop))
        {
            return false;
        }

        return true;
    }

    // polymorphic
    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return IsSupportedArgument(*dynamic_cast<const Argument*>(p_arg));
    }

    static auto MakeArgument(std::vector<const void*> p_a_vec,
                             std::vector<const void*> p_b_vec,
                             std::vector<const void*> p_b1_vec,
                             std::vector<void*> p_c_vec,
                             std::vector<void*> p_z_vec,
                             std::vector<void*> p_lse_vec,
                             std::vector<const void*> p_acc0_biases_vec,
                             std::vector<const void*> p_acc1_biases_vec,
                             std::vector<ProblemDesc>& problem_desc_vec,
                             AElementwiseOperation a_element_op,
                             BElementwiseOperation b_element_op,
                             AccElementwiseOperation acc_element_op,
                             B1ElementwiseOperation b1_element_op,
                             CElementwiseOperation c_element_op,
                             float p_dropout,
                             std::tuple<unsigned long long, unsigned long long> seeds)
    {
        return Argument{p_a_vec,
                        p_b_vec,
                        p_b1_vec,
                        p_c_vec,
                        p_z_vec,
                        p_lse_vec,
                        p_acc0_biases_vec,
                        p_acc1_biases_vec,
                        problem_desc_vec,
                        a_element_op,
                        b_element_op,
                        acc_element_op,
                        b1_element_op,
                        c_element_op,
                        p_dropout,
                        seeds};
    }

    static auto MakeInvoker() { return Invoker{}; }

    // polymorphic
    std::unique_ptr<BaseArgument>
    MakeArgumentPointer(std::vector<const void*> p_a_vec,
                        std::vector<const void*> p_b_vec,
                        std::vector<const void*> p_b1_vec,
                        std::vector<void*> p_c_vec,
                        std::vector<void*> p_z_vec,
                        std::vector<void*> p_lse_vec,
                        std::vector<const void*> p_acc0_biases_vec,
                        std::vector<const void*> p_acc1_biases_vec,
                        std::vector<ProblemDesc>& problem_desc_vec,
                        AElementwiseOperation a_element_op,
                        BElementwiseOperation b_element_op,
                        AccElementwiseOperation acc_element_op,
                        B1ElementwiseOperation b1_element_op,
                        CElementwiseOperation c_element_op,
                        float p_dropout,
                        std::tuple<unsigned long long, unsigned long long> seeds) override
    {
        return std::make_unique<Argument>(p_a_vec,
                                          p_b_vec,
                                          p_b1_vec,
                                          p_c_vec,
                                          p_z_vec,
                                          p_lse_vec,
                                          p_acc0_biases_vec,
                                          p_acc1_biases_vec,
                                          problem_desc_vec,
                                          a_element_op,
                                          b_element_op,
                                          acc_element_op,
                                          b1_element_op,
                                          c_element_op,
                                          p_dropout,
                                          seeds);
    }

    // polymorphic
    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>(Invoker{});
    }

    // polymorphic
    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        // clang-format off
        str << "DeviceGroupedMultiheadAttentionForward_Xdl_CShuffle_V2"
            << "<"
            << BlockSize << ", "
            << MPerBlock << ", "
            << NPerBlock << ", "
            << KPerBlock << ", "
            << AK1 << ", "
            << BK1 << ", "
            << MPerBlock << ", "
            << Gemm1NPerBlock << ", "
            << Gemm1KPerBlock << ", "
            << B1K1 << ", "
            << getGemmSpecializationString(GemmSpec) << ", "
            << "ASpec" << getTensorSpecializationString(ASpec) << ", "
            << "B0Spec" << getTensorSpecializationString(BSpec) << ", "
            << "B1Spec" << getTensorSpecializationString(B1Spec) << ", "
            << "CSpec" << getTensorSpecializationString(CSpec) << ", "
            << getMaskingSpecializationString(MaskingSpec) << ">";
        // clang-format on

        return str.str();
    }

    size_t GetWorkSpaceSize(const BaseArgument* p_arg) const override
    {
        return dynamic_cast<const Argument*>(p_arg)->group_count_ * sizeof(GroupKernelArg);
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
