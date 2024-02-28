// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/utility/philox_rand.hpp"
#include "ck/tensor_description/multi_index_transform_helper.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/grid/block_to_ctile_map.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_pipeline_selector.hpp"
#include "ck/tensor_operation/gpu/block/blockwise_gemm_xdlops.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v4r1.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v6r1.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/block/blockwise_softmax.hpp"
#include "ck/tensor_operation/gpu/block/blockwise_dropout.hpp"

namespace ck {

template <typename InputDataType,
          typename D0DataType,
          typename OutputDataType,
          typename ZDataType,
          typename GemmDataType,
          typename FloatGemmAcc,
          typename FloatCShuffle,
          typename FloatLSE,
          typename FloatD,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename SElementwiseOperation,
          typename B1ElementwiseOperation,
          typename CElementwiseOperation,
          InMemoryDataOperationEnum CGlobalMemoryDataOperation,
          typename QGridDesc_K0_M_K1,
          typename KGridDesc_K0_N_K1,
          typename KGridDesc_N_K,
          typename D0GridDesc_M_N,
          typename ZGridDesc_M_N,
          typename VGridDesc_O0_N_O1,
          typename YGridDesc_M_O,
          typename LSEGridDesc_M,
          index_t NumGemmKPrefetchStage,
          index_t BlockSize,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t KPerBlock,
          index_t Gemm1NPerBlock,
          index_t Gemm1KPerBlock,
          index_t Gemm2KPerBlock,
          index_t AK1Value,
          index_t BK1Value,
          index_t B1K1Value,
          index_t MPerXdl,
          index_t NPerXdl,
          index_t MXdlPerWave,
          index_t NXdlPerWave,
          index_t Gemm1NXdlPerWave,
          index_t Gemm2NXdlPerWave,
          typename ABlockTransferThreadClusterLengths_AK0_M_AK1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          index_t ABlockTransferSrcVectorDim,
          index_t ABlockTransferSrcScalarPerVector,
          index_t ABlockTransferDstScalarPerVector_AK1,
          bool AThreadTransferSrcResetCoordinateAfterRun, // ignored
          index_t ABlockLdsExtraM,
          typename BBlockTransferThreadClusterLengths_BK0_N_BK1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          index_t BBlockTransferSrcVectorDim,
          index_t BBlockTransferSrcScalarPerVector,
          index_t BBlockTransferDstScalarPerVector_BK1,
          bool BThreadTransferSrcResetCoordinateAfterRun, // ignored
          index_t BBlockLdsExtraN,
          index_t D0BlockTransferSrcScalarPerVector,
          typename B1BlockTransferThreadClusterLengths_BK0_N_BK1,
          typename B1BlockTransferThreadClusterArrangeOrder,
          typename B1BlockTransferSrcAccessOrder,
          index_t B1BlockTransferSrcVectorDim,
          index_t B1BlockTransferSrcScalarPerVector,
          index_t B1BlockTransferDstScalarPerVector_BK1,
          bool B1ThreadTransferSrcResetCoordinateAfterRun,
          index_t B1BlockLdsExtraN,
          index_t CShuffleMXdlPerWavePerShuffle,
          index_t CShuffleNXdlPerWavePerShuffle,
          typename CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          index_t CShuffleBlockTransferScalarPerVector_NPerBlock,
          LoopScheduler LoopSched,
          bool PadN,
          bool MaskOutUpperTriangle,
          bool Deterministic,
          PipelineVersion PipelineVer = PipelineVersion::v1>
struct GridwiseBatchedMultiheadAttentionBackward_Qloop_Xdl_CShuffle_Light_V2
{
    static_assert(AK1Value % ABlockTransferDstScalarPerVector_AK1 == 0);
    static_assert(BK1Value % BBlockTransferDstScalarPerVector_BK1 == 0);
    static_assert(B1K1Value % B1BlockTransferDstScalarPerVector_BK1 == 0);

    static_assert(Gemm1NPerBlock % KPerBlock == 0);
    static_assert(MPerBlock % Gemm1KPerBlock == 0);
    static_assert(NPerBlock % Gemm2KPerBlock == 0);

    static_assert(LoopSched == LoopScheduler::Default,
                  "Non-default loop scheduler is currently not supported");

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};
    static constexpr auto I4 = Number<4>{};
    static constexpr auto I5 = Number<5>{};
    static constexpr auto I6 = Number<6>{};
    static constexpr auto I7 = Number<7>{};
    static constexpr auto I8 = Number<8>{};
    static constexpr auto I9 = Number<9>{};
    static constexpr auto I10 = Number<10>{};
    static constexpr auto I11 = Number<11>{};
    static constexpr auto I12 = Number<12>{};
    static constexpr auto I13 = Number<13>{};
    static constexpr auto I14 = Number<14>{};
    static constexpr auto I15 = Number<15>{};
    static constexpr auto I16 = Number<16>{};

    static constexpr auto WaveSize = 64;
    // K1 should be Number<...>
    // Gemm0
    static constexpr auto AK0 = Number<KPerBlock / AK1Value>{};
    static constexpr auto BK0 = Number<KPerBlock / BK1Value>{};
    static constexpr auto AK1 = Number<AK1Value>{};
    static constexpr auto BK1 = Number<BK1Value>{};

    static constexpr auto Gemm0MWaves = MPerBlock / (MPerXdl * MXdlPerWave);
    static constexpr auto Gemm0NWaves = NPerBlock / (NPerXdl * NXdlPerWave);

    // Gemm1
    static constexpr auto B1K0 = Number<Gemm1KPerBlock / B1K1Value>{};
    static constexpr auto B1K1 = Number<B1K1Value>{};

    static constexpr auto mfma = MfmaSelector<GemmDataType, MPerXdl, NPerXdl>::selected_mfma;
    static constexpr auto K_K0 = Number<Gemm1NPerBlock / BK1Value>{};
    static constexpr auto V_K3 = BK1;
    static constexpr auto V_K2 = mfma.num_input_blks;
    static constexpr auto V_K1 = KPerBlock / V_K2 / V_K3;
    static constexpr auto V_K0 = Gemm1NPerBlock / KPerBlock;
    static constexpr auto V_N1 = NXdlPerWave;
    static constexpr auto DropoutNThread = mfma.num_input_blks; // 2
    // get_random_16x8() generates 16 random numbers each time
    static constexpr auto DropoutTile = Number<DropoutNThread * 16>{}; // 32

    using ThisThreadBlock = ThisThreadBlock<BlockSize>;

    using GridwiseGemmPipe = remove_cvref_t<decltype(
        GridwiseGemmPipeline_Selector<PipelineVer, NumGemmKPrefetchStage>())>;

    // C desc for source in blockwise copy
    __host__ __device__ static constexpr auto
    MakeCGridDescriptor_M0_N0_M1_N1_M2_N2_M3_M4_M5_N3(const ZGridDesc_M_N& z_grid_desc_m_n)
    {
        const auto M = z_grid_desc_m_n.GetLength(I0);
        const auto N = z_grid_desc_m_n.GetLength(I1);

        constexpr auto M3 = mfma.num_groups_per_blk;
        constexpr auto M4 = mfma.num_input_blks;
        constexpr auto M5 = mfma.group_size;

        return transform_tensor_descriptor(
            z_grid_desc_m_n,
            make_tuple(make_unmerge_transform(
                           make_tuple(M / MPerBlock, MXdlPerWave, Gemm0MWaves, M3, M4, M5)),
                       make_unmerge_transform(
                           make_tuple(N / NPerBlock, NXdlPerWave, Gemm0NWaves, NPerXdl))),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0, 2, 4, 6, 7, 8>{}, Sequence<1, 3, 5, 9>{}));
    }

    __host__ __device__ static constexpr auto GetPaddedSize(const index_t size)
    {
        return math::integer_divide_ceil(size, DropoutTile) * DropoutTile;
    }

    __device__ static auto GetGemm0WaveIdx()
    {
        const index_t thread_id = get_thread_local_1d_id();

        constexpr auto threadid_to_wave_idx_adaptor = make_single_stage_tensor_adaptor(
            make_tuple(make_merge_transform(make_tuple(Gemm0MWaves, Gemm0NWaves, WaveSize))),
            make_tuple(Sequence<0, 1, 2>{}),
            make_tuple(Sequence<0>{}));

        return threadid_to_wave_idx_adaptor.CalculateBottomIndex(make_multi_index(thread_id));
    }

    __device__ static auto GetGemm0WaveMNIdx(const index_t thread_id)
    {
        constexpr auto wave_threadid_to_mn_idx_adaptor = make_single_stage_tensor_adaptor(
            make_tuple(make_merge_transform(make_tuple(WaveSize / MPerXdl, MPerXdl))),
            make_tuple(Sequence<0, 1>{}),
            make_tuple(Sequence<0>{}));

        return wave_threadid_to_mn_idx_adaptor.CalculateBottomIndex(make_multi_index(thread_id));
    }

    __host__ __device__ static constexpr auto GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1()
    {
        // A matrix in LDS memory, dst of blockwise copy
        return make_naive_tensor_descriptor(
            make_tuple(AK0, Number<MPerBlock>{}, AK1),
            make_tuple(Number<MPerBlock + ABlockLdsExtraM>{} * AK1, AK1, I1));
    }

    __host__ __device__ static constexpr auto GetBBlockDescriptor_BK0PerBlock_NPerBlock_BK1()
    {
        // B matrix in LDS memory, dst of blockwise copy
        return make_naive_tensor_descriptor(
            make_tuple(BK0, Number<NPerBlock>{}, BK1),
            make_tuple(Number<NPerBlock + BBlockLdsExtraN>{} * BK1, BK1, I1));
    }

    __host__ __device__ static constexpr auto GetKBlockDescriptor_K0PerBlock_NPerBlock_K1()
    {
        // K matrix in LDS memory, dst of blockwise copy
        return make_naive_tensor_descriptor(
            make_tuple(K_K0, Number<NPerBlock>{}, BK1),
            make_tuple(Number<NPerBlock + BBlockLdsExtraN>{} * BK1, BK1, I1));
    }

    __host__ __device__ static constexpr auto GetVThreadDescriptor_K0_K1_K2_N0_N1_N2_N3_K3()
    {
        // V matrix in Vgpr, dst of threadwise copy
        return make_naive_tensor_descriptor_packed(
            make_tuple(I1, Number<V_K1>{}, I1, I1, Number<V_N1>{}, I1, I1, Number<V_K3>{}));
    }

    template <typename AccThreadDesc_M0_N0_M1_N1_M2_M3_M4_N2>
    __host__ __device__ static constexpr auto GetA1SrcThreadDescriptor_AK0PerBlock_MPerBlock_AK1(
        const AccThreadDesc_M0_N0_M1_N1_M2_M3_M4_N2& acc_thread_desc_m0_n0_m1_n1_m2_m3_m4_n2)
    {
        // acc_thread_desc_m0_n0_m1_n1_m2_m3_m4_n2 to a_src_thread_desc_k0_m_k1
        // m0_m1_m2_m3 -> k0
        // n0_n1_n2 -> m
        // m4 -> k1
        // NOTE: had to use merge_v3 or will spit out compilation errors
        const auto m0 = acc_thread_desc_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I0);
        const auto n0 = acc_thread_desc_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I1);
        const auto m1 = acc_thread_desc_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I2);
        const auto n1 = acc_thread_desc_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I3);
        const auto m2 = acc_thread_desc_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I4);
        const auto m3 = acc_thread_desc_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I5);
        const auto m4 = acc_thread_desc_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I6);
        const auto n2 = acc_thread_desc_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I7);

        return transform_tensor_descriptor(
            acc_thread_desc_m0_n0_m1_n1_m2_m3_m4_n2,
            make_tuple(make_merge_transform_v3_division_mod(make_tuple(m0, m1, m2, m3)),
                       make_merge_transform_v3_division_mod(make_tuple(n0, n1, n2)),
                       make_pass_through_transform(m4)),
            make_tuple(Sequence<0, 2, 4, 5>{}, Sequence<1, 3, 7>{}, Sequence<6>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));
    }

    __host__ __device__ static constexpr auto GetB1BlockDescriptor_BK0PerBlock_NPerBlock_BK1()
    {
        // B1 matrix in LDS memory, dst of blockwise copy
        return make_naive_tensor_descriptor(
            make_tuple(B1K0, Number<Gemm1NPerBlock>{}, B1K1),
            make_tuple(Number<Gemm1NPerBlock + B1BlockLdsExtraN>{} * B1K1, B1K1, I1));
    }

    __host__ __device__ static constexpr auto
    GetCShuffleBlockDescriptor_MBlock_MPerBlock_NBlock_NPerBlock()
    {
        constexpr index_t MWave = Gemm0NWaves;
        constexpr index_t NWave = Gemm0MWaves;

        constexpr auto c_shuffle_block_desc_mblock_mperblock_nblock_nperblock =
            make_naive_tensor_descriptor_packed(
                make_tuple(I1,
                           Number<CShuffleMXdlPerWavePerShuffle * MWave * MPerXdl>{},
                           I1,
                           Number<CShuffleNXdlPerWavePerShuffle * NWave * NPerXdl>{}));

        return c_shuffle_block_desc_mblock_mperblock_nblock_nperblock;
    }

    template <typename Gemm2Param>
    __host__ __device__ static constexpr auto GetA2BlockDescriptor_K0_M_K1()
    {
        return make_naive_tensor_descriptor(
            make_tuple(Number<Gemm2Param::A_K0>{},
                       Number<Gemm2Param::Gemm2_M>{},
                       Number<Gemm2Param::A_K1>{}),
            make_tuple(Number<Gemm2Param::Gemm2_M + Gemm2Param::A_LdsPad>{} *
                           Number<Gemm2Param::A_K1>{},
                       Number<Gemm2Param::A_K1>{},
                       I1));
    }

    template <typename Gemm2Param>
    __host__ __device__ static constexpr auto GetB2BlockDescriptor_K0_N_K1()
    {
        return make_naive_tensor_descriptor(
            make_tuple(Number<Gemm2Param::B_K0>{},
                       Number<Gemm2Param::Gemm2_N>{},
                       Number<Gemm2Param::B_K1>{}),
            make_tuple(Number<Gemm2Param::Gemm2_N + Gemm2Param::B_LdsPad>{} *
                           Number<Gemm2Param::B_K1>{},
                       Number<Gemm2Param::B_K1>{},
                       I1));
    }

    __host__ __device__ static constexpr bool
    CheckValidity(const QGridDesc_K0_M_K1& q_grid_desc_k0_m_k1,
                  const KGridDesc_K0_N_K1& k_grid_desc_k0_n_k1,
                  const VGridDesc_O0_N_O1& v_grid_desc_o0_n_o1,
                  const YGridDesc_M_O& y_grid_desc_m_o)
    {
        static_assert((MPerBlock % (MPerXdl * MXdlPerWave) == 0) &&
                          (NPerBlock % (NXdlPerWave * NPerXdl)) == 0,
                      "Invalid tuning param!");

        const auto M = q_grid_desc_k0_m_k1.GetLength(I1);
        const auto N = k_grid_desc_k0_n_k1.GetLength(I1);
        const auto K = q_grid_desc_k0_m_k1.GetLength(I0) * q_grid_desc_k0_m_k1.GetLength(I2);
        const auto O = v_grid_desc_o0_n_o1.GetLength(I0) * v_grid_desc_o0_n_o1.GetLength(I2);

        // This assumption reduces implemention complexity by categorizing 6 separate GEMMs into 3
        // types of GEMM operations, therefore some code body can be reused accordingly
        // P_MNK / dP_MNO Gemm (Gemm0 rcr)
        // Y_MON / dQ_MKN Gemm (Gemm1 rrr)
        // dV_NOM / dK_NKM Gemm (Gemm2 crr)
        if(O != K)
        {
            std::cerr << "SizeK must be equal to SizeO (equal attention head size)" << '\n';
            return false;
        }

        if(!(M == y_grid_desc_m_o.GetLength(I0) && O == y_grid_desc_m_o.GetLength(I1)))
        {
            return false;
        }

        if(!(M % MPerBlock == 0 && N % NPerBlock == 0 && K % KPerBlock == 0 &&
             O % Gemm1NPerBlock == 0))
        {
            return false;
        }

        // check gemm0 gridwise gemm pipeline
        const auto num_gemm0_k_loop = K / KPerBlock;
        if(!GridwiseGemmPipe::IsSupported(num_gemm0_k_loop))
        {
            return false;
        }

        // check gemm1 gridwise gemm pipeline
        if(!(NPerBlock % Gemm1KPerBlock == 0))
        {
            return false;
        }

        const auto num_gemm1_k_inner_loop = NPerBlock / Gemm1KPerBlock;
        if(!GridwiseGemmPipe::IsSupported(num_gemm1_k_inner_loop))
        {
            return false;
        }

        // TODO: also check validity of all components (blockwise-copy, threadwise-copy, etc)
        return true;
    }

    __host__ __device__ static constexpr bool CalculateHasMainKBlockLoop(index_t K)
    {
        const index_t num_loop = K / KPerBlock;

        return GridwiseGemmPipe::CalculateHasMainLoop(num_loop);
    }

    template <typename SrcBlockwiseGemm>
    __host__ __device__ static constexpr auto
    MakeLSEGridDescriptor_MB_M0_M1_M2_M3_M4(const LSEGridDesc_M& lse_grid_desc_m)
    {
        const index_t M      = lse_grid_desc_m.GetLength(I0);
        const index_t MBlock = M / MPerBlock;
        constexpr auto SrcBlockDesc_M0_N0_M1_N1_M2_M3_M4_N2 =
            SrcBlockwiseGemm::GetCBlockDescriptor_M0_N0_M1_N1_M2_M3_M4_N2();
        // M0 MXdlPerWave, M1 MWave, M2 num_groups_per_blk, M3 num_input_blks, M4 group_size
        const auto M0 = SrcBlockDesc_M0_N0_M1_N1_M2_M3_M4_N2.GetLength(I0);
        const auto M1 = SrcBlockDesc_M0_N0_M1_N1_M2_M3_M4_N2.GetLength(I2);
        const auto M2 = SrcBlockDesc_M0_N0_M1_N1_M2_M3_M4_N2.GetLength(I4);
        const auto M3 = SrcBlockDesc_M0_N0_M1_N1_M2_M3_M4_N2.GetLength(I5);
        const auto M4 = SrcBlockDesc_M0_N0_M1_N1_M2_M3_M4_N2.GetLength(I6);

        const auto lse_grid_desc_mb_m0_m1_m2_m3_m4 = transform_tensor_descriptor(
            lse_grid_desc_m,
            make_tuple(make_unmerge_transform(make_tuple(MBlock, M0, M1, M2, M3, M4))),
            make_tuple(Sequence<0>{}),
            make_tuple(Sequence<0, 1, 2, 3, 4, 5>{}));

        return lse_grid_desc_mb_m0_m1_m2_m3_m4;
    }

    __device__ static auto MakeKGradGridDesc_NBlock_NPerBlock_OBlock_OPerBlock(
        const KGridDesc_K0_N_K1& k_grid_desc_k0_n_k1)
    {
        const auto O0 = k_grid_desc_k0_n_k1.GetLength(I0);
        const auto N  = k_grid_desc_k0_n_k1.GetLength(I1);
        const auto O1 = k_grid_desc_k0_n_k1.GetLength(I2);
        const auto O  = O0 * O1;

        const auto NBlock = N / NPerBlock;
        const auto OBlock = O / Gemm1NPerBlock;

        const auto k_grid_desc_n_o = transform_tensor_descriptor(
            k_grid_desc_k0_n_k1,
            make_tuple(make_pass_through_transform(N),
                       make_merge_transform_v3_division_mod(make_tuple(O0, O1))),
            make_tuple(Sequence<1>{}, Sequence<0, 2>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        return transform_tensor_descriptor(
            k_grid_desc_n_o,
            make_tuple(make_unmerge_transform(make_tuple(NBlock, Number<NPerBlock>{})),
                       make_unmerge_transform(make_tuple(OBlock, Number<Gemm1NPerBlock>{}))),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0, 1>{}, Sequence<2, 3>{}));
    }

    __device__ static auto MakeVGradGridDesc_NBlock_NPerBlock_OBlock_OPerBlock(
        const VGridDesc_O0_N_O1& v_grid_desc_o0_n_o1)
    {
        const auto O0 = v_grid_desc_o0_n_o1.GetLength(I0);
        const auto N  = v_grid_desc_o0_n_o1.GetLength(I1);
        const auto O1 = v_grid_desc_o0_n_o1.GetLength(I2);
        const auto O  = O0 * O1;

        const auto NBlock = N / NPerBlock;
        const auto OBlock = O / Gemm1NPerBlock;

        const auto v_grid_desc_n_o = transform_tensor_descriptor(
            v_grid_desc_o0_n_o1,
            make_tuple(make_pass_through_transform(N),
                       make_merge_transform_v3_division_mod(make_tuple(O0, O1))),
            make_tuple(Sequence<1>{}, Sequence<0, 2>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        return transform_tensor_descriptor(
            v_grid_desc_n_o,
            make_tuple(make_unmerge_transform(make_tuple(NBlock, Number<NPerBlock>{})),
                       make_unmerge_transform(make_tuple(OBlock, Number<Gemm1NPerBlock>{}))),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0, 1>{}, Sequence<2, 3>{}));
    }

    __device__ static auto MakeQGradGridDesc_M_K(const QGridDesc_K0_M_K1& q_grid_desc_k0_m_k1)
    {
        const auto K0_ = q_grid_desc_k0_m_k1.GetLength(I0);
        const auto M_  = q_grid_desc_k0_m_k1.GetLength(I1);
        const auto K1_ = q_grid_desc_k0_m_k1.GetLength(I2);

        return transform_tensor_descriptor(
            q_grid_desc_k0_m_k1,
            make_tuple(make_pass_through_transform(M_),
                       make_merge_transform_v3_division_mod(make_tuple(K0_, K1_))),
            make_tuple(Sequence<1>{}, Sequence<0, 2>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));
    }

    // return block_id to C matrix tile idx (m0, n0) mapping
    __host__ __device__ static constexpr auto
    MakeDefaultBlock2CTileMap(const KGridDesc_N_K& k_grid_desc_n_k)
    {
        return BlockToCTileMap_M00_N0_M01Adapt<NPerBlock, Gemm1NPerBlock, KGridDesc_N_K>(
            k_grid_desc_n_k);
    }

    using DefaultBlock2CTileMap =
        remove_cvref_t<decltype(MakeDefaultBlock2CTileMap(KGridDesc_N_K{}))>;

    using ZGridDescriptor_M0_N0_M1_N1_M2_N2_M3_M4_M5_N3 = remove_cvref_t<decltype(
        MakeCGridDescriptor_M0_N0_M1_N1_M2_N2_M3_M4_M5_N3(ZGridDesc_M_N{}))>;

    // K / V
    struct GemmBlockwiseCopy
    {
        __device__ static auto
        MakeVGridDescriptor_K0_K1_K2_N0_N1_N2_N3_K3(const VGridDesc_O0_N_O1& v_grid_desc_o0_n_o1)
        {
            const auto K0_ = v_grid_desc_o0_n_o1.GetLength(I0);
            const auto N_  = v_grid_desc_o0_n_o1.GetLength(I1);
            const auto K1_ = v_grid_desc_o0_n_o1.GetLength(I2);

            constexpr auto V_N3 = NPerXdl;
            constexpr auto V_N2 = Gemm0NWaves;
            const auto V_N0     = N_ / NPerBlock;

            const auto v_grid_desc_n_k = transform_tensor_descriptor(
                v_grid_desc_o0_n_o1,
                make_tuple(make_pass_through_transform(N_),
                           make_merge_transform_v3_division_mod(make_tuple(K0_, K1_))),
                make_tuple(Sequence<1>{}, Sequence<0, 2>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            return transform_tensor_descriptor(
                v_grid_desc_n_k,
                make_tuple(make_unmerge_transform(make_tuple(V_N0, V_N1, V_N2, V_N3)),
                           make_unmerge_transform(make_tuple(V_K0, V_K1, V_K2, V_K3))),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<3, 4, 5, 6>{}, Sequence<0, 1, 2, 7>{}));
        }

        // K matrix in LDS, dst of blockwise copy
        static constexpr auto k_block_desc_k0_n_k1 = GetKBlockDescriptor_K0PerBlock_NPerBlock_K1();

        // V matrix in Vgpr, dst of threadwise copy
        static constexpr auto v_thread_desc_k0_k1_k2_n0_n1_n2_n3_k3 =
            GetVThreadDescriptor_K0_K1_K2_N0_N1_N2_N3_K3();

        template <typename GridDesc_K0_N_K1>
        using KBlockwiseCopy =
            ThreadGroupTensorSliceTransfer_v4r1<ThisThreadBlock,
                                                tensor_operation::element_wise::PassThrough,
                                                tensor_operation::element_wise::PassThrough,
                                                InMemoryDataOperationEnum::Set,
                                                Sequence<K_K0, NPerBlock, BK1>,
                                                BBlockTransferThreadClusterLengths_BK0_N_BK1,
                                                BBlockTransferThreadClusterArrangeOrder,
                                                InputDataType,
                                                GemmDataType,
                                                GridDesc_K0_N_K1,
                                                decltype(k_block_desc_k0_n_k1),
                                                BBlockTransferSrcAccessOrder,
                                                Sequence<1, 0, 2>,
                                                BBlockTransferSrcVectorDim,
                                                2,
                                                BBlockTransferSrcScalarPerVector,
                                                BBlockTransferDstScalarPerVector_BK1,
                                                1,
                                                1,
                                                true, // SrcResetCoord
                                                true, // DstResetCoord
                                                NumGemmKPrefetchStage>;

        template <typename GridDesc_K0_K1_k2_N0_N1_N2_N3_K3>
        using VBlockwiseCopy = ThreadwiseTensorSliceTransfer_v2<
            InputDataType,
            GemmDataType,
            GridDesc_K0_K1_k2_N0_N1_N2_N3_K3,
            decltype(v_thread_desc_k0_k1_k2_n0_n1_n2_n3_k3),
            decltype(v_thread_desc_k0_k1_k2_n0_n1_n2_n3_k3.GetLengths()),
            Sequence<0, 1, 2, 3, 4, 5, 6, 7>,
            7,
            BK1,
            1,
            true /* ResetCoordAfterRun */>;

        static constexpr auto VBlockBufferSize        = V_K0;
        static constexpr auto v_block_slice_copy_step = make_multi_index(1, 0, 0, 0, 0, 0, 0, 0);
    };

    // dP Gemm (type 1 rcc, B in Vgpr)
    template <typename BSrcThreadDesc_K0_K1_K2_N0_N1_N2_N3_K3>
    struct Gemm0
    {
        // A matrix in LDS memory, dst of blockwise copy
        static constexpr auto a_block_desc_ak0_m_ak1 =
            GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1();

        template <typename BThreadDesc_K0_K1_K2_N0_N1_N2_N3_K3>
        __host__ __device__ static constexpr auto GetBThreadDescriptor_K0_N_K1(
            const BThreadDesc_K0_K1_K2_N0_N1_N2_N3_K3& b_thread_desc_k0_k1_k2_n0_n1_n2_n3_k3)
        {
            // b_thread_desc_k0_k1_k2_n0_n1_n2_n3_k3 to b_thread_desc_k0_n_k1
            // k0_k1_k2 -> k0
            // n0_n1_n2_n3 -> n
            // k3 -> k1
            const auto k0 = b_thread_desc_k0_k1_k2_n0_n1_n2_n3_k3.GetLength(I0);
            const auto k1 = b_thread_desc_k0_k1_k2_n0_n1_n2_n3_k3.GetLength(I1);
            const auto k2 = b_thread_desc_k0_k1_k2_n0_n1_n2_n3_k3.GetLength(I2);
            const auto n0 = b_thread_desc_k0_k1_k2_n0_n1_n2_n3_k3.GetLength(I3);
            const auto n1 = b_thread_desc_k0_k1_k2_n0_n1_n2_n3_k3.GetLength(I4);
            const auto n2 = b_thread_desc_k0_k1_k2_n0_n1_n2_n3_k3.GetLength(I5);
            const auto n3 = b_thread_desc_k0_k1_k2_n0_n1_n2_n3_k3.GetLength(I6);
            const auto k3 = b_thread_desc_k0_k1_k2_n0_n1_n2_n3_k3.GetLength(I7);

            return transform_tensor_descriptor(
                b_thread_desc_k0_k1_k2_n0_n1_n2_n3_k3,
                make_tuple(make_merge_transform_v3_division_mod(make_tuple(k0, k1, k2)),
                           make_merge_transform_v3_division_mod(make_tuple(n0, n1, n2, n3)),
                           make_pass_through_transform(k3)),
                make_tuple(Sequence<0, 1, 2>{}, Sequence<3, 4, 5, 6>{}, Sequence<7>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));
        }

        static constexpr auto b_src_thread_desc_k0_n_k1 =
            GetBThreadDescriptor_K0_N_K1(BSrcThreadDesc_K0_K1_K2_N0_N1_N2_N3_K3{});

        template <typename ABlockDesc_AK0_M_AK1>
        __host__ __device__ static constexpr auto
        MakeGemm0AMmaTileDescriptor_M0_M1_M2_K(const ABlockDesc_AK0_M_AK1&)
        {
            constexpr index_t MWaves = MPerBlock / (MXdlPerWave * MPerXdl);

            return MakeGemmMmaTileDescriptor_MN0_MN1_MN2_K<MXdlPerWave, MWaves, MPerXdl>(
                ABlockDesc_AK0_M_AK1{});
        }

        template <typename BBlockDesc_BK0_N_BK1>
        __host__ __device__ static constexpr auto
        MakeGemm0BMmaTileDescriptor_N0_N1_N2_K(const BBlockDesc_BK0_N_BK1&)
        {
            return MakeGemmMmaTileDescriptor_MN0_MN1_MN2_K<NXdlPerWave, 1, 1>(
                BBlockDesc_BK0_N_BK1{});
        }

        template <typename GridDesc_K0_M_K1>
        using ABlockwiseCopy =
            ThreadGroupTensorSliceTransfer_v4r1<ThisThreadBlock,
                                                AElementwiseOperation,
                                                tensor_operation::element_wise::PassThrough,
                                                InMemoryDataOperationEnum::Set,
                                                Sequence<AK0, MPerBlock, AK1>,
                                                ABlockTransferThreadClusterLengths_AK0_M_AK1,
                                                ABlockTransferThreadClusterArrangeOrder,
                                                InputDataType,
                                                GemmDataType,
                                                GridDesc_K0_M_K1,
                                                decltype(a_block_desc_ak0_m_ak1),
                                                ABlockTransferSrcAccessOrder,
                                                Sequence<1, 0, 2>,
                                                ABlockTransferSrcVectorDim,
                                                2,
                                                ABlockTransferSrcScalarPerVector,
                                                ABlockTransferDstScalarPerVector_AK1,
                                                1,
                                                1,
                                                true, // SrcResetCoord
                                                true, // DstResetCoord
                                                NumGemmKPrefetchStage>;

        static constexpr index_t KPack = math::max(math::lcm(AK1, BK1), mfma.k_per_blk);

        // Blockwise gemm with transposed XDL output
        using BlockwiseGemm = BlockwiseGemmXdlops_v2_dl<
            BlockSize,
            GemmDataType,
            FloatGemmAcc,
            decltype(a_block_desc_ak0_m_ak1),
            decltype(b_src_thread_desc_k0_n_k1),
            decltype(MakeGemm0AMmaTileDescriptor_M0_M1_M2_K(a_block_desc_ak0_m_ak1)),
            decltype(MakeGemm0BMmaTileDescriptor_N0_N1_N2_K(b_src_thread_desc_k0_n_k1)),
            MPerBlock,
            NPerBlock,
            KPerBlock,
            MPerXdl,
            NPerXdl,
            MXdlPerWave,
            NXdlPerWave,
            KPack,
            Gemm1NPerBlock,
            false,
            KPack * XdlopsGemm<GemmDataType, MPerXdl, NPerXdl, KPack, false>{}.K0PerXdlops,
            KPack>;

        static constexpr auto a_block_slice_copy_step = make_multi_index(KPerBlock / AK1, 0, 0);
    };

    // dV / dK Gemm (type 2 rrr)
    template <typename ASrcThreadDesc_M0_N0_M1_N1_M2_M3_M4_N2,
              typename ASrcBlockDesc_M0_N0_M1_N1_M2_M3_M4_N2>
    struct Gemm1
    {
        private:
        static constexpr auto m0 = ASrcThreadDesc_M0_N0_M1_N1_M2_M3_M4_N2{}.GetLength(I0);
        static constexpr auto n0 = ASrcThreadDesc_M0_N0_M1_N1_M2_M3_M4_N2{}.GetLength(I1);
        static constexpr auto m1 = ASrcThreadDesc_M0_N0_M1_N1_M2_M3_M4_N2{}.GetLength(I2);
        static constexpr auto n1 = ASrcThreadDesc_M0_N0_M1_N1_M2_M3_M4_N2{}.GetLength(I3);
        static constexpr auto m2 = ASrcThreadDesc_M0_N0_M1_N1_M2_M3_M4_N2{}.GetLength(I4);
        static constexpr auto m3 = ASrcThreadDesc_M0_N0_M1_N1_M2_M3_M4_N2{}.GetLength(I5);
        static constexpr auto m4 = ASrcThreadDesc_M0_N0_M1_N1_M2_M3_M4_N2{}.GetLength(I6);
        static constexpr auto n2 = ASrcThreadDesc_M0_N0_M1_N1_M2_M3_M4_N2{}.GetLength(I7);

        // M2 num_groups_per_blk, M3 num_input_blks, M4 group_size
        static constexpr auto M3 = ASrcBlockDesc_M0_N0_M1_N1_M2_M3_M4_N2{}.GetLength(I5);

        public:
        static constexpr auto AThreadSliceLength_K0 = Number<Gemm1KPerBlock / m4 / M3>{};
        static constexpr auto AThreadSliceLength_M  = Number<n0 * n1 * n2>{};
        static constexpr auto AThreadSliceLength_K1 = Number<m4>{};

        // A source matrix layout in AccVGPR
        static constexpr auto a_src_thread_desc_k0_m_k1 =
            GetA1SrcThreadDescriptor_AK0PerBlock_MPerBlock_AK1(
                ASrcThreadDesc_M0_N0_M1_N1_M2_M3_M4_N2{});

        // A matrix in VGPR memory, dst of AccVGPR-to-VGPR copy
        static constexpr auto a_thread_desc_k0_m_k1 = make_naive_tensor_descriptor_packed(
            make_tuple(AThreadSliceLength_K0, AThreadSliceLength_M, AThreadSliceLength_K1));

        // B matrix in LDS memory, dst of blockwise copy
        static constexpr auto b_block_desc_bk0_n_bk1 =
            GetB1BlockDescriptor_BK0PerBlock_NPerBlock_BK1();

        template <typename ABlockDesc_AK0_M_AK1>
        __host__ __device__ static constexpr auto
        MakeGemm1AMmaTileDescriptor_M0_M1_M2_K(const ABlockDesc_AK0_M_AK1&)
        {
            return MakeGemmMmaTileDescriptor_MN0_MN1_MN2_K<NXdlPerWave, 1, 1>(
                ABlockDesc_AK0_M_AK1{});
        }

        template <typename BBlockDesc_BK0_N_BK1>
        __host__ __device__ static constexpr auto
        MakeGemm1BMmaTileDescriptor_N0_N1_N2_K(const BBlockDesc_BK0_N_BK1&)
        {
            constexpr index_t Gemm1NWaves = Gemm1NPerBlock / (Gemm1NXdlPerWave * NPerXdl);
            return MakeGemmMmaTileDescriptor_MN0_MN1_MN2_K<Gemm1NXdlPerWave, Gemm1NWaves, NPerXdl>(
                BBlockDesc_BK0_N_BK1{});
        }

        static constexpr auto ASrcScalarPerVector = m4;

        using AThreadSliceLengths_K0_M_K1 = decltype(a_thread_desc_k0_m_k1.GetLengths());

        template <typename ElementwiseOp = tensor_operation::element_wise::PassThrough>
        using ABlockwiseCopy =
            ThreadwiseTensorSliceTransfer_StaticToStatic<FloatGemmAcc,
                                                         GemmDataType,
                                                         decltype(a_src_thread_desc_k0_m_k1),
                                                         decltype(a_thread_desc_k0_m_k1),
                                                         ElementwiseOp,
                                                         AThreadSliceLengths_K0_M_K1,
                                                         Sequence<1, 0, 2>,
                                                         2,
                                                         ASrcScalarPerVector>;

        template <typename GridDesc_K0_N_K1>
        using BBlockwiseCopy =
            ThreadGroupTensorSliceTransfer_v4r1<ThisThreadBlock,
                                                BElementwiseOperation,
                                                tensor_operation::element_wise::PassThrough,
                                                InMemoryDataOperationEnum::Set,
                                                Sequence<B1K0, Gemm1NPerBlock, B1K1>,
                                                B1BlockTransferThreadClusterLengths_BK0_N_BK1,
                                                B1BlockTransferThreadClusterArrangeOrder,
                                                InputDataType,
                                                GemmDataType,
                                                GridDesc_K0_N_K1,
                                                decltype(b_block_desc_bk0_n_bk1),
                                                B1BlockTransferSrcAccessOrder,
                                                Sequence<1, 0, 2>,
                                                B1BlockTransferSrcVectorDim,
                                                2,
                                                B1BlockTransferSrcScalarPerVector,
                                                B1BlockTransferDstScalarPerVector_BK1,
                                                1,
                                                1,
                                                B1ThreadTransferSrcResetCoordinateAfterRun,
                                                true, // DstResetCoord
                                                NumGemmKPrefetchStage>;

        // for a_block_slice_copy_step to be able to address static buffers, it MUST be a
        // tuple-based container as well as containing ONLY integral constants
        static constexpr auto a_block_slice_copy_step = make_tuple(AThreadSliceLength_K0, I0, I0);
        static constexpr auto b_block_slice_copy_step =
            make_multi_index(Gemm1KPerBlock / B1K1, 0, 0);

        // selected_mfma.group_size or B1K1 <= Gemm1KPack <= selected_mfma.group_size
        // selected_mfma.k_per_blk <= Gemm1KPack
        //
        // Following similar rationale behind Gemm0KPack, let Gemm1KPack be the lowest common
        // multiples of A1K1 (predetermined by selected_mfma.group_size) and B1K1. But in this case
        // Gemm1KPack can't be higher than A1K1 itself because A1 matrix is distributed in VGPRs
        // with 'group_size' amount of contiguous elements. Having Gemm1KPack greater than A1K1 will
        // cause mismatch in summation index for example c[0:7] = a1[[0:3, 8:11]] * b1[0:7].
        // therefore we may just as well assign Gemm1KPack = group_size
        static constexpr index_t GemmKPack = mfma.group_size;

        using BlockwiseGemm = BlockwiseGemmXdlops_v2_dl<
            BlockSize,
            GemmDataType,
            FloatGemmAcc,
            decltype(a_thread_desc_k0_m_k1),
            decltype(b_block_desc_bk0_n_bk1),
            decltype(MakeGemm1AMmaTileDescriptor_M0_M1_M2_K(a_thread_desc_k0_m_k1)),
            decltype(MakeGemm1BMmaTileDescriptor_N0_N1_N2_K(b_block_desc_bk0_n_bk1)),
            NPerBlock,
            Gemm1NPerBlock,
            Gemm1KPerBlock,
            MPerXdl,
            NPerXdl,
            NXdlPerWave,
            Gemm1NXdlPerWave,
            GemmKPack,
            Gemm1NPerBlock,
            true,      // TransposeC
            GemmKPack, // AMmaKStride
            GemmKPack * XdlopsGemm<GemmDataType, MPerXdl, NPerXdl, GemmKPack, false>{}
                            .K0PerXdlops /* BMmaKStride */>;
    };

    // dQ Gemm (type 3 crr)
    // Describes tuning parameter for C2_m_n = A2_m_k * B2_k_n
    struct Gemm2Params
    {
        static constexpr index_t Gemm2_M = MPerBlock;      // 64
        static constexpr index_t Gemm2_K = NPerBlock;      // 128
        static constexpr index_t Gemm2_N = Gemm1NPerBlock; // 128
        static constexpr index_t Sum_K   = Gemm2KPerBlock;

        static constexpr index_t A_K1     = 8; // dS will be row-major
        static constexpr index_t A_K0     = Sum_K / A_K1;
        static constexpr index_t A_LdsPad = 0; // how many multiples of K1 per M * K1 elements

        static_assert(Sum_K % NPerXdl == 0, "");

        static constexpr index_t GemmNWave   = Gemm2_N / Gemm2NXdlPerWave / NPerXdl;    // 1 // 2
        static constexpr index_t GemmMWave   = BlockSize / get_warp_size() / GemmNWave; // 4 // 2
        static constexpr index_t GemmNRepeat = Gemm2NXdlPerWave;                        // 1 // 1
        static constexpr index_t GemmMRepeat = Gemm2_M / GemmMWave / MPerXdl;           // 1 // 1
        static constexpr index_t GemmKLoop   = Gemm2_K / Sum_K;                         // 2 // 2
        static constexpr index_t GemmKPack   = math::max(A_K1, mfma.k_per_blk);
        static constexpr index_t B_K3        = GemmKPack; // 8
        static constexpr index_t B_K2 =
            XdlopsGemm<GemmDataType, MPerXdl, NPerXdl, GemmKPack, false>{}.K0PerXdlops; // 2
        static constexpr index_t B_K1 = Sum_K / B_K2 / B_K3;                            // 4
        static constexpr index_t B_K0 = GemmKLoop;                                      // 2

        __host__ __device__ static constexpr auto GetABlockSliceLengths_M0_K0_M1_K1_M2_K2()
        {
            // perform manual unmerge: n -> n_repeat, n_waves, n_per_xdl
            constexpr index_t k  = Sum_K - 1;
            constexpr index_t k2 = k % NPerXdl;
            constexpr index_t k1 = k / NPerXdl % Gemm0NWaves;
            constexpr index_t k0 = k / NPerXdl / Gemm0NWaves % NXdlPerWave;

            // perform manual unmerge: m -> m_repeat, m_waves, m_per_xdl
            constexpr index_t m  = Gemm2_M - 1;
            constexpr index_t m2 = m % MPerXdl;
            constexpr index_t m1 = m / MPerXdl % Gemm0MWaves;
            constexpr index_t m0 = m / MPerXdl / Gemm0MWaves % MXdlPerWave;

            // assume 256 decomposed into 2 x 4 x 32
            // 1d idx ( 32 - 1) -> 3d idx 0, 0, 31 -> 3d dim 1 x 1 x 32
            // 1d idx (256 - 1) -> 3d idx 1, 3, 31 -> 3d dim 2 x 4 x 32
            return Sequence<m0, k0, m1, k1, m2, k2>{} + Sequence<1, 1, 1, 1, 1, 1>{};
        }

        __host__ __device__ static constexpr auto GetABlockSliceLengths_M0_K0_M1_K1()
        {
            return generate_sequence_v2(
                [](auto I) { return GetABlockSliceLengths_M0_K0_M1_K1_M2_K2().At(I); },
                Number<4>{});
        }

        using ABlockSliceLengths_M0_K0_M1_K1 =
            decltype(GetABlockSliceLengths_M0_K0_M1_K1()); //(2, 1, 1, 2) //(4, 1, 1, 2)
    };

    // dQ Gemm (type 3 crr)
    template <typename Gemm2Params, typename ASrcBlockwiseGemm, typename BSrcBlockDesc_N0_K_N1>
    struct Gemm2
    {
        private:
        static constexpr auto a_src_block_desc_m0_n0_m1_n1_m2_m3_m4_n2 =
            ASrcBlockwiseGemm::GetCBlockDescriptor_M0_N0_M1_N1_M2_M3_M4_N2();
        static constexpr auto M0 = a_src_block_desc_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I0); // repeat
        static constexpr auto N0 = a_src_block_desc_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I1);
        static constexpr auto M1 = a_src_block_desc_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I2); // wave
        static constexpr auto N1 = a_src_block_desc_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I3);
        static constexpr auto M2 = a_src_block_desc_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I4); // xdl
        static constexpr auto M3 = a_src_block_desc_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I5);
        static constexpr auto M4 = a_src_block_desc_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I6);
        static constexpr auto N2 = a_src_block_desc_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I7);

        public:
        // A source matrix layout in VGPR, src of VGPR-to-LDS copy
        static constexpr auto a_src_thread_desc_m0_n0_m1_n1_m2_m3_m4_n2 =
            ASrcBlockwiseGemm::GetCThreadDescriptor_M0_N0_M1_N1_M2_M3_M4_N2();

        // A matrix in LDS memory, dst of blockwise copy
        static constexpr auto a_block_desc_k0_m_k1 = GetA2BlockDescriptor_K0_M_K1<Gemm2Params>();

        template <typename ABlockDesc_K0_M_K1>
        __host__ __device__ static constexpr auto
        MakeGemm2AMmaTileDescriptor_M0_M1_M2_K(const ABlockDesc_K0_M_K1&)
        {
            return MakeGemmMmaTileDescriptor_MN0_MN1_MN2_K<Gemm2Params::GemmMRepeat,
                                                           Gemm2Params::GemmMWave,
                                                           MPerXdl>(ABlockDesc_K0_M_K1{});
        }

        template <typename BBlockDesc_K0_N_K1>
        __host__ __device__ static constexpr auto
        MakeGemm2BMmaTileDescriptor_N0_N1_N2_K(const BBlockDesc_K0_N_K1&)
        {
            return MakeGemmMmaTileDescriptor_MN0_MN1_MN2_K<Gemm2Params::GemmNRepeat, 1, 1>(
                BBlockDesc_K0_N_K1{});
        }

        __host__ __device__ static constexpr auto MakeABlockDesc_M0_K0_M1_K1_M2_M3_M4_K2()
        {
            const auto K0_ = a_block_desc_k0_m_k1.GetLength(I0);
            const auto M_  = a_block_desc_k0_m_k1.GetLength(I1);
            const auto K1_ = a_block_desc_k0_m_k1.GetLength(I2);

            const auto a_block_desc_m_k = transform_tensor_descriptor(
                a_block_desc_k0_m_k1,
                make_tuple(make_merge_transform_v3_division_mod(make_tuple(K0_, K1_)),
                           make_pass_through_transform(M_)),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}),
                make_tuple(Sequence<1>{}, Sequence<0>{}));

            // HACK: for unmerge transform, the length of highest dim is irrelevant so we put dummy
            // variable I1 there
            return transform_tensor_descriptor(
                a_block_desc_m_k,
                make_tuple(make_unmerge_transform(make_tuple(I1, M1, M2, M3, M4)),
                           make_unmerge_transform(make_tuple(I1, N1, N2))),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 2, 4, 5, 6>{}, Sequence<1, 3, 7>{}));
        }

        // Note: we will perform sub-workgroup VGPR-to-LDS copy to save LDS space, therefore the
        // destination coordinate can overlap between wavefronts in a workgroup as seen in the mod
        // operation before returning the values
        __host__ __device__ static auto MakeAThreadOriginOnBlock_M0_K0_M1_K1_M2_M3_M4_K2()
        {
            const auto a_thread_origin_on_block_idx =
                ASrcBlockwiseGemm::CalculateCThreadOriginDataIndex8D(I0, I0, I0, I0);

            constexpr auto a_block_slice_lengths_m0_k0_m1_k1 =
                typename Gemm2Params::ABlockSliceLengths_M0_K0_M1_K1{}; // mrepeat, nrepeat,
                                                                        // mwaves, nwaves,

            return make_tuple(
                a_thread_origin_on_block_idx[I0],                                         // mrepeat
                a_thread_origin_on_block_idx[I1],                                         // nrepeat
                a_thread_origin_on_block_idx[I2] % a_block_slice_lengths_m0_k0_m1_k1[I2], // mwave
                a_thread_origin_on_block_idx[I3] % a_block_slice_lengths_m0_k0_m1_k1[I3], // nwave
                a_thread_origin_on_block_idx[I4],                                         // xdlops
                a_thread_origin_on_block_idx[I5],
                a_thread_origin_on_block_idx[I6],
                a_thread_origin_on_block_idx[I7]);
        }

        static constexpr auto a_block_desc_m0_k0_m1_k1_m2_m3_m4_k2 =
            MakeABlockDesc_M0_K0_M1_K1_M2_M3_M4_K2();

        using ASrcBlockSliceWindowIterator =
            SpaceFillingCurve<Sequence<M0, N0, M1, N1>,
                              Sequence<0, 1, 2, 3>,
                              typename Gemm2Params::ABlockSliceLengths_M0_K0_M1_K1,
                              false>;

        template <typename ElementwiseOp = tensor_operation::element_wise::PassThrough>
        using ABlockwiseCopy = ThreadwiseTensorSliceTransfer_v1r3<
            FloatGemmAcc,
            GemmDataType,
            decltype(a_src_thread_desc_m0_n0_m1_n1_m2_m3_m4_n2),
            decltype(a_block_desc_m0_k0_m1_k1_m2_m3_m4_k2),
            ElementwiseOp,
            Sequence<Gemm2Params::ABlockSliceLengths_M0_K0_M1_K1::At(I0), // ThreadSliceLengths
                     Gemm2Params::ABlockSliceLengths_M0_K0_M1_K1::At(I1),
                     I1,
                     I1,
                     M2,
                     I1,
                     M4,
                     I1>,
            Sequence<0, 1, 2, 3, 4, 5, 6, 7>,
            7, // DstVectorDim
            1, // DstScalarPerVector
            InMemoryDataOperationEnum::Set,
            1, // DstScalarStrideInVector
            true>;


        using BThreadSlice_N0_N1_N2_K0_K1_K2_K3 =
            Sequence<Gemm2Params::GemmNRepeat, 1, 1, 1, Gemm2Params::B_K1, 1, Gemm2Params::B_K3>;

        static constexpr auto b_thread_desc_n0_n1_n2_k0_k1_k2_k3 =
            make_naive_tensor_descriptor_packed(make_tuple(Number<Gemm2Params::GemmNRepeat>{},
                                                           I1,
                                                           I1,
                                                           I1,
                                                           Number<Gemm2Params::B_K1>{},
                                                           I1,
                                                           Number<Gemm2Params::B_K3>{}));

        __host__ __device__ static constexpr auto MakeBThreadDesc_K0_N_K1()
        {
            constexpr auto b_thread_desc_n_k = transform_tensor_descriptor(
                b_thread_desc_n0_n1_n2_k0_k1_k2_k3,
                make_tuple(make_merge_transform_v3_division_mod(
                               make_tuple(Number<Gemm2Params::GemmNRepeat>{}, I1, I1)),
                           make_merge_transform_v3_division_mod(make_tuple(
                               I1, Number<Gemm2Params::B_K1>{}, I1, Number<Gemm2Params::B_K3>{}))),
                make_tuple(Sequence<0, 1, 2>{}, Sequence<3, 4, 5, 6>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            return transform_tensor_descriptor(
                b_thread_desc_n_k,
                make_tuple(make_pass_through_transform(Number<Gemm2Params::GemmNRepeat>{}),
                           make_unmerge_transform(make_tuple(Number<Gemm2Params::B_K1>{},
                                                             Number<Gemm2Params::B_K3>{}))),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<1>{}, Sequence<0, 2>{}));
        }

        static constexpr auto b_thread_desc_k0_n_k1 = MakeBThreadDesc_K0_N_K1();


        static constexpr auto b_block_slice_copy_step = make_multi_index(0, 0, 0, 1, 0, 0, 0);
        static constexpr auto b_block_reset_copy_step =
            make_multi_index(0, 0, 0, -Gemm2Params::B_K0, 0, 0, 0);

        using BlockwiseGemm = BlockwiseGemmXdlops_v2_dl<
            BlockSize,
            GemmDataType,
            FloatGemmAcc,
            decltype(a_block_desc_k0_m_k1),
            decltype(b_thread_desc_k0_n_k1),
            decltype(MakeGemm2AMmaTileDescriptor_M0_M1_M2_K(a_block_desc_k0_m_k1)),
            decltype(MakeGemm2BMmaTileDescriptor_N0_N1_N2_K(b_thread_desc_k0_n_k1)),
            MPerBlock,
            Gemm1NPerBlock,
            Gemm2Params::Sum_K,
            MPerXdl,
            NPerXdl,
            Gemm2Params::GemmMRepeat,
            Gemm2Params::GemmNRepeat,
            Gemm2Params::GemmKPack,
            Gemm1NPerBlock,
            true, // TransposeC
            Gemm2Params::GemmKPack *
                XdlopsGemm<GemmDataType, MPerXdl, NPerXdl, Gemm2Params::GemmKPack, false>{}
                    .K0PerXdlops,
            Gemm2Params::GemmKPack>;

        static constexpr auto c_block_slice_copy_step =
            make_multi_index(-Gemm2Params::GemmMRepeat, 0, 0, 0, 0, 0, 0, 0);

        template <typename CGradDesc_M_N>
        __host__ __device__ static auto
        MakeCGridDesc_M0_N0_M1_N1_M2_N2_N3_N4(const CGradDesc_M_N& c_grid_desc_m_n)
        {
            // HACK: for unmerge transform, the length of highest dim is irrelevant so we put dummy
            // variable I1 there
            const auto c_grid_desc_m0_n0_m1_n1_m2_n2 = transform_tensor_descriptor(
                c_grid_desc_m_n,
                make_tuple(make_unmerge_transform(make_tuple(I1, Gemm2Params::GemmMWave, MPerXdl)),
                           make_unmerge_transform(make_tuple(I1, Gemm2Params::GemmNWave, NPerXdl))),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 2, 4>{}, Sequence<1, 3, 5>{}));

            const auto c_grid_desc_m0_n0_m1_n1_m2_n2_n3_n4 =
                BlockwiseGemm{}.xdlops_gemm.MakeCDescriptor_M0_N0_M1_N1_M2_N2_N3_N4(
                    c_grid_desc_m0_n0_m1_n1_m2_n2);

            return c_grid_desc_m0_n0_m1_n1_m2_n2_n3_n4;
        }

        static constexpr auto c_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4 =
            BlockwiseGemm::GetCThreadDescriptor_M0_N0_M1_N1_M2_N2_N3_N4();

        __host__ __device__ static auto GetCThreadOriginOnBlock_M0_N0_M1_N1_M2_N2_N3_N4()
        {
            return to_multi_index(BlockwiseGemm::CalculateCThreadOriginDataIndex8D(I0, I0, I0, I0));
        }

        template <typename CGridDesc_M0_N0_M1_N1_M2_N2_N3_N4,
                  typename ElementwiseOp = tensor_operation::element_wise::PassThrough>
        using CBlockwiseCopy = ThreadwiseTensorSliceTransfer_v1r3<
            FloatGemmAcc,
            OutputDataType,
            decltype(c_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4),
            CGridDesc_M0_N0_M1_N1_M2_N2_N3_N4,
            ElementwiseOp,                                                // CElementwiseOperation
            decltype(c_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4.GetLengths()), // SliceLengths
            Sequence<0, 1, 2, 3, 4, 5, 6, 7>,                             // AccessOrder
            7,                                                            // VectorDim
            2,                                                            // ScalarPerVector
            InMemoryDataOperationEnum::AtomicAdd, // GlobalMemoryDataOperation
            1,                                    // DstScalarStrideInVector
            true>;
    };

    // S Gemm (type 4 rcc, B in LDS)
    template <typename BSrcBlockDesc_K0_N_K1>
    struct Gemm3
    {
        // A matrix in LDS memory, dst of blockwise copy
        static constexpr auto a_block_desc_ak0_m_ak1 =
            GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1();

        template <typename ABlockDesc_AK0_M_AK1>
        __host__ __device__ static constexpr auto
        MakeGemm3AMmaTileDescriptor_M0_M1_M2_K(const ABlockDesc_AK0_M_AK1&)
        {
            constexpr index_t MWaves = MPerBlock / (MXdlPerWave * MPerXdl);

            return MakeGemmMmaTileDescriptor_MN0_MN1_MN2_K<MXdlPerWave, MWaves, MPerXdl>(
                ABlockDesc_AK0_M_AK1{});
        }

        template <typename BBlockDesc_BK0_N_BK1>
        __host__ __device__ static constexpr auto
        MakeGemm3BMmaTileDescriptor_N0_N1_N2_K(const BBlockDesc_BK0_N_BK1&)
        {
            constexpr index_t NWaves = NPerBlock / (NXdlPerWave * NPerXdl);

            return MakeGemmMmaTileDescriptor_MN0_MN1_MN2_K<NXdlPerWave, NWaves, NPerXdl>(
                BBlockDesc_BK0_N_BK1{});
        }

        template <typename GridDesc_K0_M_K1>
        using ABlockwiseCopy =
            ThreadGroupTensorSliceTransfer_v4r1<ThisThreadBlock,
                                                AElementwiseOperation,
                                                tensor_operation::element_wise::PassThrough,
                                                InMemoryDataOperationEnum::Set,
                                                Sequence<AK0, MPerBlock, AK1>,
                                                ABlockTransferThreadClusterLengths_AK0_M_AK1,
                                                ABlockTransferThreadClusterArrangeOrder,
                                                InputDataType,
                                                GemmDataType,
                                                GridDesc_K0_M_K1,
                                                decltype(a_block_desc_ak0_m_ak1),
                                                ABlockTransferSrcAccessOrder,
                                                Sequence<1, 0, 2>,
                                                ABlockTransferSrcVectorDim,
                                                2,
                                                ABlockTransferSrcScalarPerVector,
                                                ABlockTransferDstScalarPerVector_AK1,
                                                1,
                                                1,
                                                true, // SrcResetCoord
                                                true, // DstResetCoord
                                                NumGemmKPrefetchStage>;

        static constexpr index_t KPack = math::max(math::lcm(AK1, BK1), mfma.k_per_blk);

        // Blockwise gemm with transposed XDL output
        using BlockwiseGemm = BlockwiseGemmXdlops_v2_dl<
            BlockSize,
            GemmDataType,
            FloatGemmAcc,
            decltype(a_block_desc_ak0_m_ak1),
            BSrcBlockDesc_K0_N_K1,
            decltype(MakeGemm3AMmaTileDescriptor_M0_M1_M2_K(a_block_desc_ak0_m_ak1)),
            decltype(MakeGemm3BMmaTileDescriptor_N0_N1_N2_K(BSrcBlockDesc_K0_N_K1{})),
            MPerBlock,
            NPerBlock,
            KPerBlock,
            MPerXdl,
            NPerXdl,
            MXdlPerWave,
            NXdlPerWave,
            KPack,
            NPerBlock>;

        static constexpr auto a_block_slice_copy_step = make_multi_index(KPerBlock / AK1, 0, 0);
        static constexpr auto b_block_slice_copy_step = make_multi_index(0, 0, 0, KPerBlock);
        static constexpr auto b_block_reset_copy_step = make_multi_index(0, 0, 0, -Gemm1NPerBlock);
    };

    // PGrad Gemm has the same layout as P = Q * K^T Gemm (A row-major B col-major)
    struct PGradGemmTile_M_N_O
    {
        // TODO:
        // Make all input tensors 2D and transform them into appropriate 3D form in kernel to make
        // things more concise
        template <typename YGradGridDesc_M0_O_M1_>
        __device__ static auto
        MakeYGradGridDesc_O0_M_O1(const YGradGridDesc_M0_O_M1_& ygrad_grid_desc_m0_o_m1)
        {
            const auto M0 = ygrad_grid_desc_m0_o_m1.GetLength(I0);
            const auto O  = ygrad_grid_desc_m0_o_m1.GetLength(I1);
            const auto M1 = ygrad_grid_desc_m0_o_m1.GetLength(I2);

            constexpr auto Y_O1 = AK1;
            const auto Y_O0     = O / Y_O1;

            const auto ygrad_grid_desc_o0_m_o1 = transform_tensor_descriptor(
                ygrad_grid_desc_m0_o_m1,
                make_tuple(make_unmerge_transform(make_tuple(Y_O0, Y_O1)),
                           make_merge_transform_v3_division_mod(make_tuple(M0, M1))),
                make_tuple(Sequence<1>{}, Sequence<0, 2>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            return ygrad_grid_desc_o0_m_o1;
        }
    };

    // QGrad Gemm has the same layout as Y = P * V Gemm (A in acc B row-major)
    struct QGradGemmTile_M_K_N
    {
        template <typename KGridDesc_K0_N_K1_>
        __device__ static auto MakeKGridDesc_N0_K_N1(const KGridDesc_K0_N_K1_& k_grid_desc_k0_n_k1)
        {
            const auto K0_ = k_grid_desc_k0_n_k1.GetLength(I0);
            const auto N_  = k_grid_desc_k0_n_k1.GetLength(I1);
            const auto K1_ = k_grid_desc_k0_n_k1.GetLength(I2);

            constexpr auto N1_ = B1K1;
            const auto N0_     = N_ / N1_;

            const auto k_grid_desc_n0_k_n1 = transform_tensor_descriptor(
                k_grid_desc_k0_n_k1,
                make_tuple(make_unmerge_transform(make_tuple(N0_, N1_)),
                           make_merge_transform_v3_division_mod(make_tuple(K0_, K1_))),
                make_tuple(Sequence<1>{}, Sequence<0, 2>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            return k_grid_desc_n0_k_n1;
        }
    };

    struct KGradGemmTile_N_K_M
    {
        // B position
        template <typename QGridDesc_K0_M_K1_>
        __device__ static auto MakeQGridDesc_M0_K_M1(const QGridDesc_K0_M_K1_& q_grid_desc_k0_m_k1)
        {
            const auto Q_K0 = q_grid_desc_k0_m_k1.GetLength(I0);
            const auto M    = q_grid_desc_k0_m_k1.GetLength(I1);
            const auto Q_K1 = q_grid_desc_k0_m_k1.GetLength(I2);

            constexpr auto Q_M1 = B1K1;
            const auto Q_M0     = M / Q_M1;

            const auto q_grid_desc_m0_k_m1 = transform_tensor_descriptor(
                q_grid_desc_k0_m_k1,
                make_tuple(make_unmerge_transform(make_tuple(Q_M0, Q_M1)),
                           make_merge_transform_v3_division_mod(make_tuple(Q_K0, Q_K1))),
                make_tuple(Sequence<1>{}, Sequence<0, 2>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            return q_grid_desc_m0_k_m1;
        }
    };

    // D0
    static constexpr auto D0M2 = Number<4>{};
    static constexpr auto D0M1 = Number<MPerXdl>{} / D0M2;
    static constexpr auto D0M0 = Number<MPerBlock>{} / Number<MPerXdl>{};

    __host__ __device__ static constexpr auto
    MakeD0GridDescriptor_M0_N0_M1_M2_N1_M3(const D0GridDesc_M_N& d0_grid_desc_m_n)
    {
        const auto M = d0_grid_desc_m_n.GetLength(I0);
        const auto N = d0_grid_desc_m_n.GetLength(I1);

        const auto MBlock = M / MPerBlock;
        const auto NBlock = N / NPerBlock;

        const auto d0_grid_desc_m0_n0_m1_m2_n1_m3 = transform_tensor_descriptor(
            d0_grid_desc_m_n,
            make_tuple(make_unmerge_transform(make_tuple(MBlock, D0M0, D0M1, D0M2)),
                       make_unmerge_transform(make_tuple(NBlock, Number<NPerBlock>{}))),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0, 2, 3, 5>{}, Sequence<1, 4>{}));

        return d0_grid_desc_m0_n0_m1_m2_n1_m3;
    }

    using D0GridDescriptor_M0_N0_M1_M2_N1_M3 =
        remove_cvref_t<decltype(MakeD0GridDescriptor_M0_N0_M1_M2_N1_M3(D0GridDesc_M_N{}))>;

    struct D0Operator
    {
        template <typename DataType>
        struct TypeTransform
        {
            using Type                     = DataType;
            static constexpr index_t Size0 = sizeof(DataType);
            static constexpr index_t Size  = sizeof(DataType);
        };
        template <>
        struct TypeTransform<void>
        {
            using Type                     = ck::half_t;
            static constexpr index_t Size0 = 0;
            static constexpr index_t Size  = sizeof(ck::half_t);
        };
        static constexpr index_t NThreadClusterLengths = 32;
        static_assert(NPerXdl == 32);
        static_assert(D0BlockTransferSrcScalarPerVector * NThreadClusterLengths <= NPerBlock,
                      "D0BlockTransferSrcScalarPerVector * NThreadClusterLengths <= NPerBlock");
        __host__ __device__ static constexpr auto GetD0BlockGlobalDescriptor_M0_N0_M1_M2_N1_M3()
        {
            return make_naive_tensor_descriptor_packed(
                make_tuple(I1, I1, I1, D0M1, Number<NPerBlock>{}, D0M2));
        }
        __host__ __device__ static constexpr auto GetD0BlockVgprDescriptor_N0_N1_M0_M1_M2()
        {
            constexpr auto d0_raw_m0_n_m1 =
                make_naive_tensor_descriptor_packed(make_tuple(D0M1, Number<NPerBlock>{}, D0M2));

            constexpr auto d0_n0_n1_m0_m1_m2 = transform_tensor_descriptor(
                d0_raw_m0_n_m1,
                make_tuple(make_unmerge_transform(make_tuple(D0M1 / I2, I2)),
                           make_unmerge_transform(
                               make_tuple(Number<NPerBlock / NPerXdl>{}, Number<NPerXdl>{})),
                           make_pass_through_transform(D0M2)),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                make_tuple(Sequence<2, 3>{}, Sequence<0, 1>{}, Sequence<4>{}));
            return d0_n0_n1_m0_m1_m2;
        }
        static constexpr auto d0_block_dst_desc_m0_n0_m1_m2_n1_m3 =
            GetD0BlockGlobalDescriptor_M0_N0_M1_M2_N1_M3();
        static constexpr auto d0_block_src_desc_n0_n1_m0_m1_m2 =
            GetD0BlockVgprDescriptor_N0_N1_M0_M1_M2();

        static constexpr auto d0_thread_desc_ =
            make_naive_tensor_descriptor_packed(make_tuple(I1, I1, I4, I1, D0M2));

        static constexpr auto& d0grad_block_dst_desc_n0_n1_m0_m1_m2 =
            d0_block_src_desc_n0_n1_m0_m1_m2;
        static constexpr auto& d0grad_block_src_desc_m0_n0_m1_m2_n1_m3 =
            d0_block_dst_desc_m0_n0_m1_m2_n1_m3;

        using D0BlockwiseCopyGlobalToLds = ThreadGroupTensorSliceTransfer_v4r1<
            ThisThreadBlock,
            tensor_operation::element_wise::PassThrough,
            tensor_operation::element_wise::PassThrough,
            InMemoryDataOperationEnum::Set,
            Sequence<I1, I1, I1, D0M1, NPerBlock, D0M2>, // BlockSliceLengths
            Sequence<1,
                     1,
                     1,
                     BlockSize / NThreadClusterLengths,
                     NThreadClusterLengths,
                     1>,                                   // ThreadClusterLengths
            Sequence<0, 1, 2, 3, 5, 4>,                    // ThreadClusterArrangeOrder
            typename TypeTransform<D0DataType>::Type,      // SrcData
            typename TypeTransform<D0DataType>::Type,      // DstData
            D0GridDescriptor_M0_N0_M1_M2_N1_M3,            // SrcDesc
            decltype(d0_block_dst_desc_m0_n0_m1_m2_n1_m3), // DstDesc
            Sequence<0, 1, 2, 3, 5, 4>,                    // SrcDimAccessOrder
            Sequence<0, 1, 2, 4, 3, 5>,                    // DstDimAccessOrder
            4,                                             // SrcVectorDim
            5,                                             // DstVectorDim
            4,                                             // SrcScalarPerVector
            4,                                             // DstScalarPerVector
            1,
            1,
            true,
            true, // DstResetCoord
            1>;

        using D0ThreadwiseCopyLdsToVgpr =
            ThreadwiseTensorSliceTransfer_v4<typename TypeTransform<D0DataType>::Type,   // SrcData
                                             typename TypeTransform<D0DataType>::Type,   // DstData
                                             decltype(d0_block_src_desc_n0_n1_m0_m1_m2), // SrcDesc
                                             decltype(d0_thread_desc_),                  // DstDesc
                                             Sequence<1, 1, 4, 1, 4>, // SliceLengths
                                             Sequence<0, 1, 2, 3, 4>, // DimAccessOrder
                                             4,                       // SrcVectorDim
                                             4,                       // SrcScalarPerVector
                                             2>;

        using D0GradThreadwiseCopyVgprToLds = ThreadwiseTensorSliceTransfer_v1r3<
            FloatGemmAcc,
            typename TypeTransform<D0DataType>::Type,
            decltype(d0_thread_desc_),
            decltype(d0grad_block_dst_desc_n0_n1_m0_m1_m2),
            tensor_operation::element_wise::Scale, // CElementwiseOperation
            Sequence<1, 1, 4, 1, 4>,               // SliceLengths
            Sequence<0, 1, 2, 3, 4>,               // AccessOrder
            4,                                     // VectorDim
            4,                                     // ScalarPerVector
            InMemoryDataOperationEnum::Set,        // GlobalMemoryDataOperation
            1,                                     // DstScalarStrideInVector
            true>;

        using D0GradBlockwiseCopyLdsToGlobal = ThreadGroupTensorSliceTransfer_v4r1<
            ThisThreadBlock,
            tensor_operation::element_wise::PassThrough,
            tensor_operation::element_wise::PassThrough,
            InMemoryDataOperationEnum::Set,
            Sequence<I1, I1, I1, D0M1, NPerBlock, D0M2>, // BlockSliceLengths
            Sequence<1,
                     1,
                     1,
                     BlockSize / NThreadClusterLengths,
                     NThreadClusterLengths,
                     1>,                                       // ThreadClusterLengths
            Sequence<0, 1, 2, 3, 5, 4>,                        // ThreadClusterArrangeOrder
            typename TypeTransform<D0DataType>::Type,          // SrcData
            typename TypeTransform<D0DataType>::Type,          // DstData
            decltype(d0grad_block_src_desc_m0_n0_m1_m2_n1_m3), // SrcDesc
            D0GridDescriptor_M0_N0_M1_M2_N1_M3,                // DstDesc
            Sequence<0, 1, 2, 4, 3, 5>,                        // SrcDimAccessOrder
            Sequence<0, 1, 2, 3, 5, 4>,                        // DstDimAccessOrder
            5,                                                 // SrcVectorDim
            4,                                                 // DstVectorDim
            4,                                                 // SrcScalarPerVector
            D0BlockTransferSrcScalarPerVector,                 // DstScalarPerVector
            1,
            1,
            true,
            true, // DstResetCoord
            1>;
    };

    struct SharedMemTrait
    {
        // LDS allocation for K
        static constexpr auto k_block_desc_k0_n_k1 = GetKBlockDescriptor_K0PerBlock_NPerBlock_K1();
        // LDS allocation for A and B: be careful of alignment
        static constexpr auto a_block_desc_ak0_m_ak1 =
            GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1();
        static constexpr auto b1_block_desc_bk0_n_bk1 =
            GetB1BlockDescriptor_BK0PerBlock_NPerBlock_BK1();
        static constexpr auto a2_block_desc_k0_m_k1 = GetA2BlockDescriptor_K0_M_K1<Gemm2Params>();

        static constexpr auto max_lds_align = Number<16 / sizeof(GemmDataType)>{};

        static constexpr auto k_block_space_size_aligned =
            math::integer_least_multiple(k_block_desc_k0_n_k1.GetElementSpaceSize(), max_lds_align);
        static constexpr auto a_block_space_size_aligned = math::integer_least_multiple(
            a_block_desc_ak0_m_ak1.GetElementSpaceSize(), max_lds_align);
        static constexpr auto b1_block_space_size_aligned = math::integer_least_multiple(
            b1_block_desc_bk0_n_bk1.GetElementSpaceSize(), max_lds_align);
        static constexpr auto a2_block_space_size_aligned = math::integer_least_multiple(
            a2_block_desc_k0_m_k1.GetElementSpaceSize(), max_lds_align);

        static constexpr auto k_block_space_offset  = 0;
        static constexpr auto a_block_space_offset  = k_block_space_size_aligned.value;
        static constexpr auto b1_block_space_offset = k_block_space_size_aligned.value;
        static constexpr auto a2_block_space_offset = k_block_space_size_aligned.value;

        // LDS allocation for reduction
        static constexpr index_t reduction_space_size_aligned =
            math::integer_least_multiple(BlockSize, max_lds_align);

        static constexpr auto d0_block_space_size_aligned = math::integer_least_multiple(
            D0Operator::d0_block_dst_desc_m0_n0_m1_m2_n1_m3.GetElementSpaceSize(), max_lds_align);
        static constexpr auto d0_block_space_offset =
            k_block_space_size_aligned.value * sizeof(GemmDataType) /
            D0Operator::template TypeTransform<D0DataType>::Size;

        // LDS allocation for C shuffle in LDS
        static constexpr auto c_shuffle_block_desc_mblock_mperblock_nblock_nperblock =
            GetCShuffleBlockDescriptor_MBlock_MPerBlock_NBlock_NPerBlock();
        static constexpr auto c_block_space_size =
            c_shuffle_block_desc_mblock_mperblock_nblock_nperblock.GetElementSpaceSize();
    };

    __host__ __device__ static constexpr index_t GetSharedMemoryNumberOfByte()
    {
        const index_t gemm0_bytes_end = (SharedMemTrait::k_block_space_size_aligned +
                                         SharedMemTrait::a_block_space_size_aligned) *
                                        sizeof(GemmDataType);
        const index_t gemm1_bytes_end = (SharedMemTrait::k_block_space_size_aligned +
                                         SharedMemTrait::b1_block_space_size_aligned) *
                                        sizeof(GemmDataType);
        const index_t gemm2_bytes_end = (SharedMemTrait::k_block_space_size_aligned +
                                         SharedMemTrait::a2_block_space_size_aligned) *
                                        sizeof(GemmDataType);
        const index_t gemm3_bytes_end = (SharedMemTrait::k_block_space_size_aligned +
                                         SharedMemTrait::a_block_space_size_aligned) *
                                        sizeof(GemmDataType);
        const index_t d0_bytes_end =
            (SharedMemTrait::d0_block_space_offset + SharedMemTrait::d0_block_space_size_aligned) *
            D0Operator::template TypeTransform<D0DataType>::Size0;
        const index_t c_block_bytes_end =
            SharedMemTrait::c_block_space_size * sizeof(OutputDataType);

        return math::max(gemm0_bytes_end,
                         gemm1_bytes_end,
                         gemm2_bytes_end,
                         gemm3_bytes_end,
                         d0_bytes_end,
                         c_block_bytes_end);
    }

    template <bool HasMainKBlockLoop,
              bool IsDropout,
              typename Block2CTileMap,
              typename C0MatrixMask,
              typename YGradGridDesc_M0_O_M1>
    __device__ static void
    Run(const InputDataType* __restrict__ p_q_grid,
        const InputDataType* __restrict__ p_k_grid,
        const D0DataType* __restrict__ p_d0_grid,
        ZDataType* __restrict__ p_z_grid,
        const InputDataType* __restrict__ p_v_grid,
        const FloatLSE* __restrict__ p_lse_grid,
        const FloatD* __restrict__ p_d_grid,
        const InputDataType* __restrict__ p_ygrad_grid,
        OutputDataType* __restrict__ p_qgrad_grid,
        OutputDataType* __restrict__ p_kgrad_grid,
        D0DataType* __restrict__ p_d0grad_grid,
        OutputDataType* __restrict__ p_vgrad_grid,
        void* __restrict__ p_shared,
        const AElementwiseOperation& a_element_op,
        const BElementwiseOperation& b_element_op,
        const SElementwiseOperation& s_element_op,
        const B1ElementwiseOperation& b1_element_op,
        const CElementwiseOperation& c_element_op,
        const QGridDesc_K0_M_K1& q_grid_desc_k0_m_k1,
        const KGridDesc_K0_N_K1& k_grid_desc_k0_n_k1,
        const KGridDesc_K0_N_K1& kgrad_grid_desc_k0_n_k1,
        const D0GridDescriptor_M0_N0_M1_M2_N1_M3& d0_grid_desc_m0_n0_m1_m2_n1_m3,
        const ZGridDescriptor_M0_N0_M1_N1_M2_N2_M3_M4_M5_N3&
            z_grid_desc_m0_n0_m1_n1_m2_n2_m3_m4_m5_n3,
        const VGridDesc_O0_N_O1& v_grid_desc_o0_n_o1,
        const VGridDesc_O0_N_O1& vgrad_grid_desc_o0_n_o1,
        const LSEGridDesc_M& lse_grid_desc_m,
        const YGradGridDesc_M0_O_M1& ygrad_grid_desc_m0_o_m1,
        const Block2CTileMap& block_2_ctile_map,
        const C0MatrixMask& c0_matrix_mask,
        const float p_drop,
        ck::philox& ph,
        const index_t z_random_matrix_offset,
        const index_t raw_n_padded,
        const index_t block_idx_n)
    {

        const FloatGemmAcc p_dropout  = type_convert<FloatGemmAcc>(1.0f - p_drop);
        const FloatGemmAcc rp_dropout = type_convert<FloatGemmAcc>(1.0f / p_dropout);
        const uint8_t p_dropout_in_uint8_t =
            __builtin_amdgcn_readfirstlane(uint8_t(std::floor(p_dropout * 255.0)));
        const tensor_operation::element_wise::Scale scale_rp_dropout(s_element_op.Value() *
                                                                     rp_dropout);

        const auto q_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_q_grid, q_grid_desc_k0_m_k1.GetElementSpaceSize());
        const auto k_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_k_grid, k_grid_desc_k0_n_k1.GetElementSpaceSize());
        const auto v_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_v_grid, v_grid_desc_o0_n_o1.GetElementSpaceSize());
        const auto lse_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_lse_grid, lse_grid_desc_m.GetElementSpaceSize());
        const auto d_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_d_grid, lse_grid_desc_m.GetElementSpaceSize()); // reuse lse grid descriptor
        const auto ygrad_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_ygrad_grid, ygrad_grid_desc_m0_o_m1.GetElementSpaceSize());
        auto vgrad_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_vgrad_grid, vgrad_grid_desc_o0_n_o1.GetElementSpaceSize());
        auto qgrad_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_qgrad_grid, q_grid_desc_k0_m_k1.GetElementSpaceSize());
        auto kgrad_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_kgrad_grid, kgrad_grid_desc_k0_n_k1.GetElementSpaceSize());

        // divide block work by [N, K]
        const auto block_work_idx =
            block_2_ctile_map.CalculateBottomIndex(make_multi_index(get_block_1d_id()));

        const index_t block_work_idx_n = Deterministic ? block_idx_n : block_work_idx[I0];

        // HACK: this force n_block_data_idx_on_grid into SGPR
        const index_t n_block_data_idx_on_grid =
            __builtin_amdgcn_readfirstlane(block_work_idx_n * NPerBlock);

        const index_t num_gemm0_m_block_outer_loop = q_grid_desc_k0_m_k1.GetLength(I1) / MPerBlock;

        // 6 GEMM operations are categorized into 4 buckets. SizeK == SizeO == head_dim
        // dP_MNO Gemm          (Gemm0 rcc)
        // dV_NOM / dK_NKM Gemm (Gemm1 rrr)
        // Y_MON  / dQ_MKN Gemm (Gemm2 crr)
        // S_MNK Gemm           (Gemm3 rcc)

        // LDS allocation for K
        auto k_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            static_cast<GemmDataType*>(p_shared) + SharedMemTrait::k_block_space_offset,
            GemmBlockwiseCopy::k_block_desc_k0_n_k1.GetElementSpaceSize());

        // K matrix blockwise copy
        auto gemm_tile_k_blockwise_copy =
            typename GemmBlockwiseCopy::template KBlockwiseCopy<decltype(k_grid_desc_k0_n_k1)>(
                k_grid_desc_k0_n_k1,
                make_multi_index(0, n_block_data_idx_on_grid, 0),
                b_element_op,
                GemmBlockwiseCopy::k_block_desc_k0_n_k1,
                make_multi_index(0, 0, 0),
                tensor_operation::element_wise::PassThrough{});

        // Vgpr allocation for V
        auto v_thread_buf = generate_tuple(
            [&](auto i) {
                ignore = i;
                return StaticBuffer<
                    AddressSpaceEnum::Vgpr,
                    GemmDataType,
                    GemmBlockwiseCopy::v_thread_desc_k0_k1_k2_n0_n1_n2_n3_k3.GetElementSpaceSize(),
                    true>{};
            },
            Number<GemmBlockwiseCopy::VBlockBufferSize>{});

        const auto v_grid_desc_k0_k1_k2_n0_n1_n2_n3_k3 =
            GemmBlockwiseCopy::MakeVGridDescriptor_K0_K1_K2_N0_N1_N2_N3_K3(v_grid_desc_o0_n_o1);

        const auto wave_id     = GetGemm0WaveIdx();
        const auto wave_m_n_id = GetGemm0WaveMNIdx(wave_id[I2]); // I2: 0~63

        // V matrix blockwise copy
        auto gemm_tile_v_blockwise_copy =
            typename GemmBlockwiseCopy::template VBlockwiseCopy<decltype(
                v_grid_desc_k0_k1_k2_n0_n1_n2_n3_k3)>(
                v_grid_desc_k0_k1_k2_n0_n1_n2_n3_k3,
                make_multi_index(
                    0, 0, wave_m_n_id[I0], block_work_idx_n, 0, wave_id[I1], wave_m_n_id[I1], 0));

        //
        // set up dP Gemm (type 1 rcc)
        //

        using Gemm0 = Gemm0<decltype(GemmBlockwiseCopy::v_thread_desc_k0_k1_k2_n0_n1_n2_n3_k3)>;

        // Gemm0: LDS allocation for A and B: be careful of alignment
        auto gemm0_a_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            static_cast<GemmDataType*>(p_shared) + SharedMemTrait::a_block_space_offset,
            Gemm0::a_block_desc_ak0_m_ak1.GetElementSpaceSize());

        // dP: transform input tensor descriptors
        const auto ygrad_grid_desc_o0_m_o1 =
            PGradGemmTile_M_N_O::MakeYGradGridDesc_O0_M_O1(ygrad_grid_desc_m0_o_m1);

        // dP: A matrix blockwise copy
        auto pgrad_gemm_tile_ygrad_blockwise_copy =
            typename Gemm0::template ABlockwiseCopy<decltype(ygrad_grid_desc_o0_m_o1)>(
                ygrad_grid_desc_o0_m_o1,
                make_multi_index(0,
                                 MPerBlock * (num_gemm0_m_block_outer_loop - 1),
                                 0), // will loop over GemmM dimension
                tensor_operation::element_wise::PassThrough{},
                Gemm0::a_block_desc_ak0_m_ak1,
                make_multi_index(0, 0, 0),
                tensor_operation::element_wise::PassThrough{});

        // dP: blockwise gemm
        auto pgrad_blockwise_gemm = typename Gemm0::BlockwiseGemm{};
        pgrad_blockwise_gemm.SetBBlockStartWindow(make_tuple(0, 0, 0, 0));

        auto pgrad_thread_buf = pgrad_blockwise_gemm.GetCThreadBuffer();

        const auto pgrad_gemm_tile_ygrad_block_reset_copy_step =
            make_multi_index(-ygrad_grid_desc_o0_m_o1.GetLength(I0), -MPerBlock, 0);

        //
        // set up S Gemm (type 4 rcc)
        //

        using Gemm3 = Gemm3<decltype(GemmBlockwiseCopy::k_block_desc_k0_n_k1)>;

        // Gemm3: LDS allocation for A and B: be careful of alignment
        auto gemm3_a_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            static_cast<GemmDataType*>(p_shared) + SharedMemTrait::a_block_space_offset,
            Gemm3::a_block_desc_ak0_m_ak1.GetElementSpaceSize());

        // S: A matrix blockwise copy
        auto s_gemm_tile_q_blockwise_copy =
            typename Gemm3::template ABlockwiseCopy<decltype(q_grid_desc_k0_m_k1)>(
                q_grid_desc_k0_m_k1,
                make_multi_index(0,
                                 MPerBlock * (num_gemm0_m_block_outer_loop - 1),
                                 0), // will loop over GemmM dimension
                a_element_op,
                Gemm3::a_block_desc_ak0_m_ak1,
                make_multi_index(0, 0, 0),
                tensor_operation::element_wise::PassThrough{});

        // S: blockwise gemm
        auto s_blockwise_gemm = typename Gemm3::BlockwiseGemm{}; // TransposeC

        auto s_slash_p_thread_buf = s_blockwise_gemm.GetCThreadBuffer();

        const auto s_gemm_tile_q_block_reset_copy_step =
            make_multi_index(-q_grid_desc_k0_m_k1.GetLength(I0), -MPerBlock, 0);

        //
        // set up dV / dK Gemm (type 2 rrr)
        //
        using Gemm1 =
            Gemm1<decltype(s_blockwise_gemm.GetCThreadDescriptor_M0_N0_M1_N1_M2_M3_M4_N2()),
                  decltype(s_blockwise_gemm.GetCBlockDescriptor_M0_N0_M1_N1_M2_M3_M4_N2())>;

        // Gemm1: VGPR allocation for A and LDS allocation for B
        auto gemm1_a_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, GemmDataType>(
            Gemm1::a_thread_desc_k0_m_k1.GetElementSpaceSize());

        auto gemm1_b_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            static_cast<GemmDataType*>(p_shared) + SharedMemTrait::b1_block_space_offset,
            Gemm1::b_block_desc_bk0_n_bk1.GetElementSpaceSize());

        // dV: A matrix blockwise copy
        auto vgrad_gemm_tile_p_blockwise_copy =
            typename Gemm1::template ABlockwiseCopy<tensor_operation::element_wise::Relu>{
                tensor_operation::element_wise::Relu{}}; // relu(P-dropped)

        // dV: B matrix blockwise copy
        auto vgrad_gemm_tile_ygrad_blockwise_copy =
            typename Gemm1::template BBlockwiseCopy<decltype(ygrad_grid_desc_m0_o_m1)>(
                ygrad_grid_desc_m0_o_m1,
                make_multi_index(MPerBlock / B1K1 * (num_gemm0_m_block_outer_loop - 1), 0, 0),
                b1_element_op,
                Gemm1::b_block_desc_bk0_n_bk1, // there n actually is k, k is N, so name can be
                                               // b_block_desc_bn0_k_bn1
                make_multi_index(0, 0, 0),
                tensor_operation::element_wise::PassThrough{});

        const auto vgrad_gemm_tile_ygrad_block_next_copy_step =
            make_multi_index(-2 * MPerBlock / B1K1, 0, 0);

        // dV: blockwise gemm
        auto vgrad_blockwise_gemm =
            typename Gemm1::BlockwiseGemm{make_tuple(0, 0, 0, 0)}; // A_origin

        auto vgrad_thread_buf = vgrad_blockwise_gemm.GetCThreadBuffer();

        // dV: transform input and output tensor descriptors
        auto vgrad_grid_desc_nblock_nperblock_oblock_operblock =
            MakeVGradGridDesc_NBlock_NPerBlock_OBlock_OPerBlock(vgrad_grid_desc_o0_n_o1);

        // dK: transform input and output tensor descriptors
        const auto q_grid_desc_m0_k_m1 =
            KGradGemmTile_N_K_M::MakeQGridDesc_M0_K_M1(q_grid_desc_k0_m_k1);

        // dK: A matrix blockwise copy
        auto kgrad_gemm_tile_sgrad_blockwise_copy =
            typename Gemm1::template ABlockwiseCopy<tensor_operation::element_wise::PassThrough>{
                tensor_operation::element_wise::PassThrough{}};

        // dK: B matrix blockwise copy
        auto kgrad_gemm_tile_q_blockwise_copy =
            typename Gemm1::template BBlockwiseCopy<decltype(q_grid_desc_m0_k_m1)>(
                q_grid_desc_m0_k_m1,
                make_multi_index(MPerBlock / B1K1 * (num_gemm0_m_block_outer_loop - 1), 0, 0),
                b1_element_op,
                Gemm1::b_block_desc_bk0_n_bk1, // there n actually is k, k is N, so name can be
                                               // b_block_desc_bn0_k_bn1
                make_multi_index(0, 0, 0),
                tensor_operation::element_wise::PassThrough{});

        const auto kgrad_gemm_tile_q_block_next_copy_step =
            make_multi_index(-2 * MPerBlock / B1K1, 0, 0);

        // dK: blockwise gemm
        auto kgrad_blockwise_gemm =
            typename Gemm1::BlockwiseGemm{make_tuple(0, 0, 0, 0)}; // A_origin

        auto kgrad_thread_buf = kgrad_blockwise_gemm.GetCThreadBuffer();

        // dK: transform input and output tensor descriptors
        auto kgrad_grid_desc_nblock_nperblock_oblock_operblock =
            MakeKGradGridDesc_NBlock_NPerBlock_OBlock_OPerBlock(kgrad_grid_desc_k0_n_k1);

        //
        // set up dQ Gemm (type 3 crr)
        //
        using Gemm2 = Gemm2<Gemm2Params,
                            decltype(pgrad_blockwise_gemm),
                            decltype(GemmBlockwiseCopy::k_block_desc_k0_n_k1)>;

        // Gemm2: LDS allocation for A and B: be careful of alignment
        auto gemm2_a_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            static_cast<GemmDataType*>(p_shared) + SharedMemTrait::a2_block_space_offset,
            Gemm2::a_block_desc_k0_m_k1.GetElementSpaceSize());

        // auto gemm2_b_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, GemmDataType>(
        //     Gemm2::b_thread_desc_n0_n1_n2_k0_k1_k2_k3.GetElementSpaceSize());

        // dQ: A matrix VGPR-to-LDS blockwise copy
        auto qgrad_gemm_tile_sgrad_thread_copy_vgpr_to_lds =
            typename Gemm2::template ABlockwiseCopy<tensor_operation::element_wise::PassThrough>{
                Gemm2::a_block_desc_m0_k0_m1_k1_m2_m3_m4_k2,
                Gemm2::MakeAThreadOriginOnBlock_M0_K0_M1_K1_M2_M3_M4_K2(),
                tensor_operation::element_wise::PassThrough{}};

        // dQ: blockwise gemm
        auto qgrad_blockwise_gemm = typename Gemm2::BlockwiseGemm{};
        qgrad_blockwise_gemm.SetBBlockStartWindow(make_tuple(0, 0, 0, 0));

        auto k_thread_origin = qgrad_blockwise_gemm.CalculateBThreadOriginDataIndex();

        auto qgrad_thread_buf = qgrad_blockwise_gemm.GetCThreadBuffer();

        // dQ: transform output tensor descriptors
        const auto qgrad_grid_desc_m_k = MakeQGradGridDesc_M_K(q_grid_desc_k0_m_k1);
        const auto qgrad_grid_desc_m0_o0_m1_o1_m2_o2_o3_o4 =
            Gemm2::MakeCGridDesc_M0_N0_M1_N1_M2_N2_N3_N4(qgrad_grid_desc_m_k);

        // dQ: C VGPR-to-global copy
        const auto qgrad_thread_origin_on_grid_m0_o0_m1_o1_m2_o2_o3_o4 =
            Gemm2::GetCThreadOriginOnBlock_M0_N0_M1_N1_M2_N2_N3_N4() +
            make_multi_index((num_gemm0_m_block_outer_loop - 1) * Gemm2Params::GemmMRepeat,
                             I0,
                             I0,
                             I0,
                             I0,
                             I0,
                             I0,
                             I0);

        auto qgrad_thread_copy_vgpr_to_global = typename Gemm2::template CBlockwiseCopy<
            decltype(qgrad_grid_desc_m0_o0_m1_o1_m2_o2_o3_o4),
            decltype(scale_rp_dropout)>(qgrad_grid_desc_m0_o0_m1_o1_m2_o2_o3_o4,
                                        qgrad_thread_origin_on_grid_m0_o0_m1_o1_m2_o2_o3_o4,
                                        scale_rp_dropout);

        //
        // Blockwise softmax
        //
        // get acc0 8D thread cluster
        constexpr auto thread_cluster_m0_n0_m1_n1_m2_m3_m4_n2 =
            s_blockwise_gemm.GetCBlockDescriptor_M0_N0_M1_N1_M2_M3_M4_N2().GetLengths() /
            s_blockwise_gemm.GetCThreadDescriptor_M0_N0_M1_N1_M2_M3_M4_N2().GetLengths();
        constexpr auto tm0 = thread_cluster_m0_n0_m1_n1_m2_m3_m4_n2.At(I0);
        constexpr auto tn0 = thread_cluster_m0_n0_m1_n1_m2_m3_m4_n2.At(I1);
        constexpr auto tm1 = thread_cluster_m0_n0_m1_n1_m2_m3_m4_n2.At(I2);
        constexpr auto tn1 = thread_cluster_m0_n0_m1_n1_m2_m3_m4_n2.At(I3);
        constexpr auto tm2 = thread_cluster_m0_n0_m1_n1_m2_m3_m4_n2.At(I4);
        constexpr auto tm3 = thread_cluster_m0_n0_m1_n1_m2_m3_m4_n2.At(I5);
        constexpr auto tm4 = thread_cluster_m0_n0_m1_n1_m2_m3_m4_n2.At(I6);
        constexpr auto tn2 = thread_cluster_m0_n0_m1_n1_m2_m3_m4_n2.At(I7);

        // get acc0 thread map
        constexpr auto n0_m_n1_to_m_n_adaptor = make_single_stage_tensor_adaptor(
            make_tuple(make_unmerge_transform(make_tuple(tn0 * tn1, tn2)),
                       make_pass_through_transform(I1)),
            make_tuple(Sequence<1>{}, Sequence<0>{}),
            make_tuple(Sequence<0, 2>{}, Sequence<1>{}));
        constexpr auto threadid_to_n0_m_n1_adaptor = make_single_stage_tensor_adaptor(
            make_tuple(
                make_merge_transform(make_tuple(tn0 * tn1, tm0 * tm1 * tm2 * tm3 * tm4, tn2))),
            make_tuple(Sequence<0, 1, 2>{}),
            make_tuple(Sequence<0>{}));
        const auto threadid_to_m_n_thread_cluster_adaptor =
            chain_tensor_adaptors(n0_m_n1_to_m_n_adaptor, threadid_to_n0_m_n1_adaptor);

        // get acc0 2D thread cluster & 2D thread slice
        constexpr auto thread_desc_m0_n0_m1_n1_m2_m3_m4_n2 =
            s_blockwise_gemm.GetCThreadDescriptor_M0_N0_M1_N1_M2_M3_M4_N2();
        constexpr auto m0 = thread_desc_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I0);
        constexpr auto n0 = thread_desc_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I1);
        constexpr auto m1 = thread_desc_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I2);
        constexpr auto n1 = thread_desc_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I3);
        constexpr auto m2 = thread_desc_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I4);
        constexpr auto m3 = thread_desc_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I5);
        constexpr auto m4 = thread_desc_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I6);
        constexpr auto n2 = thread_desc_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I7);

        constexpr auto thread_cluster_desc_m_n = make_naive_tensor_descriptor_packed(
            make_tuple(tm0 * tm1 * tm2 * tm3 * tm4, tn0 * tn1 * tn2));
        constexpr auto thread_slice_desc_m_n =
            make_naive_tensor_descriptor_packed(make_tuple(m0 * m1 * m2 * m3 * m4, n0 * n1 * n2));

        auto blockwise_softmax = BlockwiseSoftmax<BlockSize,
                                                  FloatGemmAcc,
                                                  decltype(threadid_to_m_n_thread_cluster_adaptor),
                                                  decltype(thread_cluster_desc_m_n),
                                                  decltype(thread_slice_desc_m_n)>{};

        auto blockwise_dropout = BlockwiseDropout<FloatGemmAcc, decltype(thread_slice_desc_m_n)>{
            p_dropout_in_uint8_t, rp_dropout};

        auto lse_grid_desc_mb_m0_m1_m2_m3_m4 =
            MakeLSEGridDescriptor_MB_M0_M1_M2_M3_M4<decltype(s_blockwise_gemm)>(lse_grid_desc_m);

        constexpr auto lse_thread_desc_mb_m0_m1_m2_m3_m4 =
            make_naive_tensor_descriptor_packed(make_tuple(I1, m0, m1, m2, m3, m4));

        auto lse_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, FloatLSE>(
            lse_thread_desc_mb_m0_m1_m2_m3_m4.GetElementSpaceSize());

        auto acc0_thread_origin = s_blockwise_gemm.CalculateCThreadOriginDataIndex8D(
            Number<0>{}, Number<0>{}, Number<0>{}, Number<0>{});

        auto lse_thread_copy_global_to_vgpr =
            ThreadwiseTensorSliceTransfer_v2<FloatLSE,
                                             FloatLSE,
                                             decltype(lse_grid_desc_mb_m0_m1_m2_m3_m4),
                                             decltype(lse_thread_desc_mb_m0_m1_m2_m3_m4),
                                             Sequence<1, m0, m1, m2, m3, m4>,
                                             Sequence<0, 1, 2, 3, 4, 5>,
                                             5,
                                             1,
                                             1,
                                             true /* ResetCoordAfterRun */>{
                lse_grid_desc_mb_m0_m1_m2_m3_m4,
                make_multi_index(num_gemm0_m_block_outer_loop - 1, // mblock
                                 acc0_thread_origin[I0],           // mrepeat
                                 acc0_thread_origin[I2],           // mwave
                                 acc0_thread_origin[I4],           // mperxdl
                                 acc0_thread_origin[I5],
                                 acc0_thread_origin[I6])};

        auto d_thread_copy_global_to_vgpr =
            ThreadwiseTensorSliceTransfer_v2<FloatD,
                                             FloatGemmAcc,
                                             decltype(lse_grid_desc_mb_m0_m1_m2_m3_m4),
                                             decltype(lse_thread_desc_mb_m0_m1_m2_m3_m4),
                                             Sequence<1, m0, m1, m2, m3, m4>,
                                             Sequence<0, 1, 2, 3, 4, 5>,
                                             5,
                                             1,
                                             1,
                                             true /* ResetCoordAfterRun */>{
                lse_grid_desc_mb_m0_m1_m2_m3_m4,
                make_multi_index(num_gemm0_m_block_outer_loop - 1, // mblock
                                 acc0_thread_origin[I0],           // mrepeat
                                 acc0_thread_origin[I2],           // mwave
                                 acc0_thread_origin[I4],           // mperxdl
                                 acc0_thread_origin[I5],
                                 acc0_thread_origin[I6])};

        //
        // z vgpr copy to global
        //
        // z matrix threadwise desc
        constexpr auto z_thread_desc_m0_n0_m1_n1_m2_n2_m3_m4_m5_n3 =
            make_naive_tensor_descriptor_packed(make_tuple(I1,   // MBlockId
                                                           I1,   // NBlockID
                                                           m0,   // MRepeat
                                                           n0,   // NRepeat
                                                           m1,   // MWaveId
                                                           n1,   // NWaveId
                                                           m2,   // MGroupNum
                                                           m3,   // MInputNum
                                                           m4,   // RegisterNum
                                                           n2)); // NPerXdl

        StaticBuffer<AddressSpaceEnum::Vgpr,
                     uint8_t,
                     z_thread_desc_m0_n0_m1_n1_m2_n2_m3_m4_m5_n3.GetElementSpaceSize(),
                     true>
            z_tensor_buffer;
        z_tensor_buffer.Clear();

        auto z_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_z_grid, z_grid_desc_m0_n0_m1_n1_m2_n2_m3_m4_m5_n3.GetElementSpaceSize());

        auto z_thread_copy_vgpr_to_global = ThreadwiseTensorSliceTransfer_v1r3<
            uint8_t,
            ZDataType,
            decltype(z_thread_desc_m0_n0_m1_n1_m2_n2_m3_m4_m5_n3),
            decltype(z_grid_desc_m0_n0_m1_n1_m2_n2_m3_m4_m5_n3),
            tensor_operation::element_wise::PassThrough,
            Sequence<I1, // MBlockId
                     I1, // NBlockID
                     m0, // MRepeat
                     n0, // NRepeat
                     m1, // MWaveId
                     n1, // NWaveId
                     m2, // MPerXdl
                     m3, // NGroupNum
                     m4, // NInputNum
                     n2>,
            Sequence<0, 1, 2, 3, 4, 5, 6, 7, 8, 9>,
            9, // DstVectorDim
            1, // DstScalarPerVector
            InMemoryDataOperationEnum::Set,
            1, // DstScalarStrideInVector
            true>{z_grid_desc_m0_n0_m1_n1_m2_n2_m3_m4_m5_n3,
                  make_multi_index(num_gemm0_m_block_outer_loop - 1, // MBlockId
                                   block_work_idx_n,                 // NBlockId
                                   0,                                // MRepeat
                                   0,                                // NRepeat
                                   wave_id[I0],                      // MWaveId
                                   wave_id[I1],                      // NWaveId
                                   0,                                // MPerXdl
                                   wave_m_n_id[I0],                  //
                                   0,                                //
                                   wave_m_n_id[I1]),                 // NPerXdl
                  tensor_operation::element_wise::PassThrough{}};

        constexpr auto y_dot_ygrad_thread_desc_mb_m0_m1_m2_m3_m4 =
            lse_thread_desc_mb_m0_m1_m2_m3_m4; // reuse LSE thread descriptor because
                                               // per-thread LSE data and y_dot_ygrad is
                                               // tiled the same way

        auto y_dot_ygrad_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, FloatGemmAcc>(
            y_dot_ygrad_thread_desc_mb_m0_m1_m2_m3_m4.GetElementSpaceSize());

        // gemm0 M loop
        index_t gemm0_m_block_outer_index = num_gemm0_m_block_outer_loop - 1;

        // D0
        auto d0_block_copy_global_to_lds = typename D0Operator::D0BlockwiseCopyGlobalToLds(
            d0_grid_desc_m0_n0_m1_m2_n1_m3,
            make_multi_index(gemm0_m_block_outer_index, block_work_idx_n, 0, 0, 0, 0),
            tensor_operation::element_wise::PassThrough{},
            D0Operator::d0_block_dst_desc_m0_n0_m1_m2_n1_m3,
            make_multi_index(0, 0, 0, 0, 0, 0),
            tensor_operation::element_wise::PassThrough{});

        auto d0_thread_copy_lds_to_vgpr = typename D0Operator::D0ThreadwiseCopyLdsToVgpr(
            make_tuple(wave_id[I1], wave_m_n_id[I1], 0, wave_m_n_id[I0], 0));

        auto& d0grad_grid_desc_m0_n0_m1_m2_n1_m3 = d0_grid_desc_m0_n0_m1_m2_n1_m3;
        auto d0grad_thread_copy_vgpr_to_lds = typename D0Operator::D0GradThreadwiseCopyVgprToLds(
            D0Operator::d0grad_block_dst_desc_n0_n1_m0_m1_m2,
            make_tuple(wave_id[I1], wave_m_n_id[I1], 0, wave_m_n_id[I0], 0),
            tensor_operation::element_wise::Scale{rp_dropout});

        auto d0grad_block_copy_lds_to_global = typename D0Operator::D0GradBlockwiseCopyLdsToGlobal(
            D0Operator::d0grad_block_src_desc_m0_n0_m1_m2_n1_m3,
            make_multi_index(0, 0, 0, 0, 0, 0),
            tensor_operation::element_wise::PassThrough{},
            d0grad_grid_desc_m0_n0_m1_m2_n1_m3,
            make_multi_index(gemm0_m_block_outer_index, block_work_idx_n, 0, 0, 0, 0),
            tensor_operation::element_wise::PassThrough{});

        if constexpr(Deterministic)
        {
            block_sync_lds();
        }

        // Initialize dK&dV
        kgrad_thread_buf.Clear();
        vgrad_thread_buf.Clear();

        // load k
        gemm_tile_k_blockwise_copy.Run(k_grid_desc_k0_n_k1,
                                       k_grid_buf,
                                       GemmBlockwiseCopy::k_block_desc_k0_n_k1,
                                       k_block_buf,
                                       I0);

        // load v
        static_for<0, GemmBlockwiseCopy::VBlockBufferSize, 1>{}([&](auto ii) {
            gemm_tile_v_blockwise_copy.Run(v_grid_desc_k0_k1_k2_n0_n1_n2_n3_k3,
                                           v_grid_buf,
                                           GemmBlockwiseCopy::v_thread_desc_k0_k1_k2_n0_n1_n2_n3_k3,
                                           make_tuple(I0, I0, I0, I0, I0, I0, I0, I0),
                                           v_thread_buf(Number<ii>{}));
            gemm_tile_v_blockwise_copy.MoveSrcSliceWindow(
                v_grid_desc_k0_k1_k2_n0_n1_n2_n3_k3, GemmBlockwiseCopy::v_block_slice_copy_step);
        });

        do
        {
            auto m_block_data_idx_on_grid =
                __builtin_amdgcn_readfirstlane(gemm0_m_block_outer_index * MPerBlock);
            if(c0_matrix_mask.IsTileSkippable(
                   m_block_data_idx_on_grid, n_block_data_idx_on_grid, MPerBlock, NPerBlock))
            {
                continue;
            }

            //
            // load d and lse
            //

            d_thread_copy_global_to_vgpr.Run(lse_grid_desc_mb_m0_m1_m2_m3_m4,
                                             d_grid_buf,
                                             lse_thread_desc_mb_m0_m1_m2_m3_m4,
                                             make_tuple(I0, I0, I0, I0, I0, I0),
                                             y_dot_ygrad_thread_buf);

            lse_thread_copy_global_to_vgpr.Run(lse_grid_desc_mb_m0_m1_m2_m3_m4,
                                               lse_grid_buf,
                                               lse_thread_desc_mb_m0_m1_m2_m3_m4,
                                               make_tuple(I0, I0, I0, I0, I0, I0),
                                               lse_thread_buf);

            // S = Q * K^T
            {
                // preload data into LDS
                s_gemm_tile_q_blockwise_copy.RunRead(q_grid_desc_k0_m_k1, q_grid_buf);

                s_gemm_tile_q_blockwise_copy.MoveSrcSliceWindow(q_grid_desc_k0_m_k1,
                                                                Gemm3::a_block_slice_copy_step);

                block_sync_lds(); // wait for previous LDS read

                s_slash_p_thread_buf.Clear();

                s_gemm_tile_q_blockwise_copy.RunWrite(Gemm3::a_block_desc_ak0_m_ak1,
                                                      gemm3_a_block_buf);

                s_gemm_tile_q_blockwise_copy.RunRead(q_grid_desc_k0_m_k1, q_grid_buf);

                block_sync_lds();
                float *cbuf= &(s_slash_p_thread_buf.GetVectorTypeReference(I0).template AsType<float>()(I0));
                s_blockwise_gemm.Run_ds(gemm3_a_block_buf.p_data_, k_block_buf.p_data_, cbuf);

                s_blockwise_gemm.MoveBBlockSrcSliceWindow(Gemm3::b_block_slice_copy_step);

                block_sync_lds();

                s_gemm_tile_q_blockwise_copy.MoveSrcSliceWindow(
                    q_grid_desc_k0_m_k1, Gemm3::a_block_slice_copy_step);

                s_gemm_tile_q_blockwise_copy.RunWrite(Gemm3::a_block_desc_ak0_m_ak1,
                                                        gemm3_a_block_buf);



                block_sync_lds();

                s_blockwise_gemm.Run_ds(gemm3_a_block_buf.p_data_, k_block_buf.p_data_+8256, cbuf);

                s_blockwise_gemm.MoveBBlockSrcSliceWindow(Gemm3::b_block_slice_copy_step);
            } // end gemm S

            // 8d thread_desc in thread scope
            constexpr auto c_thread_lengths =
                s_blockwise_gemm.GetCThreadDescriptor_M0_N0_M1_N1_M2_M3_M4_N2().GetLengths();

            // 8d block_desc in block scope
            constexpr auto c_block_lengths =
                s_blockwise_gemm.GetCBlockDescriptor_M0_N0_M1_N1_M2_M3_M4_N2().GetLengths();

            constexpr auto M0 = c_block_lengths[I0];
            constexpr auto N0 = c_block_lengths[I1];
            constexpr auto M1 = c_block_lengths[I2];
            constexpr auto N1 = c_block_lengths[I3];
            constexpr auto M2 = c_block_lengths[I4];
            constexpr auto M3 = c_block_lengths[I5];
            constexpr auto M4 = c_block_lengths[I6];
            constexpr auto N2 = c_block_lengths[I7];

            // works like multi-dimension static_for (static_ford), but provides both the linear
            // index as well as n-d index
            using Acc0TileIterator = SpaceFillingCurve<
                decltype(c_thread_lengths),
                typename arithmetic_sequence_gen<0, c_thread_lengths.Size(), 1>::type,
                typename uniform_sequence_gen<c_thread_lengths.Size(), 1>::type,
                false>; // SnakeCurved

            constexpr auto block_idx_to_m_n_adaptor = make_single_stage_tensor_adaptor(
                make_tuple(make_unmerge_transform(make_tuple(M0, M1, M2, M3, M4)),
                           make_unmerge_transform(make_tuple(N0, N1, N2))),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 2, 4, 5, 6>{}, Sequence<1, 3, 7>{}));

            // do MNK padding or upper triangular masking
            if constexpr(MaskOutUpperTriangle || PadN)
            {

                static_for<0, Acc0TileIterator::GetNumOfAccess(), 1>{}([&](auto i) {
                    auto acc0_thread_idx = Acc0TileIterator::GetIndex(i) + acc0_thread_origin;
                    auto m_local =
                        block_idx_to_m_n_adaptor.CalculateBottomIndex(acc0_thread_idx)[I0];
                    auto n_local =
                        block_idx_to_m_n_adaptor.CalculateBottomIndex(acc0_thread_idx)[I1];
                    auto m_global    = m_local + m_block_data_idx_on_grid;
                    auto n_global    = n_local + n_block_data_idx_on_grid;
                    bool masked_flag = c0_matrix_mask.IsMaskedElement(m_global, n_global);
                    s_element_op(s_slash_p_thread_buf(i),
                                 masked_flag ? -ck::NumericLimits<float>::Infinity()
                                             : s_slash_p_thread_buf[i]);
                });
            }
            else
            {
                static_for<0, s_slash_p_thread_buf.Size(), 1>{}([&](auto i) {
                    s_element_op(s_slash_p_thread_buf(i), s_slash_p_thread_buf[i]);
                });
            }

            block_sync_lds(); // wait for lds read in gemm0 blockwise gemm

            // P_i: = softmax(scalar * S_i:)
            // scaling is already performed in the preceding statements with s_element_op
            blockwise_softmax.RunWithPreCalcStats(s_slash_p_thread_buf, lse_thread_buf);

            // save z to global
            if constexpr(IsDropout)
            {
                if(p_z_grid)
                {

                    auto acc0_thread_idx = Acc0TileIterator::GetIndex(I0) + acc0_thread_origin;
                    auto m_local =
                        block_idx_to_m_n_adaptor.CalculateBottomIndex(acc0_thread_idx)[I0];
                    auto n_local =
                        block_idx_to_m_n_adaptor.CalculateBottomIndex(acc0_thread_idx)[I1];
                    auto m_global = m_local + m_block_data_idx_on_grid;
                    auto n_global = n_local + n_block_data_idx_on_grid;

                    auto global_tile_id = z_random_matrix_offset +
                                          (m_global / DropoutTile) * DropoutTile * raw_n_padded +
                                          (n_global / DropoutTile) * DropoutTile;

                    auto global_elem_id = global_tile_id + (wave_m_n_id[I0] * M4) +
                                          (n_global % DropoutTile) * raw_n_padded;

                    blockwise_dropout
                        .template ApplyDropoutAttnBwdSaveZ<decltype(s_slash_p_thread_buf),
                                                           decltype(z_tensor_buffer),
                                                           decltype(DropoutTile),
                                                           true>(s_slash_p_thread_buf,
                                                                 ph,
                                                                 global_elem_id,
                                                                 z_tensor_buffer,
                                                                 raw_n_padded);

                    z_thread_copy_vgpr_to_global.Run(
                        z_thread_desc_m0_n0_m1_n1_m2_n2_m3_m4_m5_n3,
                        make_tuple(I0, I0, I0, I0, I0, I0, I0, I0, I0, I0),
                        z_tensor_buffer,
                        z_grid_desc_m0_n0_m1_n1_m2_n2_m3_m4_m5_n3,
                        z_grid_buf);
                }
                else
                {
                    ignore = z_grid_buf;

                    auto acc0_thread_idx = Acc0TileIterator::GetIndex(I0) + acc0_thread_origin;
                    auto m_local =
                        block_idx_to_m_n_adaptor.CalculateBottomIndex(acc0_thread_idx)[I0];
                    auto n_local =
                        block_idx_to_m_n_adaptor.CalculateBottomIndex(acc0_thread_idx)[I1];
                    auto m_global = m_local + m_block_data_idx_on_grid;
                    auto n_global = n_local + n_block_data_idx_on_grid;

                    auto global_tile_id = z_random_matrix_offset +
                                          (m_global / DropoutTile) * DropoutTile * raw_n_padded +
                                          (n_global / DropoutTile) * DropoutTile;

                    auto global_elem_id = global_tile_id + (wave_m_n_id[I0] * M4) +
                                          (n_global % DropoutTile) * raw_n_padded;

                    // P_dropped
                    blockwise_dropout.template ApplyDropoutAttnBwd<decltype(s_slash_p_thread_buf),
                                                                   decltype(DropoutTile),
                                                                   true>(
                        s_slash_p_thread_buf, ph, global_elem_id, raw_n_padded);
                }
            }
            block_sync_lds(); // wait for gemm1 LDS read

            // gemm dV
            // dV = P_drop^T * dY
            {
                // TODO: explore using dynamic buffer for a1 thread buffer
                // For a1_blockwise_copy, the goal is to satisfy pipeline requirements RunRead(),
                // RunWrite(), and MoveSliceWindow(). But it is impossible to implement given that
                // the A1 source buffer is static buffer holding the output of first GEMM and
                // requires constexpr offset by design. Therefore, we pass tensor coordinate offset
                // explicitly in Run() below.

                // preload data into LDS
                vgrad_gemm_tile_ygrad_blockwise_copy.RunRead(ygrad_grid_desc_m0_o_m1,
                                                             ygrad_grid_buf);

                vgrad_gemm_tile_ygrad_blockwise_copy.MoveSrcSliceWindow(
                    ygrad_grid_desc_m0_o_m1, Gemm1::b_block_slice_copy_step);

                block_sync_lds(); // wait for previous LDS read

                vgrad_gemm_tile_ygrad_blockwise_copy.RunWrite(Gemm1::b_block_desc_bk0_n_bk1,
                                                              gemm1_b_block_buf);
                vgrad_gemm_tile_p_blockwise_copy.Run(Gemm1::a_src_thread_desc_k0_m_k1,
                                                        Gemm1::a_block_slice_copy_step * I0,
                                                        s_slash_p_thread_buf,
                                                        Gemm1::a_thread_desc_k0_m_k1,
                                                        make_tuple(I0, I0, I0),
                                                        gemm1_a_thread_buf);
                vgrad_gemm_tile_ygrad_blockwise_copy.RunRead(ygrad_grid_desc_m0_o_m1,
                                                                ygrad_grid_buf);

                block_sync_lds();
                float *cbuf= &(vgrad_thread_buf.GetVectorTypeReference(I0).template AsType<float>()(I0));
                vgrad_blockwise_gemm.Run_dv_dk(&gemm1_a_thread_buf(I0), gemm1_b_block_buf.p_data_, cbuf);

                block_sync_lds();

                vgrad_gemm_tile_ygrad_blockwise_copy.MoveSrcSliceWindow(
                    ygrad_grid_desc_m0_o_m1, Gemm1::b_block_slice_copy_step);

                vgrad_gemm_tile_ygrad_blockwise_copy.RunWrite(Gemm1::b_block_desc_bk0_n_bk1,
                                                                gemm1_b_block_buf);

                vgrad_gemm_tile_p_blockwise_copy.Run(
                    Gemm1::a_src_thread_desc_k0_m_k1,
                    Gemm1::a_block_slice_copy_step * I1,
                    s_slash_p_thread_buf,
                    Gemm1::a_thread_desc_k0_m_k1,
                    make_tuple(I0, I0, I0),
                    gemm1_a_thread_buf);

                block_sync_lds();
                vgrad_blockwise_gemm.Run_dv_dk(&gemm1_a_thread_buf(I0), gemm1_b_block_buf.p_data_, cbuf);
            } // end gemm dV

            // gemm dP
            block_sync_lds();
            // dP = dY * V^T
            // assume size K == size O so HasMainKBlockLoop is the same
            {
                // preload data into LDS
                pgrad_gemm_tile_ygrad_blockwise_copy.RunRead(ygrad_grid_desc_o0_m_o1,
                                                             ygrad_grid_buf);

                pgrad_gemm_tile_ygrad_blockwise_copy.MoveSrcSliceWindow(
                    ygrad_grid_desc_o0_m_o1, Gemm0::a_block_slice_copy_step);

                block_sync_lds(); // wait for previous LDS read

                pgrad_thread_buf.Clear();

                pgrad_gemm_tile_ygrad_blockwise_copy.RunWrite(Gemm0::a_block_desc_ak0_m_ak1,
                                                              gemm0_a_block_buf);

                pgrad_gemm_tile_ygrad_blockwise_copy.RunRead(ygrad_grid_desc_o0_m_o1,
                                                                ygrad_grid_buf);

                block_sync_lds();
                float *cbuf= &(pgrad_thread_buf.GetVectorTypeReference(I0).template AsType<float>()(I0));
                pgrad_blockwise_gemm.Run_dp(gemm0_a_block_buf.p_data_, &v_thread_buf(I0)(I0), cbuf);

                block_sync_lds();

                pgrad_gemm_tile_ygrad_blockwise_copy.MoveSrcSliceWindow(
                    ygrad_grid_desc_o0_m_o1, Gemm0::a_block_slice_copy_step);

                pgrad_gemm_tile_ygrad_blockwise_copy.RunWrite(Gemm0::a_block_desc_ak0_m_ak1,
                                                                gemm0_a_block_buf);

                block_sync_lds();

                pgrad_blockwise_gemm.Run_dp(gemm0_a_block_buf.p_data_,&v_thread_buf(I1)(I0),cbuf);

            } // end gemm dP

            // dS = P * (dP - Y_dot_dY)
            auto& sgrad_thread_buf = pgrad_thread_buf;
            constexpr auto pgrad_thread_tile_iterator =
                pgrad_blockwise_gemm.MakeCThreadTileIterator();
            constexpr auto pgrad_thread_idx_to_m_n_adaptor =
                pgrad_blockwise_gemm.MakeCThreadIndexAdaptor8DTo2D();
            static_for<0, pgrad_thread_tile_iterator.GetNumOfAccess(), 1>{}([&](auto i) {
                constexpr auto pgrad_thread_idx = pgrad_thread_tile_iterator.GetIndex(i);
                constexpr auto m =
                    pgrad_thread_idx_to_m_n_adaptor.CalculateBottomIndex(pgrad_thread_idx)[I0];
                // dS and P has same thread buf layout
                bool undropped_flag = s_slash_p_thread_buf[i] >= 0;
                sgrad_thread_buf(i) =
                    s_slash_p_thread_buf[i] *
                    (undropped_flag ? (pgrad_thread_buf[i] - y_dot_ygrad_thread_buf[Number<m>{}])
                                    : y_dot_ygrad_thread_buf[Number<m>{}]);
            });

            SubThreadBlock<BlockSize> gemm2_a_copy_subgroup(s_blockwise_gemm.GetWaveIdx()[I0],
                                                            s_blockwise_gemm.GetWaveIdx()[I1]);

            // TODO: tune gemm2 pipeline
            // gemm dQ
            // dQ = scalar * dS * K
            {
                qgrad_thread_buf.Clear();
                // load QGrad Gemm A
                const auto sgrad_slice_idx =
                    Gemm2::ASrcBlockSliceWindowIterator::GetIndexTupleOfNumber(I0);
                constexpr auto mwave_range = make_tuple(
                    sgrad_slice_idx[I2],
                    sgrad_slice_idx[I2] + Gemm2Params::ABlockSliceLengths_M0_K0_M1_K1::At(I2));
                constexpr auto nwave_range = make_tuple(
                    sgrad_slice_idx[I3],
                    sgrad_slice_idx[I3] + Gemm2Params::ABlockSliceLengths_M0_K0_M1_K1::At(I3));

                block_sync_lds(); // sync before write
                if(gemm2_a_copy_subgroup.IsBelong(mwave_range, nwave_range))
                {
                    qgrad_gemm_tile_sgrad_thread_copy_vgpr_to_lds.Run(
                        Gemm2::a_src_thread_desc_m0_n0_m1_n1_m2_m3_m4_n2,
                        make_tuple(
                            sgrad_slice_idx[I0], sgrad_slice_idx[I1], I0, I0, I0, I0, I0, I0),
                        sgrad_thread_buf,
                        Gemm2::a_block_desc_m0_k0_m1_k1_m2_m3_m4_k2,
                        gemm2_a_block_buf);
                }


                block_sync_lds(); // sync before read
                float *cbuf= &(qgrad_thread_buf.GetVectorTypeReference(I0).template AsType<float>()(I0));
                qgrad_blockwise_gemm.Run_dq(gemm2_a_block_buf.p_data_, k_block_buf.p_data_, cbuf);

                // end gemm dQ
                // atomic_add dQ
                qgrad_thread_copy_vgpr_to_global.Run(Gemm2::c_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4,
                                                    make_tuple(I0, I0, I0, I0, I0, I0, I0, I0),
                                                    qgrad_thread_buf,
                                                    qgrad_grid_desc_m0_o0_m1_o1_m2_o2_o3_o4,
                                                    qgrad_grid_buf);
            }//end dQ

            // gemm dK
            // dK = scalar * dS^T * Q
            {
                // TODO: explore using dynamic buffer for a1 thread buffer
                // For a1_blockwise_copy, the goal is to satisfy pipeline requirements RunRead(),
                // RunWrite(), and MoveSliceWindow(). But it is impossible to implement given that
                // the A1 source buffer is static buffer holding the output of first GEMM and
                // requires constexpr offset by design. Therefore, we pass tensor coordinate offset
                // explicitly in Run() below.

                // preload data into LDS
                kgrad_gemm_tile_q_blockwise_copy.RunRead(q_grid_desc_m0_k_m1, q_grid_buf);

                kgrad_gemm_tile_q_blockwise_copy.MoveSrcSliceWindow(q_grid_desc_m0_k_m1,
                                                                    Gemm1::b_block_slice_copy_step);

                block_sync_lds(); // wait for previous LDS read

                kgrad_gemm_tile_q_blockwise_copy.RunWrite(Gemm1::b_block_desc_bk0_n_bk1,
                                                          gemm1_b_block_buf);

                kgrad_gemm_tile_sgrad_blockwise_copy.Run(Gemm1::a_src_thread_desc_k0_m_k1,
                                                            Gemm1::a_block_slice_copy_step * I0,
                                                            sgrad_thread_buf,
                                                            Gemm1::a_thread_desc_k0_m_k1,
                                                            make_tuple(I0, I0, I0),
                                                            gemm1_a_thread_buf);
                kgrad_gemm_tile_q_blockwise_copy.RunRead(q_grid_desc_m0_k_m1, q_grid_buf);

                block_sync_lds();
                float *cbuf= &(kgrad_thread_buf.GetVectorTypeReference(I0).template AsType<float>()(I0));
                vgrad_blockwise_gemm.Run_dv_dk(&gemm1_a_thread_buf(I0), gemm1_b_block_buf.p_data_, cbuf);

                block_sync_lds();

                kgrad_gemm_tile_q_blockwise_copy.MoveSrcSliceWindow(
                    q_grid_desc_m0_k_m1, Gemm1::b_block_slice_copy_step);

                kgrad_gemm_tile_q_blockwise_copy.RunWrite(Gemm1::b_block_desc_bk0_n_bk1,
                                                            gemm1_b_block_buf);

                kgrad_gemm_tile_sgrad_blockwise_copy.Run(
                    Gemm1::a_src_thread_desc_k0_m_k1,
                    Gemm1::a_block_slice_copy_step * I1,
                    sgrad_thread_buf,
                    Gemm1::a_thread_desc_k0_m_k1,
                    make_tuple(I0, I0, I0),
                    gemm1_a_thread_buf);

                block_sync_lds();
                vgrad_blockwise_gemm.Run_dv_dk(&gemm1_a_thread_buf(I0), gemm1_b_block_buf.p_data_, cbuf);
            } // end gemm dK

            // move slice window
            s_gemm_tile_q_blockwise_copy.MoveSrcSliceWindow(
                q_grid_desc_k0_m_k1,
                s_gemm_tile_q_block_reset_copy_step); // rewind K and step M
            pgrad_gemm_tile_ygrad_blockwise_copy.MoveSrcSliceWindow(
                ygrad_grid_desc_o0_m_o1,
                pgrad_gemm_tile_ygrad_block_reset_copy_step); // rewind O and step M
            kgrad_gemm_tile_q_blockwise_copy.MoveSrcSliceWindow(
                q_grid_desc_m0_k_m1, kgrad_gemm_tile_q_block_next_copy_step); // step M
            vgrad_gemm_tile_ygrad_blockwise_copy.MoveSrcSliceWindow(
                ygrad_grid_desc_m0_o_m1, vgrad_gemm_tile_ygrad_block_next_copy_step); // step M
            qgrad_thread_copy_vgpr_to_global.MoveDstSliceWindow(
                qgrad_grid_desc_m0_o0_m1_o1_m2_o2_o3_o4, Gemm2::c_block_slice_copy_step); // step M
            lse_thread_copy_global_to_vgpr.MoveSrcSliceWindow(lse_grid_desc_mb_m0_m1_m2_m3_m4,
                                                              make_multi_index(-1, 0, 0, 0, 0, 0));
            d_thread_copy_global_to_vgpr.MoveSrcSliceWindow(lse_grid_desc_mb_m0_m1_m2_m3_m4,
                                                            make_multi_index(-1, 0, 0, 0, 0, 0));
            s_blockwise_gemm.MoveBBlockSrcSliceWindow(Gemm3::b_block_reset_copy_step);
            z_thread_copy_vgpr_to_global.MoveDstSliceWindow(
                z_grid_desc_m0_n0_m1_n1_m2_n2_m3_m4_m5_n3,
                make_multi_index(-1, 0, 0, 0, 0, 0, 0, 0, 0, 0));

        } while(0 < gemm0_m_block_outer_index--); // end j loop

        // shuffle dK&dV and write
        {
            static_assert(NXdlPerWave % CShuffleMXdlPerWavePerShuffle == 0 &&
                              Gemm1NXdlPerWave % CShuffleNXdlPerWavePerShuffle == 0,
                          "wrong!");

            constexpr index_t MWave = Gemm0NWaves;
            constexpr index_t NWave = Gemm0MWaves;

            // TODO: hacky, fix it!
            // thread desc same with kgrad_blockwise_gemm
            constexpr auto c_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4 =
                vgrad_blockwise_gemm.GetCThreadDescriptor_M0_N0_M1_N1_M2_N2_N3_N4();

            // TODO: hacky, fix it!
            // c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4_tmp is only used to get lengths
            // block desc same with kgrad_blockwise_gemm
            constexpr auto c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4_tmp =
                vgrad_blockwise_gemm.GetCBlockDescriptor_M0_N0_M1_N1_M2_N2_N3_N4();

            constexpr auto M0 = c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4_tmp.GetLength(I0);
            constexpr auto N0 = c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4_tmp.GetLength(I1);
            constexpr auto M1 = c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4_tmp.GetLength(I2);
            constexpr auto N1 = c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4_tmp.GetLength(I3);
            constexpr auto M2 = c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4_tmp.GetLength(I4);
            constexpr auto N2 = c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4_tmp.GetLength(I5);
            constexpr auto N3 = c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4_tmp.GetLength(I6);
            constexpr auto N4 = c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4_tmp.GetLength(I7);

            constexpr auto c_shuffle_block_desc_mblock_mperblock_nblock_nperblock =
                GetCShuffleBlockDescriptor_MBlock_MPerBlock_NBlock_NPerBlock();

            auto c_shuffle_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
                static_cast<OutputDataType*>(p_shared),
                c_shuffle_block_desc_mblock_mperblock_nblock_nperblock.GetElementSpaceSize());

            constexpr auto c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4 = transform_tensor_descriptor(
                c_shuffle_block_desc_mblock_mperblock_nblock_nperblock,
                make_tuple(
                    make_freeze_transform(I0),
                    make_unmerge_transform(make_tuple(
                        Number<CShuffleMXdlPerWavePerShuffle>{}, // M0 (MXdlPerWave) per shuffle
                        M1,                                      // M1 = MWave
                        M2)),                                    // M2 = MPerXdl
                    make_freeze_transform(I0),
                    make_unmerge_transform(make_tuple(
                        Number<CShuffleNXdlPerWavePerShuffle>{}, // N0 (NXdlPerWave) per shuffle
                        N1,                                      // N1 = NWave
                        N2,                                      // N2 * N3 * N4 = NPerXdl
                        N3,
                        N4))),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                make_tuple(
                    Sequence<>{}, Sequence<0, 2, 4>{}, Sequence<>{}, Sequence<1, 3, 5, 6, 7>{}));

            // calculate origin of thread output tensor on global memory
            //     blockwise GEMM c matrix starting index
            // index same with kgrad_blockwise_gemm
            const auto c_thread_mtx_on_block =
                vgrad_blockwise_gemm.CalculateCThreadOriginDataIndex(I0, I0, I0, I0);

            const index_t m_thread_data_on_block = c_thread_mtx_on_block[I0];
            const index_t n_thread_data_on_block = c_thread_mtx_on_block[I1];

            const auto m_thread_data_on_block_to_m0_m1_m2_adaptor =
                make_single_stage_tensor_adaptor(
                    make_tuple(make_merge_transform(make_tuple(M0, M1, M2))),
                    make_tuple(Sequence<0, 1, 2>{}),
                    make_tuple(Sequence<0>{}));

            const auto m_thread_data_on_block_idx =
                m_thread_data_on_block_to_m0_m1_m2_adaptor.CalculateBottomIndex(
                    make_multi_index(m_thread_data_on_block));

            const auto n_thread_data_on_block_to_n0_n1_n2_n3_n4_adaptor =
                make_single_stage_tensor_adaptor(
                    make_tuple(make_merge_transform(make_tuple(N0, N1, N2, N3, N4))),
                    make_tuple(Sequence<0, 1, 2, 3, 4>{}),
                    make_tuple(Sequence<0>{}));

            const auto n_thread_data_on_block_idx =
                n_thread_data_on_block_to_n0_n1_n2_n3_n4_adaptor.CalculateBottomIndex(
                    make_multi_index(n_thread_data_on_block));

            // shuffle: threadwise copy C from VGPR to LDS
            auto vgrad_thread_copy_vgpr_to_lds =
                ThreadwiseTensorSliceTransfer_v1r3<FloatGemmAcc,
                                                   OutputDataType,
                                                   decltype(c_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4),
                                                   decltype(c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4),
                                                   SElementwiseOperation,
                                                   Sequence<CShuffleMXdlPerWavePerShuffle,
                                                            CShuffleNXdlPerWavePerShuffle,
                                                            I1,
                                                            I1,
                                                            I1,
                                                            N2,
                                                            I1,
                                                            N4>,
                                                   Sequence<0, 1, 2, 3, 4, 5, 6, 7>,
                                                   7,
                                                   1,
                                                   InMemoryDataOperationEnum::Set,
                                                   1,
                                                   true>{
                    c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4,
                    make_multi_index(0,
                                     0,
                                     m_thread_data_on_block_idx[I1],
                                     n_thread_data_on_block_idx[I1],
                                     m_thread_data_on_block_idx[I2],
                                     n_thread_data_on_block_idx[I2],
                                     n_thread_data_on_block_idx[I3],
                                     n_thread_data_on_block_idx[I4]),
                    tensor_operation::element_wise::Scale{rp_dropout}};

            // shuffle: blockwise copy C from LDS to global
            auto vgrad_shuffle_block_copy_lds_to_global = ThreadGroupTensorSliceTransfer_v6r1<
                ThisThreadBlock,            // ThreadGroup
                CElementwiseOperation,      // ElementwiseOperation,
                CGlobalMemoryDataOperation, // DstInMemOp,
                Sequence<1,
                         CShuffleMXdlPerWavePerShuffle * MWave * MPerXdl,
                         1,
                         CShuffleNXdlPerWavePerShuffle * NWave * NPerXdl>, // BlockSliceLengths,
                CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
                Sequence<0, 1, 2, 3>, // typename ThreadClusterArrangeOrder,
                OutputDataType,        // typename SrcData,
                OutputDataType,       // typename DstData,
                decltype(c_shuffle_block_desc_mblock_mperblock_nblock_nperblock),
                decltype(vgrad_grid_desc_nblock_nperblock_oblock_operblock),
                Sequence<0, 1, 2, 3>,                           // typename DimAccessOrder,
                3,                                              // index_t VectorDim,
                CShuffleBlockTransferScalarPerVector_NPerBlock, // index_t ScalarPerVector,
                true,  // bool ThreadTransferSrcResetCoordinateAfterRun,
                false> // bool ThreadTransferDstResetCoordinateAfterRun>
                {c_shuffle_block_desc_mblock_mperblock_nblock_nperblock,
                 make_multi_index(0, 0, 0, 0),
                 vgrad_grid_desc_nblock_nperblock_oblock_operblock,
                 make_multi_index(block_work_idx_n, 0, block_work_idx[I1], 0),
                 c_element_op};

            // shuffle: threadwise copy C from VGPR to LDS
            auto kgrad_thread_copy_vgpr_to_lds =
                ThreadwiseTensorSliceTransfer_v1r3<FloatGemmAcc,
                                                   OutputDataType,
                                                   decltype(c_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4),
                                                   decltype(c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4),
                                                   SElementwiseOperation,
                                                   Sequence<CShuffleMXdlPerWavePerShuffle,
                                                            CShuffleNXdlPerWavePerShuffle,
                                                            I1,
                                                            I1,
                                                            I1,
                                                            N2,
                                                            I1,
                                                            N4>,
                                                   Sequence<0, 1, 2, 3, 4, 5, 6, 7>,
                                                   7,
                                                   1,
                                                   InMemoryDataOperationEnum::Set,
                                                   1,
                                                   true>{
                    c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4,
                    make_multi_index(0,
                                     0,
                                     m_thread_data_on_block_idx[I1],
                                     n_thread_data_on_block_idx[I1],
                                     m_thread_data_on_block_idx[I2],
                                     n_thread_data_on_block_idx[I2],
                                     n_thread_data_on_block_idx[I3],
                                     n_thread_data_on_block_idx[I4]),
                    scale_rp_dropout};

            // shuffle: blockwise copy C from LDS to global
            auto kgrad_shuffle_block_copy_lds_to_global = ThreadGroupTensorSliceTransfer_v6r1<
                ThisThreadBlock,            // ThreadGroup
                CElementwiseOperation,      // ElementwiseOperation,
                CGlobalMemoryDataOperation, // DstInMemOp,
                Sequence<1,
                         CShuffleMXdlPerWavePerShuffle * MWave * MPerXdl,
                         1,
                         CShuffleNXdlPerWavePerShuffle * NWave * NPerXdl>, // BlockSliceLengths,
                CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
                Sequence<0, 1, 2, 3>, // typename ThreadClusterArrangeOrder,
                OutputDataType,        // typename SrcData,
                OutputDataType,       // typename DstData,
                decltype(c_shuffle_block_desc_mblock_mperblock_nblock_nperblock),
                decltype(kgrad_grid_desc_nblock_nperblock_oblock_operblock),
                Sequence<0, 1, 2, 3>,                           // typename DimAccessOrder,
                3,                                              // index_t VectorDim,
                CShuffleBlockTransferScalarPerVector_NPerBlock, // index_t ScalarPerVector,
                true,  // bool ThreadTransferSrcResetCoordinateAfterRun,
                false> // bool ThreadTransferDstResetCoordinateAfterRun>
                {c_shuffle_block_desc_mblock_mperblock_nblock_nperblock,
                 make_multi_index(0, 0, 0, 0),
                 kgrad_grid_desc_nblock_nperblock_oblock_operblock,
                 make_multi_index(block_work_idx_n, 0, block_work_idx[I1], 0),
                 c_element_op};

            // space filling curve for threadwise C in VGPR
            constexpr auto sfc_c_vgpr =
                SpaceFillingCurve<Sequence<NXdlPerWave, Gemm1NXdlPerWave, 1, 1, 1, N2, 1, N4>,
                                  Sequence<0, 1, 2, 3, 4, 5, 6, 7>,
                                  Sequence<CShuffleMXdlPerWavePerShuffle,
                                           CShuffleNXdlPerWavePerShuffle,
                                           1,
                                           1,
                                           1,
                                           N2,
                                           1,
                                           N4>>{};

            // space filling curve for shuffled blockwise C in global mem
            constexpr auto sfc_c_global =
                SpaceFillingCurve<Sequence<1, NPerBlock, 1, Gemm1NPerBlock>,
                                  Sequence<0, 2, 1, 3>,
                                  Sequence<1,
                                           CShuffleMXdlPerWavePerShuffle * MWave * MPerXdl,
                                           1,
                                           CShuffleNXdlPerWavePerShuffle * NWave * NPerXdl>>{};

            constexpr index_t num_access = sfc_c_vgpr.GetNumOfAccess();

            static_assert(num_access == sfc_c_global.GetNumOfAccess(), "wrong!");

            // dK
            static_for<0, num_access, 1>{}([&](auto access_id) {
                // make sure it's safe to write to LDS
                block_sync_lds();

                // each thread write its data from VGPR to LDS
                kgrad_thread_copy_vgpr_to_lds.Run(c_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4,
                                                  sfc_c_vgpr.GetIndexTupleOfNumber(access_id),
                                                  kgrad_thread_buf,
                                                  c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4,
                                                  c_shuffle_block_buf);

                // make sure it's safe to read from LDS
                block_sync_lds();

                // each block copy its data from LDS to global
                kgrad_shuffle_block_copy_lds_to_global.Run(
                    c_shuffle_block_desc_mblock_mperblock_nblock_nperblock,
                    c_shuffle_block_buf,
                    kgrad_grid_desc_nblock_nperblock_oblock_operblock,
                    kgrad_grid_buf);

                if constexpr(access_id < num_access - 1)
                {
                    constexpr auto c_global_step = sfc_c_global.GetForwardStep(access_id);

                    // move on C
                    kgrad_shuffle_block_copy_lds_to_global.MoveDstSliceWindow(
                        kgrad_grid_desc_nblock_nperblock_oblock_operblock, c_global_step);
                }
            });

            // dV
            static_for<0, num_access, 1>{}([&](auto access_id) {
                // make sure it's safe to write to LDS
                block_sync_lds();

                // each thread write its data from VGPR to LDS
                vgrad_thread_copy_vgpr_to_lds.Run(c_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4,
                                                  sfc_c_vgpr.GetIndexTupleOfNumber(access_id),
                                                  vgrad_thread_buf,
                                                  c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4,
                                                  c_shuffle_block_buf);

                // make sure it's safe to read from LDS
                block_sync_lds();

                // each block copy its data from LDS to global
                vgrad_shuffle_block_copy_lds_to_global.Run(
                    c_shuffle_block_desc_mblock_mperblock_nblock_nperblock,
                    c_shuffle_block_buf,
                    vgrad_grid_desc_nblock_nperblock_oblock_operblock,
                    vgrad_grid_buf);

                if constexpr(access_id < num_access - 1)
                {
                    constexpr auto c_global_step = sfc_c_global.GetForwardStep(access_id);

                    // move on C
                    vgrad_shuffle_block_copy_lds_to_global.MoveDstSliceWindow(
                        vgrad_grid_desc_nblock_nperblock_oblock_operblock, c_global_step);
                }
            });
        }
    }
};

} // namespace ck
