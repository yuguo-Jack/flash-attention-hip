// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/data_type.hpp"
#include "ck/utility/philox_rand.hpp"

namespace ck {

template <typename DataType, typename ThreadSliceDesc_M_K>
struct BlockwiseDropout
{
    static constexpr auto I0         = Number<0>{};
    static constexpr auto I1         = Number<1>{};
    static constexpr index_t MRepeat = ThreadSliceDesc_M_K{}.GetLength(I0);
    static constexpr index_t KRepeat = ThreadSliceDesc_M_K{}.GetLength(I1);

    // template <typename CThreadBuffer, bool using_sign_bit = false>
    // __host__ __device__ void ApplyDropout(CThreadBuffer& in_thread_buf, ck::philox& ph)
    // {

    //     auto execute_dropout = [&](bool keep, DataType val) {
    //         if constexpr(using_sign_bit)
    //             return keep ? val : -val;
    //         else
    //             return keep ? val * p_dropout_rescale : float(0);
    //     };

    //     constexpr int tmp_size = MRepeat * KRepeat;

    //     int philox_calls = tmp_size / 8;

    //     ushort tmp[tmp_size];
    //     for(int i = 0; i < philox_calls; i++)
    //     {
    //         ph.get_random_8x16((tmp + i * 8));
    //     }

    //     block_sync_lds();

    //     int tmp_index = 0;
    //     static_for<0, MRepeat, 1>{}([&](auto iM) {
    //         static_for<0, KRepeat, 1>{}([&](auto iK) {
    //             auto offset = Number<ThreadSliceDesc_M_K{}.CalculateOffset(make_tuple(iM,
    //             iK))>{}; in_thread_buf(offset) =
    //                 execute_dropout(tmp[tmp_index] <= p_dropout_uint8_t, in_thread_buf(offset));
    //             tmp_index = tmp_index + 1;
    //         });
    //     });
    // }

    // template <typename CThreadBuffer, typename ZThreadBuffer, bool using_sign_bit = false>
    // __host__ __device__ void
    // ApplyDropout(CThreadBuffer& in_thread_buf, ck::philox& ph, ZThreadBuffer& z_thread_buf)
    // {

    //     auto execute_dropout = [&](bool keep, DataType val) {
    //         if constexpr(using_sign_bit)
    //             return keep ? val : -val;
    //         else
    //             return keep ? val * p_dropout_rescale : float(0);
    //     };

    //     constexpr int tmp_size = MRepeat * KRepeat;

    //     int philox_calls = tmp_size / 8;

    //     ushort tmp[tmp_size];
    //     for(int i = 0; i < philox_calls; i++)
    //     {
    //         ph.get_random_8x16((tmp + i * 8));
    //     }

    //     block_sync_lds();

    //     int tmp_index = 0;
    //     static_for<0, MRepeat, 1>{}([&](auto iM) {
    //         static_for<0, KRepeat, 1>{}([&](auto iK) {
    //             auto offset = Number<ThreadSliceDesc_M_K{}.CalculateOffset(make_tuple(iM,
    //             iK))>{}; in_thread_buf(offset) =
    //                 execute_dropout(tmp[tmp_index] <= p_dropout_uint8_t, in_thread_buf(offset));
    //             z_thread_buf(offset) = tmp[tmp_index];
    //             tmp_index            = tmp_index + 1;
    //         });
    //     });
    // }

    // template <typename CThreadBuffer,
    //           typename ZThreadBuffer,
    //           bool using_sign_bit,
    //           typename N0,
    //           typename Offset>
    // __host__ __device__ void
    // ApplyDropout(CThreadBuffer& in_thread_buf, ck::philox& ph, ZThreadBuffer& z_thread_buf)
    // {

    //     auto execute_dropout = [&](bool keep, DataType val) {
    //         if constexpr(using_sign_bit)
    //             return keep ? val : -val;
    //         else
    //             return keep ? val * p_dropout_rescale : float(0);
    //     };

    //     constexpr int tmp_size = MRepeat * KRepeat / N0{}.value;

    //     int philox_calls = tmp_size / 8;

    //     ushort tmp[tmp_size];
    //     for(int i = 0; i < philox_calls; i++)
    //     {
    //         ph.get_random_8x16((tmp + i * 8));
    //     }

    //     block_sync_lds();

    //     constexpr auto iOffset = Number<tmp_size>{} * Offset{};
    //     static_for<0, tmp_size, 1>{}([&](auto i) {
    //         in_thread_buf(i + iOffset) =
    //             execute_dropout(tmp[i.value] <= p_dropout_uint8_t, in_thread_buf(i + iOffset));
    //         z_thread_buf(i) = tmp[i.value];
    //     });
    // }

    template <typename CThreadBuffer, typename Offset, bool using_sign_bit = false>
    __host__ __device__ void ApplyDropoutAttnBwd(CThreadBuffer& in_thread_buf,
                                                 ck::philox& ph,
                                                 index_t element_global_1d_id,
                                                 index_t MRaw)
    {

        auto execute_dropout = [&](bool keep, DataType val) {
            if constexpr(using_sign_bit)
                return keep ? val : -val;
            else
                return keep ? val * p_dropout_rescale : float(0);
        };

        constexpr int tmp_size = MRepeat * KRepeat;

        int philox_calls = tmp_size / 16;

        uint8_t tmp[tmp_size];
        for(int i = 0; i < philox_calls; i++)
        {
            ph.get_random_16x8((tmp + i * 16), element_global_1d_id + i * Offset{} * MRaw);
        }

        block_sync_lds();

        int tmp_index = 0;
        static_for<0, MRepeat, 1>{}([&](auto iM) {
            static_for<0, KRepeat, 1>{}([&](auto iK) {
                auto offset = Number<ThreadSliceDesc_M_K{}.CalculateOffset(make_tuple(iM, iK))>{};
                in_thread_buf(offset) =
                    execute_dropout(tmp[tmp_index] <= p_dropout_uint8_t, in_thread_buf(offset));
                tmp_index = tmp_index + 1;
            });
        });
    }

    template <typename CThreadBuffer,
              typename ZThreadBuffer,
              typename Offset,
              bool using_sign_bit = false>
    __host__ __device__ void ApplyDropoutAttnBwdSaveZ(CThreadBuffer& in_thread_buf,
                                                      ck::philox& ph,
                                                      index_t element_global_1d_id,
                                                      ZThreadBuffer& z_thread_buf,
                                                      index_t MRaw)
    {

        auto execute_dropout = [&](bool keep, DataType val) {
            if constexpr(using_sign_bit)
                return keep ? val : -val;
            else
                return keep ? val * p_dropout_rescale : float(0);
        };

        constexpr int tmp_size = MRepeat * KRepeat;

        int philox_calls = tmp_size / 16;

        uint8_t tmp[tmp_size];
        for(int i = 0; i < philox_calls; i++)
        {
            ph.get_random_16x8((tmp + i * 16), element_global_1d_id + i * Offset{} * MRaw);
        }

        block_sync_lds();

        int tmp_index = 0;
        static_for<0, MRepeat, 1>{}([&](auto iM) {
            static_for<0, KRepeat, 1>{}([&](auto iK) {
                auto offset = Number<ThreadSliceDesc_M_K{}.CalculateOffset(make_tuple(iM, iK))>{};
                in_thread_buf(offset) =
                    execute_dropout(tmp[tmp_index] <= p_dropout_uint8_t, in_thread_buf(offset));
                z_thread_buf(offset) = tmp[tmp_index];
                tmp_index            = tmp_index + 1;
            });
        });
    }

    template <typename CThreadBuffer, typename ZThreadBuffer, typename Step, typename Offset>
    __host__ __device__ void ApplyDropoutWithZ(CThreadBuffer& in_thread_buf,
                                               ZThreadBuffer& z_thread_buf)
    {

        auto execute_dropout = [&](bool keep, DataType val) {
            return keep ? val * p_dropout_rescale : float(0);
        };

        constexpr int tmp_size = MRepeat * KRepeat / Step{}.value;
        static_for<0, tmp_size, 1>{}([&](auto i) {
            in_thread_buf(i + Offset{}) =
                execute_dropout(z_thread_buf(i) <= p_dropout_uint8_t, in_thread_buf(i + Offset{}));
        });
    }

    // get raw z matrix with random number for shuffle
    template <typename ZThreadBuffer, typename Step, typename Offset>
    __host__ __device__ void GenerateZMatrixAttnFwd(ck::philox& ph,
                                                    index_t element_global_1d_id,
                                                    ZThreadBuffer& z_thread_buf)
    {
        constexpr int tmp_size = MRepeat * KRepeat / Step{}.value;

        int philox_calls = tmp_size / 16;

        uint8_t tmp[tmp_size];
        for(int i = 0; i < philox_calls; i++)
        {
            ph.get_random_16x8((tmp + i * 16), element_global_1d_id + i * Offset{});
        }

        static_for<0, tmp_size, 1>{}([&](auto i) { z_thread_buf(i) = tmp[i.value]; });
    }

    uint8_t p_dropout_uint8_t;
    DataType p_dropout_rescale;
};

} // namespace ck
