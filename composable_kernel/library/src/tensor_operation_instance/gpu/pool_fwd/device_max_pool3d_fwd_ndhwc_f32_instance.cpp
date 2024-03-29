// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include "pool_fwd_instance_common.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

static constexpr auto ReduceOpId = ck::ReduceTensorOp::MAX;

void add_device_pool3d_fwd_ndhwc_f32_instances(
    std::vector<std::unique_ptr<DevicePoolFwd<5, 3, F32, F32, I32, ReduceOpId, false>>>& instances)
{
    add_device_operation_instances(
        instances, device_pool3d_fwd_ndhwc_instances<F32, F32, I32, F32, ReduceOpId, false>{});
}

void add_device_pool3d_fwd_ndhwc_index_f32_instances(
    std::vector<std::unique_ptr<DevicePoolFwd<5, 3, F32, F32, I32, ReduceOpId, true>>>& instances)
{
    add_device_operation_instances(
        instances, device_pool3d_fwd_ndhwc_instances<F32, F32, I32, F32, ReduceOpId, true>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
