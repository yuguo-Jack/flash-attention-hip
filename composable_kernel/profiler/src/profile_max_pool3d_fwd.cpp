// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>
#include <vector>
#include <unordered_map>

#include "profiler/data_type_enum.hpp"
#include "profiler/profile_pool3d_fwd_impl.hpp"
#include "profiler_operation_registry.hpp"

using ck::index_t;

struct maxPoolFwdArgParser
{
    std::unordered_map<std::string, std::vector<int>> long_opts = {
        {"length", {}}, {"wsize", {}}, {"wstride", {}}, {"pad1", {}}, {"pad2", {}}};

    bool parse_opt(int argc, char* argv[], const std::string& key, int i)
    {
        if(std::string("--") + key == argv[i])
        {
            int pos = i;
            while(++i < argc && argv[i][0] != '-') {}
            int end = i;
            for(int j = pos + 1; j < end; j++)
            {
                long_opts[key].push_back(std::stoi(argv[j]));
            }
            return true;
        }
        return false;
    }

    void operator()(int argc, char* argv[])
    {
        for(auto& kv : long_opts)
        {
            for(int i = 1; i < argc; i++)
            {
                if(parse_opt(argc, argv, kv.first, i))
                    break;
            }
        }
    }
};

void print_help_max_pool3d_fwd()
{
    std::cout << "arg1: data type (0: fp16; 1: fp32)\n"
              << "arg2: verification (0: no; 1: yes)\n"
              << "arg3: initialization (0: no init; 1: integer value; 2: decimal value)\n"
              << "arg4: print tensor value (0: no; 1: yes)\n"
              << "arg5: time kernel (0=no, 1=yes)\n"
              << "arg6: return index (0=no, 1=yes)\n"
              << "--length: input tensor length for NCDHW(e.g, --length 2 32 30 30 30) \n"
              << "--wsize: window size for ZYX (e.g, --wsize 2 2 2) \n"
              << "--wstride: window stride for DHW (e.g, --wstride 2 2 2) \n"
              << "--pad1: left side of padding in DHW (e.g, --pad1 1 1 1) \n"
              << "--pad2: right side of padding in DHW (e.g, --pad2 1 1 1) \n"
              << "eg: ckProfiler max_pool3d_fwd 0 1 2 0 1 0 --length 2 32 30 30 30 --wsize 2 2 2 "
                 "--wstride 2 2 2 --pad1 1 1 1 --pad2 1 1 1"
              << std::endl;
}

int profile_max_pool3d_fwd(int argc, char* argv[])
{
    ck::DataTypeEnum data_type = ck::DataTypeEnum::Half;
    bool do_verification       = true;
    int init_method            = 0;
    bool do_log                = false;
    bool time_kernel           = true;
    bool return_index          = false;

    std::vector<index_t> in_length = {2, 32, 30, 30, 30};
    std::vector<index_t> wsize     = {2, 2, 2};
    std::vector<index_t> wstride   = {2, 2, 2};
    std::vector<index_t> pad1      = {1, 1, 1};
    std::vector<index_t> pad2      = {1, 1, 1};

    if(argc != 2 && argc != 30)
    {
        print_help_max_pool3d_fwd();
        return 0;
    }
    else if(argc == 30)
    {
        data_type       = static_cast<ck::DataTypeEnum>(std::stoi(argv[2]));
        do_verification = std::stoi(argv[3]);
        init_method     = std::stoi(argv[4]);
        do_log          = std::stoi(argv[5]);
        time_kernel     = std::stoi(argv[6]);
        return_index    = std::stoi(argv[7]);

        // parse the long options
        maxPoolFwdArgParser arg_parser;
        arg_parser(argc, argv);
        in_length = arg_parser.long_opts["length"];
        wsize     = arg_parser.long_opts["wsize"];
        wstride   = arg_parser.long_opts["wstride"];
        pad1      = arg_parser.long_opts["pad1"];
        pad2      = arg_parser.long_opts["pad2"];
    }

    using F16                 = ck::half_t;
    using F32                 = float;
    using I32                 = int32_t;
    constexpr auto ReduceOpId = ck::ReduceTensorOp::MAX;

    if(data_type == ck::DataTypeEnum::Half)
    {
        if(return_index)
            ck::profiler::profile_pool3d_fwd_impl<F16, F16, F16, I32, ReduceOpId, false, true>(
                do_verification,
                init_method,
                do_log,
                time_kernel,
                in_length,
                wsize,
                wstride,
                pad1,
                pad2);
        else
            ck::profiler::profile_pool3d_fwd_impl<F16, F16, F16, I32, ReduceOpId, false, false>(
                do_verification,
                init_method,
                do_log,
                time_kernel,
                in_length,
                wsize,
                wstride,
                pad1,
                pad2);
    }
    else if(data_type == ck::DataTypeEnum::Float)
    {
        if(return_index)
            ck::profiler::profile_pool3d_fwd_impl<F32, F32, F32, I32, ReduceOpId, false, true>(
                do_verification,
                init_method,
                do_log,
                time_kernel,
                in_length,
                wsize,
                wstride,
                pad1,
                pad2);
        else
            ck::profiler::profile_pool3d_fwd_impl<F32, F32, F32, I32, ReduceOpId, false, false>(
                do_verification,
                init_method,
                do_log,
                time_kernel,
                in_length,
                wsize,
                wstride,
                pad1,
                pad2);
    }
    else
    {
        throw std::runtime_error("not implemented yet");
    }

    return 0;
}

REGISTER_PROFILER_OPERATION("max_pool3d_fwd", "max_pool3d fwd", profile_max_pool3d_fwd);
