// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

int run(int argc, char* argv[])
{
    bool do_verification = true;
    int init_method      = 1;
    bool time_kernel     = true;

    bool input_permute  = false;
    bool output_permute = true;

    float p_drop                    = 0.2;
    int h_ratio                     = 1; // G1Q / G1KV
    const unsigned long long seed   = 1;
    const unsigned long long offset = 0;

    if(argc == 1)
    {
        // use default case
    }
    else if(argc == 4)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        time_kernel     = std::stoi(argv[3]);
    }
    else if(argc == 8)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        time_kernel     = std::stoi(argv[3]);

        p_drop         = std::stoi(argv[4]);
        h_ratio        = std::stof(argv[5]);
        input_permute  = std::stoi(argv[6]);
        output_permute = std::stoi(argv[7]);
    }
    else
    {
        printf("arg1: verification (0=no, 1=yes)\n");
        printf("arg2: initialization (0=no init, 1=integer value, 2=decimal value)\n");
        printf("arg3: time kernel (0=no, 1=yes)\n");
        printf("arg4: p_drop\n");
        printf("arg5: h_ratio\n");
        printf("arg6 to 7: input / output permute\n");
        exit(0);
    }

    float p_dropout                = 1 - p_drop;
    ZDataType p_dropout_in_uint8_t = ZDataType(std::floor(p_dropout * 255.0));
    float rp_dropout               = 1.0 / p_dropout;

    float alpha = 1; // scaling after 1st gemm

    std::size_t group_count = 8;

    // Problem descs
    std::vector<DeviceGemmInstance::ProblemDesc> problem_descs;
    std::vector<const void*> p_a;
    std::vector<const void*> p_b0;
    std::vector<const void*> p_b1;
    std::vector<void*> p_c;
    std::vector<const void*> p_d;
    std::vector<void*> p_z;         // for result verification
    std::vector<void*> p_z_nullptr; // for time test
    std::vector<void*> p_lse;
    std::vector<std::vector<int>> g0_g1q_m_n_k_o;

    std::vector<Tensor<ADataType>> a_tensors;
    std::vector<Tensor<B0DataType>> b0_tensors;
    std::vector<Tensor<B1DataType>> b1_tensors;
    std::vector<Tensor<CDataType>> c_tensors;
    std::vector<Tensor<Acc0BiasDataType>> d_tensors;
    std::vector<Tensor<ZDataType>> z_tensors;
    std::vector<Tensor<LSEDataType>> lse_tensors;

    using DeviceMemPtr = std::unique_ptr<DeviceMem>;
    std::vector<DeviceMemPtr> a_tensors_device;
    std::vector<DeviceMemPtr> b0_tensors_device;
    std::vector<DeviceMemPtr> b1_tensors_device;
    std::vector<DeviceMemPtr> c_tensors_device;
    std::vector<DeviceMemPtr> d_tensors_device;
    std::vector<DeviceMemPtr> z_tensors_device;
    std::vector<DeviceMemPtr> lse_tensors_device;

    std::size_t flop = 0, num_byte = 0;

    // std::cout << "group count " << group_count << ". printing first 4 groups\n";
    for(std::size_t i = 0; i < group_count; i++)
    {

        int M    = 128 * (rand() % 8) + (rand() % 128);
        int N    = 128 * (rand() % 8) + (rand() % 128);
        int K    = DIM;
        int O    = DIM;
        int G0   = rand() % 3 + 1;
        int G1KV = rand() % 5 + 1;
        int G1Q  = G1KV * h_ratio;

        g0_g1q_m_n_k_o.push_back({G0, G1Q, M, N, K, O});

        std::vector<ck::index_t> a_gs_ms_ks_lengths{G0, G1Q, M, K};
        std::vector<ck::index_t> a_gs_ms_ks_strides =
            input_permute
                ? std::vector<ck::index_t>{M * G1Q * K, K, G1Q * K, 1} // A layout [G0, M, G1Q, K]
                : std::vector<ck::index_t>{G1Q * M * K, M * K, K, 1};  // A layout [G0, G1Q, M, K]

        std::vector<ck::index_t> b0_gs_ns_ks_lengths{G0, G1KV, N, K};
        std::vector<ck::index_t> b0_gs_ns_ks_strides =
            input_permute
                ? std::vector<ck::index_t>{N * G1KV * K, K, G1KV * K, 1}
                // B0 layout [G0, N, G1KV, K]
                : std::vector<ck::index_t>{G1KV * N * K, N * K, K, 1}; // B0 layout [G0, G1KV, N, K]

        std::vector<ck::index_t> b1_gs_os_ns_lengths{G0, G1KV, O, N};
        std::vector<ck::index_t> b1_gs_os_ns_strides =
            input_permute
                ? std::vector<ck::index_t>{N * G1KV * O, O, 1, G1KV * O}
                // B1 layout [G0, N, G1KV, O]
                : std::vector<ck::index_t>{G1KV * N * O, N * O, 1, O}; // B1 layout [G0, G1KV, N, O]

        std::vector<ck::index_t> c_gs_ms_os_lengths{G0, G1Q, M, O};
        std::vector<ck::index_t> c_gs_ms_os_strides =
            output_permute
                ? std::vector<ck::index_t>{M * G1Q * O, O, G1Q * O, 1} // C layout [G0, M, G1Q, O]
                : std::vector<ck::index_t>{G1Q * M * O, M * O, O, 1};  // C layout [G0, G1Q, M, O]

        std::vector<ck::index_t> d_gs_ms_ns_lengths{G0, G1Q, M, N};
        std::vector<ck::index_t> d_gs_ms_ns_strides =
            input_permute
                ? std::vector<ck::index_t>{M * G1Q * N, N, G1Q * N, 1} // D layout [G0, M, G1Q, N]
                : std::vector<ck::index_t>{G1Q * M * N, M * N, N, 1};  // D layout [G0, G1Q, M, N]

        std::vector<ck::index_t> z_gs_ms_ns_lengths{G0, G1Q, M, N};
        std::vector<ck::index_t> z_gs_ms_ns_strides =
            input_permute
                ? std::vector<ck::index_t>{M * G1Q * N, N, G1Q * N, 1} // Z layout [G0, M, G1Q, N]
                : std::vector<ck::index_t>{G1Q * M * N, M * N, N, 1};  // Z layout [G0, G1Q, M, N]

        std::vector<ck::index_t> lse_gs_ms_lengths{G0, G1Q, M};
        std::vector<ck::index_t> lse_gs_ms_strides =
            std::vector<ck::index_t>{G1Q * M, M, 1}; // LSE layout [G0, G1Q, M]

        problem_descs.push_back({a_gs_ms_ks_lengths,
                                 a_gs_ms_ks_strides,
                                 b0_gs_ns_ks_lengths,
                                 b0_gs_ns_ks_strides,
                                 b1_gs_os_ns_lengths,
                                 b1_gs_os_ns_strides,
                                 c_gs_ms_os_lengths,
                                 c_gs_ms_os_strides,
                                 z_gs_ms_ns_lengths,
                                 z_gs_ms_ns_strides,
                                 lse_gs_ms_lengths,
                                 lse_gs_ms_strides,
                                 d_gs_ms_ns_lengths, // acc0_biases_gs_ms_ns_lengths
                                 d_gs_ms_ns_strides, // acc0_biases_gs_ms_ns_strides
                                 {},                 // acc1_biases_gs_ms_os_lengths
                                 {}});               // acc1_biases_gs_ms_os_strides

        // C_m_o = A_m_k * B0_k_n * B1_n_o
        Tensor<ADataType> a_gs_ms_ks(a_gs_ms_ks_lengths, a_gs_ms_ks_strides);
        Tensor<B0DataType> b0_gs_ns_ks(b0_gs_ns_ks_lengths, b0_gs_ns_ks_strides);
        Tensor<B1DataType> b1_gs_os_ns(b1_gs_os_ns_lengths, b1_gs_os_ns_strides);
        Tensor<CDataType> c_gs_ms_os_device_result(c_gs_ms_os_lengths, c_gs_ms_os_strides);
        Tensor<Acc0BiasDataType> d_gs_ms_ns(d_gs_ms_ns_lengths, d_gs_ms_ns_strides);
        Tensor<ZDataType> z_gs_ms_ns(z_gs_ms_ns_lengths, z_gs_ms_ns_strides);
        Tensor<LSEDataType> lse_gs_ms_device_result(lse_gs_ms_lengths, lse_gs_ms_strides);

        int Batch = G0 * G1Q;
        flop += (size_t(M) * N * K * 2 + size_t(M) * N * O * 2) * Batch;
        num_byte +=
            (sizeof(ADataType) * M * K + sizeof(B0DataType) * K * N + sizeof(B1DataType) * N * O +
             sizeof(CDataType) * M * O +
             sizeof(Acc0BiasDataType) * M * N * (std::is_void<Acc0BiasDataType>::value ? 0 : 1)) *
            Batch;

        if(i < 4)
        {
            std::cout << "a_gs_ms_ks[" << i << "]: " << a_gs_ms_ks.mDesc << ", "
                      << "b0_gs_ns_ks[" << i << "]: " << b0_gs_ns_ks.mDesc << ", "
                      << "b1_gs_os_ns[" << i << "]: " << b1_gs_os_ns.mDesc << ", "
                      << "c_gs_ms_os[" << i << "]: " << c_gs_ms_os_device_result.mDesc << ", "
                      << "d_gs_ms_ns[" << i << "]: " << d_gs_ms_ns.mDesc << ", "
                      << "z_gs_ms_ns[" << i << "]: " << z_gs_ms_ns.mDesc << ", "
                      << "lse_gs_ms_os[" << i << "]: " << lse_gs_ms_device_result.mDesc
                      << std::endl;
        }

        z_gs_ms_ns.GenerateTensorValue(GeneratorTensor_1<ZDataType>{0});

        switch(init_method)
        {
        case 0: break;
        case 1:
            a_gs_ms_ks.GenerateTensorValue(GeneratorTensor_2<ADataType>{-2, 2});
            b0_gs_ns_ks.GenerateTensorValue(GeneratorTensor_2<B0DataType>{-2, 2});
            b1_gs_os_ns.GenerateTensorValue(GeneratorTensor_2<B1DataType>{-2, 2});
            d_gs_ms_ns.GenerateTensorValue(GeneratorTensor_2<Acc0BiasDataType>{-1, 1});
            break;
        case 2:
            a_gs_ms_ks.GenerateTensorValue(GeneratorTensor_3<ADataType>{0.0, 1.0});
            b0_gs_ns_ks.GenerateTensorValue(GeneratorTensor_3<B0DataType>{0.0, 1.0});
            b1_gs_os_ns.GenerateTensorValue(GeneratorTensor_3<B1DataType>{-0.5, 0.5});
            d_gs_ms_ns.GenerateTensorValue(GeneratorTensor_3<Acc0BiasDataType>{-0.5, 0.5});
            break;
        case 3:
            a_gs_ms_ks.GenerateTensorValue(GeneratorTensor_2<ADataType>{-2, 2});
            b0_gs_ns_ks.GenerateTensorValue(GeneratorTensor_Diagonal<B0DataType>{});
            b1_gs_os_ns.GenerateTensorValue(GeneratorTensor_Diagonal<B1DataType>{});
            d_gs_ms_ns.GenerateTensorValue(GeneratorTensor_1<Acc0BiasDataType>{1});
            break;
        default:
            a_gs_ms_ks.GenerateTensorValue(GeneratorTensor_1<ADataType>{1});
            b0_gs_ns_ks.GenerateTensorValue(GeneratorTensor_Sequential<1>{});
            b1_gs_os_ns.GenerateTensorValue(GeneratorTensor_Diagonal<B1DataType>{});
            d_gs_ms_ns.GenerateTensorValue(GeneratorTensor_1<Acc0BiasDataType>{1});
        }

        a_tensors.push_back(a_gs_ms_ks);
        b0_tensors.push_back(b0_gs_ns_ks);
        b1_tensors.push_back(b1_gs_os_ns);
        c_tensors.push_back(c_gs_ms_os_device_result);
        d_tensors.push_back(d_gs_ms_ns);
        z_tensors.push_back(z_gs_ms_ns);
        lse_tensors.push_back(lse_gs_ms_device_result);

        a_tensors_device.emplace_back(std::make_unique<DeviceMem>(
            sizeof(ADataType) * a_gs_ms_ks.mDesc.GetElementSpaceSize()));
        b0_tensors_device.emplace_back(std::make_unique<DeviceMem>(
            sizeof(B0DataType) * b0_gs_ns_ks.mDesc.GetElementSpaceSize()));
        b1_tensors_device.emplace_back(std::make_unique<DeviceMem>(
            sizeof(B1DataType) * b1_gs_os_ns.mDesc.GetElementSpaceSize()));
        c_tensors_device.emplace_back(std::make_unique<DeviceMem>(
            sizeof(CDataType) * c_gs_ms_os_device_result.mDesc.GetElementSpaceSize()));
        d_tensors_device.emplace_back(std::make_unique<DeviceMem>(
            sizeof(Acc0BiasDataType) * d_gs_ms_ns.mDesc.GetElementSpaceSize()));
        z_tensors_device.emplace_back(std::make_unique<DeviceMem>(
            sizeof(ZDataType) * z_gs_ms_ns.mDesc.GetElementSpaceSize()));
        lse_tensors_device.emplace_back(std::make_unique<DeviceMem>(
            sizeof(LSEDataType) * lse_gs_ms_device_result.mDesc.GetElementSpaceSize()));

        a_tensors_device[i]->ToDevice(a_gs_ms_ks.mData.data());
        b0_tensors_device[i]->ToDevice(b0_gs_ns_ks.mData.data());
        b1_tensors_device[i]->ToDevice(b1_gs_os_ns.mData.data());
        d_tensors_device[i]->ToDevice(d_gs_ms_ns.mData.data());
        z_tensors_device[i]->ToDevice(z_gs_ms_ns.mData.data());

        p_a.push_back(a_tensors_device[i]->GetDeviceBuffer());
        p_b0.push_back(b0_tensors_device[i]->GetDeviceBuffer());
        p_b1.push_back(b1_tensors_device[i]->GetDeviceBuffer());
        p_c.push_back(c_tensors_device[i]->GetDeviceBuffer());
        p_d.push_back(d_tensors_device[i]->GetDeviceBuffer());
        p_z.push_back(z_tensors_device[i]->GetDeviceBuffer());
        p_z_nullptr.push_back(nullptr);
        p_lse.push_back(lse_tensors_device[i]->GetDeviceBuffer());
    }

    auto a_element_op    = AElementOp{};
    auto b0_element_op   = B0ElementOp{};
    auto acc0_element_op = Acc0ElementOp{alpha};
    auto b1_element_op   = B1ElementOp{};
    auto c_element_op    = CElementOp{};

    // do GEMM
    auto gemm    = DeviceGemmInstance{};
    auto invoker = gemm.MakeInvoker();

    auto argument =
        gemm.MakeArgument(p_a,
                          p_b0,
                          p_b1,
                          p_c,
                          p_z_nullptr,
                          p_lse,
                          p_d, // p_acc0_biases
                          {},  // p_acc1_biases
                          problem_descs,
                          a_element_op,
                          b0_element_op,
                          acc0_element_op,
                          b1_element_op,
                          c_element_op,
                          p_drop,          // dropout ratio
                          {seed, offset}); // dropout random seed and offset, offset should be
                                           // at least the number of elements on a thread

    // specify workspace for problem_desc
    DeviceMem problem_desc_workspace(gemm.GetWorkSpaceSize(&argument));

    gemm.SetWorkSpacePointer(&argument, problem_desc_workspace.GetDeviceBuffer());

    if(!gemm.IsSupportedArgument(argument))
    {
        std::cout << gemm.GetTypeString() << " does not support this problem" << std::endl;

        return 0;
    }

    float ave_time = invoker.Run(argument, StreamConfig{nullptr, time_kernel});

    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

    float gb_per_sec = num_byte / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s, "
              << gemm.GetTypeString() << std::endl;

    bool pass = true;
    if(do_verification)
    {
        // data objects for hipGraph verification
        hipGraph_t graph;
        hipGraphExec_t g_instance;
        hipStream_t stream;

        std::cout << "verification with hipGraph capturing and replaying ... " << std::endl;

        HIP_CHECK_ERROR(hipStreamCreate(&stream));
        HIP_CHECK_ERROR(hipGraphCreate(&graph, 0));

        HIP_CHECK_ERROR(hipStreamBeginCapture(stream, hipStreamCaptureModeRelaxed));

        argument =
            gemm.MakeArgument(p_a,
                              p_b0,
                              p_b1,
                              p_c,
                              p_z,
                              p_lse,
                              p_d, // p_acc0_biases
                              {},  // p_acc1_biases
                              problem_descs,
                              a_element_op,
                              b0_element_op,
                              acc0_element_op,
                              b1_element_op,
                              c_element_op,
                              p_drop,          // dropout ratio
                              {seed, offset}); // dropout random seed and offset, offset should be
                                               // at least the number of elements on a thread
        // specify workspace for problem_desc
        DeviceMem problem_desc_workspace_verify(gemm.GetWorkSpaceSize(&argument));

        gemm.SetWorkSpacePointer(&argument, problem_desc_workspace_verify.GetDeviceBuffer());

        invoker.Run(argument, StreamConfig{stream, false});

        HIP_CHECK_ERROR(hipStreamEndCapture(stream, &graph));
        HIP_CHECK_ERROR(hipGraphInstantiate(&g_instance, graph, nullptr, nullptr, 0));

        HIP_CHECK_ERROR(hipGraphDebugDotPrint(graph, "grouped_fwd_debug.dot", 0x007f));

        HIP_CHECK_ERROR(hipGraphLaunch(g_instance, stream));

        HIP_CHECK_ERROR(hipStreamSynchronize(stream));

        for(std::size_t i = 0; i < group_count; i++)
        {
            const int& G0  = g0_g1q_m_n_k_o[i][0];
            const int& G1Q = g0_g1q_m_n_k_o[i][1];
            const int& M   = g0_g1q_m_n_k_o[i][2];
            const int& N   = g0_g1q_m_n_k_o[i][3];
            const int& K   = g0_g1q_m_n_k_o[i][4];
            const int& O   = g0_g1q_m_n_k_o[i][5];

            const auto& c_gs_ms_os_lengths = problem_descs[i].c_gs_ms_os_lengths;
            const auto& c_gs_ms_os_strides = problem_descs[i].c_gs_ms_os_strides;
            const auto& lse_gs_ms_lengths  = problem_descs[i].lse_gs_ms_lengths;
            const auto& lse_gs_ms_strides  = problem_descs[i].lse_gs_ms_strides;

            const auto& a_gs_ms_ks         = a_tensors[i];
            const auto& b0_gs_ns_ks        = b0_tensors[i];
            const auto& b1_gs_os_ns        = b1_tensors[i];
            const auto& d_gs_ms_ns         = d_tensors[i];
            auto& c_gs_ms_os_device_result = c_tensors[i];
            auto& z_gs_ms_ns_device_result = z_tensors[i];
            auto& lse_gs_ms_device_result  = lse_tensors[i];
            auto& c_gs_ms_os_device_buf    = *c_tensors_device[i];
            auto& z_gs_ms_ns_device_buf    = *z_tensors_device[i];
            auto& lse_gs_ms_device_buf     = *lse_tensors_device[i];

            c_gs_ms_os_device_buf.FromDevice(c_gs_ms_os_device_result.mData.data());
            z_gs_ms_ns_device_buf.FromDevice(z_gs_ms_ns_device_result.mData.data());
            lse_gs_ms_device_buf.FromDevice(lse_gs_ms_device_result.mData.data());

            Tensor<ADataType> a_g_m_k({G0 * G1Q, M, K});
            Tensor<B0DataType> b0_g_k_n({G0 * G1Q, K, N});
            Tensor<B1DataType> b1_g_n_o({G0 * G1Q, N, O});
            Tensor<AccDataType> acc0_g_m_n({G0 * G1Q, M, N}); // scratch object after gemm0
            Tensor<Acc0BiasDataType> d_g_m_n({G0 * G1Q, M, N});
            Tensor<ADataType> a1_g_m_n({G0 * G1Q, M, N});            // scratch object after softmax
            Tensor<ADataType> a1_g_m_n_drop({G0 * G1Q, M, N});       // scratch object after softmax
            Tensor<CDataType> c_g_m_o_host_result({G0 * G1Q, M, O}); // scratch object after gemm1
            Tensor<CDataType> c_gs_ms_os_host_result(c_gs_ms_os_lengths, c_gs_ms_os_strides);
            Tensor<ZDataType> z_g_m_n({G0 * G1Q, M, N});
            Tensor<LSEDataType> lse_g_m_host_result({G0 * G1Q, M}); // scratch object after gemm1
            Tensor<LSEDataType> lse_gs_ms_host_result(lse_gs_ms_lengths, lse_gs_ms_strides);

            // permute
            a_gs_ms_ks.ForEach([&](auto& self, auto idx) {
                a_g_m_k(idx[0] * G1Q + idx[1], idx[2], idx[3]) = self(idx);
            });
            b0_g_k_n.ForEach([&](auto& self, auto idx) {
                const size_t& g0   = idx[0] / G1Q;
                const size_t& g1q  = idx[0] % G1Q;
                const size_t& g1kv = g1q / h_ratio;

                self(idx) = b0_gs_ns_ks(g0, g1kv, idx[2], idx[1]);
            });
            b1_g_n_o.ForEach([&](auto& self, auto idx) {
                const size_t& g0   = idx[0] / G1Q;
                const size_t& g1q  = idx[0] % G1Q;
                const size_t& g1kv = g1q / h_ratio;

                self(idx) = b1_gs_os_ns(g0, g1kv, idx[2], idx[1]);
            });
            d_gs_ms_ns.ForEach([&](auto& self, auto idx) {
                d_g_m_n(idx[0] * G1Q + idx[1], idx[2], idx[3]) = self(idx);
            });

            z_gs_ms_ns_device_result.ForEach([&](auto& self, auto idx) {
                z_g_m_n(idx[0] * G1Q + idx[1], idx[2], idx[3]) = self(idx);
            });

            // gemm 0
            auto ref_gemm0          = ReferenceGemm0Instance{};
            auto ref_gemm0_invoker  = ref_gemm0.MakeInvoker();
            auto ref_gemm0_argument = ref_gemm0.MakeArgument(
                a_g_m_k, b0_g_k_n, acc0_g_m_n, a_element_op, b0_element_op, acc0_element_op);

            ref_gemm0_invoker.Run(ref_gemm0_argument);

            // bias
            acc0_g_m_n.ForEach([&](auto& self, auto idx) {
                self(idx) += ck::type_convert<AccDataType>(d_g_m_n(idx));
            });
            // masking
            const auto mask = DeviceGemmInstance::C0MatrixMask(M, N);
            acc0_g_m_n.ForEach([&](auto& self, auto idx) {
                if(mask.IsMaskedElement(idx[1], idx[2]))
                    self(idx) = -ck::NumericLimits<AccDataType>::Infinity();
            });

            // softmax
            auto ref_softmax         = ReferenceSoftmaxInstance{};
            auto ref_softmax_invoker = ref_softmax.MakeInvoker();
            auto ref_softmax_argument =
                ref_softmax.MakeArgument(acc0_g_m_n, a1_g_m_n, 1, 0, {2}, &lse_g_m_host_result);

            ref_softmax_invoker.Run(ref_softmax_argument);

            // printf("print z_g_m_n \n");
            // z_g_m_n.ForEach([&](auto& self, auto idx) {printf("%u ", self(idx));});

            // dropout after softmax
            auto ref_dropout         = ReferenceDropoutInstance{};
            auto ref_dropout_invoker = ref_dropout.MakeInvoker();
            auto ref_dropout_argment = ref_dropout.MakeArgument(
                z_g_m_n, a1_g_m_n, a1_g_m_n_drop, p_dropout_in_uint8_t, rp_dropout);
            ref_dropout_invoker.Run(ref_dropout_argment);

            // gemm 1
            auto ref_gemm1          = ReferenceGemm1Instance{};
            auto ref_gemm1_invoker  = ref_gemm1.MakeInvoker();
            auto ref_gemm1_argument = ref_gemm1.MakeArgument(a1_g_m_n_drop,
                                                             b1_g_n_o,
                                                             c_g_m_o_host_result,
                                                             PassThrough{},
                                                             b1_element_op,
                                                             c_element_op);

            ref_gemm1_invoker.Run(ref_gemm1_argument);

            // permute
            c_gs_ms_os_host_result.ForEach([&](auto& self, auto idx) {
                const size_t& g0  = idx[0];
                const size_t& g1q = idx[1];

                const size_t g = g0 * G1Q + g1q;

                self(idx) = c_g_m_o_host_result(g, idx[2], idx[3]);
            });
            lse_gs_ms_host_result.ForEach([&](auto& self, auto idx) {
                const size_t& g0  = idx[0];
                const size_t& g1q = idx[1];

                const size_t g = g0 * G1Q + g1q;

                self(idx) = lse_g_m_host_result(g, idx[2]);
            });

            // default absolute error and relative error is 0.001
            double rtol = 1e-3;
            double atol = 1e-3;

            // when BF16 is taken, set absolute error and relative error to 0.01
            if(std::is_same_v<DataType, ck::bhalf_t> || std::is_same_v<GemmDataType, ck::bhalf_t>)
            {
                rtol = 1e-2;
                atol = 1e-2;
            }

            // bool pass_ =
            //    ck::utils::check_err(c_gs_ms_os_device_result.mData,
            //    c_gs_ms_os_host_result.mData);

            bool pass_ = ck::utils::check_err(c_gs_ms_os_device_result.mData,
                                              c_gs_ms_os_host_result.mData,
                                              "Error: Incorrect results c!",
                                              rtol,
                                              atol) &&
                         ck::utils::check_err(lse_gs_ms_device_result.mData,
                                              lse_gs_ms_host_result.mData,
                                              "Error: Incorrect results lse!",
                                              rtol,
                                              atol);
            if(!pass_)
            {
                std::cout << "from group: " << i << std::endl;
            }
            pass &= pass_;
        }
        if(pass)
        {
            std::cout << "Verification passed." << std::endl;
        }
    }

    return pass ? 0 : 1;
}
