add_custom_target(example_grouped_gemm_xdl)

add_example_executable(example_grouped_gemm_xdl_fp32 grouped_gemm_xdl_fp32.cpp)
add_example_executable(example_grouped_gemm_xdl_fp16 grouped_gemm_xdl_fp16.cpp)
add_example_executable(example_grouped_gemm_xdl_bfp16 grouped_gemm_xdl_bfp16.cpp)
add_example_executable(example_grouped_gemm_xdl_int8 grouped_gemm_xdl_int8.cpp)
add_example_executable(example_grouped_gemm_multiple_d_dl_fp16 grouped_gemm_multiple_d_dl_fp16.cpp)
add_example_executable(example_grouped_gemm_xdl_splitk_fp16 grouped_gemm_xdl_splitk_fp16.cpp)


add_dependencies(example_grouped_gemm_xdl
                 example_grouped_gemm_xdl_fp32
                 example_grouped_gemm_xdl_fp16
                 example_grouped_gemm_xdl_bfp16
                 example_grouped_gemm_xdl_int8
                 example_grouped_gemm_multiple_d_dl_fp16
                 example_grouped_gemm_xdl_splitk_fp16)

if(USE_BITINT_EXTENSION_INT4)
  add_example_executable(example_grouped_gemm_xdl_int4 grouped_gemm_xdl_int4.cpp)
  add_dependencies(example_grouped_gemm_xdl example_grouped_gemm_xdl_int4)
endif()
