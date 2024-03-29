cmake_minimum_required(VERSION 3.15)
project(ck_app)
add_compile_options(-std=c++17)

if (DTYPES)
        add_definitions(-DDTYPES)
        if (DTYPES MATCHES "int8")
                add_definitions(-D__int8__)
        endif()
        if (DTYPES MATCHES "fp8")
                add_definitions(-D__fp8__)
        endif()
        if (DTYPES MATCHES "fp16")
                add_definitions(-D__fp16__)
        endif()
        if (DTYPES MATCHES "fp32")
                add_definitions(-D__fp32__)
        endif()
        if (DTYPES MATCHES "fp64")
                add_definitions(-D__fp64__)
        endif()
        if (DTYPES MATCHES "bf16")
                add_definitions(-D__bf16__)
        endif()
        message("DTYPES macro set to ${DTYPES}")
else()
        add_definitions(-D__int8__ -D__fp8__ -D__fp16__ -D__fp32__ -D__fp64__ -D__bf16__)
endif()

find_package(composable_kernel 1.0.0 COMPONENTS device_operations)
find_package(hip REQUIRED PATHS /opt/rocm)
message(STATUS "Build with HIP ${hip_VERSION}")

# add all example subdir
file(GLOB dir_list LIST_DIRECTORIES true *)
FOREACH(subdir ${dir_list})
    IF(IS_DIRECTORY "${subdir}" AND (NOT "${subdir}" MATCHES "build"))
        add_subdirectory(${subdir})
    ENDIF()
ENDFOREACH()
