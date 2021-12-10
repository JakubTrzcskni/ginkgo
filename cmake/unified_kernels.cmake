function(ginkgo_add_unified_library_internal target_name exec stub)
    add_library(${target_name}_${exec} STATIC ${ARGN})
    target_link_libraries(${target_name}_${exec} PUBLIC Ginkgo::ginkgo)
    if(NOT stub)
        target_compile_definitions(${target_name}_${exec} PRIVATE GKO_COMPILING_STUB GKO_DEVICE_NAMESPACE=${exec})
    else()
        string(TOUPPER ${exec} exec_upper)
        target_compile_definitions(${target_name}_${exec} PRIVATE GKO_COMPILING_${exec_upper})
    endif()
    target_link_libraries(${target_name} PRIVATE ${target_name}_${exec})
endfunction(ginkgo_add_unified_library_internal)

function(ginkgo_add_unified_kernels target_name)
    ginkgo_add_unified_library_internal(${target_name} reference ${GINKGO_BUILD_REFERENCE} ${ARGN})
    ginkgo_add_unified_library_internal(${target_name} omp ${GINKGO_BUILD_OMP} ${ARGN})
    ginkgo_add_unified_library_internal(${target_name} dpcpp ${GINKGO_BUILD_DPCPP} ${ARGN})
    if(GINKGO_BUILD_OMP)
        find_package(OpenMP 3.0 REQUIRED)
        target_link_libraries(${target_name}_omp PUBLIC OpenMP::OpenMP_CXX)
    endif()
    if(GINKGO_BUILD_DPCPP)
        target_compile_features(${target_name}_dpcpp PRIVATE cxx_std_17)
        target_link_options(${target_name}_dpcpp PRIVATE -fsycl-device-lib=all)
    endif()
    if(GINKGO_BUILD_CUDA)
        set(cuda_list)
        foreach(file IN LISTS ARGN)
            if ((IS_ABSOLUTE ${file}) OR (${file} MATCHES "\\.\\."))
                message(FATAL_ERROR "ginkgo_add_unified_kernels doesn't allow absolute paths or paths containing ..: ${file}")
            endif()
            configure_file(${file} ${file}.cu COPYONLY)
            list(APPEND cuda_list ${CMAKE_CURRENT_BINARY_DIR}/${file}.cu)
        endforeach()
        add_library(${target_name}_cuda STATIC ${cuda_list})
        target_link_libraries(${target_name}_cuda PUBLIC Ginkgo::ginkgo)
        if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
            # remove false positive CUDA warnings when calling one<T>() and zero<T>()
            # and allows the usage of std::array for nvidia GPUs
            target_compile_options(${target_name}_cuda
                PRIVATE
                    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
            if(MSVC)
                target_compile_options(${target_name}_cuda
                    PRIVATE
                        $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
            else()
                target_compile_options(${target_name}_cuda
                    PRIVATE
                        $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>)
            endif()
        endif()
        target_compile_definitions(${target_name}_cuda PRIVATE GKO_COMPILING_CUDA)
        target_link_libraries(${target_name} PRIVATE ${target_name}_cuda)
    else()
        ginkgo_add_unified_library_internal(${target_name} cuda OFF ${ARGN})
    endif()
    if(GINKGO_BUILD_HIP)
        set(hip_list)
        foreach(file IN LISTS ARGN)
            if ((IS_ABSOLUTE ${file}) OR (${file} MATCHES "\\.\\."))
                message(FATAL_ERROR "ginkgo_add_unified_kernels doesn't allow absolute paths or paths containing ..: ${file}")
            endif()
            configure_file(${file} ${file}.hip.cpp COPYONLY)
            list(APPEND hip_list ${CMAKE_CURRENT_BINARY_DIR}/${file}.hip.cpp)
        endforeach()
        set_source_files_properties(${hip_list} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT TRUE)
        hip_add_library(${target_name}_hip ${hip_list}
            HIPCC_OPTIONS ${GINKGO_HIPCC_OPTIONS} "-std=c++14 -DGKO_COMPILING_HIP"
            CLANG_OPTIONS ${GINKGO_HIP_CLANG_OPTIONS}
            NVCC_OPTIONS ${GINKGO_HIP_NVCC_OPTIONS}
            STATIC)
        target_link_libraries(${target_name}_hip PUBLIC Ginkgo::ginkgo)
        target_link_libraries(${target_name} PRIVATE ${target_name}_hip)
    else()
        ginkgo_add_unified_library_internal(${target_name} hip OFF ${ARGN})
    endif()
endfunction()
