cmake_minimum_required(VERSION 3.9)
project(gauss-seidel-visualize)

# We only need to find Ginkgo if we build this example stand-alone
if (NOT GINKGO_BUILD_EXAMPLES)
    find_package(Ginkgo 1.5.0 REQUIRED)
endif()

add_executable(gauss-seidel-visualize gauss-seidel-visualize.cpp)
target_link_libraries(gauss-seidel-visualize Ginkgo::ginkgo)

#configure_file(data/A.mtx data/A.mtx COPYONLY)
