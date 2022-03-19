#include <cstdlib>

#include <ginkgo/ginkgo.hpp>


#define INSTANTIATE_FOR_EACH_VALUE_TYPE(_macro) \
    template _macro(float);                     \
    template _macro(double);

#define MATRIX_GENERATION_KERNEL(_type) void matrix_generation_kernel(_type* x);

namespace {
template <typename ValueType>
__global__ void matrix_generation_kernel_impl()
{}
}  // namespace

template <typename ValueType>
void matrix_generation_kernel(ValueType* x)
{
    // constexpr int block_size = 512;
    // const auto grid_size = (size + block_size - 1) / block_size;
}

INSTANTIATE_FOR_EACH_VALUE_TYPE(MATRIX_GENERATION_KERNEL);