#include <cstdlib>

#include <ginkgo/ginkgo.hpp>


#define INSTANTIATE_FOR_EACH_VALUE_TYPE(_macro) \
    template _macro(float);                     \
    template _macro(double);

#define RESTRICTION_KERNEL(_type)                                           \
    void restriction_kernel(int nx, int ny, int nz, const _type* coeffs,    \
                            const _type* rhs, const int rhs_size, _type* x, \
                            const int x_size);

namespace {
// todo
template <typename ValueType>
__global__ void restriction_kernel_impl(int nx, int ny, int nz,
                                        const ValueType* coeffs,
                                        const ValueType* fine_rhs,
                                        const int rhs_size, ValueType* coarse_x,
                                        const int x_size)
{
    const auto nt_x = blockDim.x;
    const auto c_x = threadIdx.x + nt_x * blockIdx.x;
    const auto c_y = blockIdx.y;
    const auto c_z = blockIdx.z;
    const auto f_x = 2 * c_x;
    const auto f_y = 2 * c_y;
    const auto f_z = 2 * c_z;

    const auto c_id = c_z * (nx + 1) * (ny + 1) + c_y * (nx + 1) + c_x;
    const auto f_id = 2 * c_z * (2 * nx + 1) * (2 * ny + 1) +
                      2 * c_y * (2 * nx + 1) + 2 * c_x;

    if (c_x <= nx) {
        // coarse_x[c_id] =fine_rhs[f_id];//canonical injection
        // if (c_z > 0 && c_z < nz) {
        //     if (c_y > 0 && c_y < ny) {
        //         if (c_x > 0 && c_x < nx) {
        //             coarse_x[c_id] =
        //                 coeffs[1] * coeffs[1] * coeffs[1] * fine_rhs[f_id] +
        //                 coeffs[1] * coeffs[1] * coeffs[0] *
        //                     (fine_rhs[f_id + 1] + fine_rhs[f_id - 1] +
        //                      fine_rhs[f_id + (2 * nx + 1)] +
        //                      fine_rhs[f_id - (2 * nx + 1)] +
        //                      fine_rhs[f_id + (2 * nx + 1) * (2 * ny + 1)] +
        //                      fine_rhs[f_id - (2 * nx + 1) * (2 * ny + 1)]);
        //         }
        //     }
        // }
        for (auto ofs_z = 0; ofs_z < 3; ofs_z++) {
            if (f_z + ofs_z >= 1 && f_z + ofs_z <= 2 * nz + 1) {
                for (auto ofs_y = 0; ofs_y < 3; ofs_y++) {
                    if (f_y + ofs_y >= 1 && f_y + ofs_y <= 2 * ny + 1) {
                        for (auto ofs_x = 0; ofs_x < 3; ofs_x++) {
                            if (f_x + ofs_x >= 1 && f_x + ofs_x <= 2 * nx + 1) {
                                auto f_offset =
                                    (ofs_z - 1) * (2 * nx + 1) * (2 * ny + 1) +
                                    (ofs_y - 1) * (2 * nx + 1) + (ofs_x - 1);

                                coarse_x[c_id] +=
                                    coeffs[ofs_z] * coeffs[ofs_y] *
                                    coeffs[ofs_x] * fine_rhs[f_id + f_offset];
                            }
                        }
                    }
                }
            }
        }
    }
}
}  // namespace

template <typename ValueType>
void restriction_kernel(int nx, int ny, int nz, const ValueType* coeffs,
                        const ValueType* rhs, const int rhs_size, ValueType* x,
                        const int x_size)
{
    constexpr int block_size = 32;
    const auto grid_size =
        dim3((nx + 1 + block_size - 1) / block_size, ny + 1, nz + 1);
    restriction_kernel_impl<<<grid_size, block_size>>>(nx, ny, nz, coeffs, rhs,
                                                       rhs_size, x, x_size);
}

INSTANTIATE_FOR_EACH_VALUE_TYPE(RESTRICTION_KERNEL);