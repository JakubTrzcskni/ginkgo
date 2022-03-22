#include <cstdlib>

#include <ginkgo/ginkgo.hpp>


#define INSTANTIATE_FOR_EACH_VALUE_TYPE(_macro) \
    template _macro(float);                     \
    template _macro(double);

#define PROLONGATION_KERNEL(_type)                                           \
    void prolongation_kernel(int nx, int ny, int nz, const _type* coeffs,    \
                             const _type* rhs, const int rhs_size, _type* x, \
                             const int x_size);

namespace {

// geo is coarse
template <typename ValueType>
__global__ void prolongation_kernel_impl(int nx, int ny, int nz,
                                         const ValueType* coeffs,
                                         const ValueType* coarse_rhs,
                                         const int rhs_size, ValueType* fine_x,
                                         const int x_size)
{
    const auto nt_x = blockDim.x;
    const auto f_x = threadIdx.x + nt_x * blockIdx.x;
    const auto f_y = blockIdx.y;
    const auto f_z = blockIdx.z;

    const auto f_x_on_coarse = f_x % 2;
    const auto f_y_on_coarse = f_y % 2;
    const auto f_z_on_coarse = f_z % 2;

    const auto f_id =
        f_z * (2 * nx + 1) * (2 * ny + 1) + f_y * (2 * nx + 1) + f_x;
    const auto c_id = ((f_z - f_z_on_coarse) / 2) * (nx + 1) * (ny + 1) +
                      ((f_y - f_y_on_coarse) / 2) * (nx + 1) +
                      ((f_x - f_x_on_coarse) / 2);

    if (f_x <= 2 * nx) {
        if (!f_z_on_coarse) {
            if (!f_y_on_coarse) {
                if (!f_x_on_coarse) {
                    fine_x[f_id] =
                        coeffs[1] * coeffs[1] * coeffs[1] * coarse_rhs[c_id];
                } else {
                    fine_x[f_id] = coeffs[1] * coeffs[1] * coeffs[0] *
                                   (coarse_rhs[c_id] + coarse_rhs[c_id + 1]);
                }

                // alternative:
                // fine_x[f_id] = (1 - f_x_on_coarse) * coeffs[1] * coeffs[1] *
                //                    coeffs[1] * coarse_rhs[c_id] +
                //                f_x_on_coarse * coeffs[1] * coeffs[1] *
                //                    coeffs[0] *
                //                    (coarse_rhs[c_id] + coarse_rhs[c_id + 1]);
            } else {
                if (!f_x_on_coarse) {
                    fine_x[f_id] =
                        coeffs[1] * coeffs[1] * coeffs[0] *
                        (coarse_rhs[c_id] + coarse_rhs[c_id + nx + 1]);
                } else {
                    fine_x[f_id] =
                        coeffs[1] * coeffs[0] * coeffs[0] *
                        (coarse_rhs[c_id] + coarse_rhs[c_id + nx + 1] +
                         coarse_rhs[c_id + 1] + coarse_rhs[c_id + 2 + nx]);
                }
            }
        } else {
            if (!f_y_on_coarse) {
                if (!f_x_on_coarse) {
                    fine_x[f_id] = coeffs[1] * coeffs[1] * coeffs[0] *
                                   (coarse_rhs[c_id] +
                                    coarse_rhs[c_id + (nx + 1) * (ny + 1)]);
                } else {
                    fine_x[f_id] = coeffs[1] * coeffs[0] * coeffs[0] *
                                   (coarse_rhs[c_id] +
                                    coarse_rhs[c_id + (nx + 1) * (ny + 1)] +
                                    coarse_rhs[c_id + 1] +
                                    coarse_rhs[c_id + (nx + 1) * (ny + 1) + 1]);
                }
            } else {
                if (!f_x_on_coarse) {
                    fine_x[f_id] =
                        coeffs[1] * coeffs[0] * coeffs[0] *
                        (coarse_rhs[c_id] +
                         coarse_rhs[c_id + (nx + 1) * (ny + 1)] +
                         coarse_rhs[c_id + (nx + 1)] +
                         coarse_rhs[c_id + (nx + 1) * (ny + 1) + (nx + 1)]);
                } else {
                    fine_x[f_id] =
                        coeffs[0] * coeffs[0] * coeffs[0] *
                        (coarse_rhs[c_id] + coarse_rhs[c_id + 1] +
                         coarse_rhs[c_id + (nx + 1)] +
                         coarse_rhs[c_id + (nx + 1) * (ny + 1)] +
                         coarse_rhs[c_id + (nx + 1) + 1] +
                         coarse_rhs[c_id + (nx + 1) * (ny + 1) + 1] +
                         coarse_rhs[c_id + (nx + 1) + (nx + 1) * (ny + 1)] +
                         coarse_rhs[c_id + (nx + 1) + (nx + 1) * (ny + 1) + 1]);
                }
            }
        }
    }
}
}  // namespace

template <typename ValueType>
void prolongation_kernel(int nx, int ny, int nz, const ValueType* coeffs,
                         const ValueType* rhs, const int rhs_size, ValueType* x,
                         const int x_size)
{
    constexpr int block_size = 32;
    const auto grid_size =
        dim3((2 * nx + 1 + block_size - 1) / block_size, 2 * ny + 1,
             2 * nz + 1);  // cover the whole fine grid?
    prolongation_kernel_impl<<<grid_size, block_size>>>(nx, ny, nz, coeffs, rhs,
                                                        rhs_size, x, x_size);
}

INSTANTIATE_FOR_EACH_VALUE_TYPE(PROLONGATION_KERNEL);