#include <cstdlib>

#include <ginkgo/ginkgo.hpp>

#define block_size 512

#define INSTANTIATE_FOR_EACH_VALUE_TYPE(_macro) \
    template _macro(float);                     \
    template _macro(double);

#define PROLONGATION_KERNEL(_type)                                           \
    void prolongation_kernel(int nx, int ny, int nz, const _type* coeffs,    \
                             const _type* rhs, const int rhs_size, _type* x, \
                             const int x_size);

#define PROLONGATION_KERNEL_V2(_type)                                        \
    void prolongation_kernel_v2(int nx, int ny, int nz, const _type* coeffs, \
                                const _type* rhs, const int rhs_size,        \
                                _type* x, const int x_size);

namespace {

// todo correct for symmetric coefficients
// also a lot of branching
//  geo is coarse
template <typename ValueType>
__global__ void prolongation_kernel_impl(
    const int nx, const int ny, const int nz,
    const ValueType* __restrict__ coeffs,
    const ValueType* __restrict__ coarse_rhs, const int rhs_size,
    ValueType* __restrict__ fine_x, const int x_size)
{
    const auto nt_x = blockDim.x;
    const auto f_x = threadIdx.x + nt_x * blockIdx.x;
    const auto f_y = blockIdx.y;
    const auto f_z = blockIdx.z;

    const auto f_x_off_coarse_grid = f_x % 2;
    const auto f_y_off_coarse_grid = f_y % 2;
    const auto f_z_off_coarse_grid = f_z % 2;

    const auto f_id =
        f_z * (2 * nx + 1) * (2 * ny + 1) + f_y * (2 * nx + 1) + f_x;
    const auto c_id = ((f_z - f_z_off_coarse_grid) / 2) * (nx + 1) * (ny + 1) +
                      ((f_y - f_y_off_coarse_grid) / 2) * (nx + 1) +
                      ((f_x - f_x_off_coarse_grid) / 2);

    if (f_x <= 2 * nx) {
        if (!f_z_off_coarse_grid) {
            if (!f_y_off_coarse_grid) {
                if (!f_x_off_coarse_grid) {
                    fine_x[f_id] =
                        coeffs[1] * coeffs[1] * coeffs[1] * coarse_rhs[c_id];
                } else {
                    fine_x[f_id] = coeffs[1] * coeffs[1] * coeffs[0] *
                                   (coarse_rhs[c_id] + coarse_rhs[c_id + 1]);
                }
            } else {
                if (!f_x_off_coarse_grid) {
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
            if (!f_y_off_coarse_grid) {
                if (!f_x_off_coarse_grid) {
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
                if (!f_x_off_coarse_grid) {
                    fine_x[f_id] =
                        coeffs[1] * coeffs[0] * coeffs[0] *
                        (coarse_rhs[c_id] +
                         coarse_rhs[c_id + (nx + 1) * (ny + 1)] +
                         coarse_rhs[c_id + (nx + 1)] +
                         coarse_rhs[c_id + (nx + 1) * (ny + 1) + (nx + 1)]);
                } else {
                    fine_x[f_id] =
                        coeffs[0] * coeffs[0] * coeffs[0] *
                        (coarse_rhs[c_id] +
                         coarse_rhs[c_id + 1] +         //(0,0,0), (1,0,0)
                         coarse_rhs[c_id + (nx + 1)] +  //(0,1,0)
                         coarse_rhs[c_id + (nx + 1) * (ny + 1)] +      //(0,0,1)
                         coarse_rhs[c_id + (nx + 1) + 1] +             //(1,1,0)
                         coarse_rhs[c_id + (nx + 1) * (ny + 1) + 1] +  //(1,0,1)
                         coarse_rhs[c_id + (nx + 1) +
                                    (nx + 1) * (ny + 1)] +  //(0,1,1)
                         coarse_rhs[c_id + (nx + 1) + (nx + 1) * (ny + 1) +
                                    1]);  //(1,1,1)
                }
            }
        }
    }
}

// support only for symmetric stencil coefficients
template <typename ValueType>
__global__ void prolongation_kernel_impl_v2(
    const int nx, const int ny, const int nz, const ValueType* coeffs,
    const ValueType* __restrict__ coarse_rhs, const int rhs_size,
    ValueType* __restrict__ fine_x, const int x_size)
{
    const auto coeff_off_center = coeffs[0];
    const auto coeff_center = coeffs[1];

    const auto nt_x = blockDim.x;
    const auto f_x = threadIdx.x + nt_x * blockIdx.x;
    const auto f_y = blockIdx.y;
    const auto f_z = blockIdx.z;

    const auto f_x_off_coarse_grid = f_x % 2;
    const auto f_y_off_coarse_grid = f_y % 2;
    const auto f_z_off_coarse_grid = f_z % 2;

    const auto f_id =
        f_z * (2 * nx + 1) * (2 * ny + 1) + f_y * (2 * nx + 1) + f_x;
    const auto c_id = ((f_z - f_z_off_coarse_grid) / 2) * (nx + 1) * (ny + 1) +
                      ((f_y - f_y_off_coarse_grid) / 2) * (nx + 1) +
                      ((f_x - f_x_off_coarse_grid) / 2);
    const auto f_x_eq_nx = (f_x == 2 * nx);
    const auto f_y_eq_ny = (f_y == 2 * ny);
    const auto f_z_eq_nz = (f_z == 2 * nz);
    // gridpoint not on the far border
    if (!(f_x_eq_nx + f_y_eq_ny + f_z_eq_nz)) {
        const auto coarse000 = coarse_rhs[c_id];
        const auto coarse001 = coarse_rhs[c_id + 1];
        const auto coarse010 = coarse_rhs[c_id + nx + 1];
        const auto coarse011 = coarse_rhs[c_id + 2 + nx];
        const auto coarse100 = coarse_rhs[c_id + (nx + 1) * (ny + 1)];
        const auto coarse101 = coarse_rhs[c_id + (nx + 1) * (ny + 1) + 1];
        const auto coarse110 =
            coarse_rhs[c_id + (nx + 1) * (ny + 1) + (nx + 1)];
        const auto coarse111 =
            coarse_rhs[c_id + (nx + 1) + (nx + 1) * (ny + 1) + 1];
        fine_x[f_id] =
            (1 - f_z_off_coarse_grid) *
                ((1 - f_y_off_coarse_grid) *
                     ((1 - f_x_off_coarse_grid) * coeff_center * coeff_center *
                          coeff_center * coarse000 +
                      f_x_off_coarse_grid * coeff_center * coeff_center *
                          coeff_off_center * (coarse000 + coarse001)) +
                 f_y_off_coarse_grid *
                     ((1 - f_x_off_coarse_grid) * coeff_center * coeff_center *
                          coeff_off_center * (coarse000 + coarse010) +
                      f_x_off_coarse_grid * coeff_center * coeff_off_center *
                          coeff_off_center *
                          (coarse000 + coarse010 + coarse001 + coarse011))) +
            f_z_off_coarse_grid *
                ((1 - f_y_off_coarse_grid) *
                     ((1 - f_x_off_coarse_grid) * coeff_center * coeff_center *
                          coeff_off_center * (coarse000 + coarse100) +
                      f_x_off_coarse_grid * coeff_center * coeff_off_center *
                          coeff_off_center *
                          (coarse000 + coarse100 + coarse001 + coarse101)) +
                 f_y_off_coarse_grid *
                     ((1 - f_x_off_coarse_grid) * coeff_center *
                          coeff_off_center * coeff_off_center *
                          (coarse000 + coarse100 + coarse010 + coarse110) +
                      f_x_off_coarse_grid * coeff_off_center *
                          coeff_off_center * coeff_off_center *
                          (coarse000 + coarse001 + coarse010 + coarse100 +
                           coarse011 + coarse101 + coarse110 + coarse111)));
    } else {
        if (!f_z_eq_nz) {
            if (!f_y_eq_ny) {
                // x has to be on the border
                const auto coarse000 = coarse_rhs[c_id];
                const auto coarse010 = coarse_rhs[c_id + nx + 1];
                const auto coarse100 = coarse_rhs[c_id + (nx + 1) * (ny + 1)];
                const auto coarse110 =
                    coarse_rhs[c_id + (nx + 1) * (ny + 1) + (nx + 1)];
                fine_x[f_id] =
                    (1 - f_z_off_coarse_grid) *
                        ((1 - f_y_off_coarse_grid) * coeff_center *
                             coeff_center * coeff_center * coarse000 +
                         f_y_off_coarse_grid * coeff_center * coeff_center *
                             coeff_off_center * (coarse000 + coarse010)) +
                    f_z_off_coarse_grid *
                        ((1 - f_y_off_coarse_grid) * coeff_center *
                             coeff_center * coeff_off_center *
                             (coarse000 + coarse100) +
                         f_y_off_coarse_grid * coeff_center * coeff_off_center *
                             coeff_off_center *
                             (coarse000 + coarse010 + coarse100 + coarse110));
            } else {
                if (!f_x_eq_nx) {
                    const auto coarse000 = coarse_rhs[c_id];
                    const auto coarse001 = coarse_rhs[c_id + 1];
                    const auto coarse100 =
                        coarse_rhs[c_id + (nx + 1) * (ny + 1)];
                    const auto coarse101 =
                        coarse_rhs[c_id + (nx + 1) * (ny + 1) + 1];
                    fine_x[f_id] =
                        (1 - f_z_off_coarse_grid) *
                            ((1 - f_x_off_coarse_grid) * coeff_center *
                                 coeff_center * coeff_center * coarse000 +
                             f_x_off_coarse_grid * coeff_center * coeff_center *
                                 coeff_off_center * (coarse000 + coarse001)) +
                        f_z_off_coarse_grid *
                            ((1 - f_x_off_coarse_grid) * coeff_center *
                                 coeff_center * coeff_off_center *
                                 (coarse000 + coarse100) +
                             f_x_off_coarse_grid * coeff_center *
                                 coeff_off_center * coeff_off_center *
                                 (coarse000 + coarse001 + coarse100 +
                                  coarse101));
                } else {
                    const auto coarse000 = coarse_rhs[c_id];
                    const auto coarse100 =
                        coarse_rhs[c_id + (nx + 1) * (ny + 1)];
                    fine_x[f_id] = (1 - f_z_off_coarse_grid) * coeff_center *
                                       coeff_center * coeff_center * coarse000 +
                                   f_z_off_coarse_grid * coeff_center *
                                       coeff_center * coeff_off_center *
                                       (coarse000 + coarse100);
                }
            }
        } else {
            if (!f_y_eq_ny) {
                if (!f_x_eq_nx) {
                    const auto coarse000 = coarse_rhs[c_id];
                    const auto coarse001 = coarse_rhs[c_id + 1];
                    const auto coarse010 = coarse_rhs[c_id + nx + 1];
                    const auto coarse011 = coarse_rhs[c_id + 2 + nx];
                    fine_x[f_id] =
                        (1 - f_y_off_coarse_grid) *
                            ((1 - f_x_off_coarse_grid) * coeff_center *
                                 coeff_center * coeff_center * coarse000 +
                             f_x_off_coarse_grid * coeff_center * coeff_center *
                                 coeff_off_center * (coarse000 + coarse001)) +
                        f_y_off_coarse_grid *
                            ((1 - f_x_off_coarse_grid) * coeff_center *
                                 coeff_center * coeff_off_center *
                                 (coarse000 + coarse010) +
                             f_x_off_coarse_grid * coeff_center *
                                 coeff_off_center * coeff_off_center *
                                 (coarse000 + coarse010 + coarse001 +
                                  coarse011));
                } else {
                    const auto coarse000 = coarse_rhs[c_id];
                    const auto coarse010 = coarse_rhs[c_id + nx + 1];
                    fine_x[f_id] = (1 - f_y_off_coarse_grid) * coeff_center *
                                       coeff_center * coeff_center * coarse000 +
                                   f_y_off_coarse_grid * coeff_center *
                                       coeff_center * coeff_off_center *
                                       (coarse000 + coarse010);
                }
            } else {
                if (!f_x_eq_nx) {
                    const auto coarse000 = coarse_rhs[c_id];
                    const auto coarse001 = coarse_rhs[c_id + 1];
                    fine_x[f_id] = (1 - f_x_off_coarse_grid) * coeff_center *
                                       coeff_center * coeff_center * coarse000 +
                                   f_x_off_coarse_grid * coeff_center *
                                       coeff_center * coeff_off_center *
                                       (coarse000 + coarse001);

                } else {
                    const auto coarse000 = coarse_rhs[c_id];
                    fine_x[f_id] =
                        coeff_center * coeff_center * coeff_center * coarse000;
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
    const auto grid_size =
        dim3((2 * nx + 1 + block_size - 1) / block_size, 2 * ny + 1,
             2 * nz + 1);  // cover the whole fine grid?
    prolongation_kernel_impl<<<grid_size, block_size>>>(nx, ny, nz, coeffs, rhs,
                                                        rhs_size, x, x_size);
}

template <typename ValueType>
void prolongation_kernel_v2(int nx, int ny, int nz, const ValueType* coeffs,
                            const ValueType* rhs, const int rhs_size,
                            ValueType* x, const int x_size)
{
    const auto grid_size =
        dim3((2 * nx + 1 + block_size - 1) / block_size, 2 * ny + 1,
             2 * nz + 1);  // cover the whole fine grid?
    prolongation_kernel_impl_v2<<<grid_size, block_size>>>(
        nx, ny, nz, coeffs, rhs, rhs_size, x, x_size);
}

INSTANTIATE_FOR_EACH_VALUE_TYPE(PROLONGATION_KERNEL);
INSTANTIATE_FOR_EACH_VALUE_TYPE(PROLONGATION_KERNEL_V2);