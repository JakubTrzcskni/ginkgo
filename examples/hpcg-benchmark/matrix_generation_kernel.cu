#include <array>
#include <cstdlib>
#include <ginkgo/ginkgo.hpp>

#include "examples/hpcg-benchmark/include/geometric-multigrid.hpp"
#include "examples/hpcg-benchmark/include/utils.hpp"


#define block_size 256
#define warp_size 32

#define INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(_macro) \
    template _macro(float, int32_t);                      \
    template _macro(double, int32_t);                     \
    template _macro(float, int64_t);                      \
    template _macro(double, int64_t);

#define MATRIX_GEN_KERNEL(_v_type, _i_type)                                  \
    void matrix_generation_kernel(std::shared_ptr<const gko::Executor> exec, \
                                  const int nx, const int ny, const int nz,  \
                                  gko::matrix::Csr<_v_type, _i_type>* mat);

#define RHS_AND_X_GEN_KERNEL(_v_type, _i_type)             \
    void rhs_and_x_generation_kernel(                      \
        std::shared_ptr<const gko::Executor> exec,         \
        gko::matrix::Csr<_v_type, _i_type>* system_matrix, \
        gko::matrix::Dense<_v_type>* rhs,                  \
        gko::matrix::Dense<_v_type>* x_exact, gko::matrix::Dense<_v_type>* x);

namespace {

__device__ void get_ofs(const int id, int* ofs)
{
    ofs[0] = id % 3 - 1;
    int tmp = id / 3;
    ofs[1] = tmp % 3 - 1;
    tmp /= 3;
    ofs[2] = tmp % 3 - 1;
}

// todo: rename
// goal: translate ofs [0,26] to [0,7]/[0,11]/[0,17]/[0,26] depending on the
// grid position
__device__ int get_id_in_row(const int curr_id, const bool x_eq_zero,
                             const bool x_eq_nx, const bool y_eq_zero,
                             const bool y_eq_ny, const bool z_eq_zero, int* ofs)
{
    const auto x_border = x_eq_zero | x_eq_nx;
    const auto y_border = y_eq_zero | y_eq_ny;
    const auto tmp = (ofs[2] + 1) * ((x_border + y_border) * (-3) +
                                     (x_border & y_border))  // z part
                     + z_eq_zero * (-9 + 3 * (x_border + y_border) -
                                    (x_border & y_border))  // z part
                     + (ofs[1] + 1) * (x_border * (-1))     // y part
                     + y_eq_zero * (-3 + x_border)          // y part
                     + x_eq_zero * (-1);                    // x part


    return curr_id + tmp;
}

template <typename IndexType>
__device__ void init_row_ptrs(const int nx, const int ny, const int nz,
                              const int x, const int y, const int z,
                              const int row, IndexType* row_ptrs)
{
    const int corner_y[2] = {12, 8};
    const int corner_x[3] = {18, 12, 8};
    const int edge_y[2] = {18, 12};
    const int inside[3] = {27, 18, 12};
    const int general_case[4] = {27, 18, 12, 8};

    // it has the same value for all threads of all blocks with the same z
    // coordinate in the grid...
    auto if_z_zero = (z == 0);
    auto z_on_border = if_z_zero | (z == nz);

    auto z_sum_nnz =
        (1 - if_z_zero) *                     // if we're on the front face,
                                              // the whole sum should be 0
        ((z - 1) * ((nx - 1) * (ny - 1) * 27  // interior points
                    + 4 * 12                  // corners for interior faces
                    + 2 * (nx - 1) * 18 +     // edges for interior faces
                    2 * (ny - 1) * 18) +
         18 * (nx - 1) * (ny - 1)                   // inside of the front face
         + 4 * 8                                    // corners on the front face
         + 2 * (nx - 1) * 12 + 2 * (ny - 1) * 12);  // edges on the front face

    auto if_y_zero = (y == 0);
    auto y_on_border = if_y_zero | (y == ny);

    auto y_sum_nnz =
        ((1 - if_y_zero) *
         ((y - 1) * ((nx - 1) *
                         inside[z_on_border]  // interior points up to current y
                     + 2 * edge_y[z_on_border])  // edge points up to current y
          + 2 * corner_y[z_on_border]            // corners for current face
          + (nx - 1) * edge_y[z_on_border]));    // y==0 edge for current face

    auto if_x_zero = (x == 0);
    auto x_on_border = if_x_zero | (x == nx);

    auto x_sum_nnz =
        (1 - if_x_zero) *
        ((x - 1) * inside[z_on_border + y_on_border]  // interior points
         + corner_x[z_on_border + y_on_border]);  // first egde point in the row

    auto nnz_curr = z_sum_nnz + y_sum_nnz + x_sum_nnz;

    row_ptrs[row] = nnz_curr;

    // not necessary if handled by a separate kernel, todo
    if (z == nz && y == ny && x == nx) {
        row_ptrs[row + 1] =
            nnz_curr + general_case[x_on_border + y_on_border +
                                    z_on_border];  // its always the furthest
                                                   // corner -> nnz == 8
    }
}


template <typename ValueType, typename IndexType>
__global__ void matrix_generation_kernel_impl(
    const int active_threads_in_block, const int nx, const int ny, const int nz,
    const int size, ValueType* __restrict__ values,
    const IndexType* __restrict__ row_ptrs, IndexType* __restrict__ col_idxs)
{
    // cover only full rows -> no need for
    // synchronisation between threadblocks
    const int t_id = threadIdx.x;
    const int face_id = blockIdx.y * active_threads_in_block + t_id;
    const int z = blockIdx.z;

    const int row_base = z * (nx + 1) * (ny + 1);
    const int row_offset = (face_id / 27);

    if (t_id < active_threads_in_block && row_offset < ((nx + 1) * (ny + 1))) {
        // max 27 values per row, for big
        // matrices the ratio is favourable
        // block_size and grid size have to be adjusted for optimal
        // occupancy 256 -> 5% threads idle 1024 ->2,4% threads idle


        const int x = row_offset % (nx + 1);
        const int y = row_offset / (nx + 1);

        const int id_in_row = face_id % 27;
        int ofs[3];
        get_ofs(id_in_row, ofs);

        const int row = row_base + row_offset;
        const int row_ptr = row_ptrs[row];

        const int col =
            row + ofs[2] * (nx + 1) * (ny + 1) + ofs[1] * (nx + 1) + ofs[0];
        const int row_eq_col = (row == col);
        const int val =
            row_eq_col * ValueType{26} + (1 - row_eq_col) * ValueType{-1};

        const int x_eq_zero = (x == 0);
        const int y_eq_zero = (y == 0);
        const int z_eq_zero = (z == 0);
        const int x_eq_nx = (x == nx);
        const int y_eq_ny = (y == ny);
        const int curr_id =
            get_id_in_row(row_ptr + id_in_row, x_eq_zero, x_eq_nx, y_eq_zero,
                          y_eq_ny, z_eq_zero, ofs);
        const int valid = (z + ofs[2] >= 0 && z + ofs[2] <= nz) &&
                          (y + ofs[1] >= 0 && y + ofs[1] <= ny) &&
                          (x + ofs[0] >= 0 && x + ofs[0] <= nx);

        if (valid) {
            col_idxs[curr_id] = col;
            values[curr_id] = val;
        }
    }
}

template <typename IndexType>
__global__ void initialize_row_ptrs(const int nx, const int ny, const int nz,
                                    IndexType* __restrict__ row_ptrs,
                                    const int size)
{
    const auto id = blockIdx.x * blockDim.x + threadIdx.x;
    const auto z = id / ((nx + 1) * (ny + 1));
    const auto y = (id % ((nx + 1) * (ny + 1))) / (nx + 1);
    const auto x = id % (nx + 1);

    // todo proper(fast) handling of the row_ptrs[num_rows] value
    if (id < size - 1) {
        init_row_ptrs(nx, ny, nz, x, y, z, id, row_ptrs);
    }
}

template <typename ValueType, typename IndexType>
__global__ void rhs_and_x_gen_impl(const IndexType* __restrict__ row_ptrs,
                                   ValueType* __restrict__ rhs,
                                   ValueType* __restrict__ x_exact,
                                   ValueType* __restrict__ x, const int size)
{
    const auto id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size) {
        const auto nnz_in_row = row_ptrs[id + 1] - row_ptrs[id];
        rhs[id] = 27.0 - ValueType{nnz_in_row};
        x_exact[id] = ValueType{1};
        x[id] = ValueType{0};
    }
}

// todo
template <typename ValueType, typename IndexType>
__global__ void matrix_generation_kernel_impl_old(int nx, int ny, int nz,
                                                  int size, int nnz,
                                                  ValueType* values,
                                                  IndexType* row_ptrs,
                                                  IndexType* col_idxs)
{
    // one warp per row <-max 27 values per row
    // only max 27/32 Threads are active...
    // thread per row?
    const auto b_id_x = blockIdx.x;
    const int y = blockIdx.y;
    const int z = blockIdx.z;
    const auto warps_per_block = block_size / warp_size;
    const auto t_id = threadIdx.x;
    const auto warp_id = t_id / warp_size;
    const int x = b_id_x * warps_per_block + warp_id;

    const auto row = z * (nx + 1) * (ny + 1) + y * (nx + 1) + x;
    const auto id_per_row = t_id % warp_size;

    __shared__ int curr_val_ptr_offs[warps_per_block];
    if (id_per_row == 0) {
        curr_val_ptr_offs[warp_id] = 0;
    }

    if (id_per_row < 27 && row < size) {
        int ofs[3];
        get_ofs(id_per_row, ofs);
        if (z + ofs[2] >= 0 && z + ofs[2] <= nz) {
            if (y + ofs[1] >= 0 && y + ofs[1] <= ny) {
                if (x + ofs[0] >= 0 && x + ofs[0] <= nx) {
                    const auto col = row + ofs[2] * (nx + 1) * (ny + 1) +
                                     ofs[1] * (nx + 1) + ofs[0];

                    // todo
                    // number of nnz in the current row
                    // change to current_value_ptr with atomicAdd update
                    // ? firstly initialize row_ptrs -> separate kernel
                    // / before this loop
                    auto curr_id =
                        row_ptrs[row] +
                        atomicAdd(&(curr_val_ptr_offs[id_per_row]), 1);

                    if (row == col) {
                        values[curr_id] = ValueType{26};
                    } else {
                        values[curr_id] = ValueType{-1};
                    }
                    col_idxs[curr_id] = col;
                }
            }
        }
    }
}
}  // namespace

template <typename ValueType, typename IndexType>
void rhs_and_x_generation_kernel(
    std::shared_ptr<const gko::Executor> exec,
    gko::matrix::Csr<ValueType, IndexType>* system_matrix,
    gko::matrix::Dense<ValueType>* rhs, gko::matrix::Dense<ValueType>* x_exact,
    gko::matrix::Dense<ValueType>* x)
{
    using vec = gko::matrix::Dense<ValueType>;
    // GKO_ASSERT_EQ(exec, system_matrix->get_executor());  // not sure if
    // correct
    const auto mat_size = system_matrix->get_size()[0];
    const auto row_ptrs = system_matrix->get_const_row_ptrs();

    auto generated_rhs = vec::create(exec, gko::dim<2>(mat_size, 1));
    auto generated_x_exact = vec::create(exec, gko::dim<2>(mat_size, 1));
    auto generated_x = vec::create(exec, gko::dim<2>(mat_size, 1));

    const auto grid = (mat_size + block_size - 1) / block_size;
    rhs_and_x_gen_impl<<<grid, block_size>>>(
        row_ptrs, generated_rhs->get_values(), generated_x_exact->get_values(),
        generated_x->get_values(), mat_size);

    generated_rhs->move_to(rhs);
    generated_x_exact->move_to(x_exact);
    generated_x->move_to(x);
}

template <typename ValueType, typename IndexType>
void matrix_generation_kernel(std::shared_ptr<const gko::Executor> exec,
                              const int nx, const int ny, const int nz,
                              gko::matrix::Csr<ValueType, IndexType>* mat)
{
    using val_array = gko::Array<ValueType>;
    using id_array = gko::Array<IndexType>;
    using csr = gko::matrix::Csr<ValueType, IndexType>;

    const auto mat_nnz = calc_nnz(nx, ny, nz);
    const auto mat_size = gko::multigrid::get_dp_3D(nx, ny, nz);

    auto values = val_array{exec, mat_nnz};
    auto row_ptrs = id_array{exec, mat_size + 1};
    auto col_idxs = id_array{exec, mat_nnz};

    const auto init_grid = (mat_size + 1 + block_size - 1) / block_size;

    initialize_row_ptrs<<<init_grid, block_size>>>(
        nx, ny, nz, row_ptrs.get_data(), mat_size + 1);

    const auto threshold =
        (block_size / 27);  // cover only full rows -> no need for
                            // synchronisation between threadblocks
    const auto grid_y = ((nx + 1) * (ny + 1) + threshold - 1) / threshold;

    const auto grid_size = dim3(1, grid_y, nz + 1);
    matrix_generation_kernel_impl<<<grid_size, block_size>>>(
        threshold * 27, nx, ny, nz, mat_size, values.get_data(),
        row_ptrs.get_const_data(), col_idxs.get_data());

    auto generated_mat =
        csr::create(exec, gko::dim<2>(mat_size), std::move(values),
                    std::move(col_idxs), std::move(row_ptrs));
    generated_mat->move_to(mat);
}

INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(MATRIX_GEN_KERNEL);
INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(RHS_AND_X_GEN_KERNEL);
