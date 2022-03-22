#include <cstdlib>

#include <ginkgo/ginkgo.hpp>

#define block_size 512
#define warp_size 32

#define INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(_macro) \
    template _macro(float, int32_t);                      \
    template _macro(double, int32_t);                     \
    template _macro(float, int64_t);                      \
    template _macro(double, int64_t);

#define MATRIX_GEN_KERNEL(_v_type, _i_type)                                  \
    void matrix_generation_kernel(std::shared_ptr<const gko::Executor> exec, \
                                  int nx, int ny, int nz,                    \
                                  gko::matrix::Csr<_v_type, _i_type>* mat);

namespace {

// probably should be int** ofs ?
__device__ void get_ofs(const int warp_id, int* ofs)
{
    ofs[0] = warp_id % 3 - 1;
    int tmp = warp_id / 3;
    ofs[1] = tmp % 3 - 1;
    tmp /= 3;
    ofs[2] = tmp % 3 - 1;
}
template <typename IndexType>
__device__ void init_row_ptrs(int nx, int ny, int nz, int size,
                              IndexType* row_ptrs)
{}
// todo
template <typename ValueType, typename IndexType>
__global__ void matrix_generation_kernel_impl(int nx, int ny, int nz, int size,
                                              int nnz, ValueType* values,
                                              IndexType* row_ptrs,
                                              IndexType* col_idxs)
{
    // one row per warp <-max 27 values per row
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
                    // change to current_value_ptr with atomicAdd update ?
                    // firstly initialize row_ptrs -> separate kernel / before
                    // this loop
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

int calc_nnz(int nx, int ny, int nz)
{
    return (nx + 1) * (ny + 1) * (nz + 1) * 27 + 64 + ((nx + 1) - 2) * 4 * 3 +
           ((ny + 1) - 2) * 4 * 3 + ((nz + 1) - 2) * 4 * 3 -
           2 * (nx + 1) * (ny + 1) * 9 - 2 * (nx + 1) * (nz + 1) * 9 -
           2 * (ny + 1) * (nz + 1) * 9;
}

template <typename ValueType, typename IndexType>
void matrix_generation_kernel(std::shared_ptr<const gko::Executor> exec, int nx,
                              int ny, int nz,
                              gko::matrix::Csr<ValueType, IndexType>* mat)
{
    using val_array = gko::Array<ValueType>;
    using id_array = gko::Array<IndexType>;

    const auto mat_size = mat->get_size()[0];
    const auto mat_nnz = calc_nnz(nx, ny, nz);

    // how to initialize with given Arrays?
    // auto values = val_array{exec, mat_size};
    // auto row_ptrs = id_array{exec, mat_size + 1};
    // auto col_idxs = id_array{exec, mat_size};
    //  gko::device_matrix_data<ValueType, IndexType> data{
    //      exec, gko::dim<2>(mat->get_size()[0]),
    //      gko::lend(row_ptrs->get_data()), gko::lend(col_idxs->get_data()),
    //      gko::lend(values->get_data())};

    gko::device_matrix_data<ValueType, IndexType> data{
        exec, gko::dim<2>(mat->get_size()[0]), mat_nnz};

    // todo
    // adjust
    // grid ->2D, block size
    const auto row_per_block = block_size / warp_size;
    const auto grid_size = (mat_size + row_per_block - 1) / row_per_block;
    matrix_generation_kernel_impl<<<grid_size, block_size>>>(
        nx, ny, nz, mat_size, mat_nnz, data.get_values(), data.get_row_idxs(),
        data.get_col_idxs());
}

INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(MATRIX_GEN_KERNEL);
