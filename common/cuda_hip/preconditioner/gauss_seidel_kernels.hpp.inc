/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/
namespace {
template <typename IndexType>
__global__ void update_perm_idxs(IndexType* __restrict__ perm_idxs,
                                 const IndexType* __restrict__ block_ordering,
                                 const IndexType num_nodes)
{
    const auto thread_id = thread::get_thread_id_flat<IndexType>();

    if (thread_id >= num_nodes) {
        return;
    }
    perm_idxs[thread_id] = block_ordering[perm_idxs[thread_id]];
}
template <typename IndexType>
__global__ void get_color_ptrs(const IndexType* coloring,
                               const IndexType max_color,
                               const IndexType num_nodes, IndexType* color_ptrs)
{}
}  // namespace

template <typename IndexType>
void get_permutation_from_coloring(
    std::shared_ptr<const DefaultExecutor> exec, const IndexType num_nodes,
    IndexType* coloring, const IndexType max_color, IndexType* color_ptrs,
    IndexType* permutation_idxs, const IndexType* block_ordering)
{
    auto p_idxs = thrust::device_pointer_cast(permutation_idxs);
    thrust::sequence(thrust::device, p_idxs, p_idxs + num_nodes, 0);

    auto it_color = thrust::make_permutation_iterator(
        thrust::device_pointer_cast(coloring),
        thrust::device_pointer_cast(block_ordering));

    // coloring is permuted here too
    thrust::sort_by_key(thrust::device, it_color, it_color + num_nodes,
                        p_idxs);  // alternatively stable_sort_by_key

    const auto block_size = config::max_block_size;
    const auto grid_size = ceildiv(num_nodes, block_size);
    update_perm_idxs<<<grid_size, block_size>>>(permutation_idxs,
                                                block_ordering, num_nodes);
    // TODO
    // calculate color_ptrs
}
GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_GAUSS_SEIDEL_GET_PERMUTATION_FROM_COLORING_KERNEL);

namespace kernel {

template <typename IndexType>
__global__ void get_degree_of_nodes_kernel(const IndexType num_nodes,
                                           const IndexType* const row_ptrs,
                                           IndexType* const degrees)
{
    const auto thread_id = thread::get_thread_id_flat<IndexType>();

    if (thread_id >= num_nodes) {
        return;
    }
    const auto curr_row_start = row_ptrs[thread_id];
    const auto next_row_start = row_ptrs[thread_id + 1];

    const auto node_degree = next_row_start - curr_row_start;

    degrees[thread_id] = node_degree;
}

template <typename ValueType, typename IndexType>
__global__ void prepare_x_kernel(const ValueType* __restrict__ orig_x,
                                 ValueType* __restrict__ x_for_spmv,
                                 const IndexType* __restrict__ perm_idxs,
                                 const size_type stride,
                                 const size_type id_offs,
                                 const size_type num_rows,
                                 const size_type spmv_block_size)
{
    const auto thread_id = thread::get_thread_id_flat<IndexType>();

    if (thread_id >= spmv_block_size) {
        return;
    }
    for (auto i = 0; i < stride; ++i) {
        x_for_spmv[thread_id * stride + i] =
            orig_x[perm_idxs[id_offs + thread_id] * stride + i];
    }
}

// template <typename ValueType, typename IndexType>
// __global__ apply_l_spmv_block_kernel(){}

// template <typename ValueType, typename IndexType>
// __global__ apply_l_parallel_block_kernel(){}


template <typename ValueType, typename IndexType>
__device__ void apply_ltr(const IndexType* __restrict__ rows,
                          const ValueType* __restrict__ vals,
                          const size_type end_row, const size_type b_s,
                          const size_type w, const size_type num_rhs,
                          ValueType* __restrict__ b_perm,
                          ValueType* __restrict__ x,
                          const IndexType* __restrict__ perm_idxs,
                          const int* diag_LUT, const int* subblock_LUT)
{
    auto tmp = zero<ValueType>();
    for (auto i = 0; i < (diag_LUT[b_s] + 1); ++i) {  // i < nz_p_b(b_s)
        const auto curr_offs = i * w;
        const auto row = rows[curr_offs];
        const auto x_row = perm_idxs[row];

        if (row >= 0) {
            if (row >= end_row) return;
            if (i == diag_LUT[subblock_LUT[i] + 1]) {
                // on the diagonal
                const auto val = vals[curr_offs];
                GKO_ASSERT(val != zero<ValueType>());
                const auto inv_diag_val = one<ValueType>() / val;
                for (size_type k = 0; k < num_rhs; ++k) {
                    x[x_row * num_rhs + k] =
                        inv_diag_val * (b_perm[row * num_rhs + k] - tmp);
                }
                tmp = zero<ValueType>();

            } else {
                // off the diagonal
                const auto read_offs = diag_LUT[subblock_LUT[i] + 1] * w;
                const auto row_read = rows[read_offs];
                if (row_read >= 0) {
                    const auto x_row_read = perm_idxs[row_read];
                    for (size_type k = 0; k < num_rhs; ++k) {
                        tmp += vals[curr_offs] * x[x_row_read * num_rhs + k];
                    }
                }
            }
        }
    }
}

template <typename ValueType,
          typename IndexType>  // rows&vals passed already with offset
__global__ void apply_l_p_block_kernel(
    const IndexType* __restrict__ rows, const ValueType* __restrict__ vals,
    const size_type end_row, const size_type b_s, const size_type w,
    const size_type num_p_blocks, const bool res_blocks,
    const size_type num_rhs, ValueType* __restrict__ b_perm,
    ValueType* __restrict__ x, const IndexType* __restrict__ perm_idxs,
    const int* diag_LUT, const int* subblock_LUT)
{
    constexpr int warp_size = config::warp_size;
    // const auto warp =
    //     group::tiled_partition<config::warp_size>(group::this_thread_block());
    // const auto local_tid = warp.thread_rank();
    // const auto warp_id = thread::get_subwarp_id_flat<warp_size>();
    const auto tid = thread::get_thread_id_flat<int>();
    const auto warp_id = tid / warp_size;
    const auto local_tid = tid % warp_size;

    if (warp_id >= num_p_blocks) return;
    // printf(
    //     "warp_size = %d, warp_id = %d, local_tid= %u, tid_flat = %d, end_row
    //     = "
    //     "%d\n",
    //     warp_size, static_cast<int>(warp_id), local_tid,
    //     static_cast<int>(thread::get_thread_id_flat<int>()),
    //     static_cast<int>(end_row));

    const auto nz_p_b = diag_LUT[b_s] + 1;
    const auto stride = nz_p_b * w;
    const auto base_offs = warp_id * stride;  // TODO what when w !=
    // warp_size?

    if (warp_id == (num_p_blocks - 1) && res_blocks) {
        // apply base_block_agg
        // printf("nono\n");
        const auto stride_base_block = nz_p_b;
        const auto base_block_offs = base_offs + local_tid * stride_base_block;
        apply_ltr(&(rows[base_block_offs]), &(vals[base_block_offs]), end_row,
                  b_s, one<size_type>(), num_rhs, b_perm, x, perm_idxs,
                  diag_LUT, subblock_LUT);
    } else {
        // apply lvl 1
        apply_ltr(&(rows[base_offs + local_tid]),
                  &(vals[base_offs + local_tid]), end_row, b_s, w, num_rhs,
                  b_perm, x, perm_idxs, diag_LUT, subblock_LUT);
    }
}

template <typename ValueType,
          typename IndexType>  // rows&vals passed already with offset
__global__ void apply_l_lvl_1_kernel(
    const IndexType* __restrict__ rows, const ValueType* __restrict__ vals,
    const size_type b_s, const size_type w, const size_type num_rows,
    const size_type num_p_blocks, const size_type num_rhs,
    ValueType* __restrict__ b_perm, ValueType* __restrict__ x,
    const IndexType* __restrict__ perm_idxs)
{
    constexpr int warp_size = config::warp_size;
    const auto warp =
        group::tiled_partition<warp_size>(group::this_thread_block());
    const auto local_tid = warp.thread_rank();
    const auto warp_id = thread::get_subwarp_id_flat<warp_size>();

    if (warp_id >= num_p_blocks) const auto lvl_1_size = b_s * w;
    const auto stride = precomputed_nz_p_b(b_s) * w;
    const auto base_offs =
        warp_id * stride + local_tid;  // TODO what when w != warp_size?

    ValueType tmp{0};
    for (auto i = 0; i < b_s; ++i) {
        const auto curr_offs = base_offs + i * w;
        const auto row = rows[curr_offs];
        const auto x_row = perm_idxs[row];

        if (row >= 0) {
            if (i == precomputed_diag(precomputed_block(i))) {
                // on the diagonal
                const auto inv_diag_val = ValueType{1} / vals[curr_offs];
                for (size_type k = 0; k < num_rhs; ++k) {
                    x[x_row * num_rhs + k] =
                        inv_diag_val * (b_perm[row * num_rhs + k] - tmp);
                }
                tmp = ValueType{0};

            } else {
                // off the diagonal
                const auto read_offs =
                    base_offs + precomputed_diag(precomputed_block(i)) * w;
                const auto row_read = rows[read_offs];
                if (row_read >= 0) {
                    const auto x_row_read = perm_idxs[row_read];
                    for (size_type k = 0; k < num_rhs; ++k) {
                        tmp += vals[curr_offs] * x[x_row_read * num_rhs + k];
                    }
                }
            }
        }
    }
}
}  // namespace kernel
