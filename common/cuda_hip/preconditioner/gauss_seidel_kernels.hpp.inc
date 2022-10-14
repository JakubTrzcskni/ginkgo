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

template <typename ValueType, typename IndexType>
__device__ void apply_ltr(
    const IndexType* __restrict__ rows, const ValueType* __restrict__ vals,
    const size_type end_row, const size_type num_rows_in_block,
    const size_type b_s, const size_type w, const size_type num_rhs,
    ValueType* __restrict__ b_perm, ValueType* __restrict__ x,
    const IndexType* __restrict__ perm_idxs, const int* diag_LUT,
    const int* subblock_LUT)
{
    const auto tid = thread::get_thread_id_flat<int>();
    const auto curr_b_s = (num_rows_in_block - tid * b_s) < b_s
                              ? (num_rows_in_block - tid * b_s)
                              : b_s;
    const auto nz_p_b = (diag_LUT[curr_b_s] + 1);
    for (auto i = 0; i < nz_p_b; ++i) {
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
                        inv_diag_val * (b_perm[row * num_rhs + k]);
                }

            } else {
                // off the diagonal
                const auto read_offs = diag_LUT[subblock_LUT[i] + 1] * w;
                const auto row_read = rows[read_offs];
                if (row_read >= 0) {
                    const auto x_row_read = perm_idxs[row_read];
                    for (size_type k = 0; k < num_rhs; ++k) {
                        auto tmp =
                            vals[curr_offs] * x[x_row_read * num_rhs + k];
                        b_perm[row * num_rhs + k] -= tmp;
                    }
                }
            }
        }
    }
}

// TODO subwarp_size as a template param isn't being used atm
template <bool prepermuted, int subwarp_size, typename ValueType,
          typename IndexType, typename Closure>
__device__ void apply_ltr(
    const IndexType* __restrict__ rows, const ValueType* __restrict__ vals,
    const size_type end_row, const size_type num_rows_in_block,
    const size_type b_s, const size_type w, const size_type num_rhs,
    ValueType* __restrict__ b_perm, ValueType* __restrict__ x,
    const IndexType* __restrict__ perm_idxs, const int* diag_LUT,
    const int* subblock_LUT, Closure scale)
{
    const auto tid = thread::get_thread_id_flat<int>();
    const auto curr_b_s = (num_rows_in_block - tid * b_s) < b_s
                              ? (num_rows_in_block - tid * b_s)
                              : b_s;
    const auto nz_p_b = (diag_LUT[curr_b_s] + 1);
    for (auto i = 0; i < nz_p_b; ++i) {
        const auto curr_offs = i * w;
        const auto row = rows[curr_offs];
        auto x_row = 0;

        if (row >= 0) {
            if (row >= end_row) return;
            if (prepermuted)
                x_row = row;
            else
                x_row = perm_idxs[row];
            if (i == diag_LUT[subblock_LUT[i] + 1]) {
                // on the diagonal
                const auto val = vals[curr_offs];
                GKO_ASSERT(val != zero<ValueType>());
                const auto inv_diag_val = one<ValueType>() / val;
                for (size_type k = 0; k < num_rhs; ++k) {
                    x[x_row * num_rhs + k] =
                        scale(inv_diag_val * (b_perm[row * num_rhs + k]),
                              x[x_row * num_rhs + k]);
                }

            } else {
                // off the diagonal
                const auto read_offs = diag_LUT[subblock_LUT[i] + 1] * w;
                const auto row_read = rows[read_offs];
                if (row_read >= 0) {
                    auto x_row_read = 0;
                    if (prepermuted)
                        x_row_read = row_read;
                    else
                        x_row_read = perm_idxs[row_read];
                    for (size_type k = 0; k < num_rhs; ++k) {
                        auto tmp =
                            vals[curr_offs] * x[x_row_read * num_rhs + k];
                        b_perm[row * num_rhs + k] -= tmp;
                    }
                }
            }
        }
    }
}

template <bool prepermuted, int subwarp_size, int b_s, typename ValueType,
          typename IndexType, typename Closure>
__device__ void apply_ltr_v2(
    const IndexType* __restrict__ rows, const ValueType* __restrict__ vals,
    const size_type end_row, const size_type num_rows_in_block,
    const size_type w, const size_type num_rhs,
    const ValueType* __restrict__ b_perm, ValueType* __restrict__ x,
    const IndexType* __restrict__ perm_idxs, const int* diag_LUT,
    const int* subblock_LUT, Closure scale)
{
    const auto tid = thread::get_thread_id_flat<int>();
    const auto curr_b_s = (num_rows_in_block - tid * b_s) < b_s
                              ? (num_rows_in_block - tid * b_s)
                              : b_s;
    const auto nz_p_b = (diag_LUT[curr_b_s] + 1);
    ValueType tmp_rhs[b_s];
    const auto base_row = rows[0];
    for (size_type k = 0; k < num_rhs; ++k) {
        for (auto i = 0; i < b_s; ++i) {
            tmp_rhs[i] = 0;
        }
        for (auto i = 0; i < nz_p_b; ++i) {
            const auto curr_offs = i * w;
            const auto row = rows[curr_offs];
            auto x_row = 0;
            if (row >= 0) {
                if (row >= end_row) return;
                if (prepermuted)
                    x_row = row;
                else
                    x_row = perm_idxs[row];
                const auto rhs_id = (row - base_row) / w;
                if (i == diag_LUT[subblock_LUT[i] + 1]) {
                    // on the diagonal
                    const auto val = vals[curr_offs];
                    assert(val != zero<ValueType>());
                    const auto inv_diag_val = one<ValueType>() / val;
                    tmp_rhs[rhs_id] += b_perm[row * num_rhs + k];

                    x[x_row * num_rhs + k] = scale(
                        inv_diag_val * tmp_rhs[rhs_id], x[x_row * num_rhs + k]);
                } else {
                    // off the diagonal
                    const auto read_offs = diag_LUT[subblock_LUT[i] + 1] * w;
                    const auto row_read = rows[read_offs];
                    if (row_read >= 0) {
                        auto x_row_read = 0;
                        if (prepermuted)
                            x_row_read = row_read;
                        else
                            x_row_read = perm_idxs[row_read];

                        tmp_rhs[rhs_id] -=
                            vals[curr_offs] * x[x_row_read * num_rhs + k];
                    }
                }
            }
        }
    }
}

template <bool prepermuted, int subwarp_size, int b_s, typename ValueType,
          typename IndexType, typename Closure>
__device__ void apply_ltr_v3(
    const IndexType* __restrict__ rows, const ValueType* __restrict__ vals,
    const size_type end_row, const size_type num_rows_in_block,
    const size_type w, const size_type num_rhs,
    const ValueType* __restrict__ b_perm, ValueType* __restrict__ x,
    const IndexType* __restrict__ perm_idxs, const int* diag_LUT,
    const int* subblock_LUT, Closure scale)
{
    const auto tid = thread::get_thread_id_flat<int>();
    const auto col_id = blockIdx.y;
    const auto curr_b_s = (num_rows_in_block - tid * b_s) < b_s
                              ? (num_rows_in_block - tid * b_s)
                              : b_s;
    const auto nz_p_b = (diag_LUT[curr_b_s] + 1);
    ValueType tmp_rhs[b_s];
    const auto base_row = rows[0];

    for (auto i = 0; i < b_s; ++i) {
        tmp_rhs[i] = 0;
    }
    for (auto i = 0; i < nz_p_b; ++i) {
        const auto curr_offs = i * w;
        const auto row = rows[curr_offs];
        auto x_row = 0;
        if (row >= 0) {
            if (row >= end_row) return;
            if (prepermuted)
                x_row = row;
            else
                x_row = perm_idxs[row];
            const auto rhs_id = (row - base_row) / w;
            if (i == diag_LUT[subblock_LUT[i] + 1]) {
                // on the diagonal
                const auto val = vals[curr_offs];
                assert(val != zero<ValueType>());
                const auto inv_diag_val = one<ValueType>() / val;
                tmp_rhs[rhs_id] += b_perm[row * num_rhs + col_id];

                x[x_row * num_rhs + col_id] =
                    scale(inv_diag_val * tmp_rhs[rhs_id],
                          x[x_row * num_rhs + col_id]);
            } else {
                // off the diagonal
                const auto read_offs = diag_LUT[subblock_LUT[i] + 1] * w;
                const auto row_read = rows[read_offs];
                if (row_read >= 0) {
                    const auto x_row_read = perm_idxs[row_read];

                    tmp_rhs[rhs_id] -=
                        vals[curr_offs] * x[x_row_read * num_rhs + col_id];
                }
            }
        }
    }
}

template <bool prepermuted, int subwarp_size, typename ValueType,
          typename IndexType>
__global__ void apply_l_p_block_kernel(
    const IndexType* __restrict__ rows, const ValueType* __restrict__ vals,
    const size_type end_row, const size_type num_rows_in_block,
    const size_type b_s, const size_type num_p_blocks, const bool res_blocks,
    const size_type num_rhs, ValueType* __restrict__ b_perm,
    ValueType* __restrict__ x, const IndexType* __restrict__ perm_idxs,
    const int* diag_LUT, const int* subblock_LUT)
{
    const auto tid = thread::get_thread_id_flat<int>();
    const auto subwarp_id = tid / subwarp_size;
    const auto local_tid = tid % subwarp_size;
    const auto l_tid = tid % blockDim.x;
    // const auto l_tid = thread::get_local_thread_id<subwarp_size>();
    // auto subwarp =
    //     group::tiled_partition<subwarp_size>(group::this_thread_block());
    // const auto local_tid = subwarp.thread_rank();
    // const auto subwarp_id = thread::get_subwarp_id_flat<subwarp_size, int>();

    // __shared__ int shared_diag_LUT[max_block_size + 1];
    // __shared__ int shared_subblock_LUT[max_nz_block + 1];
    // if (l_tid < max_nz_block + 1) {
    //     printf("subblock lid = %u\n", l_tid);
    //     shared_subblock_LUT[l_tid] = subblock_LUT[l_tid];
    // } else if (l_tid <= max_nz_block + max_block_size + 2) {
    //     printf("diag lid = %u\n", l_tid);
    //     shared_diag_LUT[l_tid - max_nz_block - 1] =
    //         diag_LUT[l_tid - max_nz_block - 1];
    // }

    if (subwarp_id >= num_p_blocks || tid * b_s >= num_rows_in_block) return;
    const auto nz_p_b = diag_LUT[b_s] + 1;
    const auto stride = nz_p_b * subwarp_size;
    const auto base_offs = subwarp_id * stride;
    if (subwarp_id == (num_p_blocks - 1) && res_blocks) {
        // apply base_block_agg
        const auto stride_base_block = nz_p_b;
        const auto base_block_offs = base_offs + local_tid * stride_base_block;

        apply_ltr<prepermuted, subwarp_size>(
            &(rows[base_block_offs]), &(vals[base_block_offs]), end_row,
            num_rows_in_block, b_s, one<size_type>(), num_rhs, b_perm, x,
            perm_idxs, diag_LUT,
            subblock_LUT,  // shared_diag_LUT, shared_subblock_LUT,//
            [](const ValueType& x, const ValueType& y) { return x; });

    } else {
        // apply lvl 1
        apply_ltr<prepermuted, subwarp_size>(
            &(rows[base_offs + local_tid]), &(vals[base_offs + local_tid]),
            end_row, num_rows_in_block, b_s, subwarp_size, num_rhs, b_perm, x,
            perm_idxs, diag_LUT,
            subblock_LUT,  // shared_diag_LUT, shared_subblock_LUT,//
            [](const ValueType& x, const ValueType& y) { return x; });
    }
}

template <bool prepermuted, int subwarp_size, typename ValueType,
          typename IndexType>
__global__ void apply_l_p_block_kernel(
    const IndexType* __restrict__ rows, const ValueType* __restrict__ vals,
    const size_type end_row, const size_type num_rows_in_block,
    const size_type b_s, const size_type num_p_blocks, const bool res_blocks,
    const size_type num_rhs, ValueType* __restrict__ b_perm,
    const ValueType* __restrict__ alpha, ValueType* __restrict__ x,
    const ValueType* __restrict__ beta, const IndexType* __restrict__ perm_idxs,
    const int* diag_LUT, const int* subblock_LUT)
{
    const auto alpha_val = alpha[0];
    const auto beta_val = beta[0];
    const auto tid = thread::get_thread_id_flat<int>();
    const auto subwarp_id = tid / subwarp_size;
    const auto local_tid = tid % subwarp_size;
    // auto subwarp =
    //     group::tiled_partition<subwarp_size>(group::this_thread_block());
    // const auto local_tid = subwarp.thread_rank();
    // const auto subwarp_id = thread::get_subwarp_id_flat<subwarp_size, int>();
    if (subwarp_id >= num_p_blocks || tid * b_s >= num_rows_in_block) return;

    const auto nz_p_b = diag_LUT[b_s] + 1;
    const auto stride = nz_p_b * subwarp_size;
    const auto base_offs = subwarp_id * stride;
    if (subwarp_id == (num_p_blocks - 1) && res_blocks) {
        // apply base_block_agg
        const auto stride_base_block = nz_p_b;
        const auto base_block_offs = base_offs + local_tid * stride_base_block;

        apply_ltr<prepermuted, subwarp_size>(
            &(rows[base_block_offs]), &(vals[base_block_offs]), end_row,
            num_rows_in_block, b_s, one<size_type>(), num_rhs, b_perm, x,
            perm_idxs, diag_LUT, subblock_LUT,
            [&alpha_val, &beta_val](const ValueType& x, const ValueType& y) {
                return alpha_val * x + beta_val * y;
            });
    } else {
        // apply lvl 1
        apply_ltr<prepermuted, subwarp_size>(
            &(rows[base_offs + local_tid]), &(vals[base_offs + local_tid]),
            end_row, num_rows_in_block, b_s, subwarp_size, num_rhs, b_perm, x,
            perm_idxs, diag_LUT, subblock_LUT,
            [&alpha_val, &beta_val](const ValueType& x, const ValueType& y) {
                return alpha_val * x + beta_val * y;
            });
    }
}

template <bool prepermuted, int subwarp_size, int b_s, typename ValueType,
          typename IndexType>
__global__ void apply_l_p_block_kernel(
    const IndexType* __restrict__ rows, const ValueType* __restrict__ vals,
    const size_type end_row, const size_type num_rows_in_block,
    const size_type num_p_blocks, const bool res_blocks,
    const size_type num_rhs, ValueType* __restrict__ b_perm,
    ValueType* __restrict__ x, const IndexType* __restrict__ perm_idxs,
    const int* diag_LUT, const int* subblock_LUT)
{
    const auto tid = thread::get_thread_id_flat<int>();
    const auto subwarp_id = tid / subwarp_size;
    const auto local_tid = tid % subwarp_size;
    const auto l_tid = tid % blockDim.x;

    if (subwarp_id >= num_p_blocks || tid * b_s >= num_rows_in_block) return;
    const auto nz_p_b = diag_LUT[b_s] + 1;
    const auto stride = nz_p_b * subwarp_size;
    const auto base_offs = subwarp_id * stride;
    if (subwarp_id == (num_p_blocks - 1) && res_blocks) {
        // apply base_block_agg
        const auto stride_base_block = nz_p_b;
        const auto base_block_offs = base_offs + local_tid * stride_base_block;
        apply_ltr_v2<prepermuted, subwarp_size, b_s>(
            &(rows[base_block_offs]), &(vals[base_block_offs]), end_row,
            num_rows_in_block, one<size_type>(), num_rhs, b_perm, x, perm_idxs,
            diag_LUT, subblock_LUT,
            [](const ValueType& x, const ValueType& y) { return x; });

    } else {
        // apply lvl 1
        apply_ltr_v2<prepermuted, subwarp_size, b_s>(
            &(rows[base_offs + local_tid]), &(vals[base_offs + local_tid]),
            end_row, num_rows_in_block, subwarp_size, num_rhs, b_perm, x,
            perm_idxs, diag_LUT, subblock_LUT,
            [](const ValueType& x, const ValueType& y) { return x; });
    }
}

template <bool prepermuted, int subwarp_size, int b_s, typename ValueType,
          typename IndexType>
__global__ void apply_l_p_block_kernel(
    const IndexType* __restrict__ rows, const ValueType* __restrict__ vals,
    const size_type end_row, const size_type num_rows_in_block,
    const size_type num_p_blocks, const bool res_blocks,
    const size_type num_rhs, ValueType* __restrict__ b_perm,
    const ValueType* __restrict__ alpha, ValueType* __restrict__ x,
    const ValueType* __restrict__ beta, const IndexType* __restrict__ perm_idxs,
    const int* diag_LUT, const int* subblock_LUT)
{
    const auto alpha_val = alpha[0];
    const auto beta_val = beta[0];
    const auto tid = thread::get_thread_id_flat<int>();
    const auto subwarp_id = tid / subwarp_size;
    const auto local_tid = tid % subwarp_size;

    if (subwarp_id >= num_p_blocks || tid * b_s >= num_rows_in_block) return;

    const auto nz_p_b = diag_LUT[b_s] + 1;
    const auto stride = nz_p_b * subwarp_size;
    const auto base_offs = subwarp_id * stride;
    if (subwarp_id == (num_p_blocks - 1) && res_blocks) {
        // apply base_block_agg
        const auto stride_base_block = nz_p_b;
        const auto base_block_offs = base_offs + local_tid * stride_base_block;

        apply_ltr_v2<prepermuted, subwarp_size, b_s>(
            &(rows[base_block_offs]), &(vals[base_block_offs]), end_row,
            num_rows_in_block, one<size_type>(), num_rhs, b_perm, x, perm_idxs,
            diag_LUT, subblock_LUT,
            [&alpha_val, &beta_val](const ValueType& x, const ValueType& y) {
                return alpha_val * x + beta_val * y;
            });
    } else {
        // apply lvl 1
        apply_ltr_v2<prepermuted, subwarp_size, b_s>(
            &(rows[base_offs + local_tid]), &(vals[base_offs + local_tid]),
            end_row, num_rows_in_block, subwarp_size, num_rhs, b_perm, x,
            perm_idxs, diag_LUT, subblock_LUT,
            [&alpha_val, &beta_val](const ValueType& x, const ValueType& y) {
                return alpha_val * x + beta_val * y;
            });
    }
}

template <typename ValueType, typename IndexType>
__global__ void fill_with_vals_kernel()
{}

}  // namespace kernel

namespace host_kernel {

// template <bool prepermuted, int subwarp_size, int b_s, typename ValueType,
//           typename IndexType>
// void apply_hbmc_v2(syn::value_list<int, b_s>,
//                    std::shared_ptr<const Executor> exec,
//                    const IndexType* l_diag_rows, const ValueType*
//                    l_diag_vals, const preconditioner::parallel_block*
//                    p_block, matrix::Dense<ValueType>* b_perm,
//                    matrix::Dense<ValueType>* x, const int* diag_LUT,
//                    const int* subblock_LUT,
//                    const IndexType* permutation_idxs = nullptr,
//                    const matrix::Dense<ValueType>* alpha = nullptr,
//                    const matrix::Dense<ValueType>* beta = nullptr)
// {
//     GKO_ASSERT(permutation_idxs || prepermuted);
//     const auto num_rows_p_block =
//         p_block->end_row_global_ - p_block->start_row_global_;
//     const auto num_rhs = b_perm->get_size()[1];
//     const auto num_rows = b_perm->get_size()[0];
//     auto id_offs = p_block->val_storage_id_;
//     const auto num_involved_subwarps = p_block->degree_of_parallelism_;
//     const auto min_num_threads =
//         config::min_warps_per_block * config::warp_size;
//     const auto num_involved_threads =
//         num_involved_subwarps * subwarp_size < min_num_threads
//             ? min_num_threads
//             : num_involved_subwarps * subwarp_size;
//     const auto block_size = (num_involved_threads > config::max_block_size)
//                                 ? config::max_block_size
//                                 : num_involved_threads;
//     const auto grid_size = ceildiv(num_involved_threads, block_size);
//     if (alpha == nullptr && beta == nullptr) {
//         kernel::apply_l_p_block_kernel<prepermuted, subwarp_size, b_s>
//             <<<block_size, grid_size>>>(
//                 &(l_diag_rows[id_offs]),
//                 as_device_type(&(l_diag_vals[id_offs])),
//                 p_block->end_row_global_, num_rows_p_block,
//                 p_block->degree_of_parallelism_, p_block->residual_, num_rhs,
//                 as_device_type(b_perm->get_values()),
//                 as_device_type(x->get_values()), permutation_idxs, diag_LUT,
//                 subblock_LUT);
//     } else if (alpha != nullptr && beta != nullptr) {
//         kernel::apply_l_p_block_kernel<prepermuted, subwarp_size, b_s>
//             <<<block_size, grid_size>>>(
//                 &(l_diag_rows[id_offs]),
//                 as_device_type(&(l_diag_vals[id_offs])),
//                 p_block->end_row_global_, num_rows_p_block,
//                 p_block->degree_of_parallelism_, p_block->residual_, num_rhs,
//                 as_device_type(b_perm->get_values()),
//                 as_device_type(alpha->get_const_values()),
//                 as_device_type(x->get_values()),
//                 as_device_type(beta->get_const_values()), permutation_idxs,
//                 diag_LUT, subblock_LUT);

//     } else {
//         GKO_KERNEL_NOT_FOUND;
//     }
// }
// GKO_ENABLE_IMPLEMENTATION_SELECTION(select_apply_hbmc_v2, apply_hbmc_v2);

template <bool prepermuted, int subwarp_size, int version = 1,
          typename ValueType, typename IndexType>
void apply_hbmc(syn::value_list<int, subwarp_size>,
                std::shared_ptr<const Executor> exec,
                const IndexType* l_diag_rows, const ValueType* l_diag_vals,
                const preconditioner::parallel_block* p_block,
                matrix::Dense<ValueType>* b_perm, matrix::Dense<ValueType>* x,
                const int* diag_LUT, const int* subblock_LUT,
                const IndexType* permutation_idxs = nullptr,
                const matrix::Dense<ValueType>* alpha = nullptr,
                const matrix::Dense<ValueType>* beta = nullptr)
{
    GKO_ASSERT(permutation_idxs || prepermuted);
    const auto num_rows_p_block =
        p_block->end_row_global_ - p_block->start_row_global_;
    const auto num_rhs = b_perm->get_size()[1];
    const auto num_rows = b_perm->get_size()[0];
    auto id_offs = p_block->val_storage_id_;
    const auto num_involved_subwarps = p_block->degree_of_parallelism_;
    const auto min_num_threads =
        config::min_warps_per_block * config::warp_size;
    const auto num_involved_threads =
        num_involved_subwarps * subwarp_size < min_num_threads
            ? min_num_threads
            : num_involved_subwarps * subwarp_size;
    const auto block_size = (num_involved_threads > config::max_block_size)
                                ? config::max_block_size
                                : num_involved_threads;
    const auto grid_size = ceildiv(num_involved_threads, block_size);
    const auto b_s = p_block->base_block_size_;
    if (version == 1) {
        if (alpha == nullptr && beta == nullptr) {
            kernel::apply_l_p_block_kernel<prepermuted, subwarp_size>
                <<<block_size, grid_size>>>(
                    &(l_diag_rows[id_offs]),
                    as_device_type(&(l_diag_vals[id_offs])),
                    p_block->end_row_global_, num_rows_p_block, b_s,
                    p_block->degree_of_parallelism_, p_block->residual_,
                    num_rhs, as_device_type(b_perm->get_values()),
                    as_device_type(x->get_values()), permutation_idxs, diag_LUT,
                    subblock_LUT);
        } else if (alpha != nullptr && beta != nullptr) {
            kernel::apply_l_p_block_kernel<prepermuted, subwarp_size>
                <<<block_size, grid_size>>>(
                    &(l_diag_rows[id_offs]),
                    as_device_type(&(l_diag_vals[id_offs])),
                    p_block->end_row_global_, num_rows_p_block, b_s,
                    p_block->degree_of_parallelism_, p_block->residual_,
                    num_rhs, as_device_type(b_perm->get_values()),
                    as_device_type(alpha->get_const_values()),
                    as_device_type(x->get_values()),
                    as_device_type(beta->get_const_values()), permutation_idxs,
                    diag_LUT, subblock_LUT);
        }
    } else if (version == 2) {
        // select_apply_hbmc_v2(
        //     hbmc_block_sizes(),
        //     [&](int compiled_b_s) { return compiled_b_s == b_s; },
        //     syn::value_list<int, prepermuted, subwarp_size>(),
        //     syn::type_list<>(), exec, l_diag_rows, l_diag_vals, p_block,
        //     b_perm, x, diag_LUT, subblock_LUT, permutation_idxs, alpha,
        //     beta);
    } else if (version == 3) {
        GKO_KERNEL_NOT_FOUND;
    } else {
        GKO_KERNEL_NOT_FOUND;
    }
}
GKO_ENABLE_IMPLEMENTATION_SELECTION(select_apply_hbmc, apply_hbmc);


}  // namespace host_kernel

template <typename ValueType, typename IndexType>
void prepermuted_apply(
    std::shared_ptr<const DefaultExecutor> exec, const IndexType* l_diag_rows,
    const ValueType* l_diag_vals, const IndexType* l_spmv_row_ptrs,
    const IndexType* l_spmv_col_idxs, const ValueType* l_spmv_vals,
    const preconditioner::storage_scheme& storage_scheme,
    const IndexType* permutation_idxs, const matrix::Dense<ValueType>* alpha,
    matrix::Dense<ValueType>* b_perm, const matrix::Dense<ValueType>* beta,
    matrix::Dense<ValueType>* x_perm)
{
    GKO_ASSERT(!storage_scheme.symm_);
    const auto block_ptrs = storage_scheme.forward_solve_;
    const auto num_blocks = storage_scheme.num_blocks_;
    const auto num_rhs = b_perm->get_size()[1];
    const auto num_rows = b_perm->get_size()[0];

    auto diag_LUT = gko::array<gko::int32>(exec, max_block_size + 1);

    exec->copy_from<gko::int32>(exec->get_master().get(),
                                static_cast<gko::size_type>(max_block_size + 1),
                                diag_lut.data(), diag_LUT.get_data());
    auto subblock_LUT =
        gko::array<gko::int32>(exec, get_nz_block(max_block_size) + 1);

    exec->copy_from<gko::int32>(
        exec->get_master().get(),
        static_cast<gko::size_type>(get_nz_block(max_block_size) + 1),
        sub_block_lut.data(), subblock_LUT.get_data());

    for (auto block = 0; block < num_blocks; block += 2) {
        auto p_block = static_cast<preconditioner::parallel_block*>(
            block_ptrs[block].get());
        const auto w = p_block->lvl_2_block_size_;

        host_kernel::select_apply_hbmc(
            hbmc_kernels(),
            [&](int compiled_subwarp_size) {
                return compiled_subwarp_size == w;
            },
            syn::value_list<int, true>(), syn::type_list<>(), exec, l_diag_rows,
            l_diag_vals, p_block, b_perm, x_perm, diag_LUT.get_const_data(),
            subblock_LUT.get_const_data(), permutation_idxs, alpha, beta);

        if (block < num_blocks - 1) {
            auto spmv_block = static_cast<preconditioner::spmv_block*>(
                block_ptrs[block + 1].get());
            const auto spmv_size_row =
                spmv_block->end_row_global_ - spmv_block->start_row_global_;
            const auto spmv_size_col =
                spmv_block->end_col_global_ - spmv_block->start_col_global_;
            const auto spmv_nnz =
                l_spmv_row_ptrs[spmv_block->row_ptrs_storage_id_ +
                                spmv_size_row];

            auto tmp_csr = gko::matrix::Csr<ValueType, IndexType>::create_const(
                exec, gko::dim<2>{spmv_size_row, spmv_size_col},
                gko::array<ValueType>::const_view(
                    exec, spmv_nnz,
                    &(l_spmv_vals[spmv_block->val_storage_id_])),
                gko::array<IndexType>::const_view(
                    exec, spmv_nnz,
                    &(l_spmv_col_idxs[spmv_block->val_storage_id_])),
                gko::array<IndexType>::const_view(
                    exec, spmv_size_row + 1,
                    &(l_spmv_row_ptrs[spmv_block->row_ptrs_storage_id_])));
            auto tmp_b_perm = b_perm->create_submatrix(
                gko::span{spmv_block->start_row_global_,
                          spmv_block->end_row_global_},
                gko::span{0, num_rhs});
            auto tmp_x = x_perm->create_submatrix(
                gko::span{spmv_block->start_col_global_,
                          spmv_block->end_col_global_},
                gko::span{0, num_rhs});
            auto alpha_spmv =
                gko::initialize<gko::matrix::Dense<ValueType>>({-1.}, exec);
            auto beta_spmv =
                gko::initialize<gko::matrix::Dense<ValueType>>({1.}, exec);

            csr::advanced_spmv(exec, lend(alpha_spmv), lend(tmp_csr),
                               lend(tmp_x), lend(beta_spmv), lend(tmp_b_perm));
        }
    }
}
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_GAUSS_SEIDEL_PREPERMUTED_APPLY_KERNEL);

template <typename ValueType, typename IndexType>
void prepermuted_simple_apply(
    std::shared_ptr<const DefaultExecutor> exec, const IndexType* l_diag_rows,
    const ValueType* l_diag_vals, const IndexType* l_spmv_row_ptrs,
    const IndexType* l_spmv_col_idxs, const ValueType* l_spmv_vals,
    const preconditioner::storage_scheme& storage_scheme,
    const IndexType* permutation_idxs, matrix::Dense<ValueType>* b_perm,
    matrix::Dense<ValueType>* x_perm)
{
    GKO_ASSERT(!storage_scheme.symm_);
    const auto block_ptrs = storage_scheme.forward_solve_;
    const auto num_blocks = storage_scheme.num_blocks_;
    const auto num_rhs = b_perm->get_size()[1];
    const auto num_rows = b_perm->get_size()[0];

    auto diag_LUT = gko::array<gko::int32>(exec, max_block_size + 1);

    exec->copy_from<gko::int32>(exec->get_master().get(),
                                static_cast<gko::size_type>(max_block_size + 1),
                                diag_lut.data(), diag_LUT.get_data());
    auto subblock_LUT =
        gko::array<gko::int32>(exec, get_nz_block(max_block_size) + 1);

    exec->copy_from<gko::int32>(
        exec->get_master().get(),
        static_cast<gko::size_type>(get_nz_block(max_block_size) + 1),
        sub_block_lut.data(), subblock_LUT.get_data());

    for (auto block = 0; block < num_blocks; block += 2) {
        auto p_block = static_cast<preconditioner::parallel_block*>(
            block_ptrs[block].get());
        const auto w = p_block->lvl_2_block_size_;

        host_kernel::select_apply_hbmc(
            hbmc_kernels(),
            [&](int compiled_subwarp_size) {
                return compiled_subwarp_size == w;
            },
            syn::value_list<int, true>(), syn::type_list<>(), exec, l_diag_rows,
            l_diag_vals, p_block, b_perm, x_perm, diag_LUT.get_const_data(),
            subblock_LUT.get_const_data());

        if (block < num_blocks - 1) {
            auto spmv_block = static_cast<preconditioner::spmv_block*>(
                block_ptrs[block + 1].get());
            const auto spmv_size_row =
                spmv_block->end_row_global_ - spmv_block->start_row_global_;
            const auto spmv_size_col =
                spmv_block->end_col_global_ - spmv_block->start_col_global_;
            const auto spmv_nnz =
                l_spmv_row_ptrs[spmv_block->row_ptrs_storage_id_ +
                                spmv_size_row];

            auto tmp_csr = gko::matrix::Csr<ValueType, IndexType>::create_const(
                exec, gko::dim<2>{spmv_size_row, spmv_size_col},
                gko::array<ValueType>::const_view(
                    exec, spmv_nnz,
                    &(l_spmv_vals[spmv_block->val_storage_id_])),
                gko::array<IndexType>::const_view(
                    exec, spmv_nnz,
                    &(l_spmv_col_idxs[spmv_block->val_storage_id_])),
                gko::array<IndexType>::const_view(
                    exec, spmv_size_row + 1,
                    &(l_spmv_row_ptrs[spmv_block->row_ptrs_storage_id_])));
            auto tmp_b_perm = b_perm->create_submatrix(
                gko::span{spmv_block->start_row_global_,
                          spmv_block->end_row_global_},
                gko::span{0, num_rhs});
            auto tmp_x = x_perm->create_submatrix(
                gko::span{spmv_block->start_col_global_,
                          spmv_block->end_col_global_},
                gko::span{0, num_rhs});
            auto alpha =
                gko::initialize<gko::matrix::Dense<ValueType>>({-1.}, exec);
            auto beta =
                gko::initialize<gko::matrix::Dense<ValueType>>({1.}, exec);

            csr::advanced_spmv(exec, lend(alpha), lend(tmp_csr), lend(tmp_x),
                               lend(beta), lend(tmp_b_perm));
        }
    }
}
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_GAUSS_SEIDEL_PREPERMUTED_SIMPLE_APPLY_KERNEL);