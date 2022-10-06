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
template <int subwarp_size, typename ValueType, typename IndexType,
          typename Closure>
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
                        scale(inv_diag_val * (b_perm[row * num_rhs + k]),
                              x[x_row * num_rhs + k]);
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

template <int subwarp_size, typename ValueType, typename IndexType>
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

    if (subwarp_id >= num_p_blocks || tid * b_s >= num_rows_in_block) return;
    const auto nz_p_b = diag_LUT[b_s] + 1;
    const auto stride = nz_p_b * subwarp_size;
    const auto base_offs = subwarp_id * stride;
    if (subwarp_id == (num_p_blocks - 1) && res_blocks) {
        // apply base_block_agg
        const auto stride_base_block = nz_p_b;
        const auto base_block_offs = base_offs + local_tid * stride_base_block;
        apply_ltr<subwarp_size>(
            &(rows[base_block_offs]), &(vals[base_block_offs]), end_row,
            num_rows_in_block, b_s, one<size_type>(), num_rhs, b_perm, x,
            perm_idxs, diag_LUT, subblock_LUT,
            [](const ValueType& x, const ValueType& y) { return x; });
    } else {
        // apply lvl 1
        apply_ltr<subwarp_size>(
            &(rows[base_offs + local_tid]), &(vals[base_offs + local_tid]),
            end_row, num_rows_in_block, b_s, subwarp_size, num_rhs, b_perm, x,
            perm_idxs, diag_LUT, subblock_LUT,
            [](const ValueType& x, const ValueType& y) { return x; });
    }
}
template <int subwarp_size, typename ValueType, typename IndexType>
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
    if (subwarp_id >= num_p_blocks || tid * b_s >= num_rows_in_block) return;

    const auto nz_p_b = diag_LUT[b_s] + 1;
    const auto stride = nz_p_b * subwarp_size;
    const auto base_offs = subwarp_id * stride;
    if (subwarp_id == (num_p_blocks - 1) && res_blocks) {
        // apply base_block_agg
        const auto stride_base_block = nz_p_b;
        const auto base_block_offs = base_offs + local_tid * stride_base_block;

        apply_ltr<subwarp_size>(
            &(rows[base_block_offs]), &(vals[base_block_offs]), end_row,
            num_rows_in_block, b_s, one<size_type>(), num_rhs, b_perm, x,
            perm_idxs, diag_LUT, subblock_LUT,
            [&alpha_val, &beta_val](const ValueType& x, const ValueType& y) {
                return alpha_val * x + beta_val * y;
            });
    } else {
        // apply lvl 1

        apply_ltr<subwarp_size>(
            &(rows[base_offs + local_tid]), &(vals[base_offs + local_tid]),
            end_row, num_rows_in_block, b_s, subwarp_size, num_rhs, b_perm, x,
            perm_idxs, diag_LUT, subblock_LUT,
            [&alpha_val, &beta_val](const ValueType& x, const ValueType& y) {
                return alpha_val * x + beta_val * y;
            });
    }
}

template <typename ValueType,
          typename IndexType>  // rows&vals passed already with offset
__global__ void apply_l_p_block_kernel(
    const IndexType* __restrict__ rows, const ValueType* __restrict__ vals,
    const size_type end_row, const size_type num_rows_in_block,
    const size_type b_s, const size_type w, const size_type num_p_blocks,
    const bool res_blocks, const size_type num_rhs,
    ValueType* __restrict__ b_perm, ValueType* __restrict__ x,
    const IndexType* __restrict__ perm_idxs, const int* diag_LUT,
    const int* subblock_LUT)
{
    const int subwarp_size = w;
    // constexpr int warp_size = config::warp_size;
    auto subwarp =
        group::tiled_partition<config::warp_size>(group::this_thread_block());

    // const auto local_tid = subwarp.thread_rank();
    // const auto warp_id = thread::get_subwarp_id_flat<config::warp_size,
    // int>();
    const auto tid = thread::get_thread_id_flat<int>();
    const auto warp_id = tid / subwarp_size;
    const auto local_tid = tid % subwarp_size;

    if (warp_id >= num_p_blocks || tid * b_s >= num_rows_in_block) return;
    // printf("subw id = %d, thread rank in subwarp = %d\n", subwarp_id,
    // subwarp.thread_rank());
    const auto nz_p_b = diag_LUT[b_s] + 1;
    const auto stride = nz_p_b * w;
    const auto base_offs = warp_id * stride;
    if (warp_id == (num_p_blocks - 1) && res_blocks) {
        // apply base_block_agg
        const auto stride_base_block = nz_p_b;
        const auto base_block_offs = base_offs + local_tid * stride_base_block;

        apply_ltr(&(rows[base_block_offs]), &(vals[base_block_offs]), end_row,
                  num_rows_in_block, b_s, one<size_type>(), num_rhs, b_perm, x,
                  perm_idxs, diag_LUT, subblock_LUT);
    } else {
        // apply lvl 1
        apply_ltr(&(rows[base_offs + local_tid]),
                  &(vals[base_offs + local_tid]), end_row, num_rows_in_block,
                  b_s, w, num_rhs, b_perm, x, perm_idxs, diag_LUT,
                  subblock_LUT);
    }
}

template <typename ValueType, typename IndexType>
__global__ void fill_with_vals_kernel()
{}

}  // namespace kernel

namespace host_kernel {

template <int subwarp_size, typename ValueType, typename IndexType>
void apply_hbmc(syn::value_list<int, subwarp_size>,
                std::shared_ptr<const HipExecutor> exec,
                const IndexType* l_diag_rows, const ValueType* l_diag_vals,
                const IndexType* permutation_idxs,
                const preconditioner::parallel_block* p_block,
                matrix::Dense<ValueType>* b_perm, matrix::Dense<ValueType>* x,
                const int* diag_LUT, const int* subblock_LUT,
                const matrix::Dense<ValueType>* alpha = nullptr,
                const matrix::Dense<ValueType>* beta = nullptr)
{
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
    if (alpha == nullptr && beta == nullptr) {
        kernel::apply_l_p_block_kernel<subwarp_size><<<block_size, grid_size>>>(
            &(l_diag_rows[id_offs]), as_hip_type(&(l_diag_vals[id_offs])),
            p_block->end_row_global_, num_rows_p_block,
            p_block->base_block_size_, p_block->degree_of_parallelism_,
            p_block->residual_, num_rhs, as_hip_type(b_perm->get_values()),
            as_hip_type(x->get_values()), permutation_idxs, diag_LUT,
            subblock_LUT);
    } else if (alpha != nullptr && beta != nullptr) {
        kernel::apply_l_p_block_kernel<subwarp_size><<<block_size, grid_size>>>(
            &(l_diag_rows[id_offs]), as_hip_type(&(l_diag_vals[id_offs])),
            p_block->end_row_global_, num_rows_p_block,
            p_block->base_block_size_, p_block->degree_of_parallelism_,
            p_block->residual_, num_rhs, as_hip_type(b_perm->get_values()),
            as_hip_type(alpha->get_const_values()),
            as_hip_type(x->get_values()), as_hip_type(beta->get_const_values()),
            permutation_idxs, diag_LUT, subblock_LUT);
    } else {
        GKO_KERNEL_NOT_FOUND;
    }
}
GKO_ENABLE_IMPLEMENTATION_SELECTION(select_apply_hbmc, apply_hbmc);

}  // namespace host_kernel
