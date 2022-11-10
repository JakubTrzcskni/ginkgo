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

using hbmc_kernels =
    syn::value_list<int, config::warp_size, 32, 16, 8, 4>;  // ,2, 1>;

using hbmc_block_sizes = syn::value_list<int, 2, 4, 8>;  // 1, 2, 3, 4, 5, 6, 7,
// 8, 9, 10, 11, 12, 13, 14, 15, 16>;

using hbmc_kernel_version = syn::value_list<int, 1, 2, 3, 4, 5, 6, 7, 8, 9>;

constexpr int default_kernel_version = 1;

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

template <int subwarp_size, typename ValueType, typename IndexType>
__device__ void fill_row()
{}

template <int subwarp_size, typename ValueType, typename IndexType>
__global__ void fill_with_vals_kernel(
    const IndexType* mtx_row_ptrs, const IndexType* mtx_col_idxs,
    const ValueType* mtx_vals, const IndexType* perm_idxs,
    const size_type diag_num_elems, const size_type l_spmv_row_mem_requirement,
    const size_type u_spmv_row_mem_requirement, const size_type l_spmv_row_offs,
    const IndexType* l_diag_rows, const IndexType* l_diag_mtx_col_idxs,
    ValueType* l_diag_vals, const IndexType* l_spmv_row_ptrs,
    const IndexType* l_spmv_col_idxs, const IndexType* l_spmv_mtx_col_idxs,
    ValueType* l_spmv_vals, const IndexType* u_diag_rows,
    const IndexType* u_diag_mtx_col_idxs, ValueType* u_diag_vals,
    const IndexType* u_spmv_row_ptrs, const IndexType* u_spmv_col_idxs,
    const IndexType* u_spmv_mtx_col_idxs, ValueType* u_spmv_vals,
    const bool symm)
{
    const auto tid = thread::get_thread_id_flat<int>();
    if (tid < diag_num_elems) {
        const auto row = l_diag_rows[tid];
        if (row >= 0) {
            const auto mtx_row = perm_idxs[row];
            l_diag_vals[tid] =
                mtx_vals[mtx_row_ptrs[mtx_row] + l_diag_mtx_col_idxs[tid]];
            if (symm) {
                const auto u_row = u_diag_rows[tid];
                if (u_row >= 0) {
                    const auto u_mtx_row = perm_idxs[u_row];
                    u_diag_vals[tid] = mtx_vals[mtx_row_ptrs[u_mtx_row] +
                                                u_diag_mtx_col_idxs[tid]];
                }
            }
        }
    }
    auto subwarp =
        group::tiled_partition<subwarp_size>(group::this_thread_block());
    const auto local_tid = subwarp.thread_rank();
    const auto subwarp_id = thread::get_subwarp_id_flat<subwarp_size, int>();
    // if (subwarp_id < l_spmv_row_mem_requirement - 1) {
    //     if () }
    // if (subwarp_id < u_spmv_row_mem_requirement - 1) {
    // }
}

#include "gauss_seidel_apply_kernels.hpp.inc"

#include "gs_advanced_apply.hpp.inc"

}  // namespace kernel

namespace host_kernel {

template <int subwarp_size, typename ValueType, typename IndexType>
void fill_with_vals(const IndexType* mtx_row_ptrs,
                    const IndexType* mtx_col_idxs, const ValueType* mtx_vals,
                    const IndexType* perm_idxs, const size_type diag_num_elems,
                    const size_type spmv_num_rows, const IndexType* l_diag_rows,
                    const IndexType* l_diag_mtx_col_idxs,
                    ValueType* l_diag_vals, const IndexType* l_spmv_row_ptrs,
                    const IndexType* l_spmv_col_idxs,
                    const IndexType* l_spmv_mtx_col_idxs,
                    ValueType* l_spmv_vals, const IndexType* u_diag_rows,
                    const IndexType* u_diag_mtx_col_idxs,
                    ValueType* u_diag_vals, const IndexType* u_spmv_row_ptrs,
                    const IndexType* u_spmv_col_idxs,
                    const IndexType* u_spmv_mtx_col_idxs,
                    ValueType* u_spmv_vals, const bool symm)
{}
GKO_ENABLE_IMPLEMENTATION_SELECTION(selection_fill_with_vals, fill_with_vals);

#include "gs_select_b_s.hpp.inc"

#include "gs_select_subwarp.hpp.inc"

#include "gs_select_kernel_version.hpp.inc"

}  // namespace host_kernel

constexpr int kernel_version = kernel::default_kernel_version;

template <typename ValueType, typename IndexType>
void prepermuted_simple_apply(
    std::shared_ptr<const DefaultExecutor> exec, const IndexType* l_diag_rows,
    const ValueType* l_diag_vals, const IndexType* l_spmv_row_ptrs,
    const IndexType* l_spmv_col_idxs, const ValueType* l_spmv_vals,
    const preconditioner::storage_scheme& storage_scheme,
    const IndexType* permutation_idxs, const matrix::Dense<ValueType>* b_perm,
    matrix::Dense<ValueType>* x_perm, int kernel_ver)
{
    GKO_ASSERT(!storage_scheme.symm_);
    GKO_ASSERT(kernel_ver >= 2);
    const auto block_ptrs = storage_scheme.forward_solve_;
    const auto num_blocks = storage_scheme.num_blocks_;
    const auto num_rhs = b_perm->get_size()[1];
    const auto num_rows = b_perm->get_size()[0];


    auto diag_LUT = gko::array<gko::int32>();
    auto subblock_LUT = gko::array<gko::int32>();
    if (kernel_ver != 9) {
        diag_LUT = gko::array<gko::int32>(exec, max_b_s + 1);
        exec->copy_from<gko::int32>(exec->get_master().get(),
                                    static_cast<gko::size_type>(max_b_s + 1),
                                    diag_lut.data(), diag_LUT.get_data());
        subblock_LUT = gko::array<gko::int32>(exec, max_nz_block + 1);
        exec->copy_from<gko::int32>(
            exec->get_master().get(),
            static_cast<gko::size_type>(max_nz_block + 1), sub_block_lut.data(),
            subblock_LUT.get_data());
    }
    auto alpha_spmv =
        gko::initialize<gko::matrix::Dense<ValueType>>({-1.}, exec);
    auto beta_spmv = gko::initialize<gko::matrix::Dense<ValueType>>({0.}, exec);

    for (auto block = 0; block < num_blocks; block += 2) {
        auto p_block = static_cast<preconditioner::parallel_block*>(
            block_ptrs[block].get());

        host_kernel::select_const_hbmc_kernel_version(
            kernel::hbmc_kernel_version(),
            [&](int compiled_version) {
                return compiled_version == kernel_ver;
            },
            syn::value_list<int, true>(), syn::type_list<>(), exec, l_diag_rows,
            l_diag_vals, p_block, b_perm, x_perm, diag_LUT.get_const_data(),
            subblock_LUT.get_const_data(), permutation_idxs);

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
            auto x_in = x_perm->create_submatrix(
                gko::span{spmv_block->start_col_global_,
                          spmv_block->end_col_global_},
                gko::span{0, num_rhs});
            auto x_out_tmp = x_perm->create_submatrix(
                gko::span{spmv_block->start_row_global_,
                          spmv_block->end_row_global_},
                gko::span{0, num_rhs});

            csr::advanced_spmv(exec, lend(alpha_spmv), lend(tmp_csr),
                               lend(x_in), lend(beta_spmv), lend(x_out_tmp));
        }
    }
}
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_GAUSS_SEIDEL_PREPERMUTED_SIMPLE_APPLY_KERNEL);

template <typename ValueType, typename IndexType>
void advanced_apply(
    std::shared_ptr<const DefaultExecutor> exec, const IndexType* l_diag_rows,
    const ValueType* l_diag_vals, const IndexType* l_spmv_row_ptrs,
    const IndexType* l_spmv_col_idxs, const ValueType* l_spmv_vals,
    const IndexType* u_diag_rows, const ValueType* u_diag_vals,
    const IndexType* u_spmv_row_ptrs, const IndexType* u_spmv_col_idxs,
    const ValueType* u_spmv_vals, const IndexType* permutation_idxs,
    const preconditioner::storage_scheme& storage_scheme,
    const gko::remove_complex<ValueType> omega,
    matrix::Dense<ValueType>* b_perm, matrix::Dense<ValueType>* x_perm,
    int kernel_version)
{
    GKO_ASSERT(storage_scheme.symm_);
    GKO_ASSERT(kernel_version == 9);
    const auto forward_solve = storage_scheme.forward_solve_;
    const auto backward_solve = storage_scheme.backward_solve_;

    const auto num_blocks = storage_scheme.num_blocks_;
    const auto num_rhs = b_perm->get_size()[1];
    const auto num_rows = b_perm->get_size()[0];
    auto alpha_spmv =
        gko::initialize<gko::matrix::Dense<ValueType>>({-1.}, exec);
    auto beta_spmv_forward =
        gko::initialize<gko::matrix::Dense<ValueType>>({0.}, exec);

    // forward solve
    for (auto block = 0; block < num_blocks; block += 2) {
        auto p_block = static_cast<preconditioner::parallel_block*>(
            forward_solve[block].get());
        const auto w = p_block->lvl_2_block_size_;
        host_kernel::select_const_adv_apply_hbmc(
            kernel::hbmc_kernels(),
            [&](int compiled_subwarp_size) {
                return compiled_subwarp_size == w;
            },
            syn::value_list<int, true, true>(), syn::type_list<>(), exec,
            l_diag_rows, l_diag_vals, p_block, b_perm, x_perm,
            permutation_idxs);

        if (block < num_blocks - 1) {
            auto spmv_block = static_cast<preconditioner::spmv_block*>(
                forward_solve[block + 1].get());
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
            auto x_in = x_perm->create_submatrix(
                gko::span{spmv_block->start_col_global_,
                          spmv_block->end_col_global_},
                gko::span{0, num_rhs});
            auto x_out_tmp = x_perm->create_submatrix(
                gko::span{spmv_block->start_row_global_,
                          spmv_block->end_row_global_},
                gko::span{0, num_rhs});

            csr::advanced_spmv(exec, lend(alpha_spmv), lend(tmp_csr),
                               lend(x_in), lend(beta_spmv_forward),
                               lend(x_out_tmp));
        }
    }
    auto beta_spmv_backward =
        gko::initialize<gko::matrix::Dense<ValueType>>({1.}, exec);
    for (auto block = 0; block < num_blocks; block += 2) {
        auto p_block = static_cast<preconditioner::parallel_block*>(
            backward_solve[block].get());

        const auto w = p_block->lvl_2_block_size_;
        host_kernel::select_const_adv_apply_hbmc(
            kernel::hbmc_kernels(),
            [&](int compiled_subwarp_size) {
                return compiled_subwarp_size == w;
            },
            syn::value_list<int, false, true>(), syn::type_list<>(), exec,
            u_diag_rows, u_diag_vals, p_block, b_perm, x_perm,
            permutation_idxs);

        if (block < num_blocks - 1) {
            auto spmv_block = static_cast<preconditioner::spmv_block*>(
                backward_solve[block + 1].get());
            const auto spmv_size_row =
                spmv_block->end_row_global_ - spmv_block->start_row_global_;
            const auto spmv_size_col =
                spmv_block->end_col_global_ - spmv_block->start_col_global_;
            const auto spmv_nnz =
                u_spmv_row_ptrs[spmv_block->row_ptrs_storage_id_ +
                                spmv_size_row];

            auto tmp_csr = gko::matrix::Csr<ValueType, IndexType>::create_const(
                exec, gko::dim<2>{spmv_size_row, spmv_size_col},
                gko::array<ValueType>::const_view(
                    exec, spmv_nnz,
                    &(u_spmv_vals[spmv_block->val_storage_id_])),
                gko::array<IndexType>::const_view(
                    exec, spmv_nnz,
                    &(u_spmv_col_idxs[spmv_block->val_storage_id_])),
                gko::array<IndexType>::const_view(
                    exec, spmv_size_row + 1,
                    &(u_spmv_row_ptrs[spmv_block->row_ptrs_storage_id_])));
            auto x_in = x_perm->create_submatrix(
                gko::span{spmv_block->start_col_global_,
                          spmv_block->end_col_global_},
                gko::span{0, num_rhs});
            auto x_out_tmp = x_perm->create_submatrix(
                gko::span{spmv_block->start_row_global_,
                          spmv_block->end_row_global_},
                gko::span{0, num_rhs});

            csr::advanced_spmv(exec, lend(alpha_spmv), lend(tmp_csr),
                               lend(x_in), lend(beta_spmv_backward),
                               lend(x_out_tmp));
        }
    }
}
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_GAUSS_SEIDEL_ADVANCED_APPLY_KERNEL);

#include "gs_simple_apply.hpp.inc"
