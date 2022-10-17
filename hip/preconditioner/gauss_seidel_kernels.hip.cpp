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

#include "core/preconditioner/gauss_seidel_kernels.hpp"

#include <algorithm>

#include <hip/hip_runtime.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>

#include "core/base/allocator.hpp"
#include "core/synthesizer/implementation_selection.hpp"
#include "hip/base/config.hip.hpp"
#include "hip/base/math.hip.hpp"
#include "hip/base/types.hip.hpp"
#include "hip/components/cooperative_groups.hip.hpp"
#include "hip/components/merging.hip.hpp"
#include "hip/components/reduction.hip.hpp"
#include "hip/components/thread_ids.hip.hpp"
#include "hip/components/uninitialized_array.hip.hpp"
#include "hip/components/warp_blas.hip.hpp"
#include "hip/matrix/csr_kernels.hip.cpp"

namespace gko {
namespace kernels {
namespace hip {
namespace gauss_seidel {

using hbmc_kernels =
    syn::value_list<int, config::warp_size, 32, 16, 8, 4>;  // ,2, 1>;

using hbmc_block_sizes =
    syn::value_list<int, 2, 4, 8>;  // 1, 2, 3, 4, 5, 6, 7,
                                    // 8, 9, 10, 11, 12, 13, 14, 15, 16>;

constexpr int default_kernel_version = 1;

#include "common/cuda_hip/preconditioner/gauss_seidel_kernels.hpp.inc"

template <typename IndexType>
void get_degree_of_nodes(std::shared_ptr<const HipExecutor> exec,
                         const IndexType num_vertices,
                         const IndexType* const row_ptrs,
                         IndexType* const degrees)
{
    const auto block_size = config::max_block_size;
    const auto grid_size = ceildiv(num_vertices, block_size);
    kernel::get_degree_of_nodes_kernel<<<grid_size, block_size>>>(
        num_vertices, row_ptrs, degrees);
}
GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_GAUSS_SEIDEL_GET_DEGREE_OF_NODES_KERNEL);

template <typename ValueType, typename IndexType>
void get_coloring(
    std::shared_ptr<const HipExecutor> exec,
    const matrix::SparsityCsr<ValueType, IndexType>* adjacency_matrix,
    array<IndexType>& vertex_colors, IndexType* max_color) GKO_NOT_IMPLEMENTED;
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_GAUSS_SEIDEL_GET_COLORING_KERNEL);

template <typename ValueType, typename IndexType>
void get_block_coloring(
    std::shared_ptr<const HipExecutor> exec,
    const matrix::SparsityCsr<ValueType, IndexType>* adjacency_matrix,
    const IndexType* block_ordering, const IndexType block_size,
    IndexType* vertex_colors, IndexType* max_color) GKO_NOT_IMPLEMENTED;
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_GAUSS_SEIDEL_GET_BLOCK_COLORING_KERNEL);

template <typename ValueType, typename IndexType>
void assign_to_blocks(
    std::shared_ptr<const HipExecutor> exec,
    const matrix::SparsityCsr<ValueType, IndexType>* adjacency_matrix,
    IndexType* block_ordering, const IndexType* degrees, int8* visited,
    const IndexType block_size) GKO_NOT_IMPLEMENTED;
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_GAUSS_SEIDEL_ASSIGN_TO_BLOCKS_KERNEL);

template <typename IndexType>
void get_secondary_ordering(std::shared_ptr<const HipExecutor> exec,
                            IndexType* permutation_idxs,
                            preconditioner::storage_scheme& storage_scheme,
                            const IndexType base_block_size,
                            const IndexType lvl_2_block_size,
                            const IndexType* color_block_ptrs,
                            const IndexType max_color,
                            const bool use_padding) GKO_NOT_IMPLEMENTED;
GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_GAUSS_SEIDEL_GET_SECONDARY_ORDERING_KERNEL);

template <typename ValueType, typename IndexType>
void setup_blocks(std::shared_ptr<const HipExecutor> exec,
                  const matrix::Csr<ValueType, IndexType>* system_matrix,
                  const IndexType* permutation_idxs,
                  const IndexType* inv_permutation_idxs,
                  preconditioner::storage_scheme& storage_scheme,
                  IndexType* l_diag_rows, IndexType* l_diag_mtx_col_idxs,
                  ValueType* l_diag_vals, IndexType* l_spmv_row_ptrs,
                  IndexType* l_spmv_col_idxs, IndexType* l_spmv_mtx_col_idxs,
                  ValueType* l_spmv_vals, IndexType* u_diag_rows,
                  IndexType* u_diag_mtx_col_idxs, ValueType* u_diag_vals,
                  IndexType* u_spmv_row_ptrs, IndexType* u_spmv_col_idxs,
                  IndexType* u_spmv_mtx_col_idxs,
                  ValueType* u_spmv_vals) GKO_NOT_IMPLEMENTED;
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_GAUSS_SEIDEL_SETUP_BLOCKS_KERNEL);

template <typename ValueType, typename IndexType>
void fill_with_vals(
    std::shared_ptr<const HipExecutor> exec,
    const matrix::Csr<ValueType, IndexType>* system_matrix,
    const IndexType* permutation_idxs,
    const preconditioner::storage_scheme& storage_scheme,
    const IndexType diag_num_elems, const IndexType* l_diag_rows,
    const IndexType* l_diag_mtx_col_idxs, ValueType* l_diag_vals,
    const IndexType* l_spmv_row_ptrs, const IndexType* l_spmv_col_idxs,
    const IndexType* l_spmv_mtx_col_idxs, ValueType* l_spmv_vals,
    const IndexType* u_diag_rows, const IndexType* u_diag_mtx_col_idxs,
    ValueType* u_diag_vals, const IndexType* u_spmv_row_ptrs,
    const IndexType* u_spmv_col_idxs, const IndexType* u_spmv_mtx_col_idxs,
    ValueType* u_spmv_vals)
{
    const auto mtx_row_ptrs = system_matrix->get_const_row_ptrs();
    const auto mtx_col_idxs = system_matrix->get_const_col_idxs();
    const auto mtx_vals = system_matrix->get_const_values();
    if (storage_scheme.symm_) {
        GKO_NOT_IMPLEMENTED;
    } else {
        // fill the diagonal block
        // const auto work_p_thread;
        // const auto max_grid_size;
        const auto combined_nnz_spmv_blocks =
            storage_scheme
                .combined_nnz_spmv_blocks_;  // TODO save this in the storage
        // scheme, set/update during setup
        const auto combined_storage_size =
            combined_nnz_spmv_blocks + diag_num_elems;
        const auto max_storage_block_length = std::max(
            combined_nnz_spmv_blocks, static_cast<size_type>(diag_num_elems));
        const auto min_block_size =
            config::min_warps_per_block * config::warp_size;
        const auto block_size =
            config::max_block_size <
                    max_storage_block_length  // or take combined size
                ? config::max_block_size
                : (min_block_size < max_storage_block_length
                       ? max_storage_block_length
                       : min_block_size);
        const auto grid_size_diag_part = ceildiv(diag_num_elems, block_size);
        const auto grid_size_spmv_part =
            ceildiv(combined_nnz_spmv_blocks, block_size);
        const auto grid_size = grid_size_diag_part + grid_size_spmv_part;

        // kernel::fill_with_vals_kernel<<<block_size, grid_size>>>();

        for (auto i = 0; i < diag_num_elems; i++) {
            if (l_diag_rows[i] >= 0) {
                const auto mtx_row = permutation_idxs[l_diag_rows[i]];
                const auto mtx_col = l_diag_mtx_col_idxs[i];
                l_diag_vals[i] = mtx_vals[mtx_row_ptrs[mtx_row] + mtx_col];
            }
        }

        auto main_blocks = storage_scheme.forward_solve_;
        const auto num_blocks = storage_scheme.num_blocks_;
        // fill the spmv blocks
        if (num_blocks >= 3) {  // if there are spmv blocks
            for (auto i = 1; i < num_blocks; i += 2) {
                auto spmv_block = static_cast<preconditioner::spmv_block*>(
                    main_blocks[i].get());
                const auto block_row_offset = spmv_block->start_row_global_;
                const auto num_rows =
                    spmv_block->end_row_global_ - block_row_offset;
                const auto row_ptrs_start = spmv_block->row_ptrs_storage_id_;
                const auto val_start = spmv_block->val_storage_id_;
                for (auto row = 0; row < num_rows; row++) {
                    const auto mtx_row =
                        permutation_idxs[block_row_offset + row];
                    for (auto id = l_spmv_row_ptrs[row_ptrs_start + row];
                         id < l_spmv_row_ptrs[row_ptrs_start + row + 1]; id++) {
                        const auto mtx_col =
                            l_spmv_mtx_col_idxs[val_start + id];
                        l_spmv_vals[val_start + id] =
                            mtx_vals[mtx_row_ptrs[mtx_row] + mtx_col];
                    }
                }
            }
        }
    }
}
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_GAUSS_SEIDEL_FILL_WITH_VALS_KERNEL);
}  // namespace gauss_seidel
}  // namespace hip
}  // namespace kernels
}  // namespace gko