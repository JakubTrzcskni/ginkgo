// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

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
#include <ginkgo/core/base/mtx_io.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/base/allocator.hpp"
#include "core/matrix/csr_kernels.hpp"
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

namespace gko {
namespace kernels {
namespace hip {
namespace gauss_seidel {

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
                  const gko::remove_complex<ValueType> omega,
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
    const auto combined_nnz_spmv_blocks =
        storage_scheme.combined_nnz_spmv_blocks_;
    const auto avg_nnz_per_row_spmv =
        static_cast<float>(combined_nnz_spmv_blocks) /
        system_matrix->get_size()[0];


    const auto max_storage_block_length = std::max(
        combined_nnz_spmv_blocks, static_cast<size_type>(diag_num_elems));
    const auto min_block_size = config::min_warps_per_block * config::warp_size;
    const auto block_size =
        config::max_block_size <
                max_storage_block_length  // or take combined size
            ? config::max_block_size
            : (min_block_size < max_storage_block_length
                   ? max_storage_block_length
                   : min_block_size);
    // const auto grid_size_diag_part = ceildiv(diag_num_elems, block_size);
    // const auto grid_size_spmv_part =
    //     ceildiv(combined_nnz_spmv_blocks, block_size);
    // const auto grid_size = grid_size_diag_part + grid_size_spmv_part;

    const auto grid_size = ceildiv(max_storage_block_length, block_size);
}
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_GAUSS_SEIDEL_FILL_WITH_VALS_KERNEL);
}  // namespace gauss_seidel
}  // namespace hip
}  // namespace kernels
}  // namespace gko
