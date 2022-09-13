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

#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/csr.hpp>

#include "core/base/allocator.hpp"

namespace gko {
namespace kernels {
namespace omp {
namespace gauss_seidel {

template <typename IndexType>
void get_degree_of_nodes(std::shared_ptr<const OmpExecutor> exec,
                         const IndexType num_vertices,
                         const IndexType* const row_ptrs,
                         IndexType* const degrees)
{
#pragma omp parallel for
    for (IndexType i = 0; i < num_vertices; ++i) {
        degrees[i] = row_ptrs[i + 1] - row_ptrs[i];
    }
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_GAUSS_SEIDEL_GET_DEGREE_OF_NODES_KERNEL);

template <typename ValueType>
void ref_apply(std::shared_ptr<const OmpExecutor> exec, const LinOp* solver,
               const matrix::Dense<ValueType>* alpha,
               const matrix::Dense<ValueType>* b,
               const matrix::Dense<ValueType>* beta,
               matrix::Dense<ValueType>* x)
{
    solver->apply(alpha, b, beta, x);
}
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_GAUSS_SEIDEL_REFERENCE_APPLY_KERNEL);

template <typename ValueType>
void ref_simple_apply(std::shared_ptr<const OmpExecutor> exec,
                      const LinOp* solver, const matrix::Dense<ValueType>* b,
                      matrix::Dense<ValueType>* x)
{
    solver->apply(b, x);
}
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_GAUSS_SEIDEL_REFERENCE_SIMPLE_APPLY_KERNEL);

template <typename ValueType, typename IndexType>
void apply(std::shared_ptr<const OmpExecutor> exec,
           const matrix::Csr<ValueType, IndexType>* A,
           const matrix::Dense<ValueType>* alpha,
           const matrix::Dense<ValueType>* rhs,
           const matrix::Dense<ValueType>* beta,
           matrix::Dense<ValueType>* x) GKO_NOT_IMPLEMENTED;
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_GAUSS_SEIDEL_APPLY_KERNEL);

template <typename ValueType, typename IndexType>
void simple_apply(std::shared_ptr<const OmpExecutor> exec,
                  const matrix::Csr<ValueType, IndexType>* A,
                  const matrix::Dense<ValueType>* rhs,
                  matrix::Dense<ValueType>* x) GKO_NOT_IMPLEMENTED;
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_GAUSS_SEIDEL_SIMPLE_APPLY_KERNEL);


template <typename ValueType, typename IndexType>
void get_coloring(
    std::shared_ptr<const OmpExecutor> exec,
    const matrix::SparsityCsr<ValueType, IndexType>* adjacency_matrix,
    array<IndexType>& vertex_colors, IndexType* max_color) GKO_NOT_IMPLEMENTED;
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_GAUSS_SEIDEL_GET_COLORING_KERNEL);

template <typename ValueType, typename IndexType>
void get_block_coloring(
    std::shared_ptr<const OmpExecutor> exec,
    const matrix::SparsityCsr<ValueType, IndexType>* adjacency_matrix,
    const IndexType* block_ordering, const IndexType block_size,
    IndexType* vertex_colors, IndexType* max_color) GKO_NOT_IMPLEMENTED;
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_GAUSS_SEIDEL_GET_BLOCK_COLORING_KERNEL);

template <typename ValueType, typename IndexType>
void assign_to_blocks(
    std::shared_ptr<const OmpExecutor> exec,
    const matrix::SparsityCsr<ValueType, IndexType>* adjacency_matrix,
    IndexType* block_ordering, const IndexType* degrees, int8* visited,
    const IndexType block_size) GKO_NOT_IMPLEMENTED;
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_GAUSS_SEIDEL_ASSIGN_TO_BLOCKS_KERNEL);

template <typename IndexType>
void get_permutation_from_coloring(
    std::shared_ptr<const OmpExecutor> exec, const IndexType num_nodes,
    IndexType* coloring, const IndexType max_color, IndexType* color_ptrs,
    IndexType* permutation_idxs,
    const IndexType* block_ordering) GKO_NOT_IMPLEMENTED;
GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_GAUSS_SEIDEL_GET_PERMUTATION_FROM_COLORING_KERNEL);

template <typename IndexType>
void get_secondary_ordering(std::shared_ptr<const OmpExecutor> exec,
                            IndexType* permutation_idxs,
                            preconditioner::storage_scheme& storage_scheme,
                            const IndexType base_block_size,
                            const IndexType lvl_2_block_size,
                            const IndexType* color_block_ptrs,
                            const IndexType max_color) GKO_NOT_IMPLEMENTED;
GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_GAUSS_SEIDEL_GET_SECONDARY_ORDERING_KERNEL);

template <typename ValueType, typename IndexType>
void setup_blocks(std::shared_ptr<const OmpExecutor> exec,
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
    std::shared_ptr<const OmpExecutor> exec,
    const matrix::Csr<ValueType, IndexType>* system_matrix,
    const IndexType* permutation_idxs,
    preconditioner::storage_scheme& storage_scheme,
    const IndexType diag_num_elems, const IndexType* l_diag_rows,
    const IndexType* l_diag_mtx_col_idxs, ValueType* l_diag_vals,
    const IndexType* l_spmv_row_ptrs, const IndexType* l_spmv_col_idxs,
    const IndexType* l_spmv_mtx_col_idxs, ValueType* l_spmv_vals,
    const IndexType* u_diag_rows, const IndexType* u_diag_mtx_col_idxs,
    ValueType* u_diag_vals, const IndexType* u_spmv_row_ptrs,
    const IndexType* u_spmv_col_idxs, const IndexType* u_spmv_mtx_col_idxs,
    ValueType* u_spmv_vals) GKO_NOT_IMPLEMENTED;
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_GAUSS_SEIDEL_FILL_WITH_VALS_KERNEL);


}  // namespace gauss_seidel
}  // namespace omp
}  // namespace kernels
}  // namespace gko