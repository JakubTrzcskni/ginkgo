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

#include <array>
#include <cstddef>
#include <cstring>
#include <iterator>
#include <limits>
#include <list>
#include <set>
#include <utility>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>

#include "core/base/allocator.hpp"
#include "core/matrix/csr_kernels.hpp"
#include "core/utils/matrix_utils.hpp"

namespace gko {
namespace kernels {
namespace omp {
namespace gauss_seidel {

namespace {

enum struct nodeSelectionPolicy { fifo, maxNumEdges, score };
enum struct seedSelectionPolicy { noPolicy, minDegree };

template <typename IndexType>
IndexType find_next_candidate(
    const IndexType* block_ordering, const IndexType* curr_block,
    const IndexType block_fill_level, std::list<IndexType>& candidates,
    const IndexType num_nodes, const IndexType block_size,
    const IndexType* degrees, int8* visited, const IndexType* row_ptrs,
    const IndexType* col_idxs,
    nodeSelectionPolicy policy = nodeSelectionPolicy::fifo,
    seedSelectionPolicy seed_policy = seedSelectionPolicy::minDegree)
{
    if (candidates.empty()) {
        switch (seed_policy) {
        case seedSelectionPolicy::minDegree: {
            IndexType index_min_node = -1;
            IndexType min_node_degree = std::numeric_limits<IndexType>::max();
            for (auto i = 0; i < num_nodes; ++i) {
                if (degrees[i] < min_node_degree && visited[i] == 0) {
                    index_min_node = i;
                    min_node_degree = degrees[i];
                }
            }
            if (index_min_node >= 0) {
                visited[index_min_node] = 1;
                for (auto i = row_ptrs[index_min_node];
                     i < row_ptrs[index_min_node + 1]; i++) {
                    if (visited[col_idxs[i]] == 0)
                        candidates.push_back(col_idxs[i]);
                }
            }
            return index_min_node;
            break;
        }
        case seedSelectionPolicy::noPolicy: {
            IndexType seed = -1;
            for (auto i = 0; i < num_nodes; ++i) {
                if (visited[i] == 0) {
                    seed = i;
                    visited[i] = 1;
                    return seed;
                }
            }
            return seed;  // if no node found return -1
            break;
        }
        default:
            GKO_NOT_SUPPORTED(seed_policy);
            break;
        }

    } else {
        switch (policy) {
        case nodeSelectionPolicy::fifo: {
            IndexType next_candidate;
            for (IndexType candidate : candidates) {
                if (visited[candidate] == 0) {
                    next_candidate = candidate;
                    visited[candidate] = 1;
                    candidates.remove(candidate);
                    for (auto i = row_ptrs[next_candidate];
                         i < row_ptrs[next_candidate + 1]; i++) {
                        if (visited[col_idxs[i]] == 0)
                            candidates.push_back(col_idxs[i]);
                    }
                    return next_candidate;  // return first candidate in the
                                            // list (if found)
                } else {
                    candidates.remove(
                        candidate);  // delete already visited nodes
                }
            }
            return -1;  // return -1 if all candidates have been visited
            break;
        }
        case nodeSelectionPolicy::score: {
            GKO_NOT_IMPLEMENTED;
            break;
        }
        case nodeSelectionPolicy::maxNumEdges: {
            auto best_joint_edges = -1;
            auto best_candidate = -1;
            for (IndexType candidate : candidates) {
                if (visited[candidate] == 0) {
                    auto curr_joint_edges = 0;
                    for (auto i = row_ptrs[candidate];
                         i < row_ptrs[candidate + 1]; i++) {
                        auto candidate_neighbour = col_idxs[i];
                        for (auto i = 0; i < block_fill_level; ++i) {
                            auto node_in_block = curr_block[i];
                            if (candidate_neighbour == node_in_block)
                                curr_joint_edges++;
                        }
                    }
                    if (curr_joint_edges > best_joint_edges) {
                        best_joint_edges = curr_joint_edges;
                        best_candidate = candidate;
                    }
                } else {
                    candidates.remove(
                        candidate);  // delete already visited nodes
                }
            }
            if (best_candidate >= 0) {
                visited[best_candidate] = 1;
                candidates.remove(best_candidate);
                for (auto i = row_ptrs[best_candidate];
                     i < row_ptrs[best_candidate + 1]; i++) {
                    if (visited[col_idxs[i]] == 0)
                        candidates.push_back(col_idxs[i]);
                }
            }
            return best_candidate;
            break;
        }
        default:
            GKO_NOT_SUPPORTED(policy);
            break;
        }
    }
}
}  // namespace

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

template <typename ValueType, typename IndexType>
void prepermuted_simple_apply(
    std::shared_ptr<const OmpExecutor> exec, const IndexType* l_diag_rows,
    const ValueType* l_diag_vals, const IndexType* l_spmv_row_ptrs,
    const IndexType* l_spmv_col_idxs, const ValueType* l_spmv_vals,
    const preconditioner::storage_scheme& storage_scheme,
    const IndexType* permutation_idxs, const matrix::Dense<ValueType>* b_perm,
    matrix::Dense<ValueType>* x_perm, int kernel_version) GKO_NOT_IMPLEMENTED;
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_GAUSS_SEIDEL_PREPERMUTED_SIMPLE_APPLY_KERNEL);

template <typename ValueType, typename IndexType>
void simple_apply(std::shared_ptr<const OmpExecutor> exec,
                  const IndexType* l_diag_rows, const ValueType* l_diag_vals,
                  const IndexType* l_spmv_row_ptrs,
                  const IndexType* l_spmv_col_idxs,
                  const ValueType* l_spmv_vals,
                  const IndexType* permutation_idxs,
                  const preconditioner::storage_scheme& storage_scheme,
                  matrix::Dense<ValueType>* b_perm, matrix::Dense<ValueType>* x,
                  int kernel_version) GKO_NOT_IMPLEMENTED;
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_GAUSS_SEIDEL_SIMPLE_APPLY_KERNEL);

template <typename ValueType, typename IndexType>
void advanced_apply(
    std::shared_ptr<const OmpExecutor> exec, const IndexType* l_diag_rows,
    const ValueType* l_diag_vals, const IndexType* l_spmv_row_ptrs,
    const IndexType* l_spmv_col_idxs, const ValueType* l_spmv_vals,
    const IndexType* u_diag_rows, const ValueType* u_diag_vals,
    const IndexType* u_spmv_row_ptrs, const IndexType* u_spmv_col_idxs,
    const ValueType* u_spmv_vals, const IndexType* permutation_idxs,
    const preconditioner::storage_scheme& storage_scheme,
    const gko::remove_complex<ValueType> omega,
    matrix::Dense<ValueType>* b_perm, matrix::Dense<ValueType>* x,
    int kernel_version) GKO_NOT_IMPLEMENTED;
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_GAUSS_SEIDEL_ADVANCED_APPLY_KERNEL);

template <typename ValueType, typename IndexType>
void advanced_prepermuted_apply(
    std::shared_ptr<const DefaultExecutor> exec, const IndexType* l_diag_rows,
    const ValueType* l_diag_vals, const IndexType* l_spmv_row_ptrs,
    const IndexType* l_spmv_col_idxs, const ValueType* l_spmv_vals,
    const IndexType* u_diag_rows, const ValueType* u_diag_vals,
    const IndexType* u_spmv_row_ptrs, const IndexType* u_spmv_col_idxs,
    const ValueType* u_spmv_vals, const IndexType* permutation_idxs,
    const preconditioner::storage_scheme& storage_scheme,
    const matrix::Dense<ValueType>* b_perm, matrix::Dense<ValueType>* x_perm,
    int kernel_version) GKO_NOT_IMPLEMENTED;
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_GAUSS_SEIDEL_ADVANCED_PREPERMUTED_APPLY_KERNEL);

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
    const IndexType block_size)
{
    const IndexType num_nodes = adjacency_matrix->get_size()[0];
    const auto row_ptrs = adjacency_matrix->get_const_row_ptrs();
    const auto col_idxs = adjacency_matrix->get_const_col_idxs();
    for (IndexType k = 0; k < num_nodes; k += block_size) {
        std::list<IndexType>
            candidates;  // not sure if std::list is the right one
        auto curr_block = &(block_ordering[k]);
        for (IndexType i = 0; i < block_size && k + i < num_nodes; i++) {
            IndexType next_node = find_next_candidate(
                block_ordering, curr_block, i, candidates, num_nodes,
                block_size, degrees, visited, row_ptrs, col_idxs,
                nodeSelectionPolicy::maxNumEdges,
                seedSelectionPolicy::minDegree);
            if (next_node >= 0)
                curr_block[i] = next_node;
            else
                break;  // last block which cannot be filled fully (i <
                        // block_size, but all nodes already visited)
                        // second check, loop bounds are the first
        }
    }
}
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
                            const IndexType max_color,
                            const bool use_padding) GKO_NOT_IMPLEMENTED;
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
    const preconditioner::storage_scheme& storage_scheme,
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