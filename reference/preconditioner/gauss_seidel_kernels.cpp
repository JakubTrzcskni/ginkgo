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

#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>

// for testing LUTs
#include <array>
#include <cstddef>
#include <utility>

#include <cstring>
#include <iterator>
#include <limits>
#include <list>
#include <set>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>
#include <ginkgo/core/solver/lower_trs.hpp>


#include "core/base/allocator.hpp"
#include "core/utils/matrix_utils.hpp"

namespace gko {
namespace kernels {
namespace reference {
namespace gauss_seidel {
namespace {

enum struct nodeSelectionPolicy { fifo, maxNumEdges, score };
enum struct seedSelectionPolicy { noPolicy, minDegree };

template <typename IndexType>
IndexType find_next_candidate(
    const IndexType* block_ordering, const IndexType curr_block,
    std::list<IndexType>& candidates, const IndexType num_nodes,
    const IndexType block_size, const IndexType* degrees, int8* visited,
    const IndexType* row_ptrs, const IndexType* col_idxs,
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
                        for (auto node = curr_block;
                             node < curr_block + block_size && node < num_nodes;
                             node++) {
                            if (candidate_neighbour == node) curr_joint_edges++;
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
void get_degree_of_nodes(std::shared_ptr<const ReferenceExecutor> exec,
                         const IndexType num_vertices,
                         const IndexType* const row_ptrs,
                         IndexType* const degrees)
{
    for (IndexType i = 0; i < num_vertices; ++i) {
        degrees[i] = row_ptrs[i + 1] - row_ptrs[i];
    }
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_GAUSS_SEIDEL_GET_DEGREE_OF_NODES_KERNEL);

template <typename ValueType>
void ref_apply(std::shared_ptr<const ReferenceExecutor> exec,
               const LinOp* solver, const matrix::Dense<ValueType>* alpha,
               const matrix::Dense<ValueType>* b,
               const matrix::Dense<ValueType>* beta,
               matrix::Dense<ValueType>* x)
{
    solver->apply(alpha, b, beta, x);
}
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_GAUSS_SEIDEL_REFERENCE_APPLY_KERNEL);

template <typename ValueType>
void ref_simple_apply(std::shared_ptr<const ReferenceExecutor> exec,
                      const LinOp* solver, const matrix::Dense<ValueType>* b,
                      matrix::Dense<ValueType>* x)
{
    solver->apply(b, x);
}
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_GAUSS_SEIDEL_REFERENCE_SIMPLE_APPLY_KERNEL);

template <typename ValueType, typename IndexType>
void apply(std::shared_ptr<const ReferenceExecutor> exec,
           const matrix::Csr<ValueType, IndexType>* A,
           const matrix::Dense<ValueType>* alpha,
           const matrix::Dense<ValueType>* b,
           const matrix::Dense<ValueType>* beta,
           matrix::Dense<ValueType>* x) GKO_NOT_IMPLEMENTED;
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_GAUSS_SEIDEL_APPLY_KERNEL);

template <typename ValueType, typename IndexType>
void simple_apply(std::shared_ptr<const ReferenceExecutor> exec,
                  const matrix::Csr<ValueType, IndexType>* A,
                  const matrix::Dense<ValueType>* b,
                  matrix::Dense<ValueType>* x) GKO_NOT_IMPLEMENTED;
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_GAUSS_SEIDEL_SIMPLE_APPLY_KERNEL);

template <typename ValueType, typename IndexType>
void get_coloring(
    std::shared_ptr<const ReferenceExecutor> exec,
    const matrix::SparsityCsr<ValueType, IndexType>* adjacency_matrix,
    array<IndexType>& vertex_colors, IndexType* max_color)
{
    const IndexType* row_ptrs = adjacency_matrix->get_const_row_ptrs();
    const IndexType* col_idxs = adjacency_matrix->get_const_col_idxs();
    IndexType highest_color = 0;
    for (auto i = 0; i < vertex_colors.get_num_elems(); i++) {
        typename std::set<IndexType> neighbour_colors;
        for (auto j = row_ptrs[i]; j < row_ptrs[i + 1]; j++) {
            auto col = col_idxs[j];
            auto adjacent_vertex_color = vertex_colors.get_const_data()[col];
            neighbour_colors.insert(adjacent_vertex_color);
            if (adjacent_vertex_color > highest_color)
                highest_color = adjacent_vertex_color;
        }
        bool color_found = false;
        for (auto color = 0; !color_found && color <= highest_color; color++) {
            typename std::set<IndexType>::iterator it;
            it = neighbour_colors.find(color);
            if (it == neighbour_colors.end()) {
                vertex_colors.get_data()[i] = color;
                color_found = true;
            }
        }
        if (!color_found) {
            highest_color++;
            vertex_colors.get_data()[i] = highest_color;
        }
    }
    *max_color = highest_color;
}
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_GAUSS_SEIDEL_GET_COLORING_KERNEL);

template <typename ValueType, typename IndexType>
void get_block_coloring(
    std::shared_ptr<const ReferenceExecutor> exec,
    const matrix::SparsityCsr<ValueType, IndexType>* adjacency_matrix,
    const IndexType* block_ordering, const IndexType block_size,
    IndexType* vertex_colors, IndexType* max_color)
{
    IndexType highest_color = 0;
    const IndexType* colors = vertex_colors;
    const IndexType num_nodes = adjacency_matrix->get_size()[0];
    const auto row_ptrs = adjacency_matrix->get_const_row_ptrs();
    const auto col_idxs = adjacency_matrix->get_const_col_idxs();
    for (auto k = 0; k < num_nodes; k += block_size) {
        typename std::set<IndexType>
            neighbour_colors;  // colors of all neighbours of nodes in the block
        auto curr_block = &block_ordering[k];
        for (auto i = 0; i < block_size && k + i < num_nodes; i++) {
            auto curr_node = curr_block[i];
            for (auto j = row_ptrs[curr_node]; j < row_ptrs[curr_node + 1];
                 j++) {
                auto col = col_idxs[j];
                auto adjacent_vertex_color = colors[col];
                neighbour_colors.insert(adjacent_vertex_color);
                if (adjacent_vertex_color > highest_color)
                    highest_color = adjacent_vertex_color;
            }
        }
        bool color_found = false;
        IndexType best_color_found = 0;
        for (auto color = 0; !color_found && color <= highest_color; color++) {
            typename std::set<IndexType>::iterator it;
            it = neighbour_colors.find(color);
            if (it == neighbour_colors.end()) {
                best_color_found = color;
                color_found = true;
            }
        }
        if (!color_found) {
            highest_color++;
            best_color_found = highest_color;
        }
        for (auto i = 0; i < block_size && k + i < num_nodes; i++) {
            auto curr_node = curr_block[i];
            vertex_colors[curr_node] = best_color_found;
        }
    }
    *max_color = highest_color;
}
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_GAUSS_SEIDEL_GET_BLOCK_COLORING_KERNEL);

// TODO include the selection policies in the template params
template <typename ValueType, typename IndexType>
void assign_to_blocks(
    std::shared_ptr<const ReferenceExecutor> exec,
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
        for (auto i = 0; i < block_size && i + k < num_nodes; i++) {
            auto next_node = find_next_candidate(
                block_ordering, k, candidates, num_nodes, block_size, degrees,
                visited, row_ptrs, col_idxs, nodeSelectionPolicy::maxNumEdges,
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
    std::shared_ptr<const ReferenceExecutor> exec, const IndexType num_nodes,
    IndexType* coloring, const IndexType max_color, IndexType* color_ptrs,
    IndexType* permutation_idxs, const IndexType* block_ordering)
{
    IndexType tmp{0};
    if (!block_ordering) {
        for (auto color = 0; color <= max_color; color++) {
            for (auto i = 0; i < num_nodes; i++) {
                if (i == 0) {
                    color_ptrs[color] = tmp;
                }
                if (coloring[i] == color) {
                    permutation_idxs[tmp] = i;
                    tmp++;
                }
            }
        }
    } else {
        for (auto color = 0; color <= max_color; color++) {
            for (auto i = 0; i < num_nodes; i++) {
                auto node = block_ordering[i];
                if (i == 0) {
                    color_ptrs[color] = tmp;
                }
                if (coloring[node] == color) {
                    permutation_idxs[tmp] = node;
                    tmp++;
                }
            }
        }
    }
    GKO_ASSERT_EQ(tmp, num_nodes);
    color_ptrs[max_color + 1] = num_nodes;
}
GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_GAUSS_SEIDEL_GET_PERMUTATION_FROM_COLORING_KERNEL);

namespace {
/// @brief equivalent to number of full blocks up to the current node * number
/// of nz per block of size
/// @tparam IndexType
/// @param curr_node
/// @param base_block_size
/// @return
template <typename IndexType>
IndexType get_curr_storage_offset(const IndexType curr_node,
                                  const IndexType base_block_size,
                                  const IndexType num_nodes = 0)
{
    if (num_nodes == 0) {
        return static_cast<IndexType>(
            static_cast<float>(curr_node) *
            (static_cast<float>(base_block_size + 1) / 2));
    } else {  // storage offset backward
        return static_cast<IndexType>(
            static_cast<float>(num_nodes - curr_node) *
            (static_cast<float>(base_block_size + 1) / 2));
    }
}
template <typename IndexType>
constexpr IndexType get_nz_block(const IndexType block_size)
{
    return (block_size * block_size - block_size) / 2 + block_size;
}
// Source: https://joelfilho.com/blog/2020/compile_time_lookup_tables_in_cpp/
template <std::size_t Length, typename Generator, std::size_t... Indexes>
constexpr auto lut_impl(Generator&& f, std::index_sequence<Indexes...>)
{
    using content_type = decltype(f(std::size_t{0}));
    return std::array<content_type, Length>{{f(Indexes)...}};
}

template <std::size_t Length, typename Generator>
constexpr auto lut(Generator&& f)
{
    return lut_impl<Length>(std::forward<Generator>(f),
                            std::make_index_sequence<Length>{});
}
constexpr auto max_block_size = 8;
constexpr auto max_nz_block = get_nz_block(max_block_size);
constexpr unsigned diag(unsigned n)
{
    unsigned result = 0;
    for (unsigned i = 0; i <= n; i++) {
        result += i;
    }
    return result - 1;
}
constexpr auto diag_lut = lut<max_block_size + 1>(diag);
constexpr unsigned sub_block(unsigned n)
{
    unsigned result = 0;
    for (unsigned i = 0; diag_lut[i + 1] <= n && i < max_block_size - 1; i++) {
        result = i;
    }
    unsigned tmp = n - diag_lut[result + 1];
    return tmp == 0 ? result : tmp - 1;
}

constexpr auto sub_block_lut = lut<max_nz_block + 1>(sub_block);
unsigned precomputed_block(unsigned n) { return sub_block_lut[n]; }
unsigned precomputed_diag(unsigned n) { return diag_lut[n + 1]; }


int32 get_id_for_storage(preconditioner::parallel_block* p_block,
                         const int32 row, const int32 col, bool* lvl_1)
{
    if (row<p_block->start_row_global_ | row> p_block->end_row_global_ |
        col<p_block->start_row_global_ | col> p_block->end_row_global_)
        return -1;
    const auto b_block_size = p_block->base_block_size_;
    const auto local_row =
        row - p_block->start_row_global_;  // local row within the p_block
    const auto block_id = local_row / p_block->lvl_1_block_size_;
    const auto row_id_block =
        local_row %
        p_block
            ->lvl_1_block_size_;  // local row within the lvl1/block_agg block
    const auto curr_block = p_block->parallel_blocks_[block_id].get();
    const auto base_offset = curr_block->val_storage_id_;
    const auto nz_per_block = p_block->nz_p_b_block_;
    const auto col_offs = row - col;
    if (p_block->residual_ && block_id == p_block->degree_of_parallelism_ -
                                              1) {  // case base_block_agg
        if (col_offs > row_id_block % b_block_size | col_offs < 0) return -1;
        *lvl_1 = false;
        return base_offset + (row_id_block / b_block_size) * nz_per_block +
               precomputed_diag(row_id_block % b_block_size) - col_offs;

    } else {  // case lvl_1_block
        const auto l2_block_size = p_block->lvl_2_block_size_;
        if (col_offs % l2_block_size != 0 |
            col_offs > l2_block_size * (b_block_size - 1) | col_offs < 0)
            return -1;
        *lvl_1 = true;
        return base_offset +
               precomputed_diag(row_id_block / l2_block_size) * l2_block_size +
               row_id_block % l2_block_size - col_offs;  //* l2_block_size;
    }
}


}  // namespace

template <typename ValueType, typename IndexType>
void fill_with_vals(
    std::shared_ptr<const ReferenceExecutor> exec,
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
    ValueType* u_spmv_vals)
{
    const auto mtx_row_ptrs = system_matrix->get_const_row_ptrs();
    const auto mtx_col_idxs = system_matrix->get_const_col_idxs();
    const auto mtx_vals = system_matrix->get_const_values();
    if (storage_scheme.symm_) {
        GKO_NOT_IMPLEMENTED;
    } else {
        // fill the diagonal blokcs
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

template <typename ValueType, typename IndexType>
void setup_blocks(std::shared_ptr<const ReferenceExecutor> exec,
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
                  IndexType* u_spmv_mtx_col_idxs, ValueType* u_spmv_vals)
{
    const auto mtx_row_ptrs = system_matrix->get_const_row_ptrs();
    const auto mtx_col_idxs = system_matrix->get_const_col_idxs();
    const auto mtx_vals = system_matrix->get_const_values();
    if (storage_scheme.symm_) {
        GKO_NOT_IMPLEMENTED;
    } else {
        auto main_blocks = storage_scheme.forward_solve_;
        const auto num_blocks = storage_scheme.num_blocks_;
        GKO_ASSERT(num_blocks >= 3);

        // fill the first parallel block
        auto first_p_block =
            static_cast<preconditioner::parallel_block*>(main_blocks[0].get());
        for (auto row = first_p_block->start_row_global_;
             row < first_p_block->end_row_global_; row++) {
            const auto mtx_row = permutation_idxs[row];
            const auto mtx_start_id = mtx_row_ptrs[mtx_row];
            const auto mtx_end_id = mtx_row_ptrs[mtx_row + 1];
            const auto nnz_in_row =
                mtx_row_ptrs[mtx_row + 1] - mtx_row_ptrs[mtx_row];

            array<ValueType> tmp_mtx_vals(exec, &mtx_vals[mtx_start_id],
                                          &mtx_vals[mtx_end_id]);
            array<IndexType> tmp_mtx_col_idxs(exec, nnz_in_row);

            thrust::sequence(tmp_mtx_col_idxs.get_data(),
                             tmp_mtx_col_idxs.get_data() + nnz_in_row);

            array<IndexType> tmp_perm(exec, nnz_in_row);

            for (auto j = 0; j < nnz_in_row; j++) {
                tmp_perm.get_data()[j] =
                    inv_permutation_idxs[mtx_col_idxs[mtx_start_id + j]];
            }

            thrust::sort_by_key(tmp_perm.get_data(),
                                tmp_perm.get_data() + nnz_in_row,
                                tmp_mtx_col_idxs.get_data());
            // after that tmp_perm holds the global column idxs in our permuted
            // matrix and tmp_mtx_col_idxs the storage idxs to those values in
            // in the input matrix
            for (auto j = 0; j < nnz_in_row; j++) {
                tmp_mtx_vals.get_data()[j] =
                    mtx_vals[mtx_start_id + tmp_mtx_col_idxs.get_const_data()
                                                [j]];  // wouldn't be necessary
                                                       // with a sort by key
                                                       // over a zip iterator
            }
            auto tmp = 0;
            for (auto k = 0; k < nnz_in_row; k++) {
                if (tmp_perm.get_const_data()[k] <= row) {
                    bool lvl_1 = true;
                    auto id = get_id_for_storage(first_p_block, row,
                                                 tmp_perm.get_const_data()[k],
                                                 &lvl_1);
                    if (id >= 0) {
                        l_diag_rows[id] = row;
                        l_diag_vals[id] =
                            (lvl_1 && row == tmp_perm.get_const_data()[k])
                                ? static_cast<ValueType>(1) /
                                      tmp_mtx_vals.get_const_data()[k]
                                : tmp_mtx_vals.get_const_data()[k];
                        l_diag_mtx_col_idxs[id] =
                            tmp_mtx_col_idxs.get_const_data()[k];
                        tmp++;
                    }
                }
            }
            GKO_ASSERT(tmp > 0);  // at least the diagonal must be filled
        }


        auto next_p_block =
            static_cast<preconditioner::parallel_block*>(main_blocks[2].get());
        auto first_spmv_block =
            static_cast<preconditioner::spmv_block*>(main_blocks[1].get());
        first_spmv_block->update(0, 0, next_p_block->start_row_global_,
                                 next_p_block->end_row_global_, 0,
                                 next_p_block->start_row_global_);

        for (auto i = 1; i < num_blocks; i += 2) {
            //  i even - diagonal(parallel) block, uneven - spmv block
            auto spmv_block =
                static_cast<preconditioner::spmv_block*>(main_blocks[i].get());
            auto parallel_block = static_cast<preconditioner::parallel_block*>(
                main_blocks[i + 1].get());

            GKO_ASSERT(spmv_block->start_row_global_ ==
                       parallel_block->start_row_global_);
            GKO_ASSERT(spmv_block->end_col_global_ ==
                       parallel_block->start_row_global_);

            l_spmv_row_ptrs[spmv_block->row_ptrs_storage_id_] = 0;


            for (auto row = parallel_block->start_row_global_;
                 row < parallel_block->end_row_global_; row++) {
                const auto mtx_row = permutation_idxs[row];
                const auto mtx_start_id = mtx_row_ptrs[mtx_row];
                const auto mtx_end_id = mtx_row_ptrs[mtx_row + 1];
                const auto nnz_in_row =
                    mtx_row_ptrs[mtx_row + 1] - mtx_row_ptrs[mtx_row];

                array<ValueType> tmp_mtx_vals(exec, &mtx_vals[mtx_start_id],
                                              &mtx_vals[mtx_end_id]);
                array<IndexType> tmp_mtx_col_idxs(exec, nnz_in_row);

                thrust::sequence(tmp_mtx_col_idxs.get_data(),
                                 tmp_mtx_col_idxs.get_data() + nnz_in_row);

                array<IndexType> tmp_perm(exec, nnz_in_row);

                for (auto j = 0; j < nnz_in_row; j++) {
                    tmp_perm.get_data()[j] =
                        inv_permutation_idxs[mtx_col_idxs[mtx_start_id + j]];
                }
                thrust::sort_by_key(tmp_perm.get_data(),
                                    tmp_perm.get_data() + nnz_in_row,
                                    tmp_mtx_col_idxs.get_data());
                for (auto j = 0; j < nnz_in_row; j++) {
                    tmp_mtx_vals.get_data()[j] =
                        mtx_vals[mtx_start_id +
                                 tmp_mtx_col_idxs
                                     .get_const_data()[j]];  // wouldn't be
                                                             // necessary with a
                                                             // sort by key over
                                                             // a zip iterator
                }

                const auto id_of_row_in_block =
                    row - spmv_block->start_row_global_;
                const auto curr_id_spmv_row =
                    spmv_block->row_ptrs_storage_id_ + id_of_row_in_block;
                const auto curr_id_spmv_val_col =
                    spmv_block->val_storage_id_ +
                    l_spmv_row_ptrs[curr_id_spmv_row];


                auto nnz_in_spmv_block_row = 0;
                for (auto j = 0;
                     j < nnz_in_row && tmp_perm.get_const_data()[j] <
                                           parallel_block->start_row_global_;
                     j++) {
                    l_spmv_vals[curr_id_spmv_val_col + j] =
                        tmp_mtx_vals.get_const_data()[j];
                    l_spmv_mtx_col_idxs[curr_id_spmv_val_col + j] =
                        tmp_mtx_col_idxs.get_const_data()[j];
                    l_spmv_col_idxs[curr_id_spmv_val_col + j] =
                        tmp_perm.get_const_data()[j];
                    nnz_in_spmv_block_row++;
                }


                l_spmv_row_ptrs[curr_id_spmv_row + 1] =
                    l_spmv_row_ptrs[curr_id_spmv_row] + nnz_in_spmv_block_row;

                // setup of the next spmv_block
                if (row == parallel_block->end_row_global_ - 1 &&
                    i + 3 < num_blocks) {  // last row of the curr block and not
                                           // in the last spmv block
                    const auto next_row_ptrs_id = curr_id_spmv_row + 2;
                    const auto next_val_col_id =
                        curr_id_spmv_val_col + nnz_in_spmv_block_row;
                    auto next_parallel_block =
                        static_cast<preconditioner::parallel_block*>(
                            main_blocks[i + 3].get());
                    auto next_spmv_block =
                        static_cast<preconditioner::spmv_block*>(
                            main_blocks[i + 2].get());
                    next_spmv_block->update(
                        next_row_ptrs_id, next_val_col_id,
                        spmv_block->end_row_global_,  // end of curr ==
                                                      // start of next
                        next_parallel_block->end_row_global_, 0,
                        next_parallel_block->start_row_global_);
                }
                auto tmp = 0;
                for (auto k = nnz_in_spmv_block_row;
                     k < nnz_in_row && tmp_perm.get_const_data()[k] <= row;
                     k++) {
                    bool lvl_1 = true;
                    auto id = get_id_for_storage(parallel_block, row,
                                                 tmp_perm.get_const_data()[k],
                                                 &lvl_1);
                    if (id >= 0) {
                        l_diag_rows[id] = row;
                        l_diag_vals[id] =
                            (lvl_1 && row == tmp_perm.get_const_data()[k])
                                ? static_cast<ValueType>(1) /
                                      tmp_mtx_vals.get_const_data()[k]
                                : tmp_mtx_vals.get_const_data()[k];
                        l_diag_mtx_col_idxs[id] =
                            tmp_mtx_col_idxs.get_const_data()[k];
                        tmp++;
                    }
                }
                GKO_ASSERT(tmp > 0);  // at least the diagonal must be filled
            }
        }
    }
}
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_GAUSS_SEIDEL_SETUP_BLOCKS_KERNEL);

template <typename IndexType>
void get_secondary_ordering(std::shared_ptr<const ReferenceExecutor> exec,
                            IndexType* permutation_idxs,
                            preconditioner::storage_scheme& storage_scheme,
                            const IndexType base_block_size,
                            const IndexType lvl_2_block_size,
                            const IndexType* color_block_ptrs,
                            const IndexType max_color)
{
    using namespace preconditioner;

    GKO_ASSERT(storage_scheme.num_blocks_ == 2 * max_color + 1);

    auto lvl_1_block_size = lvl_2_block_size * base_block_size;
    const auto num_nodes = color_block_ptrs[max_color + 1];
    const bool backward_solve = storage_scheme.symm_;

    for (auto color = 0; color <= max_color; color++) {
        const auto curr_color_offset = color_block_ptrs[color];
        const auto next_color_offset = color_block_ptrs[color + 1];
        const auto nodes_in_curr_color = next_color_offset - curr_color_offset;
        const auto full_lvl_1_blocks_in_curr_color =
            nodes_in_curr_color / lvl_1_block_size;
        const auto base_block_residual =
            (nodes_in_curr_color % lvl_1_block_size) > 0;

        const auto l_p_block_storage_offset =
            get_curr_storage_offset(curr_color_offset, base_block_size);
        auto curr_l_p_block = parallel_block(
            l_p_block_storage_offset, curr_color_offset, next_color_offset,
            full_lvl_1_blocks_in_curr_color + base_block_residual,
            base_block_size, lvl_2_block_size,
            static_cast<bool>(base_block_residual));
        parallel_block curr_u_p_block{};
        if (backward_solve) {
            const auto u_p_block_storage_offset = get_curr_storage_offset(
                next_color_offset, base_block_size, num_nodes);
            auto curr_u_p_block = parallel_block(
                u_p_block_storage_offset, curr_color_offset, next_color_offset,
                full_lvl_1_blocks_in_curr_color + base_block_residual,
                base_block_size, lvl_2_block_size,
                static_cast<bool>(base_block_residual));
        }

        for (auto curr_lvl_1_block_id = 0;
             curr_lvl_1_block_id < full_lvl_1_blocks_in_curr_color;
             curr_lvl_1_block_id++) {
            const auto curr_lvl_1_block_offset =
                curr_lvl_1_block_id * lvl_1_block_size;
            array<IndexType> new_lvl_1_block_ordering(exec, lvl_1_block_size);

            const auto l_lvl1_block_storage_offset = get_curr_storage_offset(
                curr_color_offset + curr_lvl_1_block_offset, base_block_size);

            curr_l_p_block.parallel_blocks_
                .emplace_back(  // at curr_lvl_1_block_id hopefully
                    std::make_shared<lvl_1_block>(lvl_1_block(
                        l_lvl1_block_storage_offset,
                        curr_color_offset + curr_lvl_1_block_offset,
                        curr_color_offset + curr_lvl_1_block_offset +
                            lvl_1_block_size)));
            if (backward_solve) {
                const auto u_lvl1_block_storage_offset =
                    get_curr_storage_offset(curr_color_offset +
                                                curr_lvl_1_block_offset +
                                                lvl_1_block_size,
                                            base_block_size, num_nodes);

                curr_u_p_block.parallel_blocks_.emplace_back(
                    std::make_shared<lvl_1_block>(lvl_1_block(
                        u_lvl1_block_storage_offset,
                        curr_color_offset + curr_lvl_1_block_offset,
                        curr_color_offset + curr_lvl_1_block_offset +
                            lvl_1_block_size)));
            }
            auto lvl_1_reordering = new_lvl_1_block_ordering.get_data();
            for (auto curr_node_base_lvl_block = 0;
                 curr_node_base_lvl_block < base_block_size;
                 curr_node_base_lvl_block++) {
                for (auto curr_node_lvl_2_block = 0;
                     curr_node_lvl_2_block < lvl_2_block_size;
                     curr_node_lvl_2_block++) {
                    const auto curr_id =
                        curr_node_base_lvl_block * lvl_2_block_size +
                        curr_node_lvl_2_block;

                    const auto id_to_swap =
                        curr_color_offset + curr_lvl_1_block_offset +
                        curr_node_lvl_2_block * base_block_size +
                        curr_node_base_lvl_block;

                    lvl_1_reordering[curr_id] = permutation_idxs[id_to_swap];
                }
            }
            auto dest = &(
                permutation_idxs[curr_color_offset + curr_lvl_1_block_offset]);
            const auto source = new_lvl_1_block_ordering.get_const_data();
            auto count = sizeof(IndexType) * lvl_1_block_size;
            std::memcpy(dest, source, count);
        }
        if (base_block_residual) {
            const auto residual_start_row_global =
                curr_color_offset +
                full_lvl_1_blocks_in_curr_color * lvl_1_block_size;
            const auto residual_end_row_global = next_color_offset;
            const auto num_residual_blocks =
                nodes_in_curr_color -
                full_lvl_1_blocks_in_curr_color * lvl_1_block_size;
            curr_l_p_block.parallel_blocks_
                .emplace_back(  // at full_lvl_1_blocks_in_curr_color
                    std::make_shared<base_block_aggregation>(
                        base_block_aggregation(
                            get_curr_storage_offset(residual_start_row_global,
                                                    base_block_size),
                            residual_start_row_global, residual_end_row_global,
                            num_residual_blocks)));
            if (backward_solve) {
                curr_u_p_block.parallel_blocks_.emplace_back(
                    std::make_shared<base_block_aggregation>(
                        base_block_aggregation(
                            get_curr_storage_offset(residual_end_row_global,
                                                    base_block_size, num_nodes),
                            residual_start_row_global, residual_end_row_global,
                            num_residual_blocks)));
            }
        }
        storage_scheme.forward_solve_.emplace_back(
            std::make_shared<parallel_block>(curr_l_p_block));
        storage_scheme.forward_solve_.emplace_back(std::make_shared<spmv_block>(
            spmv_block(next_color_offset, 0, next_color_offset)));
        if (backward_solve) {
            storage_scheme.backward_solve_.emplace(
                storage_scheme.backward_solve_.begin(),
                std::make_shared<parallel_block>(curr_u_p_block));
            if (color < max_color) {
                storage_scheme.backward_solve_.emplace(
                    storage_scheme.backward_solve_.begin(),
                    std::make_shared<spmv_block>(
                        spmv_block(0, 0, curr_color_offset, next_color_offset,
                                   next_color_offset, num_nodes)));
            }
        }
    }
}
GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_GAUSS_SEIDEL_GET_SECONDARY_ORDERING_KERNEL);


}  // namespace gauss_seidel
}  // namespace reference
}  // namespace kernels
}  // namespace gko