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

template <typename ValueType, typename IndexType>
void assign_to_blocks(
    std::shared_ptr<const ReferenceExecutor> exec,
    const matrix::SparsityCsr<ValueType, IndexType>* adjacency_matrix,
    IndexType* block_ordering, const IndexType* degrees, int8* visited,
    const IndexType block_size, const IndexType lvl_2_block_size)
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
    const IndexType* coloring, const IndexType max_color, IndexType* color_ptrs,
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

template <typename IndexType>
void get_secondary_ordering(std::shared_ptr<const ReferenceExecutor> exec,
                            IndexType* block_ordering,
                            const IndexType base_block_size,
                            const IndexType lvl_2_block_size,
                            const IndexType* color_block_ptrs,
                            const IndexType max_color)
{
    auto lvl_1_block_size = lvl_2_block_size * base_block_size;
    for (auto color = 0; color <= max_color; color++) {
        const auto nodes_in_curr_color =
            color_block_ptrs[color + 1] - color_block_ptrs[color];
        const auto curr_color_offset = color_block_ptrs[color];
        const auto full_lvl_1_blocks_in_curr_color =
            nodes_in_curr_color / lvl_1_block_size;
        for (auto curr_lvl_1_block_id = 0;
             curr_lvl_1_block_id < full_lvl_1_blocks_in_curr_color;
             curr_lvl_1_block_id++) {
            const auto curr_lvl_1_block_offset =
                curr_lvl_1_block_id * lvl_1_block_size;
            array<IndexType> new_lvl_1_block_ordering(exec, lvl_1_block_size);
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

                    lvl_1_reordering[curr_id] = block_ordering[id_to_swap];
                }
            }
            auto dest =
                &(block_ordering[curr_color_offset + curr_lvl_1_block_offset]);
            const auto source = new_lvl_1_block_ordering.get_const_data();
            auto count = sizeof(IndexType) * lvl_1_block_size;
            std::memcpy(dest, source, count);
        }
    }
}
GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_GAUSS_SEIDEL_GET_SECONDARY_ORDERING_KERNEL);


}  // namespace gauss_seidel
}  // namespace reference
}  // namespace kernels
}  // namespace gko