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

#include <ginkgo/core/preconditioner/gauss_seidel.hpp>

#include <vector>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/matrix_data.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>
#include <ginkgo/core/reorder/rcm.hpp>

#include "core/base/utils.hpp"
#include "core/preconditioner/gauss_seidel_kernels.hpp"
#include "core/preconditioner/jacobi_kernels.hpp"
#include "core/reorder/rcm_kernels.hpp"
#include "core/utils/matrix_utils.hpp"

namespace gko {
namespace preconditioner {
namespace gauss_seidel {
namespace {
GKO_REGISTER_OPERATION(apply, gauss_seidel::apply);
GKO_REGISTER_OPERATION(ref_apply, gauss_seidel::ref_apply);
GKO_REGISTER_OPERATION(simple_apply, gauss_seidel::simple_apply);
GKO_REGISTER_OPERATION(ref_simple_apply, gauss_seidel::ref_simple_apply);
GKO_REGISTER_OPERATION(get_coloring, gauss_seidel::get_coloring);
GKO_REGISTER_OPERATION(invert_diagonal, jacobi::invert_diagonal);
GKO_REGISTER_OPERATION(get_degree_of_nodes, rcm::get_degree_of_nodes);
}  // namespace
}  // namespace gauss_seidel

// TODO
template <typename ValueType, typename IndexType>
GaussSeidel<ValueType, IndexType>& GaussSeidel<ValueType, IndexType>::operator=(
    const GaussSeidel& other)
{
    if (&other != this) {
        EnableLinOp<GaussSeidel>::operator=(other);
        lower_triangular_matrix_ = other.lower_triangular_matrix_;
        relaxation_factor_ = other.relaxation_factor_;
        parameters_ = other.parameters_;
    }
    return *this;
}


// TODO
template <typename ValueType, typename IndexType>
GaussSeidel<ValueType, IndexType>& GaussSeidel<ValueType, IndexType>::operator=(
    GaussSeidel&& other)
{
    if (&other != this) {
        EnableLinOp<GaussSeidel>::operator=(std::move(other));
        lower_triangular_matrix_ = std::move(other.lower_triangular_matrix_);
        relaxation_factor_ = std::exchange(other.relaxation_factor_, 1.0);
        parameters_ = std::exchange(other.parameters_, parameters_type{});
    }
    return *this;
}


template <typename ValueType, typename IndexType>
GaussSeidel<ValueType, IndexType>::GaussSeidel(const GaussSeidel& other)
    : GaussSeidel{other.get_executor()}
{
    *this = other;
}


template <typename ValueType, typename IndexType>
GaussSeidel<ValueType, IndexType>::GaussSeidel(GaussSeidel&& other)
    : GaussSeidel{other.get_executor()}
{
    *this = std::move(other);
}

template <typename ValueType, typename IndexType>
void GaussSeidel<ValueType, IndexType>::update_system(ValueType* values)
{}

template <typename ValueType, typename IndexType>
void GaussSeidel<ValueType, IndexType>::apply_impl(const LinOp* b,
                                                   LinOp* x) const
{
    using Dense = matrix::Dense<ValueType>;
    using Csr = matrix::Csr<ValueType, IndexType>;
    using Diagonal = matrix::Diagonal<ValueType>;
    using dim_type = gko::dim<2>::dimension_type;
    const auto exec = this->get_executor();
    bool permuted = permutation_idxs_.get_num_elems() > 0;

    if (permuted) {
        const auto b_perm = share(as<const Dense>(
            as<const Dense>(b)->row_permute(&permutation_idxs_)));
        auto x_perm =
            share(as<Dense>(as<Dense>(x)->row_permute(&permutation_idxs_)));

        if (use_reference_) {
            GKO_NOT_SUPPORTED(this);
        } else {
            const auto block_ptrs = color_ptrs_.get_const_data();
            const auto num_rows = lower_triangular_matrix_->get_size()[0];

            auto tmp_rhs_block = Dense::create(exec);

            for (auto color_block = 0;
                 color_block < color_ptrs_.get_num_elems() - 1; color_block++) {
                dim_type block_start = block_ptrs[color_block];
                dim_type block_end = block_ptrs[color_block + 1];
                dim_type block_size = block_end - block_start;

                if (color_block == 0) {
                    const auto curr_b_block = Dense::create_const(
                        exec, dim<2>{block_size, b_perm->get_size()[1]},
                        gko::array<ValueType>::const_view(
                            exec, block_size * b_perm->get_size()[1],
                            &(b_perm
                                  ->get_const_values()[block_start *
                                                       b_perm->get_size()[1]])),
                        b_perm->get_size()[1]);
                    tmp_rhs_block->copy_from(lend(curr_b_block));
                }

                auto curr_x_block = Dense::create(
                    exec, dim<2>{block_size, x_perm->get_size()[1]},
                    gko::make_array_view(
                        exec, block_size * x_perm->get_size()[1],
                        &(x_perm->get_values()[block_start *
                                               x_perm->get_size()[1]])),
                    x_perm->get_size()[1]);

                block_ptrs_[2 * color_block]->apply(lend(tmp_rhs_block),
                                                    lend(curr_x_block));

                if (block_end < num_rows) {
                    dim_type next_block_start = block_ptrs[color_block + 1];
                    dim_type next_block_end = block_ptrs[color_block + 2];
                    dim_type next_block_size =
                        next_block_end - next_block_start;

                    const auto next_b_block = Dense::create_const(
                        exec, dim<2>{next_block_size, b_perm->get_size()[1]},
                        gko::array<ValueType>::const_view(
                            exec, next_block_size * b_perm->get_size()[1],
                            &(b_perm
                                  ->get_const_values()[next_block_start *
                                                       b_perm->get_size()[1]])),
                        b_perm->get_size()[1]);

                    auto up_to_curr_x_block = Dense::create(
                        exec, dim<2>{block_end, x_perm->get_size()[1]},
                        gko::make_array_view(exec,
                                             block_end * x_perm->get_size()[1],
                                             &(x_perm->get_values()[0])),
                        x_perm->get_size()[1]);

                    tmp_rhs_block->copy_from(lend(next_b_block));

                    auto one = gko::initialize<Dense>({1.0}, exec);
                    auto neg_one = gko::initialize<Dense>({-1.0}, exec);
                    block_ptrs_[2 * color_block + 1]->apply(
                        lend(neg_one), lend(up_to_curr_x_block), lend(one),
                        lend(tmp_rhs_block));
                }
            }
        }
        as<Dense>(x)->copy_from(std::move(
            as<Dense>(x_perm->inverse_row_permute(&permutation_idxs_))));
    } else {
        if (use_reference_) {
            exec->run(gauss_seidel::make_ref_simple_apply(
                lend(lower_trs_), as<const Dense>(b), as<Dense>(x)));
        } else {
            GKO_NOT_SUPPORTED(this);
        }
    }
}


template <typename ValueType, typename IndexType>
void GaussSeidel<ValueType, IndexType>::apply_impl(const LinOp* alpha,
                                                   const LinOp* b,
                                                   const LinOp* beta,
                                                   LinOp* x) const
{
    using Dense = matrix::Dense<ValueType>;
    using Csr = matrix::Csr<ValueType, IndexType>;
    const auto exec = this->get_executor();
    bool permuted = permutation_idxs_.get_num_elems() > 0;

    if (use_reference_) {
        if (permuted) {
            GKO_NOT_SUPPORTED(this);
        } else {
            exec->run(gauss_seidel::make_ref_apply(
                lend(lower_trs_), as<const Dense>(alpha), as<const Dense>(b),
                as<const Dense>(beta), as<Dense>(x)));
        }
    } else {
        // TODO
        auto alpha_b = Dense::create(exec, b->get_size());
        as<const Dense>(b)->apply(as<const Dense>(alpha), lend(alpha_b));

        auto x_copy = x->clone();

        this->apply_impl(as<const LinOp>(lend(alpha_b)), x);

        as<Dense>(x)->add_scaled(beta, lend(x_copy));
    }
}

template <typename ValueType, typename IndexType>
std::unique_ptr<LinOp> GaussSeidel<ValueType, IndexType>::transpose() const
    GKO_NOT_IMPLEMENTED;


template <typename ValueType, typename IndexType>
std::unique_ptr<LinOp> GaussSeidel<ValueType, IndexType>::conj_transpose() const
    GKO_NOT_IMPLEMENTED;

template <typename ValueType, typename IndexType>
std::unique_ptr<matrix::SparsityCsr<ValueType, IndexType>>
GaussSeidel<ValueType, IndexType>::get_adjacency_matrix(
    matrix_data<ValueType, IndexType>& mat_data, bool is_symmetric)
{
    using SparsityMatrix = matrix::SparsityCsr<ValueType, IndexType>;
    using MatData = matrix_data<ValueType, IndexType>;

    auto exec = this->get_executor();

    auto tmp = SparsityMatrix::create(exec);

    if (!is_symmetric) {
        utils::make_symmetric(mat_data);
    }
    tmp->read(mat_data);

    return give(tmp->to_adjacency_matrix());
}

template <typename ValueType, typename IndexType>
IndexType GaussSeidel<ValueType, IndexType>::get_coloring(
    matrix_data<ValueType, IndexType>& mat_data, bool is_symmetric)
{
    using SparsityMatrix = matrix::SparsityCsr<ValueType, IndexType>;
    using MatData = matrix_data<ValueType, IndexType>;

    auto exec = this->get_executor();

    auto adjacency_matrix = SparsityMatrix::create(exec);
    adjacency_matrix = get_adjacency_matrix(mat_data, is_symmetric);

    vertex_colors_.fill(IndexType{-1});
    IndexType max_color{0};
    exec->run(gauss_seidel::make_get_coloring(lend(adjacency_matrix),
                                              vertex_colors_, &max_color));

    return max_color;
}

template <typename ValueType, typename IndexType>
void GaussSeidel<ValueType, IndexType>::compute_permutation_idxs(
    IndexType max_color)
{
    auto num_rows = vertex_colors_.get_num_elems();
    const auto coloring = vertex_colors_.get_const_data();
    permutation_idxs_.resize_and_reset(num_rows);
    auto permutation = permutation_idxs_.get_data();
    auto block_ptrs = color_ptrs_.get_data();
    IndexType tmp{0};

    for (auto color = 0; color <= max_color; color++) {
        for (auto i = 0; i < num_rows; i++) {
            if (i == 0) {
                block_ptrs[color] = tmp;
            }
            if (coloring[i] == color) {
                permutation[tmp] = i;
                tmp++;
            }
        }
    }
    GKO_ASSERT_EQ(tmp, num_rows);
    block_ptrs[max_color + 1] = num_rows;
}

// TODO not finished
template <typename ValueType, typename IndexType>
void GaussSeidel<ValueType, IndexType>::initialize_blocks()
{
    using Diagonal = gko::matrix::Diagonal<ValueType>;
    using dim_type = gko::dim<2>::dimension_type;
    auto exec = this->get_executor();
    const auto block_ptrs = color_ptrs_.get_const_data();
    const auto num_rows = lower_triangular_matrix_->get_size()[0];
    const auto num_of_colors = color_ptrs_.get_num_elems() - 1;

    for (auto block_row = 0; block_row < num_of_colors; block_row++) {
        dim_type block_start = block_ptrs[block_row];
        dim_type block_end = block_ptrs[block_row + 1];
        dim_type block_size = block_end - block_start;


        auto curr_diag_block = lower_triangular_matrix_->create_submatrix(
            gko::span(block_start, block_end),
            gko::span(block_start, block_end));

        auto diag = gko::make_array_view(exec, block_size,
                                         curr_diag_block->get_values());

        auto curr_inv_diag_block = Diagonal::create(exec, block_size);
        auto inv_diag_view = gko::make_array_view(
            exec, block_size, curr_inv_diag_block->get_values());
        exec->run(gauss_seidel::make_invert_diagonal(diag, inv_diag_view));

        // TODO
        block_ptrs_.push_back(as<LinOp>(give(curr_inv_diag_block)));

        if (block_row < num_of_colors - 1) {
            dim_type next_block_start = block_end;
            dim_type next_block_end = block_ptrs[block_row + 2];
            auto curr_under_diag_block =
                lower_triangular_matrix_->create_submatrix(
                    gko::span(next_block_start, next_block_end),
                    gko::span(0, next_block_start));

            block_ptrs_.push_back(as<LinOp>(give(curr_under_diag_block)));
        }
    }
}

template <typename ValueType, typename IndexType>
void GaussSeidel<ValueType, IndexType>::generate(
    std::shared_ptr<const LinOp> system_matrix, bool skip_sorting)
{
    using Csr = matrix::Csr<ValueType, IndexType>;
    const auto exec = this->get_executor();
    GKO_ASSERT_IS_SQUARE_MATRIX(system_matrix);

    auto csr_matrix =
        convert_to_with_sorting<Csr>(exec, system_matrix, skip_sorting);

    matrix_data<ValueType, IndexType> mat_data{csr_matrix->get_size()};
    csr_matrix->write(mat_data);

    if (use_coloring_) {
        auto max_color =  // colors start with 0 so there are max_colors + 1
                          // different colors
            get_coloring(mat_data);  // matrix data is made symmetric here
        color_ptrs_.resize_and_reset(max_color + 2);
        // block_ptrs_.resize_and_reset(
        //     (max_color + 1) +
        //     max_color);  // num_of_colors (=max_color +1) for the diagonal
        // blocks and num_of_colors (=max_color) for the blocks
        // under the diagonal
        compute_permutation_idxs(max_color);
    }

    if (!symmetric_preconditioner_) {
        if (permutation_idxs_.get_num_elems() > 0) {
            auto tmp = Csr::create(exec);
            tmp->read(mat_data);
            tmp = as<Csr>(tmp->permute(&permutation_idxs_));
            tmp->write(mat_data);
        }
        utils::make_lower_triangular(mat_data);
        lower_triangular_matrix_->read(mat_data);
        if (use_reference_) {
            lower_trs_ =
                share(lower_trs_factory_->generate(lower_triangular_matrix_));
        } else {
            lower_trs_ = nullptr;
            initialize_blocks();
            GKO_ASSERT_EQ(block_ptrs_.size(),
                          2 * color_ptrs_.get_num_elems() - 3);
        }
    } else
        GKO_NOT_IMPLEMENTED;
}

template <typename ValueType, typename IndexType>
void GaussSeidel<ValueType, IndexType>::generate_RACE(
    std::shared_ptr<const LinOp> system_matrix, bool skip_sorting)
{
    using Csr = matrix::Csr<ValueType, IndexType>;
    auto exec = this->get_executor();
    auto csr_matrix =
        convert_to_with_sorting<Csr>(exec, system_matrix, skip_sorting);

    matrix_data<ValueType, IndexType> mat_data{csr_matrix->get_size()};
    csr_matrix->write(mat_data);
    auto adjacency_matrix =
        matrix::SparsityCsr<ValueType, IndexType>::create(exec);
    adjacency_matrix = get_adjacency_matrix(mat_data);


    // generate levels
    // Algorithm 3 from
    // https://arxiv.org/pdf/1907.06487.pdf
    // find starting node as in reorder/rcm
    IndexType num_rows = adjacency_matrix->get_size()[0];
    array<IndexType> degrees(exec, num_rows);
    exec->run(gauss_seidel::make_get_degree_of_nodes(
        num_rows, adjacency_matrix->get_const_row_ptrs(), degrees.get_data()));

    IndexType index_min_node = 0;
    IndexType min_node_degree = std::numeric_limits<IndexType>::max();
    for (IndexType i = 0; i < num_rows; ++i) {
        if (degrees.get_data()[i] < min_node_degree) {
            index_min_node = i;
            min_node_degree = degrees.get_data()[i];
        }
    }
    IndexType* row_ptrs = adjacency_matrix->get_row_ptrs();
    IndexType* col_idxs = adjacency_matrix->get_col_idxs();
    auto root = index_min_node;  // starting node;
    bool marked_all = false;
    array<IndexType> distFromRoot(exec, num_rows);
    distFromRoot.fill(-1);
    std::vector<IndexType> curr_children;
    curr_children.push_back(root);
    permutation_idxs_.resize_and_reset(num_rows);

    level_ptrs_.push_back(0);
    auto curr_lvl = 0;
    auto curr_node = 0;
    while (!marked_all) {
        marked_all = true;
        std::vector<IndexType> next_children;
        for (auto i = 0; i < curr_children.size(); i++) {
            if (distFromRoot.get_data()[curr_children.at(i)] == -1) {
                distFromRoot.get_data()[curr_children.at(i)] = curr_lvl;
                permutation_idxs_.get_data()[curr_node] = curr_children.at(i);
                curr_node++;
                // TODO not sure if correct i.e., if equivalent to "for j in
                // graph[curr children[i]].children do"
                for (auto j = row_ptrs[curr_children.at(i)];
                     j < row_ptrs[curr_children.at(i) + 1]; j++) {
                    if (distFromRoot.get_data()[col_idxs[j]] == -1) {
                        next_children.push_back(col_idxs[j]);
                    }
                }
            }
        }
        level_ptrs_.push_back(curr_node);
        curr_children = next_children;
        if (!curr_children.empty()) marked_all = false;
        curr_lvl++;
    }
    GKO_ASSERT(curr_node == num_rows);
    GKO_ASSERT(curr_lvl + 1 == level_ptrs_.size());


    // load-balancing
}

//TODO 
//there must be already a fast parallel version of this?(or something similar)
template <typename IndexType>
IndexType get_id_min_node(IndexType* degrees, size_t num_nodes, bool* visited){
IndexType index_min_node = 0;
    IndexType min_node_degree = std::numeric_limits<IndexType>::max();
    for (size_t i = 0; i < num_nodes; ++i) {
        if (degrees[i] < min_node_degree && visited[i] == false) {
            visited[i] = true;
            index_min_node = i;
            min_node_degree = degrees[i];
        }
    }
    return index_min_node;
}

template <typename ValueType, typename IndexType>
void GaussSeidel<ValueType, IndexType>::generate_block_structure(matrix::SparsityCsr<ValueType, IndexType>* adjacency_matrix, size_t block_size, size_t lvl_2_block_size){
    
    auto exec = this->get_executor()->get_master();
    auto num_nodes = adjacency_matrix->get_size()[0];
    auto num_base_blocks = ceildiv(num_nodes, block_size);
    array<std::vector<IndexType>> block_pointers(exec, num_base_blocks);
    array<IndexType> degrees(exec, num_nodes);
    array<bool> visited(exec, num_nodes);
    visited->fill(false);
    exec->run(gauss_seidel::make_get_degree_of_nodes(
        num_nodes, adjacency_matrix->get_const_row_ptrs(), degrees.get_data()));
    
    //loop over the blocks
    for(auto k = 0; k < num_base_blocks; k++){
        std::vector<IndexType> candidates;
        auto curr_block = block_pointers.get_data()[k];
        for(auto i = 0; i < block_size; i++){
            if(candidates.empty()){
                auto seed_node_id = get_id_min_node(degrees.get_data(), num_nodes, visited.get_data());
            }else{
                
            }
        }
    }
    

}

#define GKO_DECLARE_GAUSS_SEIDEL(ValueType, IndexType) \
    class GaussSeidel<ValueType, IndexType>

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_GAUSS_SEIDEL);

}  // namespace preconditioner
}  // namespace gko