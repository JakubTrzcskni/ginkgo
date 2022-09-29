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

#include <cstring>
#include <list>
#include <vector>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/matrix_data.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>
#include <ginkgo/core/reorder/rcm.hpp>

#include "core/base/allocator.hpp"
#include "core/base/utils.hpp"
#include "core/matrix/csr_kernels.hpp"
#include "core/preconditioner/gauss_seidel_kernels.hpp"
#include "core/preconditioner/jacobi_kernels.hpp"
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
GKO_REGISTER_OPERATION(get_block_coloring, gauss_seidel::get_block_coloring);
GKO_REGISTER_OPERATION(assign_to_blocks, gauss_seidel::assign_to_blocks);
GKO_REGISTER_OPERATION(get_permutation_from_coloring,
                       gauss_seidel::get_permutation_from_coloring);
GKO_REGISTER_OPERATION(get_secondary_ordering,
                       gauss_seidel::get_secondary_ordering);
GKO_REGISTER_OPERATION(invert_diagonal, jacobi::invert_diagonal);
GKO_REGISTER_OPERATION(get_degree_of_nodes, gauss_seidel::get_degree_of_nodes);
GKO_REGISTER_OPERATION(invert_permutation, csr::invert_permutation);
GKO_REGISTER_OPERATION(setup_blocks, gauss_seidel::setup_blocks);
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
    const auto exec = this->get_executor()->get_master();
    bool permuted = permutation_idxs_.get_num_elems() > 0;

    if (permuted) {
        auto b_perm = share(
            as<Dense>(as<const Dense>(b)->row_permute(&permutation_idxs_)));
        if (use_HBMC_) {
            std::cout << "apply" << std::endl;
            this->get_executor()->run(gauss_seidel::make_simple_apply(
                l_diag_rows_.get_const_data(), l_diag_vals_.get_const_data(),
                l_spmv_row_ptrs_.get_const_data(),
                l_spmv_col_idxs_.get_const_data(),
                l_spmv_vals_.get_const_data(),
                permutation_idxs_.get_const_data(), hbmc_storage_scheme_,
                lend(b_perm), as<Dense>(x)));
            // }
        } else {
            auto x_perm =
                share(as<Dense>(as<Dense>(x)->row_permute(&permutation_idxs_)));

            if (use_reference_) {
                GKO_NOT_SUPPORTED(this);
            } else {
                const auto block_ptrs = color_ptrs_.get_const_data();
                const auto num_rows = lower_triangular_matrix_->get_size()[0];

                auto tmp_rhs_block = Dense::create(exec);

                for (auto color_block = 0;
                     color_block < color_ptrs_.get_num_elems() - 1;
                     color_block++) {
                    dim_type block_start = block_ptrs[color_block];
                    dim_type block_end = block_ptrs[color_block + 1];
                    dim_type block_size = block_end - block_start;

                    if (color_block == 0) {
                        const auto curr_b_block = Dense::create_const(
                            exec, dim<2>{block_size, b_perm->get_size()[1]},
                            gko::array<ValueType>::const_view(
                                exec, block_size * b_perm->get_size()[1],
                                &(b_perm->get_const_values()
                                      [block_start * b_perm->get_size()[1]])),
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
                            exec,
                            dim<2>{next_block_size, b_perm->get_size()[1]},
                            gko::array<ValueType>::const_view(
                                exec, next_block_size * b_perm->get_size()[1],
                                &(b_perm->get_const_values()
                                      [next_block_start *
                                       b_perm->get_size()[1]])),
                            b_perm->get_size()[1]);

                        auto up_to_curr_x_block = Dense::create(
                            exec, dim<2>{block_end, x_perm->get_size()[1]},
                            gko::make_array_view(
                                exec, block_end * x_perm->get_size()[1],
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
        }
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
    const auto exec = this->get_executor()->get_master();
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

    auto exec = this->get_executor()->get_master();

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

    auto exec = this->get_executor()->get_master();

    auto adjacency_matrix = SparsityMatrix::create(exec);
    adjacency_matrix = get_adjacency_matrix(mat_data, is_symmetric);

    vertex_colors_.fill(IndexType{-1});
    IndexType max_color{0};
    exec->run(gauss_seidel::make_get_coloring(lend(adjacency_matrix),
                                              vertex_colors_, &max_color));

    return max_color;
}

// TODO not finished
template <typename ValueType, typename IndexType>
void GaussSeidel<ValueType, IndexType>::initialize_blocks()
{
    using Diagonal = gko::matrix::Diagonal<ValueType>;
    using dim_type = gko::dim<2>::dimension_type;
    auto exec = this->get_executor()->get_master();
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
    const auto exec = this->get_executor()->get_master();
    GKO_ASSERT_IS_SQUARE_MATRIX(system_matrix);

    auto csr_matrix =
        convert_to_with_sorting<Csr>(exec, system_matrix, skip_sorting);
    const IndexType num_nodes = csr_matrix->get_size()[0];
    matrix_data<ValueType, IndexType> mat_data{csr_matrix->get_size()};
    csr_matrix->write(mat_data);

    if (!symmetric_preconditioner_) {
        if (use_reference_) {
            utils::make_lower_triangular(mat_data);
            lower_triangular_matrix_->read(mat_data);
            lower_trs_ =
                share(lower_trs_factory_->generate(lower_triangular_matrix_));
        } else {
            lower_trs_ = nullptr;
            auto max_color =  // colors start with 0 so there are max_colors + 1
                              // different colors
                get_coloring(mat_data);  // matrix data is made symmetric here
            color_ptrs_.resize_and_reset(max_color + 2);
            permutation_idxs_.resize_and_reset(num_nodes);
            exec->run(gauss_seidel::make_get_permutation_from_coloring(
                num_nodes, vertex_colors_.get_data(), max_color,
                color_ptrs_.get_data(), permutation_idxs_.get_data(),
                static_cast<IndexType*>(nullptr)));
            auto tmp = Csr::create(exec);
            tmp->read(mat_data);
            tmp = as<Csr>(tmp->permute(&permutation_idxs_));
            tmp->write(mat_data);
            utils::make_lower_triangular(mat_data);
            lower_triangular_matrix_->read(mat_data);
            initialize_blocks();
            GKO_ASSERT_EQ(block_ptrs_.size(),
                          2 * color_ptrs_.get_num_elems() - 3);
        }

    } else
        GKO_NOT_IMPLEMENTED;
}

template <typename ValueType, typename IndexType>
void GaussSeidel<ValueType, IndexType>::generate_HBMC(
    std::shared_ptr<const LinOp> system_matrix, bool skip_sorting)
{
    using Csr = matrix::Csr<ValueType, IndexType>;
    auto exec = this->get_executor()->get_master();
    const auto is_gpu_executor = this->get_executor() != exec;

    auto csr_matrix =
        convert_to_with_sorting<Csr>(exec, system_matrix, skip_sorting);

    matrix_data<ValueType, IndexType> mat_data{csr_matrix->get_size()};
    csr_matrix->write(mat_data);
    auto adjacency_matrix =
        matrix::SparsityCsr<ValueType, IndexType>::create(exec);
    adjacency_matrix = get_adjacency_matrix(
        mat_data);  // this assumes that the matrix is not symmetric

    auto block_ordering = generate_block_structure(
        lend(adjacency_matrix), base_block_size_,
        lvl2_block_size_);  // TODO a lot of functionality in this
                            // function, split up needed ?

    if (hbmc_storage_scheme_.symm_) GKO_NOT_IMPLEMENTED;
    ValueType* dummyVal;
    IndexType* dummyInd;

    exec->run(gauss_seidel::make_setup_blocks(
        lend(csr_matrix), permutation_idxs_.get_const_data(),
        inv_permutation_idxs_.get_const_data(), hbmc_storage_scheme_,
        l_diag_rows_.get_data(), l_diag_mtx_col_idxs_.get_data(),
        l_diag_vals_.get_data(), l_spmv_row_ptrs_.get_data(),
        l_spmv_col_idxs_.get_data(), l_spmv_mtx_col_idxs_.get_data(),
        l_spmv_vals_.get_data(), dummyInd, dummyInd, dummyVal, dummyInd,
        dummyInd, dummyInd, dummyVal));

    GKO_ASSERT(hbmc_storage_scheme_.num_blocks_ ==
               hbmc_storage_scheme_.forward_solve_.size());


    const auto d_exec = this->get_executor();
    l_diag_rows_.set_executor(d_exec);
    l_diag_mtx_col_idxs_.set_executor(d_exec);
    l_diag_vals_.set_executor(d_exec);
    l_spmv_row_ptrs_.set_executor(d_exec);
    l_spmv_col_idxs_.set_executor(d_exec);
    l_spmv_mtx_col_idxs_.set_executor(d_exec);
    l_spmv_vals_.set_executor(d_exec);
    permutation_idxs_.set_executor(d_exec);


    // for testing only
    lower_triangular_matrix_->copy_from(
        give(as<Csr>(csr_matrix->permute(&permutation_idxs_))));
}

template <typename ValueType, typename IndexType>
void GaussSeidel<ValueType, IndexType>::reserve_mem_for_block_structure(
    const matrix::SparsityCsr<ValueType, IndexType>* adjacency_matrix,
    const IndexType num_base_blocks, const IndexType base_block_size,
    const IndexType num_colors)
{
    auto exec = this->get_executor()->get_master();
    const auto num_nodes = adjacency_matrix->get_size()[0];
    const auto nnz_triangle = adjacency_matrix->get_num_nonzeros() / 2 +
                              num_nodes;  // should work if matrix is symmetric

    // best case all blocks are dense
    const auto diag_mem_requirement =
        num_base_blocks * base_block_size * base_block_size;

    // worst case all diag blocks are only a diagonal
    const auto l_spmv_val_col_mem_requirement =
        nnz_triangle - num_nodes;  // more memory than needed
    const auto l_spmv_row_mem_requirement =
        num_nodes - color_ptrs_.get_const_data()[1] +
        color_ptrs_.get_num_elems() - 2;  // optimal

    l_diag_rows_.resize_and_reset(diag_mem_requirement);
    l_diag_mtx_col_idxs_.resize_and_reset(diag_mem_requirement);
    l_diag_vals_.resize_and_reset(diag_mem_requirement);
    l_diag_vals_.fill(ValueType{0});
    l_diag_mtx_col_idxs_.fill(IndexType{-1});
    l_diag_rows_.fill(IndexType{-1});
    l_spmv_row_ptrs_.resize_and_reset(l_spmv_row_mem_requirement);
    l_spmv_col_idxs_.resize_and_reset(l_spmv_val_col_mem_requirement);
    l_spmv_mtx_col_idxs_.resize_and_reset(l_spmv_val_col_mem_requirement);
    l_spmv_vals_.resize_and_reset(l_spmv_val_col_mem_requirement);

    const auto num_blocks = num_colors * 2 - 1;

    if (symmetric_preconditioner_) {
        const auto u_spmv_val_col_mem_requirement =
            l_spmv_val_col_mem_requirement;
        const auto u_spmv_row_mem_requirement =
            color_ptrs_.get_const_data()[color_ptrs_.get_num_elems() - 2] +
            color_ptrs_.get_num_elems() - 2;

        u_diag_rows_.resize_and_reset(diag_mem_requirement);
        u_diag_mtx_col_idxs_.resize_and_reset(diag_mem_requirement);
        u_diag_vals_.resize_and_reset(diag_mem_requirement);
        u_diag_vals_.fill(ValueType{0});
        u_diag_mtx_col_idxs_.fill(IndexType{-1});
        u_diag_rows_.fill(IndexType{-1});
        u_spmv_row_ptrs_.resize_and_reset(u_spmv_row_mem_requirement);
        u_spmv_col_idxs_.resize_and_reset(u_spmv_val_col_mem_requirement);
        u_spmv_mtx_col_idxs_.resize_and_reset(u_spmv_val_col_mem_requirement);
        u_spmv_vals_.resize_and_reset(u_spmv_val_col_mem_requirement);
    }

    hbmc_storage_scheme_ =
        storage_scheme(num_blocks, symmetric_preconditioner_);
}

template <typename ValueType, typename IndexType>
array<IndexType> GaussSeidel<ValueType, IndexType>::generate_block_structure(
    const matrix::SparsityCsr<ValueType, IndexType>* adjacency_matrix,
    const IndexType block_size, const IndexType lvl_2_block_size)
{
    auto exec = this->get_executor()->get_master();
    const IndexType num_nodes = adjacency_matrix->get_size()[0];
    const IndexType num_base_blocks = ceildiv(num_nodes, block_size);

    array<IndexType> block_ordering(exec, num_nodes);
    array<IndexType> degrees(exec, num_nodes);
    array<int8> visited(exec, num_nodes);

    exec->run(gauss_seidel::make_get_degree_of_nodes(
        num_nodes, adjacency_matrix->get_const_row_ptrs(), degrees.get_data()));

    std::fill_n(visited.get_data(), num_nodes, int8{0});
    exec->run(gauss_seidel::make_assign_to_blocks(
        adjacency_matrix, block_ordering.get_data(), degrees.get_const_data(),
        visited.get_data(), block_size));

    // TODO move to generate_HBMC / to gs kernels
    IndexType max_color = 0;
    vertex_colors_.fill(IndexType{-1});
    exec->run(gauss_seidel::make_get_block_coloring(
        adjacency_matrix, block_ordering.get_const_data(), block_size,
        vertex_colors_.get_data(), &max_color));

    color_ptrs_.resize_and_reset(max_color + 2);
    permutation_idxs_.resize_and_reset(num_nodes);
    exec->run(gauss_seidel::make_get_permutation_from_coloring(
        num_nodes, vertex_colors_.get_data(), max_color, color_ptrs_.get_data(),
        permutation_idxs_.get_data(), block_ordering.get_const_data()));

    if (lvl_2_block_size > 0) {
        reserve_mem_for_block_structure(adjacency_matrix, num_base_blocks,
                                        block_size, max_color + 1);

        // secondary ordering
        exec->run(gauss_seidel::make_get_secondary_ordering(
            permutation_idxs_.get_data(), hbmc_storage_scheme_, block_size,
            lvl_2_block_size, color_ptrs_.get_const_data(), max_color));

        inv_permutation_idxs_.resize_and_reset(num_nodes);
        exec->run(gauss_seidel::make_invert_permutation(
            num_nodes, permutation_idxs_.get_const_data(),
            inv_permutation_idxs_.get_data()));
    }

    return block_ordering;  // won't be needed at all
    // return storage scheme maybe? or nothing
}

#define GKO_DECLARE_GAUSS_SEIDEL(ValueType, IndexType) \
    class GaussSeidel<ValueType, IndexType>

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_GAUSS_SEIDEL);

}  // namespace preconditioner
}  // namespace gko