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

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/matrix_data.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>

#include "core/base/utils.hpp"
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
GKO_REGISTER_OPERATION(invert_diagonal, jacobi::invert_diagonal);
}  // namespace
}  // namespace gauss_seidel

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
void GaussSeidel<ValueType, IndexType>::apply_impl(const LinOp* b,
                                                   LinOp* x) const
{
    using Dense = matrix::Dense<ValueType>;
    using Csr = matrix::Csr<ValueType, IndexType>;
    using Diagonal = matrix::Diagonal<ValueType>;
    const auto exec = this->get_executor();
    bool permuted = permutation_idxs_.get_num_elems() > 0;

    if (permuted) {
        const auto b_perm = as<const Dense>(
            as<const Dense>(b)->row_permute(&permutation_idxs_));
        auto x_perm = as<Dense>(as<Dense>(x)->row_permute(&permutation_idxs_));

        if (use_reference_) {
            exec->run(gauss_seidel::make_ref_simple_apply(
                lend(lower_trs_), lend(b_perm), lend(x_perm)));
        } else {
            // TODO
            if (b->get_size()[1] > 1) {
                GKO_NOT_IMPLEMENTED;
            }
            const auto col_ptrs = color_ptrs_.get_const_data();
            const auto num_rows = lower_triangular_matrix_->get_size()[0];

            auto tmp_rhs_block = Dense::create(exec);

            for (auto color_block = 0;
                 color_block < color_ptrs_.get_num_elems() - 1; color_block++) {
                auto block_start = col_ptrs[color_block];
                auto block_end = col_ptrs[color_block + 1];
                auto block_size = block_end - block_start;

                if (color_block == 0) {
                    const auto curr_b_block = Dense::create_const(
                        exec, dim<2>{block_size, 1},
                        gko::array<ValueType>::const_view(
                            exec, block_size,
                            &(b_perm->get_const_values()[block_start])),
                        1);
                    tmp_rhs_block->copy_from(lend(curr_b_block));
                }

                // seg fault while debugging???
                // when no variables are being observed -> no errors
                auto curr_x_block = Dense::create(
                    exec, dim<2>{block_size, 1},
                    gko::make_array_view(exec, block_size,
                                         &(x_perm->get_values()[block_start])),
                    1);

                auto curr_diag_block =
                    lower_triangular_matrix_->create_submatrix(
                        gko::span(block_start, block_end),
                        gko::span(block_start, block_end));

                auto diag = gko::make_array_view(exec, block_size,
                                                 curr_diag_block->get_values());

                auto curr_inv_diag_block = Diagonal::create(exec, block_size);
                auto inv_diag_view = gko::make_array_view(
                    exec, block_size, curr_inv_diag_block->get_values());
                exec->run(
                    gauss_seidel::make_invert_diagonal(diag, inv_diag_view));


                curr_inv_diag_block->apply(lend(tmp_rhs_block),
                                           lend(curr_x_block));

                if (block_end < num_rows) {
                    auto next_block_start = col_ptrs[color_block + 1];
                    auto next_block_end = col_ptrs[color_block + 2];
                    auto next_block_size = next_block_end - next_block_start;

                    auto curr_under_diag_block =
                        lower_triangular_matrix_->create_submatrix(
                            gko::span(next_block_start, next_block_end),
                            gko::span(0, next_block_start));

                    const auto next_b_block = Dense::create_const(
                        exec, dim<2>{next_block_size, 1},
                        gko::array<ValueType>::const_view(
                            exec, next_block_size,
                            &(b_perm->get_const_values()[next_block_start])),
                        1);

                    auto up_to_curr_x_block = Dense::create(
                        exec, dim<2>{block_end, 1},
                        gko::make_array_view(exec, block_end,
                                             &(x_perm->get_values()[0])),
                        1);

                    tmp_rhs_block->copy_from(lend(next_b_block));

                    auto one = gko::initialize<Dense>({1.0}, exec);
                    auto neg_one = gko::initialize<Dense>({-1.0}, exec);
                    curr_under_diag_block->apply(
                        lend(neg_one), lend(up_to_curr_x_block), lend(one),
                        lend(tmp_rhs_block));
                }
            }
        }
        as<Dense>(x)->copy_from(std::move(x_perm));
    } else {
        if (use_reference_) {
            exec->run(gauss_seidel::make_ref_simple_apply(
                lend(lower_trs_), as<const Dense>(b), as<Dense>(x)));
        } else {
            // TODO
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
    if (use_reference_) {
        exec->run(gauss_seidel::make_ref_apply(
            lend(lower_trs_), as<const Dense>(alpha), as<const Dense>(b),
            as<const Dense>(beta), as<Dense>(x)));
    } else {
        auto system_matrix = lower_triangular_matrix_;
        exec->run(gauss_seidel::make_apply(
            as<const Csr>(lend(system_matrix)), as<const Dense>(alpha),
            as<const Dense>(b), as<const Dense>(beta), as<Dense>(x)));
    }
}

template <typename ValueType, typename IndexType>
std::unique_ptr<LinOp> GaussSeidel<ValueType, IndexType>::transpose() const
    GKO_NOT_IMPLEMENTED;


template <typename ValueType, typename IndexType>
std::unique_ptr<LinOp> GaussSeidel<ValueType, IndexType>::conj_transpose() const
    GKO_NOT_IMPLEMENTED;

template <typename ValueType, typename IndexType>
IndexType GaussSeidel<ValueType, IndexType>::get_coloring(
    matrix_data<ValueType, IndexType>& mat_data, bool is_symmetric)
{
    using SparsityMatrix = matrix::SparsityCsr<ValueType, IndexType>;
    using MatData = matrix_data<ValueType, IndexType>;

    auto exec = this->get_executor();

    auto tmp = SparsityMatrix::create(exec);

    auto adjacency_matrix = SparsityMatrix::create(exec);

    if (!is_symmetric) {
        utils::make_symmetric(mat_data);
    }
    tmp->read(mat_data);
    adjacency_matrix = std::move(tmp->to_adjacency_matrix());

    vertex_colors_.fill(IndexType{-1});
    IndexType max_color{0};
    exec->run(gauss_seidel::make_get_coloring(lend(adjacency_matrix),
                                              vertex_colors_, &max_color));

    return max_color;
}

template <typename ValueType, typename IndexType>
void GaussSeidel<ValueType, IndexType>::reorder_with_colors(IndexType max_color)
{
    auto num_rows = vertex_colors_.get_num_elems();
    const auto coloring = vertex_colors_.get_const_data();
    permutation_idxs_.resize_and_reset(num_rows);
    auto permutation = permutation_idxs_.get_data();
    auto col_ptrs = color_ptrs_.get_data();
    IndexType tmp{0};

    for (auto color = 0; color <= max_color; color++) {
        for (auto i = 0; i < num_rows; i++) {
            if (i == 0) {
                col_ptrs[color] = tmp;
            }
            if (coloring[i] == color) {
                permutation[tmp] = i;
                tmp++;
            }
        }
    }
    GKO_ASSERT_EQ(tmp, num_rows);
    col_ptrs[max_color + 1] = num_rows;
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
        auto max_color =
            get_coloring(mat_data);  // matrix data is made symmetric here
        color_ptrs_.resize_and_reset(max_color + 2);
        reorder_with_colors(max_color);
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
        }
    } else
        GKO_NOT_IMPLEMENTED;
}

#define GKO_DECLARE_GAUSS_SEIDEL(ValueType, IndexType) \
    class GaussSeidel<ValueType, IndexType>

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_GAUSS_SEIDEL);

}  // namespace preconditioner
}  // namespace gko