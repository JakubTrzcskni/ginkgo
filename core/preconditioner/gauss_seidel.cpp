/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
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

#include <memory>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/precision_dispatch.hpp>

#include "core/base/utils.hpp"
#include "core/factorization/factorization_kernels.hpp"
#include "core/preconditioner/jacobi_kernels.hpp"

namespace gko {
namespace preconditioner {
namespace gauss_seidel {
GKO_REGISTER_OPERATION(invert_diagonal, jacobi::invert_diagonal);
GKO_REGISTER_OPERATION(initialize_l, factorization::initialize_l);
GKO_REGISTER_OPERATION(initialize_l_u, factorization::initialize_l_u);
GKO_REGISTER_OPERATION(initialize_row_ptrs_l,
                       factorization::initialize_row_ptrs_l);
GKO_REGISTER_OPERATION(initialize_row_ptrs_l_u,
                       factorization::initialize_row_ptrs_l_u);
namespace {}
}  // namespace gauss_seidel


template <typename ValueType, typename IndexType>
GaussSeidel<ValueType, IndexType>& GaussSeidel<ValueType, IndexType>::operator=(
    const GaussSeidel& other)
{
    if (&other != this) {
        EnableLinOp<GaussSeidel>::operator=(other);
        auto exec = this->get_executor();
        lower_trs_ = other.lower_trs_;
        upper_trs_ = other.upper_trs_;
        parameters_ = other.parameters_;
        if (lower_trs_ && other.lower_trs_->get_executor() != exec) {
            lower_trs_ = gko::clone(exec, lower_trs_);
        }
        if (upper_trs_ && other.upper_trs_->get_executor() != exec) {
            upper_trs_ = gko::clone(exec, upper_trs_);
        }
    }
    return *this;
}


template <typename ValueType, typename IndexType>
GaussSeidel<ValueType, IndexType>& GaussSeidel<ValueType, IndexType>::operator=(
    GaussSeidel&& other)
{
    if (&other != this) {
        EnableLinOp<GaussSeidel>::operator=(std::move(other));
        auto exec = this->get_executor();
        lower_trs_ = std::move(other.lower_trs_);
        upper_trs_ = std::move(other.upper_trs_);
        parameters_ = std::exchange(other.parameters_, parameters_type{});
        if (lower_trs_ && other.lower_trs_->get_executor() != exec) {
            lower_trs_ = gko::clone(exec, lower_trs_);
        }
        if (upper_trs_ && other.upper_trs_->get_executor() != exec) {
            upper_trs_ = gko::clone(exec, upper_trs_);
        }
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
std::unique_ptr<LinOp> GaussSeidel<ValueType, IndexType>::transpose() const
    GKO_NOT_IMPLEMENTED;


template <typename ValueType, typename IndexType>
std::unique_ptr<LinOp> GaussSeidel<ValueType, IndexType>::conj_transpose() const
    GKO_NOT_IMPLEMENTED;

template <typename ValueType, typename IndexType>
void GaussSeidel<ValueType, IndexType>::generate(
    std::shared_ptr<const LinOp> system_matrix, bool skip_sorting)
{
    using Csr = matrix::Csr<ValueType, IndexType>;
    const auto exec = this->get_executor();
    GKO_ASSERT_IS_SQUARE_MATRIX(system_matrix);

    auto csr_matrix =
        convert_to_with_sorting<Csr>(exec, system_matrix, skip_sorting);

    const auto mat_size = csr_matrix->get_size();
    const auto num_rows = mat_size[0];
    auto mat_strategy = csr_matrix->get_strategy();
    if (!symmetric_preconditioner_) {
        // init row pointers
        array<IndexType> l_row_ptrs{exec, num_rows + 1};
        exec->run(gauss_seidel::make_initialize_row_ptrs_l(
            csr_matrix.get(), l_row_ptrs.get_data()));

        // Get nnz from device memory
        auto l_nnz = static_cast<size_type>(
            exec->copy_val_to_host(l_row_ptrs.get_data() + num_rows));

        // Init arrays
        array<IndexType> l_col_idxs{exec, l_nnz};
        array<ValueType> l_vals{exec, l_nnz};

        std::shared_ptr<Csr> l_factor = Csr::create(
            exec, mat_size, std::move(l_vals), std::move(l_col_idxs),
            std::move(l_row_ptrs), mat_strategy);

        exec->run(gauss_seidel::make_initialize_l(csr_matrix.get(),
                                                  l_factor.get(), false));

        lower_trs_ = share(solver::LowerTrs<ValueType, IndexType>::build()
                               .with_num_rhs(parameters_.num_rhs)
                               .with_algorithm(parameters_.algorithm)
                               .on(exec)
                               ->generate(l_factor));
    } else {
        // TODO
        auto diag = share(csr_matrix->extract_diagonal());
        auto inv_diag = diag->clone();
        auto diag_view = make_array_view(exec, mat_size[0], diag->get_values());
        auto inv_diag_view =
            make_array_view(exec, mat_size[0], inv_diag->get_values());

        exec->run(gauss_seidel::make_invert_diagonal(diag_view, inv_diag_view));

        array<IndexType> l_row_ptrs{exec, num_rows + 1};
        array<IndexType> u_row_ptrs{exec, num_rows + 1};
        exec->run(gauss_seidel::make_initialize_row_ptrs_l_u(
            csr_matrix.get(), l_row_ptrs.get_data(), u_row_ptrs.get_data()));

        auto l_nnz = static_cast<size_type>(
            exec->copy_val_to_host(l_row_ptrs.get_data() + num_rows));
        auto u_nnz = static_cast<size_type>(
            exec->copy_val_to_host(u_row_ptrs.get_data() + num_rows));

        // Init arrays
        array<IndexType> l_col_idxs{exec, l_nnz};
        array<ValueType> l_vals{exec, l_nnz};
        std::shared_ptr<Csr> l_factor = Csr::create(
            exec, mat_size, std::move(l_vals), std::move(l_col_idxs),
            std::move(l_row_ptrs), mat_strategy);
        array<IndexType> u_col_idxs{exec, u_nnz};
        array<ValueType> u_vals{exec, u_nnz};
        std::shared_ptr<Csr> u_factor = Csr::create(
            exec, mat_size, std::move(u_vals), std::move(u_col_idxs),
            std::move(u_row_ptrs), mat_strategy);

        // Separate L and U: columns and values
        exec->run(gauss_seidel::make_initialize_l_u(
            csr_matrix.get(), l_factor.get(), u_factor.get(),
            relaxation_factor_));

        inv_diag->apply(l_factor, l_factor);

        lower_trs_ = share(solver::LowerTrs<ValueType, IndexType>::build()
                               .with_num_rhs(parameters_.num_rhs)
                               .with_algorithm(parameters_.algorithm)
                               .with_unit_diagonal(true)
                               .on(exec)
                               ->generate(l_factor));
        upper_trs_ = share(solver::UpperTrs<ValueType, IndexType>::build()
                               .with_num_rhs(parameters_.num_rhs)
                               .with_algorithm(parameters_.algorithm)
                               .on(exec)
                               ->generate(u_factor));
    }
}

template <typename ValueType, typename IndexType>
void GaussSeidel<ValueType, IndexType>::apply_impl(const LinOp* b,
                                                   LinOp* x) const
{
    precision_dispatch_real_complex<ValueType>(
        [this](auto dense_b, auto dense_x) {
            if (!symmetric_preconditioner_)
                lower_trs_->apply(dense_b, dense_x);
            else {
                lower_trs_->apply(dense_b, dense_x);
                upper_trs_->apply(dense_x, dense_x);
            }
        },
        b, x);
}

template <typename ValueType, typename IndexType>
void GaussSeidel<ValueType, IndexType>::apply_impl(const LinOp* alpha,
                                                   const LinOp* b,
                                                   const LinOp* beta,
                                                   LinOp* x) const
{
    precision_dispatch_real_complex<ValueType>(
        [this](auto dense_alpha, auto dense_b, auto dense_beta, auto dense_x) {
            if (!symmetric_preconditioner_) {
                lower_trs_->apply(dense_alpha, dense_b, dense_beta, dense_x);
            } else {
                auto x_clone = dense_x->clone();
                this->apply_impl(dense_b, x_clone.get());
                dense_x->scale(dense_beta);
                dense_x->add_scaled(dense_alpha, x_clone);
            }
        },
        alpha, b, beta, x);
}

#define GKO_DECLARE_GAUSS_SEIDEL(ValueType, IndexType) \
    class GaussSeidel<ValueType, IndexType>

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_GAUSS_SEIDEL);

}  // namespace preconditioner
}  // namespace gko