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

namespace gko {
namespace preconditioner {
namespace gauss_seidel {
GKO_REGISTER_OPERATION(initialize_w_scaling_l,
                       factorization::initialize_w_scaling_l);
GKO_REGISTER_OPERATION(initialize_w_scaling_l_u,
                       factorization::initialize_w_scaling_l_u);
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
        solver_comp_ = other.solver_comp_;
        parameters_ = other.parameters_;
        if (solver_comp_ && other.solver_comp_->get_executor() != exec) {
            solver_comp_ = gko::clone(exec, solver_comp_);
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
        solver_comp_ = std::move(other.solver_comp_);
        parameters_ = std::exchange(other.parameters_, parameters_type{});
        if (solver_comp_ && other.solver_comp_->get_executor() != exec) {
            solver_comp_ = gko::clone(exec, solver_comp_);
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

        // Init the L factor
        std::shared_ptr<Csr> l_factor = Csr::create(
            exec, mat_size, array<ValueType>{exec, l_nnz},
            array<IndexType>{exec, l_nnz}, std::move(l_row_ptrs), mat_strategy);

        // fill the L factor with col_idxs & values
        // values not on the diagonal are scaled with the relaxation factor,
        exec->run(gauss_seidel::make_initialize_w_scaling_l(
            csr_matrix.get(), l_factor.get(), false, relaxation_factor_));

        // create the solver
        auto lower_trs = share(solver::LowerTrs<ValueType, IndexType>::build()
                                   .with_num_rhs(parameters_.num_rhs)
                                   .with_algorithm(parameters_.algorithm)
                                   .on(exec)
                                   ->generate(l_factor));
        solver_comp_ = Composition<ValueType>::create(lower_trs);
    } else {
        // extract the diagonal to scale the lower factor on initialization
        auto diag = share(csr_matrix->extract_diagonal());

        // init row pointers
        array<IndexType> l_row_ptrs{exec, num_rows + 1};
        array<IndexType> u_row_ptrs{exec, num_rows + 1};
        exec->run(gauss_seidel::make_initialize_row_ptrs_l_u(
            csr_matrix.get(), l_row_ptrs.get_data(), u_row_ptrs.get_data()));

        // Get nnz from device memory
        auto l_nnz = static_cast<size_type>(
            exec->copy_val_to_host(l_row_ptrs.get_data() + num_rows));
        auto u_nnz = static_cast<size_type>(
            exec->copy_val_to_host(u_row_ptrs.get_data() + num_rows));

        // Init L and U factors
        std::shared_ptr<Csr> l_factor = Csr::create(
            exec, mat_size, array<ValueType>{exec, l_nnz},
            array<IndexType>{exec, l_nnz}, std::move(l_row_ptrs), mat_strategy);
        std::shared_ptr<Csr> u_factor = Csr::create(
            exec, mat_size, array<ValueType>{exec, u_nnz},
            array<IndexType>{exec, u_nnz}, std::move(u_row_ptrs), mat_strategy);

        // fill the L and U factors with col_idxs & values
        // values not on the diagonal are scaled with the relaxation factor,
        // values of the strict lower factor are column scaled with the inverse
        // of the diagonal
        exec->run(gauss_seidel::make_initialize_w_scaling_l_u(
            csr_matrix.get(), l_factor.get(), u_factor.get(), diag.get(),
            relaxation_factor_));

        auto lower_trs = share(solver::LowerTrs<ValueType, IndexType>::build()
                                   .with_num_rhs(parameters_.num_rhs)
                                   .with_algorithm(parameters_.algorithm)
                                   .with_unit_diagonal(true)
                                   .on(exec)
                                   ->generate(l_factor));
        auto upper_trs = share(solver::UpperTrs<ValueType, IndexType>::build()
                                   .with_num_rhs(parameters_.num_rhs)
                                   .with_algorithm(parameters_.algorithm)
                                   .on(exec)
                                   ->generate(u_factor));

        solver_comp_ = Composition<ValueType>::create(upper_trs, lower_trs);
    }
}

template <typename ValueType, typename IndexType>
void GaussSeidel<ValueType, IndexType>::apply_impl(const LinOp* b,
                                                   LinOp* x) const
{
    precision_dispatch_real_complex<ValueType>(
        [this](auto dense_b, auto dense_x) {
            solver_comp_->apply(dense_b, dense_x);
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
            solver_comp_->apply(dense_alpha, dense_b, dense_beta, dense_x);
        },
        alpha, b, beta, x);
}

#define GKO_DECLARE_GAUSS_SEIDEL(ValueType, IndexType) \
    class GaussSeidel<ValueType, IndexType>

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_GAUSS_SEIDEL);

}  // namespace preconditioner
}  // namespace gko