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

#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/preconditioner/gauss_seidel.hpp>

namespace gko {
namespace preconditioner {
namespace gauss_seidel {
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
        convert_to_with_sorting<Csr>(host_exec_, system_matrix, skip_sorting);
    if (!symmetric_preconditioner_) {
        matrix_data<ValueType, IndexType> mat_data{csr_matrix->get_size()};
        csr_matrix->write(mat_data);
        utils::make_lower_triangular(mat_data);
        auto lower_triangular_matrix = Csr::create(exec);
        lower_triangular_matrix->read(mat_data);
        lower_trs_ = share(solver::LowerTrs<ValueType, IndexType>::build()
                               .with_num_rhs(parameters_.num_rhs)
                               .with_algorithm(parameters_.algorithm)
                               .on(exec)
                               ->generate(lower_triangular_matrix));
    } else {
        GKO_NOT_IMPLEMENTED;
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
            else
                GKO_NOT_IMPLEMENTED;
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
            if (symmetric_preconditioner_) {
                lower_trs_->apply(dense_alpha, dense_b, dense_beta, dense_x);
            } else {
                GKO_NOT_IMPLEMENTED;
            }
        },
        alpha, b, beta, x);
}

#define GKO_DECLARE_GAUSS_SEIDEL(ValueType, IndexType) \
    class GaussSeidel<ValueType, IndexType>

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_GAUSS_SEIDEL);

}  // namespace preconditioner
}  // namespace gko