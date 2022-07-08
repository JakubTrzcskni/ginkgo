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

#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>

#include "core/base/utils.hpp"
#include "core/preconditioner/gauss_seidel_kernels.hpp"

namespace gko {
namespace preconditioner {
namespace gauss_seidel {
namespace {
// names of the implemented kernels, they will be run exclusively from here
GKO_REGISTER_OPERATION(apply, gauss_seidel::apply);
GKO_REGISTER_OPERATION(simple_apply, gauss_seidel::simple_apply);

// GKO_REGISTER_OPERATION(get_permutation, gauss_seidel::get_permutation);
}  // namespace
}  // namespace gauss_seidel


template <typename ValueType, typename IndexType>
GaussSeidel<ValueType, IndexType>& GaussSeidel<ValueType, IndexType>::operator=(
    const GaussSeidel& other)
{
    if (&other != this) {
        EnableLinOp<GaussSeidel>::operator=(other);
        system_matrix_ = other.system_matrix_;
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
        system_matrix_ = std::move(other.system_matrix_);
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
    this->get_executor()->run(gauss_seidel::make_simple_apply(
        lend(this->system_matrix_), dynamic_cast<const Dense*>(b),
        dynamic_cast<Dense*>(x)));
}

template <typename ValueType, typename IndexType>
void GaussSeidel<ValueType, IndexType>::apply_impl(const LinOp* alpha,
                                                   const LinOp* b,
                                                   const LinOp* beta,
                                                   LinOp* x) const
{
    using Dense = matrix::Dense<ValueType>;
    this->get_executor()->run(gauss_seidel::make_apply(
        lend(this->system_matrix_), dynamic_cast<const Dense*>(alpha),
        dynamic_cast<const Dense*>(b), dynamic_cast<const Dense*>(beta),
        dynamic_cast<Dense*>(x)));
}

template <typename ValueType, typename IndexType>
std::unique_ptr<LinOp> GaussSeidel<ValueType, IndexType>::transpose() const
    GKO_NOT_IMPLEMENTED;


template <typename ValueType, typename IndexType>
std::unique_ptr<LinOp> GaussSeidel<ValueType, IndexType>::conj_transpose() const
    GKO_NOT_IMPLEMENTED;

template <typename ValueType, typename IndexType>
void GaussSeidel<ValueType, IndexType>::generate(const LinOp* system_matrix,
                                                 bool skip_sorting)
{
    GKO_ASSERT_IS_SQUARE_MATRIX(system_matrix);
    using Csr = matrix::Csr<ValueType, IndexType>;
    const auto exec = this->get_executor();
    // TODO how to initialise the system matrix?
    // this->system_matrix_ = std::move(
    //     convert_to_with_sorting<Csr>(exec, system_matrix, skip_sorting));
}

#define GKO_DECLARE_GAUSS_SEIDEL(ValueType, IndexType) \
    class GaussSeidel<ValueType, IndexType>

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_GAUSS_SEIDEL);

}  // namespace preconditioner
}  // namespace gko