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

#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>
#include <ginkgo/core/solver/lower_trs.hpp>

#include "core/base/allocator.hpp"

namespace gko {
namespace kernels {
namespace reference {
namespace gauss_seidel {
namespace {
// local functions
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
           const matrix::Dense<ValueType>* beta, matrix::Dense<ValueType>* x)
{
    const ValueType* values = A->get_const_values();
    const IndexType* row_ptrs = A->get_const_row_ptrs();
    const IndexType* col_idxs = A->get_const_col_idxs();
    auto diag = A->extract_diagonal();
    const ValueType* diag_values = diag->get_const_values();
    // TODO
    // non-scalar alpha and beta impl
    GKO_ASSERT_IS_SCALAR(alpha);
    GKO_ASSERT_IS_SCALAR(beta);

    // beta<->x; alpha<->b

    for (size_type i = 0; i < x->get_size()[0]; i++) {
        for (size_type j = 0; j < x->get_size()[1]; j++) {
            ValueType tmp = alpha->get_const_values()[0] * b->at(i, j);

            for (size_type k = row_ptrs[i]; k < row_ptrs[i + 1]; k++) {
                IndexType curr_col = col_idxs[k];
                if (curr_col != i) {
                    tmp -= values[k] * beta->get_const_values()[0] *
                           x->at(curr_col, j);  // not sure if correct with
                }
            }
            x->at(i, j) = tmp / diag_values[i];
        }
    }
}
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_GAUSS_SEIDEL_APPLY_KERNEL);

template <typename ValueType, typename IndexType>
void simple_apply(std::shared_ptr<const ReferenceExecutor> exec,
                  const matrix::Csr<ValueType, IndexType>* A,
                  const matrix::Dense<ValueType>* b,
                  matrix::Dense<ValueType>* x)
{
    const ValueType* values = A->get_const_values();
    const IndexType* row_ptrs = A->get_const_row_ptrs();
    const IndexType* col_idxs = A->get_const_col_idxs();
    auto diag = A->extract_diagonal();
    const ValueType* diag_values = diag->get_const_values();

    for (size_type i = 0; i < x->get_size()[0]; i++) {
        for (size_type j = 0; j < x->get_size()[1]; j++) {
            ValueType tmp = b->at(i, j);

            for (size_type k = row_ptrs[i]; k < row_ptrs[i + 1]; k++) {
                IndexType curr_col = col_idxs[k];
                if (curr_col != i) {
                    tmp -= values[k] * x->at(curr_col, j);
                }
            }
            x->at(i, j) = tmp / diag_values[i];
        }
    }
}
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_GAUSS_SEIDEL_SIMPLE_APPLY_KERNEL);


}  // namespace gauss_seidel
}  // namespace reference
}  // namespace kernels
}  // namespace gko