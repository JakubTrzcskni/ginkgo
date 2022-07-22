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

#include "core/base/allocator.hpp"

namespace gko {
namespace kernels {
namespace cuda {
namespace gauss_seidel {

template <typename ValueType>
void ref_apply(std::shared_ptr<const CudaExecutor> exec, const LinOp* solver,
               const matrix::Dense<ValueType>* alpha,
               const matrix::Dense<ValueType>* b,
               const matrix::Dense<ValueType>* beta,
               matrix::Dense<ValueType>* x)
{
    solver->apply(alpha, b, beta, x);
}
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_GAUSS_SEIDEL_REFERENCE_APPLY_KERNEL);

template <typename ValueType>
void ref_simple_apply(std::shared_ptr<const CudaExecutor> exec,
                      const LinOp* solver, const matrix::Dense<ValueType>* b,
                      matrix::Dense<ValueType>* x)
{
    solver->apply(b, x);
}
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_GAUSS_SEIDEL_REFERENCE_SIMPLE_APPLY_KERNEL);

template <typename ValueType, typename IndexType>
void apply(std::shared_ptr<const CudaExecutor> exec,
           const matrix::Csr<ValueType, IndexType>* A,
           const matrix::Dense<ValueType>* alpha,
           const matrix::Dense<ValueType>* rhs,
           const matrix::Dense<ValueType>* beta,
           matrix::Dense<ValueType>* x) GKO_NOT_IMPLEMENTED;
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_GAUSS_SEIDEL_APPLY_KERNEL);


template <typename ValueType, typename IndexType>
void simple_apply(std::shared_ptr<const CudaExecutor> exec,
                  const matrix::Csr<ValueType, IndexType>* A,
                  const matrix::Dense<ValueType>* rhs,
                  matrix::Dense<ValueType>* x) GKO_NOT_IMPLEMENTED;
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_GAUSS_SEIDEL_SIMPLE_APPLY_KERNEL);

template <typename ValueType, typename IndexType>
void get_coloring(
    std::shared_ptr<const CudaExecutor> exec,
    const matrix::SparsityCsr<ValueType, IndexType>* adjacency_matrix,
    array<IndexType>& vertex_colors, IndexType* max_color) GKO_NOT_IMPLEMENTED;
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_GAUSS_SEIDEL_GET_COLORING_KERNEL);


}  // namespace gauss_seidel
}  // namespace cuda
}  // namespace kernels
}  // namespace gko