// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/preconditioner/gauss_seidel_kernels.hpp"


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/csr.hpp>


#include "core/base/allocator.hpp"

namespace gko {
namespace kernels {
namespace dpcpp {
namespace gauss_seidel {

template <typename ValueType>
void ref_apply(std::shared_ptr<const DpcppExecutor> exec, const LinOp* solver,
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
void ref_simple_apply(std::shared_ptr<const DpcppExecutor> exec,
                      const LinOp* solver, const matrix::Dense<ValueType>* b,
                      matrix::Dense<ValueType>* x)
{
    solver->apply(b, x);
}
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_GAUSS_SEIDEL_REFERENCE_SIMPLE_APPLY_KERNEL);

template <typename ValueType, typename IndexType>
void apply(std::shared_ptr<const DpcppExecutor> exec,
           const matrix::Csr<ValueType, IndexType>* A,
           const matrix::Dense<ValueType>* alpha,
           const matrix::Dense<ValueType>* rhs,
           const matrix::Dense<ValueType>* beta,
           matrix::Dense<ValueType>* x) GKO_NOT_IMPLEMENTED;
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_GAUSS_SEIDEL_APPLY_KERNEL);

template <typename ValueType, typename IndexType>
void simple_apply(std::shared_ptr<const DpcppExecutor> exec,
                  const matrix::Csr<ValueType, IndexType>* A,
                  const matrix::Dense<ValueType>* rhs,
                  matrix::Dense<ValueType>* x) GKO_NOT_IMPLEMENTED;
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_GAUSS_SEIDEL_SIMPLE_APPLY_KERNEL);

template <typename ValueType, typename IndexType>
void get_coloring(
    std::shared_ptr<const DpcppExecutor> exec,
    const matrix::SparsityCsr<ValueType, IndexType>* adjacency_matrix,
    array<IndexType>& vertex_colors, IndexType* max_color) GKO_NOT_IMPLEMENTED;
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_GAUSS_SEIDEL_GET_COLORING_KERNEL);

}  // namespace gauss_seidel
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
