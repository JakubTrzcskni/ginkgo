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

#ifndef GKO_CORE_PRECONDITIONER_GAUSS_SEIDEL_KERNELS_HPP_
#define GKO_CORE_PRECONDITIONER_GAUSS_SEIDEL_KERNELS_HPP_


#include <ginkgo/core/preconditioner/gauss_seidel.hpp>


#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>


#include "core/base/kernel_declaration.hpp"

namespace gko {
namespace kernels {

#define GKO_DECLARE_GAUSS_SEIDEL_APPLY_KERNEL(ValueType, IndexType) \
    void apply(std::shared_ptr<const DefaultExecutor> exec,         \
               const matrix::Csr<ValueType, IndexType>* A,          \
               const matrix::Dense<ValueType>* alpha,               \
               const matrix::Dense<ValueType>* rhs,                 \
               const matrix::Dense<ValueType>* beta,                \
               matrix::Dense<ValueType>* x)

#define GKO_DECLARE_GAUSS_SEIDEL_REFERENCE_APPLY_KERNEL(ValueType)             \
    void ref_apply(std::shared_ptr<const DefaultExecutor> exec,                \
                   const LinOp* solver, const matrix::Dense<ValueType>* alpha, \
                   const matrix::Dense<ValueType>* rhs,                        \
                   const matrix::Dense<ValueType>* beta,                       \
                   matrix::Dense<ValueType>* x)

#define GKO_DECLARE_GAUSS_SEIDEL_SIMPLE_APPLY_KERNEL(ValueType, IndexType) \
    void simple_apply(std::shared_ptr<const DefaultExecutor> exec,         \
                      const matrix::Csr<ValueType, IndexType>* A,          \
                      const matrix::Dense<ValueType>* rhs,                 \
                      matrix::Dense<ValueType>* x)

#define GKO_DECLARE_GAUSS_SEIDEL_REFERENCE_SIMPLE_APPLY_KERNEL(ValueType) \
    void ref_simple_apply(                                                \
        std::shared_ptr<const DefaultExecutor> exec, const LinOp* solver, \
        const matrix::Dense<ValueType>* b, matrix::Dense<ValueType>* x)

#define GKO_DECLARE_GAUSS_SEIDEL_GET_COLORING_KERNEL(ValueType, IndexType) \
    void get_coloring(                                                     \
        std::shared_ptr<const DefaultExecutor> exec,                       \
        const matrix::SparsityCsr<ValueType, IndexType>* adjacency_matrix, \
        array<IndexType>& vertex_colors, IndexType* max_color)

#define GKO_DECLARE_GAUSS_SEIDEL_GET_BLOCK_COLORING_KERNEL(ValueType,      \
                                                           IndexType)      \
    void get_block_coloring(                                               \
        std::shared_ptr<const DefaultExecutor> exec,                       \
        const matrix::SparsityCsr<ValueType, IndexType>* adjacency_matrix, \
        const IndexType* block_ordering, const IndexType block_size,       \
        IndexType* vertex_colors, IndexType* max_color)

#define GKO_DECLARE_GAUSS_SEIDEL_GET_SECONDARY_ORDERING_KERNEL(IndexType)    \
    void get_secondary_ordering(                                             \
        std::shared_ptr<const DefaultExecutor> exec,                         \
        IndexType* block_ordering, const IndexType base_block_size,          \
        const IndexType lvl_2_block_size, const IndexType* color_block_ptrs, \
        const IndexType max_color)

#define GKO_DECLARE_GAUSS_SEIDEL_ASSIGN_TO_BLOCKS_KERNEL(ValueType, IndexType) \
    void assign_to_blocks(                                                     \
        std::shared_ptr<const DefaultExecutor> exec,                           \
        const matrix::SparsityCsr<ValueType, IndexType>* adjacency_matrix,     \
        IndexType* block_ordering, const IndexType* degrees, int8* visited,    \
        const IndexType block_size, const IndexType lvl_2_block_size)

#define GKO_DECLARE_GAUSS_SEIDEL_GET_PERMUTATION_FROM_COLORING_KERNEL( \
    IndexType)                                                         \
    void get_permutation_from_coloring(                                \
        std::shared_ptr<const DefaultExecutor> exec,                   \
        const IndexType num_nodes, const IndexType* coloring,          \
        const IndexType max_color, IndexType* color_ptrs,              \
        IndexType* permutation_idxs, const IndexType* block_ordering)

#define GKO_DECLARE_ALL_AS_TEMPLATES                                          \
    template <typename ValueType, typename IndexType>                         \
    GKO_DECLARE_GAUSS_SEIDEL_APPLY_KERNEL(ValueType, IndexType);              \
    template <typename ValueType>                                             \
    GKO_DECLARE_GAUSS_SEIDEL_REFERENCE_APPLY_KERNEL(ValueType);               \
    template <typename ValueType, typename IndexType>                         \
    GKO_DECLARE_GAUSS_SEIDEL_SIMPLE_APPLY_KERNEL(ValueType, IndexType);       \
    template <typename ValueType>                                             \
    GKO_DECLARE_GAUSS_SEIDEL_REFERENCE_SIMPLE_APPLY_KERNEL(ValueType);        \
    template <typename ValueType, typename IndexType>                         \
    GKO_DECLARE_GAUSS_SEIDEL_GET_COLORING_KERNEL(ValueType, IndexType);       \
    template <typename ValueType, typename IndexType>                         \
    GKO_DECLARE_GAUSS_SEIDEL_GET_BLOCK_COLORING_KERNEL(ValueType, IndexType); \
    template <typename ValueType, typename IndexType>                         \
    GKO_DECLARE_GAUSS_SEIDEL_ASSIGN_TO_BLOCKS_KERNEL(ValueType, IndexType);   \
    template <typename IndexType>                                             \
    GKO_DECLARE_GAUSS_SEIDEL_GET_PERMUTATION_FROM_COLORING_KERNEL(IndexType); \
    template <typename IndexType>                                             \
    GKO_DECLARE_GAUSS_SEIDEL_GET_SECONDARY_ORDERING_KERNEL(IndexType)

GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(gauss_seidel,
                                        GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES

}  // namespace kernels
}  // namespace gko

#endif  // GKO_CORE_PRECONDITIONER_GAUSS_SEIDEL_KERNELS_HPP_