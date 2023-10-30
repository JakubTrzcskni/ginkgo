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

#ifndef GKO_CORE_FACTORIZATION_LU_KERNELS_HPP_
#define GKO_CORE_FACTORIZATION_LU_KERNELS_HPP_


#include <memory>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/csr.hpp>


#include "core/base/kernel_declaration.hpp"


namespace gko {
namespace kernels {


#define GKO_DECLARE_LU_INITIALIZE(ValueType, IndexType)                       \
    void initialize(std::shared_ptr<const DefaultExecutor> exec,              \
                    const matrix::Csr<ValueType, IndexType>* mtx,             \
                    const IndexType* factor_lookup_offsets,                   \
                    const int64* factor_lookup_descs,                         \
                    const int32* factor_lookup_storage, IndexType* diag_idxs, \
                    matrix::Csr<ValueType, IndexType>* factors)


#define GKO_DECLARE_LU_FACTORIZE(ValueType, IndexType)                         \
    void factorize(std::shared_ptr<const DefaultExecutor> exec,                \
                   const IndexType* lookup_offsets, const int64* lookup_descs, \
                   const int32* lookup_storage, const IndexType* diag_idxs,    \
                   matrix::Csr<ValueType, IndexType>* factors,                 \
                   array<int>& tmp_storage)


#define GKO_DECLARE_LU_SYMMETRIC_FACTORIZE_SIMPLE(IndexType)                  \
    void symbolic_factorize_simple(                                           \
        std::shared_ptr<const DefaultExecutor> exec,                          \
        const IndexType* row_ptrs, const IndexType* col_idxs,                 \
        const IndexType* factor_lookup_offsets,                               \
        const int64* factor_lookup_descs, const int32* factor_lookup_storage, \
        matrix::Csr<float, IndexType>* factors, IndexType* out_row_nnz)


#define GKO_DECLARE_LU_SYMMETRIC_FACTORIZE_SIMPLE_FINALIZE(IndexType) \
    void symbolic_factorize_simple_finalize(                          \
        std::shared_ptr<const DefaultExecutor> exec,                  \
        const matrix::Csr<float, IndexType>* factors, IndexType* col_idxs)


#define GKO_DECLARE_ALL_AS_TEMPLATES                      \
    template <typename ValueType, typename IndexType>     \
    GKO_DECLARE_LU_INITIALIZE(ValueType, IndexType);      \
    template <typename ValueType, typename IndexType>     \
    GKO_DECLARE_LU_FACTORIZE(ValueType, IndexType);       \
    template <typename IndexType>                         \
    GKO_DECLARE_LU_SYMMETRIC_FACTORIZE_SIMPLE(IndexType); \
    template <typename IndexType>                         \
    GKO_DECLARE_LU_SYMMETRIC_FACTORIZE_SIMPLE_FINALIZE(IndexType)


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(lu_factorization,
                                        GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_FACTORIZATION_LU_KERNELS_HPP_
