/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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

#include <ginkgo/core/matrix/sub_matrix.hpp>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/matrix/sub_matrix_kernels.hpp"


namespace gko {
namespace matrix {
namespace sub_matrix {


GKO_REGISTER_OPERATION(spmv, sub_matrix::spmv);
GKO_REGISTER_OPERATION(advanced_spmv, sub_matrix::advanced_spmv);
GKO_REGISTER_OPERATION(compute_block_ptrs, sub_matrix::compute_block_ptrs);


}  // namespace sub_matrix


template <typename MatrixType>
void SubMatrix<MatrixType>::generate(
    const MatrixType *matrix, const gko::span &row_span,
    const gko::span &col_span, const std::vector<gko::span> &overlap_row_span,
    const std::vector<gko::span> &overlap_col_span)
{
    this->get_executor()->run(sub_matrix::make_compute_block_ptrs(
        num_blocks, block_sizes.get_const_data(), block_ptrs_.get_data()));

    for (size_type j = 0; j < block_mtxs.size(); ++j) {
        block_mtxs_.emplace_back(std::move(block_mtxs[j]));
        overlap_mtxs_.emplace_back(std::move(overlap_mtxs[j]));
        block_dims_.emplace_back(block_mtxs_.back()->get_size());
        block_nnzs_.emplace_back(block_mtxs_.back()->get_num_stored_elements());
    }
}


template <typename MatrixType>
void SubMatrix<MatrixType>::apply_impl(const LinOp *b, LinOp *x) const
{
    using value_type = typename MatrixType::value_type;
    using index_type = typename MatrixType::index_type;
    using Dense = Dense<value_type>;

    auto dense_b = as<Dense>(b);
    auto dense_x = as<Dense>(x);
    this->get_executor()->run(sub_matrix::make_spmv(this, dense_b, dense_x));
}


template <typename MatrixType>
void SubMatrix<MatrixType>::apply_impl(const LinOp *alpha, const LinOp *b,
                                       const LinOp *beta, LinOp *x) const
{
    using value_type = typename MatrixType::value_type;
    using index_type = typename MatrixType::index_type;
    using Dense = Dense<value_type>;

    auto dense_b = as<Dense>(b);
    auto dense_x = as<Dense>(x);
    this->get_executor()->run(sub_matrix::make_advanced_spmv(
        as<Dense>(alpha), this, dense_b, as<Dense>(beta), dense_x));
}


#define GKO_DECLARE_SUB_MATRIX_CSR_GENERATE(ValueType, IndexType) \
    void SubMatrix<Csr<ValueType, IndexType>>::generate(          \
        const Array<size_type> &, const Overlap<size_type> &,     \
        const Csr<ValueType, IndexType> *x)

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SUB_MATRIX_CSR_GENERATE);


#define GKO_DECLARE_SUB_MATRIX_CSR_APPLY(ValueType, IndexType)            \
    void SubMatrix<Csr<ValueType, IndexType>>::apply_impl(const LinOp *b, \
                                                          LinOp *x) const
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_SUB_MATRIX_CSR_APPLY);


#define GKO_DECLARE_SUB_MATRIX_CSR_APPLY2(ValueType, IndexType) \
    void SubMatrix<Csr<ValueType, IndexType>>::apply_impl(      \
        const LinOp *alpha, const LinOp *b, const LinOp *beta, LinOp *x) const
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SUB_MATRIX_CSR_APPLY2);


}  // namespace matrix
}  // namespace gko
