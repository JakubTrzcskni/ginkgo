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

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/matrix_data.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>

#include "core/base/utils.hpp"
#include "core/preconditioner/gauss_seidel_kernels.hpp"
#include "core/utils/matrix_utils.hpp"

namespace gko {
namespace preconditioner {
namespace gauss_seidel {
namespace {
GKO_REGISTER_OPERATION(apply, gauss_seidel::apply);
GKO_REGISTER_OPERATION(ref_apply, gauss_seidel::ref_apply);
GKO_REGISTER_OPERATION(simple_apply, gauss_seidel::simple_apply);
GKO_REGISTER_OPERATION(ref_simple_apply, gauss_seidel::ref_simple_apply);
GKO_REGISTER_OPERATION(get_coloring, gauss_seidel::get_coloring);
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
    using Csr = matrix::Csr<ValueType, IndexType>;
    const auto exec = this->get_executor();
    if (use_reference_) {
        exec->run(gauss_seidel::make_ref_simple_apply(
            lend(lower_trs_), as<const Dense>(b), as<Dense>(x)));
    } else {
        auto system_matrix =
            convert_to_ltr_ ? lower_triangular_matrix_ : system_matrix_;
        exec->run(
            gauss_seidel::make_simple_apply(as<const Csr>(lend(system_matrix)),
                                            as<const Dense>(b), as<Dense>(x)));
    }
}

template <typename ValueType, typename IndexType>
void GaussSeidel<ValueType, IndexType>::apply_impl(const LinOp* alpha,
                                                   const LinOp* b,
                                                   const LinOp* beta,
                                                   LinOp* x) const
{
    using Dense = matrix::Dense<ValueType>;
    using Csr = matrix::Csr<ValueType, IndexType>;
    const auto exec = this->get_executor();
    if (use_reference_) {
        exec->run(gauss_seidel::make_ref_apply(
            lend(lower_trs_), as<const Dense>(alpha), as<const Dense>(b),
            as<const Dense>(beta), as<Dense>(x)));
    } else {
        auto system_matrix =
            convert_to_ltr_ ? lower_triangular_matrix_ : system_matrix_;
        exec->run(gauss_seidel::make_apply(
            as<const Csr>(lend(system_matrix)), as<const Dense>(alpha),
            as<const Dense>(b), as<const Dense>(beta), as<Dense>(x)));
    }
}

template <typename ValueType, typename IndexType>
std::unique_ptr<LinOp> GaussSeidel<ValueType, IndexType>::transpose() const
    GKO_NOT_IMPLEMENTED;


template <typename ValueType, typename IndexType>
std::unique_ptr<LinOp> GaussSeidel<ValueType, IndexType>::conj_transpose() const
    GKO_NOT_IMPLEMENTED;

template <typename ValueType, typename IndexType>
IndexType GaussSeidel<ValueType, IndexType>::get_coloring(
    std::shared_ptr<matrix::Csr<ValueType, IndexType>> system_matrix,
    bool is_symmetric)
{
    using SparsityMatrix = matrix::SparsityCsr<ValueType, IndexType>;
    using MatData = matrix_data<ValueType, IndexType>;

    auto exec = this->get_executor();

    auto tmp = copy_and_convert_to<SparsityMatrix>(exec, system_matrix);

    auto adjacency_matrix = SparsityMatrix::create(exec);

    adjacency_matrix = std::move(tmp->to_adjacency_matrix());

    if (!is_symmetric) {
        MatData tmp_mat_data(system_matrix->get_size());
        adjacency_matrix->write(tmp_mat_data);
        utils::make_symmetric_generic(
            tmp_mat_data, [](auto val) { return ValueType{2} * val; });
        adjacency_matrix->read(tmp_mat_data);
    }

    vertex_colors_.fill(IndexType{-1});
    exec->run(gauss_seidel::make_get_coloring(lend(adjacency_matrix),
                                              vertex_colors_));

    // TODO save the max color of the computed coloring (and return the value)
    return IndexType{-1};
}

template <typename ValueType, typename IndexType>
void GaussSeidel<ValueType, IndexType>::reorder_with_colors(IndexType max_color)
{
    auto num_rows = vertex_colors_.get_num_elems();
    const auto coloring = vertex_colors_.get_const_data();
    auto permutation = permutation_idxs_.get_data();
    IndexType tmp{0};
    for (auto color = 0; color < max_color; color++) {
        for (auto i = 0; i < num_rows; i++) {
            if (coloring[i] == color) {
                permutation[tmp] = i;
                tmp++;
            }
        }
    }
    GKO_ASSERT_EQ(tmp, num_rows);
}

template <typename ValueType, typename IndexType>
void GaussSeidel<ValueType, IndexType>::generate(bool skip_sorting)
{
    using Csr = matrix::Csr<ValueType, IndexType>;
    const auto exec = this->get_executor();
    GKO_ASSERT_IS_SQUARE_MATRIX(system_matrix_);

    if (convert_to_ltr_) {
        auto csr_matrix =
            convert_to_with_sorting<Csr>(exec, system_matrix_, skip_sorting);

        matrix_data<ValueType, IndexType> tmp_mat_data{csr_matrix->get_size()};
        csr_matrix->write(tmp_mat_data);
        utils::make_lower_triangular(tmp_mat_data);
        lower_triangular_matrix_->read(tmp_mat_data);

        if (use_reference_) {
            lower_trs_ =
                share(lower_trs_factory_->generate(lower_triangular_matrix_));
        } else {
            lower_trs_ = {};
        }
    }
    // not necessary anymore, matrix is assumed not to be lower triangular.
    //  else {
    //      lower_trs_ = lower_trs_factory_->generate(system_matrix_);
    //  }
    if (!symmetric_) {
        auto max_color = get_coloring(lower_triangular_matrix_, false);
        // TODO
        reorder_with_colors(8);  // hard coded for the example
    }
}

#define GKO_DECLARE_GAUSS_SEIDEL(ValueType, IndexType) \
    class GaussSeidel<ValueType, IndexType>

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_GAUSS_SEIDEL);

}  // namespace preconditioner
}  // namespace gko