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

#ifndef GKO_CORE_COMPONENTS_VALIDATION_HELPERS_HPP_
#define GKO_CORE_COMPONENTS_VALIDATION_HELPERS_HPP_

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/lin_op.hpp>

namespace gko {
namespace validate {

/**
 * Tests whether a given matrix is symmetric
 *
 * @param A the input matrix which is tested
 */
bool is_symmetric(const LinOp *A, const float tolerance = 0.0);

/**
 * Tests whether a given matrix has zero elements on the diagonal
 *
 * @param A the input matrix which is tested
 */
bool has_non_zero_diagonal(const LinOp *A);

/**
 * Tests whether the given row_ptrs are in an ascending order
 *
 * @param row_ptrs the sorted array which is to be tested
 * @param num_entries length of the array which is to be tested
 */
template <typename IndexType>
bool is_row_ordered(const IndexType *row_ptrs, const size_type num_entries);

/**
 * Tests whether the elements of the given index array are within unique
 *
 * @param idxs the sorted array which is to be tested
 * @param num_entries length of the array which is to be tested
 */
template <typename IndexType>
bool has_unique_idxs(const IndexType *idxs, const size_type num_entries);

/**
 * Tests whether the elements of the given array are within [lower_bound,
 * upper_bound)
 *
 * @param row_ptrs the array which is to be tested
 * @param num_entries length of the array which is to be tested
 * @param lower_bound the lower bound
 * @param upper_bound the upper bound
 */
template <typename IndexType>
bool is_within_bounds(const IndexType *idxs, const size_type num_entries,
                      const IndexType lower_bound, const IndexType upper_bound);


/**
 * Tests whether all elements of the given array are finite
 *
 * @param values the array which is to be tested
 * @param num_entries length of the array which is to be tested
 */
template <typename ValueType>
bool is_finite(const ValueType *values, const size_type num_entries);


/**
 * Tests whether the difference between adjacent elements is below a threshold
 *
 * @param idxs the sorted array which is to be tested
 * @param num_entries length of the array which is to be tested
 */
template <typename IndexType>
bool is_consecutive(const IndexType *idxs, const size_type num_entries,
                    const IndexType max_gap);
}  // namespace validate
}  // namespace gko

#endif  // GKO_CORE_COMPONENTS_VALIDATION_HELPERS_HPP_
