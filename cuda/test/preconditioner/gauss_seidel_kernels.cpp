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


#include <random>


#include <gtest/gtest.h>


#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>

#include "core/preconditioner/gauss_seidel_kernels.hpp"
#include "core/test/utils.hpp"
#include "core/utils/matrix_utils.hpp"
#include "cuda/test/utils.hpp"

namespace {
template <typename ValueIndexType>
class GaussSeidel : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    GaussSeidel() {}

    void SetUp()
    {
        ASSERT_GT(gko::CudaExecutor::get_num_devices(), 0);
        ref = gko::ReferenceExecutor::create();
        cuda = gko::CudaExecutor::create(0, ref);
    }

    void TearDown()
    {
        if (cuda != nullptr) {
            ASSERT_NO_THROW(cuda->synchronize());
        }
    }

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<gko::CudaExecutor> cuda;
};

TYPED_TEST_SUITE(GaussSeidel, gko::test::ValueIndexTypes,
                 PairTypenameNameGenerator);

TYPED_TEST(GaussSeidel, SimpleTest)
{
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    auto cuda_exec = this->cuda;

    const IndexType num_nodes = 8;
    gko::array<IndexType> coloring(cuda_exec,
                                   I<IndexType>({1, 1, 1, 1, 0, 0, 0, 0}));
    const IndexType max_color = 1;
    gko::array<IndexType> color_ptrs(cuda_exec, max_color + 2);
    gko::array<IndexType> permutation_idxs(cuda_exec, num_nodes);
    gko::array<IndexType> block_ordering(
        cuda_exec, I<IndexType>({0, 1, 2, 3, 4, 5, 6, 7}));

    gko::kernels::cuda::gauss_seidel::get_permutation_from_coloring(
        cuda_exec, num_nodes, coloring.get_data(), max_color,
        color_ptrs.get_data(), permutation_idxs.get_data(),
        block_ordering.get_const_data());

    GKO_ASSERT_ARRAY_EQ(permutation_idxs,
                        I<IndexType>({4, 5, 6, 7, 0, 1, 2, 3}));
    GKO_ASSERT_ARRAY_EQ(color_ptrs, I<IndexType>({0, 4, 8}));
}

TYPED_TEST(GaussSeidel, GetDegreeOfNodesKernel)
{
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using Csr = gko::matrix::Csr<ValueType, IndexType>;
    auto cuda_exec = this->cuda;

    auto mtx = gko::initialize<Csr>({{1., 0., 0., 0.},
                                     {1., 2., 0., 0.},
                                     {1., 2., 3., 0.},
                                     {1., 2., 3., 4.}},
                                    cuda_exec);
    const IndexType num_nodes = mtx->get_size()[0];
    gko::array<IndexType> degrees(cuda_exec, num_nodes);

    gko::kernels::cuda::gauss_seidel::get_degree_of_nodes(
        cuda_exec, num_nodes, mtx->get_const_row_ptrs(), degrees.get_data());

    GKO_ASSERT_ARRAY_EQ(degrees, I<IndexType>({1, 2, 3, 4}));
}
}  // namespace