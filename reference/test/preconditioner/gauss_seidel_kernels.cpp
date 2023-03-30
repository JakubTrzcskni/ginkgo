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
#include <ginkgo/core/preconditioner/gauss_seidel.hpp>

#include <memory>

#include <gtest/gtest.h>

#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/utils_helper.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>

#include "core/test/utils.hpp"

namespace {
template <typename ValueIndexType>
class GaussSeidel : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using Csr = gko::matrix::Csr<value_type, index_type>;
    using Dense = gko::matrix::Dense<value_type>;
    using GS = gko::preconditioner::GaussSeidel<value_type, index_type>;

    GaussSeidel()
        : exec(gko::ReferenceExecutor::create()),
          mtx(gko::initialize<Csr>({{1., 2., 3.}, {4., 5., 6.}, {7., 8., 9.}},
                                   exec)),
          ref_l_mtx(gko::initialize<Csr>(
              {{1., 0., 0.}, {4., 5., 0.}, {7., 8., 9.}}, exec)),
          x(gko::initialize<Dense>({0., 0., 0.}, exec)),
          rhs(gko::initialize<Dense>({1., 1., 1.}, exec)),
          gs_factory(GS::build().on(exec)),
          tol{r<value_type>::value}
    {}

    std::shared_ptr<const gko::Executor> exec;
    std::shared_ptr<Csr> mtx;
    std::shared_ptr<Csr> ref_l_mtx;
    std::shared_ptr<Dense> x;
    std::shared_ptr<Dense> rhs;
    std::unique_ptr<typename GS::Factory> gs_factory;
    gko::remove_complex<value_type> tol;
};

TYPED_TEST_SUITE(GaussSeidel, gko::test::ValueIndexTypes,
                 PairTypenameNameGenerator);
TYPED_TEST(GaussSeidel, ExtractsLowerTriangularCorrectly)
{
    using index_type = typename TestFixture::index_type;
    using value_type = typename TestFixture::value_type;

    auto gs = this->gs_factory->generate(gko::as<gko::LinOp>(this->mtx));
    auto ref_ltrs = gko::solver::LowerTrs<value_type, index_type>::build()
                        .on(this->exec)
                        ->generate(this->ref_l_mtx);
    auto x_copy = this->x->clone();
    gs->apply(this->rhs, x_copy);
    ref_ltrs->apply(this->rhs, this->x);
    GKO_ASSERT_MTX_NEAR(x_copy, this->x, this->tol);
}

}  // namespace