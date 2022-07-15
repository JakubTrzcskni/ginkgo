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

#include <gtest/gtest.h>


#include <ginkgo/core/matrix/dense.hpp>


#include "core/test/utils.hpp"

namespace {
template <typename ValueIndexType>
class GaussSeidelFactory : public ::testing ::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using GS = gko::preconditioner::GaussSeidel<value_type, index_type>;
    using Mtx = gko::matrix::Dense<value_type>;

    GaussSeidelFactory()
        : exec{gko::ReferenceExecutor::create()},
          mtx(gko::initialize<Mtx>(
              {{2, -1.0, 0.0}, {-1.0, 2, -1.0}, {0.0, -1.0, 2}}, exec)),
          gs_factory(GS::build()
                         .with_skip_sorting(false)
                         .with_relaxation_factor(1.5)
                         .with_convert_to_lower_triangular(false)
                         .with_use_reference(true)
                         .with_symmetric_preconditioner(true)
                         .on(exec)),
          precondtioner(gs_factory->generate(mtx))
    {}

    std::shared_ptr<const gko::Executor> exec;
    std::shared_ptr<Mtx> mtx;
    std::shared_ptr<typename GS::Factory> gs_factory;
    std::unique_ptr<gko::LinOp> precondtioner;
};

TYPED_TEST_SUITE(GaussSeidelFactory, gko::test::ValueIndexTypes,
                 PairTypenameNameGenerator);


TYPED_TEST(GaussSeidelFactory, KnowsItsExecutor)
{
    ASSERT_EQ(this->gs_factory->get_executor(), this->exec);
}

TYPED_TEST(GaussSeidelFactory, SetsSkipSortingCorrectly)
{
    ASSERT_EQ(this->gs_factory->get_parameters().skip_sorting, false);
}

TYPED_TEST(GaussSeidelFactory, SetsDefaultSkipSortingCorrectly)
{
    using GS = typename TestFixture::GS;

    auto gs_factory = GS::build().on(this->exec);

    ASSERT_EQ(gs_factory->get_parameters().skip_sorting, true);
}


TYPED_TEST(GaussSeidelFactory, SetsRelaxationFactorCorrectly)
{
    ASSERT_EQ(this->gs_factory->get_parameters().relaxation_factor, 1.5);
}

TYPED_TEST(GaussSeidelFactory, SetsDefaultRelaxationFactorCorrectly)
{
    using GS = typename TestFixture::GS;

    auto gs_factory = GS::build().on(this->exec);

    ASSERT_EQ(gs_factory->get_parameters().relaxation_factor, 1.0);
}

TYPED_TEST(GaussSeidelFactory, SetsConvertToLTrCorrectly)
{
    ASSERT_EQ(this->gs_factory->get_parameters().convert_to_lower_triangular,
              false);
}

TYPED_TEST(GaussSeidelFactory, SetsDefaultConvertToLTrCorrectly)
{
    using GS = typename TestFixture::GS;

    auto gs_factory = GS::build().on(this->exec);

    ASSERT_EQ(gs_factory->get_parameters().convert_to_lower_triangular, true);
}

TYPED_TEST(GaussSeidelFactory, SetsUseReferenceCorrectly)
{
    ASSERT_EQ(this->gs_factory->get_parameters().use_reference, true);
}

TYPED_TEST(GaussSeidelFactory, SetsDefaultUseReferenceCorrectly)
{
    using GS = typename TestFixture::GS;

    auto gs_factory = GS::build().on(this->exec);

    ASSERT_EQ(gs_factory->get_parameters().use_reference, false);
}

TYPED_TEST(GaussSeidelFactory, SetsSymmPrecondCorrectly)
{
    ASSERT_EQ(this->gs_factory->get_parameters().symmetric_preconditioner,
              true);
}

TYPED_TEST(GaussSeidelFactory, SetsDefaultSymmPrecondCorrectly)
{
    using GS = typename TestFixture::GS;

    auto gs_factory = GS::build().on(this->exec);

    ASSERT_EQ(gs_factory->get_parameters().symmetric_preconditioner, false);
}

}  // namespace