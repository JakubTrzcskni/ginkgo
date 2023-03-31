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


#include <gtest/gtest.h>


#include <ginkgo/core/matrix/csr.hpp>

#include "core/test/utils.hpp"

namespace {

template <typename ValueIndexType>
class GaussSeidelFactory : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using GS = gko::preconditioner::GaussSeidel<value_type, index_type>;
    using Csr = gko::matrix::Csr<value_type, index_type>;
    GaussSeidelFactory()
        : exec(gko::ReferenceExecutor::create()),
          gs_factory(GS::build().on(exec))
    {}

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<typename GS::Factory> gs_factory;
};

TYPED_TEST_SUITE(GaussSeidelFactory, gko::test::ValueIndexTypes,
                 PairTypenameNameGenerator);

TYPED_TEST(GaussSeidelFactory, KnowsItsExecutor)
{
    ASSERT_EQ(this->gs_factory->get_executor(), this->exec);
}
TYPED_TEST(GaussSeidelFactory, SetsSkipSortingCorrectly)
{
    using GS = typename TestFixture::GS;
    ASSERT_EQ(this->gs_factory->get_parameters().skip_sorting, true);
    auto newFactory = GS::build().with_skip_sorting(false).on(this->exec);
    ASSERT_EQ(newFactory->get_parameters().skip_sorting, false);
}
TYPED_TEST(GaussSeidelFactory, SetsRelaxationFactorCorrectly)
{
    using GS = typename TestFixture::GS;
    ASSERT_EQ(this->gs_factory->get_parameters().relaxation_factor, 1.0);
    auto newFactory = GS::build().with_relaxation_factor(1.5).on(this->exec);
    ASSERT_EQ(newFactory->get_parameters().relaxation_factor, 1.5);
}
TYPED_TEST(GaussSeidelFactory, SetsSymmetricPreconditionerCorrectly)
{
    using GS = typename TestFixture::GS;
    ASSERT_EQ(this->gs_factory->get_parameters().symmetric_preconditioner,
              false);
    auto newFactory =
        GS::build().with_symmetric_preconditioner(true).on(this->exec);
    ASSERT_EQ(newFactory->get_parameters().symmetric_preconditioner, true);
}
//  throws on dimensions, conversion etc. ?

}  // namespace