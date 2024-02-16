// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

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
