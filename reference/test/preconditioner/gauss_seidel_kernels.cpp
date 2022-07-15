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

#include <algorithm>
#include <fstream>
#include <memory>
#include <type_traits>

#include <gtest/gtest.h>

#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/polymorphic_object.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/solver/ir.hpp>
#include <ginkgo/core/solver/lower_trs.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm.hpp>

#include "core/preconditioner/gauss_seidel_kernels.hpp"
#include "core/test/utils.hpp"
#include "matrices/config.hpp"

namespace {
template <typename ValueIndexType>
class GaussSeidel : public ::testing ::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using GS = gko::preconditioner::GaussSeidel<value_type, index_type>;
    using Ir = gko::solver::Ir<value_type>;
    using Csr = gko::matrix::Csr<value_type, index_type>;
    using Vec = gko::matrix::Dense<value_type>;


    GaussSeidel()
        : exec{gko::ReferenceExecutor::create()},
          gs_factory(GS::build().on(exec)),
          ref_gs_factory(GS::build().with_use_reference(true).on(exec)),
          ir_gs_factory(
              Ir::build()
                  .with_solver(gs_factory)
                  .with_criteria(
                      gko::stop::Iteration::build().with_max_iters(10u).on(
                          exec),
                      gko::stop::ResidualNorm<value_type>::build()
                          .with_reduction_factor(r<value_type>::value)
                          .on(exec))
                  .on(exec)),
          mtx_dense_2{gko::initialize<Vec>(
              {{0.9, -1.0, 3.0}, {0.0, 1.0, 3.0}, {0.0, 0.0, 1.1}}, exec)},
          mtx_csr_2{Csr::create(exec)},
          mtx_dense_3{gko::initialize<Vec>({{10.0, -1.0, 2.0, 0.0},
                                            {-1.0, 11.0, -1.0, 3.0},
                                            {2.0, -1.0, 10.0, -1.0},
                                            {0.0, 3.0, -1.0, 8.0}},
                                           exec)},
          mtx_csr_3{Csr::create(exec)},
          mtx_csr_4{Csr::create(exec, gko::dim<2>{5}, 10)},
          rhs_2{gko::initialize<Vec>({3.9, 9.0, 2.2}, exec)},
          ans_2{gko::initialize<Vec>({1.0, 3.0, 2.0}, exec)},
          rhs_3{gko::initialize<Vec>({6.0, 25.0, -11.0, 15.0}, exec)},
          ans_3{gko::initialize<Vec>({1.0, 2.0, -1.0, 1.0}, exec)},
          ltr_ans_3{gko::initialize<Vec>(
              {value_type{6.0 / 10.0}, value_type{256.0 / 110.0},
               value_type{-1086.0 / 1100.0}, value_type{7734.0 / 8800.0}},
              exec)},
          x_1_3{
              gko::initialize<Vec>({0.6, 2.32727, -0.987273, 0.878864}, exec)},
          rhs_4{gko::initialize<Vec>({6.0, 25.0, -11.0, 15.0, -3.0}, exec)},
          ans_4{gko::initialize<Vec>(
              {value_type{6.0 / 10.0}, value_type{256.0 / 110.0},
               value_type{-122.0 / 100.0}, value_type{882.0 / 880.0},
               value_type{-45956.0 / 61600.0}},
              exec)}

    {
        mtx_dense_2->convert_to(gko::lend(mtx_csr_2));
        mtx_dense_3->convert_to(gko::lend(mtx_csr_3));
        init_array<index_type>(mtx_csr_4->get_row_ptrs(), {0, 1, 3, 5, 7, 10});
        init_array<index_type>(mtx_csr_4->get_col_idxs(),
                               {0, 0, 1, 0, 2, 1, 3, 2, 3, 4});
        init_array<value_type>(
            mtx_csr_4->get_values(),
            {10.0, -1.0, 11.0, 2.0, 10.0, 3.0, 8.0, -1.0, 1.0, 7.0});
    }

    // Source: jacobi_kernels.cpp (test)
    template <typename T>
    void init_array(T* arr, std::initializer_list<T> vals)
    {
        for (auto elem : vals) {
            *(arr++) = elem;
        }
    }

    std::shared_ptr<const gko::ReferenceExecutor> exec;
    std::shared_ptr<typename GS::Factory> gs_factory;
    std::shared_ptr<typename GS::Factory> ref_gs_factory;
    std::unique_ptr<typename Ir::Factory> ir_gs_factory;
    std::shared_ptr<Vec> mtx_dense_2;
    std::shared_ptr<Csr> mtx_csr_2;
    // example wiki 2
    std::shared_ptr<Vec> mtx_dense_3;
    std::shared_ptr<Csr> mtx_csr_3;
    std::shared_ptr<Csr> mtx_csr_4;
    std::shared_ptr<Vec> rhs_2;
    std::shared_ptr<Vec> ans_2;
    std::shared_ptr<Vec> rhs_3;
    std::shared_ptr<Vec> ans_3;
    std::shared_ptr<Vec> ltr_ans_3;
    std::shared_ptr<Vec> x_1_3;
    std::shared_ptr<Vec> rhs_4;
    std::shared_ptr<Vec> ans_4;
};

TYPED_TEST_SUITE(GaussSeidel, gko::test::ValueIndexTypes,
                 PairTypenameNameGenerator);

TYPED_TEST(GaussSeidel, CanBeGenerated)
{
    auto gs = this->gs_factory->generate(this->mtx_csr_2);

    ASSERT_NE(gs, nullptr);
    EXPECT_EQ(gs->get_executor(), this->exec);
    ASSERT_EQ(gs->get_size(), gko::dim<2>(3, 3));
}

TYPED_TEST(GaussSeidel, ReferenceSimpleApplyKernel)
{
    using Vec = typename TestFixture::Vec;
    using value_type = typename TestFixture::value_type;

    auto ans = Vec::create_with_config_of(lend(this->rhs_4));
    auto ref_gs = this->ref_gs_factory->generate(this->mtx_csr_4);
    ref_gs->apply(lend(this->rhs_4), lend(ans));

    GKO_ASSERT_MTX_NEAR(ans, this->ans_4, r<value_type>::value);
}

TYPED_TEST(GaussSeidel, SimpleApplyKernel)
{
    using Vec = typename TestFixture::Vec;
    using value_type = typename TestFixture::value_type;

    auto ans = Vec::create_with_config_of(lend(this->rhs_4));
    auto gs = this->gs_factory->generate(this->mtx_csr_4);
    gs->apply(lend(this->rhs_4), lend(ans));

    GKO_ASSERT_MTX_NEAR(ans, this->ans_4, r<value_type>::value);
}

TYPED_TEST(GaussSeidel, ReferenceSimpleApplyNonTriangularMatrix)
{
    using Vec = typename TestFixture::Vec;
    using value_type = typename TestFixture::value_type;

    auto ans = Vec::create_with_config_of(lend(this->rhs_3));
    auto ref_gs = this->ref_gs_factory->generate(this->mtx_csr_3);
    ref_gs->apply(lend(this->rhs_3), lend(ans));

    // GKO_ASSERT_MTX_NEAR(ref_gs->get_ltr_system_matrix(), this->mtx_csr_3,
    //                     r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(ans, this->ltr_ans_3, r<value_type>::value);
}


TYPED_TEST(GaussSeidel, SystemSolveReferenceApply)
{
    using Csr = typename TestFixture::Csr;
    using Vec = typename TestFixture::Vec;
    using GS = typename TestFixture::GS;
    using Ir = typename TestFixture::Ir;
    using value_type = typename TestFixture::value_type;

    auto exec = this->exec;
    auto ir_ref_gs_factory =
        Ir::build()
            .with_solver(this->ref_gs_factory)
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(10u).on(exec),
                gko::stop::ResidualNorm<value_type>::build()
                    .with_reduction_factor(r<value_type>::value)
                    .on(exec))
            .on(exec);
    auto irs = ir_ref_gs_factory->generate(this->mtx_csr_4);

    auto result = Vec::create_with_config_of(lend(this->rhs_4));
    result->fill(0.0);

    irs->apply(lend(this->rhs_4), result.get());

    GKO_ASSERT_MTX_NEAR(result, this->ans_4, r<value_type>::value);
}

// TYPED_TEST(GaussSeidel, SystemSolvePureIR)
// {
//     using Ir = typename TestFixture::Ir;
//     using Csr = typename TestFixture::Csr;
//     using Vec = typename TestFixture::Vec;
//     using value_type = typename TestFixture::value_type;
//     auto exec = this->exec;

//     auto result = Vec::create_with_config_of(lend(this->rhs_4));
//     result->fill(0.0);

//     auto ir_factory = share(
//         Ir::build()
//             .with_criteria(
//                 gko::stop::Iteration::build().with_max_iters(10u).on(exec),
//                 gko::stop::ResidualNorm<value_type>::build()
//                     .with_reduction_factor(r<value_type>::value)
//                     .on(exec))
//             .on(exec));
//     auto irs = ir_factory->generate(this->mtx_csr_4);
//     irs->apply(lend(this->rhs_4), result.get());

//     GKO_ASSERT_MTX_NEAR(result, this->ans_4, r<value_type>::value);
// }

}  // namespace