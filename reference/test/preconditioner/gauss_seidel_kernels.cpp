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
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/solver/lower_trs.hpp>

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
    using Trs = gko::solver::LowerTrs<value_type, index_type>;
    using Csr = gko::matrix::Csr<value_type, index_type>;
    using Vec = gko::matrix::Dense<value_type>;


    GaussSeidel()
        : exec{gko::ReferenceExecutor::create()},
          scalar_gs_factory(GS::build().on(exec)),
          lower_trs_factory(Trs::build().on(exec)),
          mtx_dense{gko::initialize<Vec>({{16.0, 3.0}, {7.0, -11.0}}, exec)},
          mtx_csr{Csr::create(exec)},
          T_dense{gko::initialize<Vec>({{0, -0.1875}, {0, -0.1194}}, exec)},
          T_csr{Csr::create(exec)},
          C_dense{gko::initialize<Vec>({0.6875, -0.7439}, exec)},
          rhs{gko::initialize<Vec>({11.0, 13.0}, exec)},
          x_0{gko::initialize<Vec>({1.0, 1.0}, exec)},
          x_1{gko::initialize<Vec>({0.5, -0.8636}, exec)}

    {
        mtx_dense->convert_to(gko::lend(mtx_csr));
        T_dense->convert_to(gko::lend(T_csr));
    }

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<typename GS::Factory> scalar_gs_factory;
    std::unique_ptr<typename Trs::Factory> lower_trs_factory;
    std::shared_ptr<Vec> mtx_dense;
    std::shared_ptr<Csr> mtx_csr;
    std::shared_ptr<Vec> T_dense;
    std::shared_ptr<Csr> T_csr;
    std::shared_ptr<Vec> C_dense;
    std::shared_ptr<Vec> rhs;
    std::shared_ptr<Vec> x_0;
    std::shared_ptr<Vec> x_1;
};

TYPED_TEST_SUITE(GaussSeidel, gko::test::ValueIndexTypes,
                 PairTypenameNameGenerator);

// TYPED_TEST(GaussSeidel, )

TYPED_TEST(GaussSeidel, SolveCompareVsLowerTrs)
{
    // using Trs = typename TestFixture::Trs;
    // using Csr = typename TestFixture::Csr;
    // using value_type = typename TestFixture::value_type;
    // auto result_trs =
    //     Vec::create(this->exec, gko::dim<2>(this->mtx_csr->get_size()[1],
    //     1));
    // auto result = Vec::create_with_config_of(gko::lend(result_trs));

    // auto gs =
    //     this->scalar_gs_factory->generate(gko::lend(this->mtx_csr));  //
    //     lend?

    // gs->apply(gko::lend(this->rhs), result.get());

    // auto trs = this->lower_trs_factory->generate(gko::lend(this->mtx_csr));

    // ts->apply(gko::lend(this->rhs), result_trs.get());

    // GKO_ASSERT_MTX_NEAR(result, result_trs, r<value_type>::value);
}

}  // namespace