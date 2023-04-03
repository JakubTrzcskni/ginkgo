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

#include <random>

#include <gtest/gtest.h>

#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/matrix_data.hpp>
#include <ginkgo/core/base/mtx_io.hpp>
#include <ginkgo/core/base/utils_helper.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/preconditioner/gauss_seidel.hpp>

#include "core/test/utils.hpp"
#include "core/utils/matrix_utils.hpp"
#include "test/utils/executor.hpp"

enum struct matrix_type { lower, upper, general, spd };

namespace {
// template <typename ValueIndexType>
class GaussSeidel : public ::CommonTestFixture {
protected:
    /* using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type; */
    using Csr = gko::matrix::Csr<value_type, index_type>;
    using Dense = gko::matrix::Dense<value_type>;
    using GS = gko::preconditioner::GaussSeidel<value_type, index_type>;
    GaussSeidel() : rand_engine(42), tol(r<value_type>::value) {}

    // source isai_kernels.cpp test
    void initialize_data(matrix_type type, gko::size_type n,
                         gko::size_type row_limit, gko::size_type num_rhs)
    {
        const bool for_lower_tm = type == matrix_type::lower;
        auto nz_dist = std::uniform_int_distribution<index_type>(1, row_limit);
        auto val_dist =
            std::uniform_real_distribution<gko::remove_complex<value_type>>(-1.,
                                                                            1.);
        mtx = gko::share(Csr::create(ref));
        if (type == matrix_type::general) {
            auto dense_mtx = gko::test::generate_random_matrix<Dense>(
                n, n, nz_dist, val_dist, rand_engine, ref, gko::dim<2>{n, n});
            ensure_diagonal(dense_mtx.get());
            mtx->copy_from(dense_mtx);
        } else if (type == matrix_type::spd) {
            auto dense_mtx = gko::test::generate_random_band_matrix<Dense>(
                n, row_limit / 4, row_limit / 4, val_dist, rand_engine, ref,
                gko::dim<2>{n, n});
            ensure_diagonal(dense_mtx.get());
            auto transp = gko::as<Dense>(dense_mtx->transpose());
            auto spd_mtx = Dense::create(ref, gko::dim<2>{n, n});
            dense_mtx->apply(transp, spd_mtx);
            mtx->copy_from(spd_mtx);
        } else {
            mtx = gko::test::generate_random_triangular_matrix<Csr>(
                n, true, for_lower_tm, nz_dist, val_dist, rand_engine, ref,
                gko::dim<2>{n, n});
        }
        d_mtx = gko::clone(exec, mtx);
        b = gko::share(gko::test::generate_random_matrix<Dense>(
            n, num_rhs, std::uniform_int_distribution<>(num_rhs, num_rhs),
            std::normal_distribution<>(0.0, 1.0), rand_engine, ref));
        d_b = gko::clone(exec, b);
        x = gko::share(gko::test::generate_random_matrix<Dense>(
            n, num_rhs, std::uniform_int_distribution<>(num_rhs, num_rhs),
            std::normal_distribution<>(0.0, 1.0), rand_engine, ref));
        d_x = gko::clone(exec, x);
    }

    void ensure_diagonal(Dense* mtx)
    {
        for (int i = 0; i < mtx->get_size()[0]; ++i) {
            mtx->at(i, i) = gko::one<value_type>();
        }
    }
    std::default_random_engine rand_engine;
    std::shared_ptr<Csr> mtx;
    std::shared_ptr<Csr> d_mtx;
    std::shared_ptr<Dense> x;
    std::shared_ptr<Dense> b;
    std::shared_ptr<Dense> d_x;
    std::shared_ptr<Dense> d_b;
    gko::remove_complex<value_type> tol;
};

/* TYPED_TEST_SUITE(
    GaussSeidel,
    gko::test::RealValueIndexTypes,
    PairTypenameNameGenerator); */


TEST_F(GaussSeidel, GSApplyEqualToLTRSRef)
{
    /* using GS = typename TestFixture::GS;
    using Csr = typename TestFixture::Csr;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type; */

    this->initialize_data(matrix_type::spd, 401, 23, 1);
    auto ref_gs = GS::build()
                      .on(CommonTestFixture::ref)
                      ->generate(gko::as<gko::LinOp>(this->mtx));
    auto ref_x = this->x->clone();
    ref_gs->apply(this->b, this->x);
    auto mat_data = gko::matrix_data<value_type, index_type>(
        this->mtx->get_size(), this->mtx->get_num_stored_elements());
    this->mtx->write(mat_data);
    gko::utils::make_lower_triangular(mat_data);
    mat_data.ensure_row_major_order();
    auto ref_mtx = gko::share(Csr::create(CommonTestFixture::ref));
    ref_mtx->read(mat_data);
    auto ref_ltrs = gko::solver::LowerTrs<value_type, index_type>::build()
                        .on(CommonTestFixture::ref)
                        ->generate(gko::as<gko::LinOp>(ref_mtx));
    ref_ltrs->apply(this->b, ref_x);
    GKO_ASSERT_MTX_NEAR(this->x, ref_x, this->tol);
}
TEST_F(GaussSeidel, GSApplyEqualToRef)
{
    /* using GS = typename TestFixture::GS; */
    this->initialize_data(matrix_type::spd, 401, 23, 1);
    auto ref_gs = GS::build()
                      .on(CommonTestFixture::ref)
                      ->generate(gko::as<gko::LinOp>(this->mtx));
    auto d_gs = GS::build()
                    .on(CommonTestFixture::exec)
                    ->generate(gko::as<gko::LinOp>(this->d_mtx));
    ref_gs->apply(this->b, this->x);
    d_gs->apply(this->d_b, this->d_x);
    GKO_ASSERT_MTX_NEAR(this->x, this->d_x, this->tol);
}

TEST_F(GaussSeidel, SSORApplyEqualToRef)
{
    this->initialize_data(matrix_type::spd, 521, 33, 1);
    auto ref_gs = GS::build()
                      .with_symmetric_preconditioner(true)
                      .with_relaxation_factor(value_type{1.5})
                      .on(CommonTestFixture::ref)
                      ->generate(gko::as<gko::LinOp>(this->mtx));
    auto d_gs = GS::build()
                    .with_symmetric_preconditioner(true)
                    .with_relaxation_factor(value_type{1.5})
                    .on(CommonTestFixture::exec)
                    ->generate(gko::as<gko::LinOp>(this->d_mtx));
    ref_gs->apply(this->b, this->x);
    d_gs->apply(this->d_b, this->d_x);
    GKO_ASSERT_MTX_NEAR(this->x, this->d_x, this->tol);
}

}  // namespace