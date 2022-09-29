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
#include <random>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/matrix_data.hpp>
#include <ginkgo/core/base/polymorphic_object.hpp>
#include <ginkgo/core/log/convergence.hpp>
#include <ginkgo/core/log/logger.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>
#include <ginkgo/core/solver/cg.hpp>
#include <ginkgo/core/solver/ir.hpp>
#include <ginkgo/core/solver/lower_trs.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm.hpp>

#include "core/matrix/csr_kernels.hpp"
#include "core/preconditioner/gauss_seidel_kernels.hpp"
#include "core/preconditioner/sparse_display.hpp"
#include "core/test/utils.hpp"
#include "core/utils/matrix_utils.hpp"
#include "matrices/config.hpp"

namespace {
template <typename ValueIndexType>
class GaussSeidel : public ::testing ::Test,
                    public ::testing::WithParamInterface<std::tuple<int, int>> {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using GS = gko::preconditioner::GaussSeidel<value_type, index_type>;
    using Ir = gko::solver::Ir<value_type>;
    using CG = gko::solver::Cg<value_type>;
    using Iter = gko::stop::Iteration;
    using ResNorm = gko::stop::ResidualNorm<value_type>;
    using Csr = gko::matrix::Csr<value_type, index_type>;
    using Vec = gko::matrix::Dense<value_type>;
    using Diagonal = gko::matrix::Diagonal<value_type>;
    using MatData = gko::matrix_data<value_type, index_type>;
    using Log = gko::log::Convergence<value_type>;


    GaussSeidel()
        : exec{gko::ReferenceExecutor::create()},
          rand_engine{15},
          iter_logger(Log::create()),
          gs_factory(GS::build().on(exec)),
          ref_gs_factory(GS::build().with_use_reference(true).on(exec)),
          iter_criterion_factory(Iter::build().with_max_iters(100u).on(exec)),
          res_norm_criterion_factory(
              ResNorm::build()
                  .with_reduction_factor(r<value_type>::value)
                  .on(exec)),
          ir_gs_factory(Ir::build()
                            .with_solver(gs_factory)
                            .with_criteria(iter_criterion_factory,
                                           res_norm_criterion_factory)
                            .on(exec)),
          mtx_dense_2{gko::initialize<Vec>(
              {{0.9, -1.0, 3.0}, {0.0, 1.0, 3.0}, {0.0, 0.0, 1.1}}, exec)},
          mtx_csr_2{Csr::create(exec)},
          mtx_csr_3{gko::initialize<Csr>({{10.0, -1.0, 2.0, 0.0},
                                          {-1.0, 11.0, -1.0, 3.0},
                                          {2.0, -1.0, 10.0, -1.0},
                                          {0.0, 3.0, -1.0, 8.0}},
                                         exec)},
          mtx_csr_4{Csr::create(exec, gko::dim<2>{5}, 15)},
          mtx_rand{Csr::create(exec, gko::dim<2>{500})},
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
          ltr_ans_4{gko::initialize<Vec>(
              {value_type{6.0 / 10.0}, value_type{253.0 / 110.0},
               value_type{-116.0 / 100.0}, value_type{1155.0 / 800.0},
               value_type{-6883.0 / 11200.0}},
              exec)},
          rhs_rand{gko::test::generate_random_matrix<Vec>(
              mtx_rand->get_size()[0], 1,
              std::uniform_int_distribution<index_type>(
                  mtx_rand->get_size()[0], mtx_rand->get_size()[0]),
              std::normal_distribution<gko::remove_complex<value_type>>(-1.0,
                                                                        1.0),
              rand_engine, exec)}

    {
        mtx_dense_2->convert_to(gko::lend(mtx_csr_2));
        init_array<index_type>(mtx_csr_4->get_row_ptrs(), {0, 3, 6, 9, 12, 15});
        init_array<index_type>(mtx_csr_4->get_col_idxs(),
                               {0, 1, 2, 0, 1, 3, 0, 2, 4, 1, 3, 4, 2, 3, 4});
        init_array<value_type>(mtx_csr_4->get_values(),
                               {10.0, -0.5, 1.0, -0.5, 11.0, 1.5, 1.0, 10.0,
                                -0.5, 1.5, 8.0, 0.5, -0.5, 0.5, 7.0});

        auto rand_mat_data =
            gko::test::generate_random_matrix_data<value_type, index_type>(
                mtx_rand->get_size()[0], mtx_rand->get_size()[1],
                std::uniform_int_distribution<index_type>(5, 10),
                std::normal_distribution<gko::remove_complex<value_type>>(-1.0,
                                                                          1.0),
                rand_engine);
        gko::utils::make_hpd(rand_mat_data, 2.0);
        rand_mat_data.ensure_row_major_order();
        mtx_rand->read(rand_mat_data);
    }

    template <typename ValueType, typename IndexType>
    std::unique_ptr<gko::matrix::Csr<ValueType, IndexType>>
    generate_rand_matrix(IndexType size, IndexType num_elems_lo,
                         IndexType num_elems_hi, ValueType deduction_help,
                         gko::remove_complex<ValueType> values_lo = -1.0,
                         gko::remove_complex<ValueType> values_hi = 1.0)
    {
        auto mtx = gko::matrix::Csr<ValueType, IndexType>::create(
            exec, gko::dim<2>(size));
        auto mat_data =
            gko::test::generate_random_matrix_data<ValueType, IndexType>(
                mtx->get_size()[0], mtx->get_size()[1],
                std::uniform_int_distribution<IndexType>(num_elems_lo,
                                                         num_elems_hi),
                std::normal_distribution<gko::remove_complex<ValueType>>(
                    values_lo, values_hi),
                rand_engine);

        gko::utils::make_hpd(mat_data, 2.0);
        mat_data.ensure_row_major_order();
        mtx->read(mat_data);

        return give(mtx);
    }
    template <typename ValueType>
    std::unique_ptr<gko::matrix::Dense<ValueType>> generate_rand_dense(
        ValueType deduction_help, size_t num_rows, size_t num_cols = 1,
        gko::remove_complex<ValueType> values_lo = -1.0,
        gko::remove_complex<ValueType> values_hi = 1.0)
    {
        auto rhs_rand{
            gko::test::generate_random_matrix<gko::matrix::Dense<ValueType>>(
                num_rows, num_cols,
                std::uniform_int_distribution<size_t>(num_rows, num_rows),
                std::normal_distribution<gko::remove_complex<ValueType>>(
                    values_lo, values_hi),
                rand_engine, exec)};

        return give(rhs_rand);
    }

    // Source: jacobi_kernels.cpp (test)
    template <typename T>
    void init_array(T* arr, I<T> vals)
    {
        for (auto elem : vals) {
            *(arr++) = elem;
        }
    }

    template <typename ValueType, typename IndexType>
    void print_csr(const gko::matrix::Csr<ValueType, IndexType>* matrix)
    {
        const IndexType* row_ptrs = matrix->get_const_row_ptrs();
        const IndexType* col_idxs = matrix->get_const_col_idxs();
        const ValueType* values = matrix->get_const_values();
        for (auto row = 0; row < matrix->get_size()[0]; row++) {
            for (auto j = row_ptrs[row]; j < row_ptrs[row + 1]; j++) {
                auto col = col_idxs[j];
                auto val = values[j];
                std::cout << "(" << row << ", " << col << ", " << val << ")"
                          << std::endl;
            }
        }
    }

    // represents a 5/9-point stencil on a regular grid
    template <typename ValueType, typename size_type>
    std::unique_ptr<Csr> generate_2D_regular_grid_matrix(
        size_type size, ValueType deduction_help, bool nine_point = false)
    {
        gko::dim<2>::dimension_type grid_points = size * size;
        MatData data(gko::dim<2>{grid_points});

        auto matrix = Csr::create(exec, gko::dim<2>{grid_points});

        for (auto iy = 0; iy < size; iy++) {
            for (auto ix = 0; ix < size; ix++) {
                auto current_row = iy * size + ix;
                for (auto ofs_y : {-1, 0, 1}) {
                    if (iy + ofs_y > -1 && iy + ofs_y < size) {
                        for (auto ofs_x : {-1, 0, 1}) {
                            if (ix + ofs_x > -1 && ix + ofs_x < size) {
                                if (nine_point) {
                                    auto current_col =
                                        current_row + ofs_y * size + ofs_x;
                                    if (current_col == current_row) {
                                        data.nonzeros.emplace_back(
                                            current_row, current_col, 8.0);
                                    } else {
                                        data.nonzeros.emplace_back(
                                            current_row, current_col, -1.0);
                                    }

                                } else {
                                    if (std::abs(ofs_x) + std::abs(ofs_y) < 2) {
                                        auto current_col =
                                            current_row + ofs_y * size + ofs_x;
                                        if (current_col == current_row) {
                                            data.nonzeros.emplace_back(
                                                current_row, current_col, 4.0);
                                        } else {
                                            data.nonzeros.emplace_back(
                                                current_row, current_col, -1.0);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        data.ensure_row_major_order();
        matrix->read(data);
        return std::move(matrix);
    }

    template <typename ValueType>
    void print_array(gko::array<ValueType>& arr)
    {
        for (auto i = 0; i < arr.get_num_elems(); i++) {
            std::cout << arr.get_data()[i] << " ";
        }
        std::cout << std::endl;
    }

    template <typename ValueType, typename IndexType>
    void visualize(gko::matrix::Csr<ValueType, IndexType>* csr_mat,
                   std::string plot_label)
    {
        auto dense_mat = Vec::create(exec);
        csr_mat->convert_to(lend(dense_mat));
        auto num_rows = dense_mat->get_size()[0];
        gko::preconditioner::visualize::spy_ge(
            num_rows, num_rows, dense_mat->get_values(), plot_label);
    }

    std::shared_ptr<const gko::ReferenceExecutor> exec;
    std::default_random_engine rand_engine;
    std::shared_ptr<Log> iter_logger;
    std::shared_ptr<typename GS::Factory> gs_factory;
    std::shared_ptr<typename GS::Factory> ref_gs_factory;
    std::shared_ptr<typename Iter::Factory> iter_criterion_factory;
    std::shared_ptr<typename ResNorm::Factory> res_norm_criterion_factory;
    std::unique_ptr<typename Ir::Factory> ir_gs_factory;
    std::shared_ptr<Vec> mtx_dense_2;
    std::shared_ptr<Csr> mtx_csr_2;
    // example wiki 2
    std::shared_ptr<Csr> mtx_csr_3;
    std::shared_ptr<Csr> mtx_csr_4;
    std::shared_ptr<Csr> mtx_rand;
    std::shared_ptr<Vec> rhs_2;
    std::shared_ptr<Vec> ans_2;
    std::shared_ptr<Vec> rhs_3;
    std::shared_ptr<Vec> ans_3;
    std::shared_ptr<Vec> ltr_ans_3;
    std::shared_ptr<Vec> x_1_3;
    std::shared_ptr<Vec> rhs_4;
    std::shared_ptr<Vec> ltr_ans_4;
    std::shared_ptr<Vec> rhs_rand;
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
    using ValueType = typename TestFixture::value_type;

    auto ans = Vec::create_with_config_of(lend(this->rhs_4));
    ans->fill(ValueType{0});
    auto ref_gs = this->ref_gs_factory->generate(this->mtx_csr_4);
    ref_gs->apply(lend(this->rhs_4), lend(ans));

    GKO_ASSERT_MTX_NEAR(ans, this->ltr_ans_4, r<ValueType>::value);
}

TYPED_TEST(GaussSeidel, ReferenceSimpleApply_2)
{
    using Vec = typename TestFixture::Vec;
    using ValueType = typename TestFixture::value_type;

    auto ans = Vec::create_with_config_of(lend(this->rhs_3));
    auto ref_gs = this->ref_gs_factory->generate(this->mtx_csr_3);
    ref_gs->apply(lend(this->rhs_3), lend(ans));

    GKO_ASSERT_MTX_NEAR(ans, this->ltr_ans_3, r<ValueType>::value);
}

TYPED_TEST(GaussSeidel, ReferenceSimpleApplyKernel_rand_mat_spd)
{
    using Csr = typename TestFixture::Csr;
    using Vec = typename TestFixture::Vec;
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    auto exec = this->exec;

    auto mtx_rand = this->mtx_rand;
    auto ref_mtx_rand = share(Csr::create(exec));
    gko::matrix_data<ValueType, IndexType> rand_mat_data;
    mtx_rand->write(rand_mat_data);
    gko::utils::make_lower_triangular(rand_mat_data);
    ref_mtx_rand->read(rand_mat_data);

    auto rhs_rand = this->rhs_rand;

    auto x = Vec::create_with_config_of(lend(rhs_rand));
    x->fill(ValueType{0});
    auto ref_x = Vec::create_with_config_of(lend(rhs_rand));
    ref_x->fill(ValueType{0});

    auto ref_gs = this->ref_gs_factory->generate(mtx_rand);
    auto ltrs_factory =
        gko::solver::LowerTrs<ValueType, IndexType>::build().on(exec);
    auto ref_ltrs = ltrs_factory->generate(ref_mtx_rand);

    ref_gs->apply(lend(rhs_rand), lend(x));
    ref_ltrs->apply(lend(rhs_rand), lend(ref_x));

    GKO_ASSERT_MTX_NEAR(x, ref_x, r<ValueType>::value);
}

TYPED_TEST(GaussSeidel, SimpleApplyKernel)
{
    using Vec = typename TestFixture::Vec;
    using ValueType = typename TestFixture::value_type;

    auto ans = Vec::create_with_config_of(lend(this->rhs_4));
    ans->fill(ValueType{0});
    auto gs = this->gs_factory->generate(this->mtx_csr_4);

    gs->apply(lend(this->rhs_4), lend(ans));

    GKO_ASSERT_MTX_NEAR(ans,
                        l({ValueType{6.0 / 10.0}, ValueType{3598.0 / 1760.0},
                           ValueType{-116.0 / 100.0}, ValueType{15.0 / 8.0},
                           ValueType{-1807.0 / 2800.0}}),
                        r<ValueType>::value);
}

TYPED_TEST(GaussSeidel, SimpleApplyKernel_2)
{
    using Vec = typename TestFixture::Vec;
    using ValueType = typename TestFixture::value_type;

    auto ans = Vec::create_with_config_of(lend(this->rhs_3));
    ans->fill(ValueType{0});
    auto gs = this->gs_factory->generate(this->mtx_csr_3);
    gs->apply(lend(this->rhs_3), lend(ans));

    GKO_ASSERT_MTX_NEAR(ans,
                        l({ValueType{6.0 / 10.0}, ValueType{799.0 / 440.0},
                           ValueType{-3744.0 / 4400.0}, ValueType{15.0 / 8.0}}),
                        r<ValueType>::value);
}

TYPED_TEST(GaussSeidel, SimpleApplyKernel_rand_mat_spd)
{
    using Csr = typename TestFixture::Csr;
    using Vec = typename TestFixture::Vec;
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    auto exec = this->exec;

    auto mtx_rand = this->mtx_rand;
    auto rhs_rand = this->rhs_rand;

    auto x = Vec::create_with_config_of(lend(rhs_rand));
    x->fill(ValueType{0});
    auto ref_x = Vec::create_with_config_of(lend(rhs_rand));
    ref_x->fill(ValueType{0});

    auto gs = this->gs_factory->generate(mtx_rand);

    auto ltrs_factory =
        gko::solver::LowerTrs<ValueType, IndexType>::build().on(exec);

    gko::matrix_data<ValueType, IndexType> ref_data;
    gs->get_ltr_matrix()->write(ref_data);
    ref_data.ensure_row_major_order();

    auto ref_mtx = share(Csr::create(exec));
    ref_mtx->read(ref_data);
    auto ref_ltrs = ltrs_factory->generate(ref_mtx);

    auto perm_idxs_view =
        gko::array<IndexType>(exec, gs->get_permutation_idxs());

    const auto rhs_rand_perm =
        gko::as<const Vec>(lend(rhs_rand)->row_permute(&perm_idxs_view));

    gs->apply(lend(rhs_rand), lend(x));
    ref_ltrs->apply(lend(rhs_rand_perm), lend(ref_x));

    auto ref_ans = Vec::create(exec);
    ref_ans->copy_from(
        std::move(gko::as<Vec>(ref_x->inverse_row_permute(&perm_idxs_view))));

    GKO_ASSERT_MTX_NEAR(x, ref_ans, r<ValueType>::value);
}

TYPED_TEST(GaussSeidel, SimpleApplyDiagonalMatrix)
{
    using Vec = typename TestFixture::Vec;
    using ValueType = typename TestFixture::value_type;
    using Diagonal = typename TestFixture::Diagonal;

    auto diag_vals =
        gko::array<ValueType>(this->exec, I<ValueType>({1, 2, 3, 4, 5}));

    auto diag_mat = share(
        Diagonal::create(this->exec, diag_vals.get_num_elems(), diag_vals));

    auto ans = Vec::create_with_config_of(lend(this->rhs_4));
    ans->fill(ValueType{0});
    auto gs = this->gs_factory->generate(diag_mat);
    gs->apply(lend(this->rhs_4), lend(ans));


    GKO_ASSERT_MTX_NEAR(
        ans,
        l({ValueType{6.0 / 1.0}, ValueType{25.0 / 2.0}, ValueType{-11.0 / 3.0},
           ValueType{15.0 / 4.0}, ValueType{-3.0 / 5.0}}),
        r<ValueType>::value);
}

TYPED_TEST(GaussSeidel, SimpleApplyKernel_multi_rhs)
{
    using Vec = typename TestFixture::Vec;
    using ValueType = typename TestFixture::value_type;

    auto mtx = this->mtx_csr_4;
    auto rhs = gko::initialize<Vec>(
        {I<ValueType>{6.0, 3.0}, I<ValueType>{25.0, 12.5},
         I<ValueType>{-11.0, -5.5}, I<ValueType>{15.0, 7.5},
         I<ValueType>{-3.0, -1.5}},
        this->exec);
    auto ans = Vec::create_with_config_of(gko::lend(rhs));
    ans->fill(ValueType{0});
    auto gs = this->gs_factory->generate(mtx);

    gs->apply(gko::lend(rhs), gko::lend(ans));

    GKO_ASSERT_MTX_NEAR(
        ans,
        l({{ValueType{6.0 / 10.0}, ValueType{3.0 / 10.0}},
           {ValueType{3598.0 / 1760.0}, ValueType{1799.0 / 1760.0}},
           {ValueType{-116.0 / 100.0}, ValueType{-58.0 / 100.0}},
           {ValueType{15.0 / 8.0}, ValueType{7.5 / 8.0}},
           {ValueType{-1807.0 / 2800.0}, ValueType{-903.5 / 2800.0}}}),
        r<ValueType>::value);
}

TYPED_TEST(GaussSeidel, SystemSolveIRGS)
{
    using Csr = typename TestFixture::Csr;
    using Vec = typename TestFixture::Vec;
    using GS = typename TestFixture::GS;
    using Ir = typename TestFixture::Ir;
    using ValueType = typename TestFixture::value_type;

    auto exec = this->exec;
    auto ir_gs_factory = Ir::build()
                             .with_solver(this->gs_factory)
                             .with_criteria(this->iter_criterion_factory,
                                            this->res_norm_criterion_factory)
                             .on(exec);
    auto irs = ir_gs_factory->generate(this->mtx_csr_3);

    auto result = share(Vec::create_with_config_of(lend(this->rhs_3)));
    result->fill(0.0);

    irs->apply(lend(this->rhs_3), lend(result));

    GKO_ASSERT_MTX_NEAR(result, this->ans_3, r<ValueType>::value);
}

TYPED_TEST(GaussSeidel, SystemSolveIRRefGS)
{
    using Ir = typename TestFixture::Ir;
    using Csr = typename TestFixture::Csr;
    using Vec = typename TestFixture::Vec;
    using ValueType = typename TestFixture::value_type;
    auto exec = this->exec;

    auto result = Vec::create_with_config_of(lend(this->rhs_3));
    result->fill(0.0);

    auto ir_factory = Ir::build()
                          .with_solver(this->ref_gs_factory)
                          .with_criteria(this->iter_criterion_factory,
                                         this->res_norm_criterion_factory)
                          .on(exec);
    auto irs = ir_factory->generate(this->mtx_csr_3);
    irs->apply(lend(this->rhs_3), result.get());

    GKO_ASSERT_MTX_NEAR(result, this->ans_3, r<ValueType>::value);
}

/* TYPED_TEST(GaussSeidel, SystemSolveGSPCG)
{
    using CG = typename TestFixture::CG;
    using Csr = typename TestFixture::Csr;
    using Vec = typename TestFixture::Vec;
    using GS = typename TestFixture::GS;
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using Log = typename TestFixture::Log;
    using namespace gko;

    auto exec = this->exec;

    auto mtx = share(this->generate_rand_matrix(IndexType{1000}, IndexType{5},
                                                IndexType{15}, ValueType{0}));
    auto rhs =
        share(this->generate_rand_dense(ValueType{0}, mtx->get_size()[0]));
    auto x = Vec::create_with_config_of(lend(rhs));
    x->fill(0.0);
    auto x_clone = clone(exec, x);
    auto rhs_clone = clone(exec, rhs);

    auto iter_crit = this->iter_criterion_factory;
    iter_crit->add_logger(this->iter_logger);
    auto res_norm = this->res_norm_criterion_factory;
    res_norm->add_logger(this->iter_logger);

    auto cg_factory = CG::build().with_criteria(iter_crit, res_norm).on(exec);
    auto cg = cg_factory->generate(mtx);
    cg->apply(lend(rhs), lend(x));
    auto cg_num_iters = this->iter_logger->get_num_iterations();

    auto pcg_factory = CG::build()
                           .with_criteria(iter_crit, res_norm)
                           .with_preconditioner(this->gs_factory)
                           .on(exec);
    auto pcg = pcg_factory->generate(mtx);
    pcg->apply(lend(rhs_clone), lend(x_clone));

    auto pcg_num_iters = this->iter_logger->get_num_iterations();

    GKO_ASSERT_EQ(pcg_num_iters < cg_num_iters, 1);
    // GKO_ASSERT_MTX_NEAR(x, x_clone, r<ValueType>::value);
} */

TYPED_TEST(GaussSeidel, CorrectColoringRegularGrid)
{
    using IndexType = typename TestFixture::index_type;
    using ValueType = typename TestFixture::value_type;
    using SparsityCsr = typename gko::matrix::SparsityCsr<ValueType, IndexType>;
    using Csr = typename TestFixture::Csr;
    auto exec = this->exec;
    size_t grid_size = 10;
    auto regular_grid_matrix =
        share(this->generate_2D_regular_grid_matrix(grid_size, ValueType{0}));

    gko::array<IndexType> vertex_colors{exec, grid_size * grid_size};
    vertex_colors.fill(IndexType{-1});
    gko::array<IndexType> ans{exec, vertex_colors};
    for (auto i = 0; i < ans.get_num_elems(); i++) {
        if (grid_size % 2 == 1) {
            ans.get_data()[i] = i % 2;
        } else {
            ans.get_data()[i] = (i % grid_size + (i / grid_size) % 2) % 2;
        }
    }

    auto tmp = gko::copy_and_convert_to<SparsityCsr>(exec, regular_grid_matrix);
    auto adjacency_matrix = SparsityCsr::create(exec);
    adjacency_matrix = std::move(tmp->to_adjacency_matrix());
    IndexType max_color{0};
    gko::kernels::reference::gauss_seidel::get_coloring(
        exec, lend(adjacency_matrix), vertex_colors, &max_color);

    GKO_ASSERT_ARRAY_EQ(vertex_colors, ans);
}

// TODO not sure if this test is needed anymore
// TYPED_TEST(GaussSeidel, CorrectReorderingRegularGrid)
// {
//     using IndexType = typename TestFixture::index_type;
//     using ValueType = typename TestFixture::value_type;
//     auto exec = this->exec;
//     size_t grid_size = 3;
//     auto regular_grid_matrix =
//         share(this->generate_2D_regular_grid_matrix(grid_size,
//         ValueType{0}));
//     auto gs = this->gs_factory->generate(regular_grid_matrix);
//     auto perm_arr = gs->get_permutation_idxs();
//     GKO_ASSERT_EQ(0, 1);
// }

TYPED_TEST(GaussSeidel, GetSecondaryOrderingKernel)
{
    using Vec = typename TestFixture::Vec;
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using Csr = typename TestFixture::Csr;
    auto exec = this->exec;
    auto mtx = Csr::create(exec, gko::dim<2>{8}, 16);


    this->template init_array<IndexType>(mtx->get_row_ptrs(),
                                         {0, 2, 4, 6, 8, 10, 12, 14, 16});
    this->template init_array<IndexType>(
        mtx->get_col_idxs(), {0, 1, 0, 1, 2, 3, 2, 3, 4, 5, 4, 5, 6, 7, 6, 7});
    this->template init_array<ValueType>(
        mtx->get_values(), {1.0, 12.0, 12.0, 2., 3., 34., 34., 4., 5., 56., 56.,
                            6., 7., 78., 78., 8.});

    gko::array<IndexType> color_ptrs(exec, 2);
    this->template init_array<IndexType>(
        color_ptrs.get_data(), {0, 8});  // all nodes have the same color

    gko::array<IndexType> block_ordering(exec, 8);
    this->template init_array<IndexType>(block_ordering.get_data(),
                                         {0, 1, 2, 3, 4, 5, 6, 7});

    const IndexType base_block_size = 2;
    const IndexType lvl_2_block_size = 4;
    const IndexType max_color = 0;

    auto dummy_storage = gko::preconditioner::storage_scheme(1);

    gko::kernels::reference::gauss_seidel::get_secondary_ordering(
        exec, block_ordering.get_data(), dummy_storage, base_block_size,
        lvl_2_block_size, color_ptrs.get_const_data(), max_color);

    GKO_ASSERT_ARRAY_EQ(block_ordering, I<IndexType>({0, 2, 4, 6, 1, 3, 5, 7}));
}

TYPED_TEST(GaussSeidel, SimpleAssignToBlocksKernel)
{
    using IndexType = typename TestFixture::index_type;
    using ValueType = typename TestFixture::value_type;
    using SparsityCsr = typename gko::matrix::SparsityCsr<ValueType, IndexType>;

    auto exec = this->exec;

    auto adjacency_mtx =
        gko::initialize<SparsityCsr>({{0., 0., 0., 1., 0., 0.},
                                      {0., 0., 1., 1., 1., 0.},
                                      {0., 1., 0., 1., 0., 0.},
                                      {1., 1., 1., 0., 1., 0.},
                                      {0., 1., 0., 1., 0., 0.},
                                      {0., 0., 0., 0., 0., 0.}},
                                     exec);
    auto num_nodes = adjacency_mtx->get_size()[0];
    gko::array<IndexType> block_ordering(exec, num_nodes);
    gko::array<IndexType> degrees(exec, I<IndexType>({1, 3, 2, 4, 2, 0}));
    gko::array<gko::int8> visited(exec, num_nodes);
    std::fill_n(visited.get_data(), num_nodes, gko::int8{0});
    const IndexType block_size = 2;

    gko::kernels::reference::gauss_seidel::assign_to_blocks(
        exec, gko::lend(adjacency_mtx), block_ordering.get_data(),
        degrees.get_const_data(), visited.get_data(), block_size);

    // for minDegree seedPolicy & maxNumEdges policy
    GKO_ASSERT_ARRAY_EQ(block_ordering, I<IndexType>({5, 0, 2, 1, 4, 3}));
}

TYPED_TEST(GaussSeidel, AssignToBlocksKernel)
{
    using IndexType = typename TestFixture::index_type;
    using ValueType = typename TestFixture::value_type;
    using SparsityCsr = typename gko::matrix::SparsityCsr<ValueType, IndexType>;

    auto exec = this->exec;

    auto adjacency_mtx =
        gko::initialize<SparsityCsr>({{0., 0., 0., 1., 0., 0., 1., 0., 0.},
                                      {0., 0., 1., 1., 1., 0., 0., 1., 0.},
                                      {0., 1., 0., 1., 0., 0., 0., 0., 1.},
                                      {1., 1., 1., 0., 1., 0., 0., 1., 0.},
                                      {0., 1., 0., 1., 0., 0., 0., 0., 1.},
                                      {0., 0., 0., 0., 0., 0., 1., 0., 0.},
                                      {1., 0., 0., 0., 0., 1., 0., 1., 1.},
                                      {0., 1., 0., 1., 0., 0., 1., 0., 0.},
                                      {0., 0., 1., 0., 1., 0., 1., 0., 0.}},
                                     exec);
    auto num_nodes = adjacency_mtx->get_size()[0];
    gko::array<IndexType> block_ordering(exec, num_nodes);
    gko::array<IndexType> degrees(exec,
                                  I<IndexType>({2, 4, 3, 5, 3, 1, 4, 3, 3}));
    gko::array<gko::int8> visited(exec, num_nodes);
    std::fill_n(visited.get_data(), num_nodes, gko::int8{0});

    IndexType block_size = 4;
    gko::kernels::reference::gauss_seidel::assign_to_blocks(
        exec, gko::lend(adjacency_mtx), block_ordering.get_data(),
        degrees.get_const_data(), visited.get_data(), block_size);

    // // for minDegree seedPolicy & maxNumEdges policy
    GKO_ASSERT_ARRAY_EQ(block_ordering,
                        I<IndexType>({5, 6, 0, 7, 2, 1, 3, 4, 8}));


    block_size = 6;
    std::fill_n(visited.get_data(), num_nodes, gko::int8{0});
    block_ordering.resize_and_reset(num_nodes);
    gko::kernels::reference::gauss_seidel::assign_to_blocks(
        exec, gko::lend(adjacency_mtx), block_ordering.get_data(),
        degrees.get_const_data(), visited.get_data(), block_size);

    // // for minDegree seedPolicy & maxNumEdges policy
    GKO_ASSERT_ARRAY_EQ(block_ordering,
                        I<IndexType>({5, 6, 0, 7, 3, 1, 2, 8, 4}));
}

TYPED_TEST(GaussSeidel, GetBlockColoringKernel) {}

// TODO
// Test if nodes within blocks are contigous after permuting (similar
// formulation) / test assign_to_blocks

TYPED_TEST(GaussSeidel, SecondaryOrderingSetupBlocksKernel)
{
    using IndexType = typename TestFixture::index_type;
    using ValueType = typename TestFixture::value_type;
    using Csr = typename TestFixture::Csr;
    using Vec = typename TestFixture::Vec;
    auto exec = this->exec;

    auto mtx = Csr::create(exec, gko::dim<2>{18}, 56);
    this->template init_array<IndexType>(
        mtx->get_row_ptrs(), {0, 4, 9, 12, 15, 18, 20, 22, 27, 30, 33, 36, 39,
                              42, 43, 46, 49, 52, 56});
    this->template init_array<IndexType>(
        mtx->get_col_idxs(),
        {0,  1,  2,  15, 0, 1,  5,  7,  15, 0,  2,  17, 3,  7,
         9,  4,  9,  14, 1, 5,  6,  16, 1,  3,  7,  8,  12, 7,
         8,  17, 3,  4,  9, 10, 12, 14, 11, 16, 17, 7,  10, 12,
         13, 4,  10, 14, 0, 1,  15, 6,  11, 16, 2,  8,  11, 17});
    this->template init_array<ValueType>(
        mtx->get_values(),
        {10., 5.,  2.,  1.,  5.,  6.,  0.5, 2., 5.,  2.,  9.7, 6., 3., 4.,
         1.,  5.2, 12., 4.,  0.5, 15., 2.,  2., 2.,  4.,  2.,  3., 4., 3.,
         3.,  6.,  1.,  12., 4.7, 1.5, 7.,  7., 1.3, 3.,  9.,  4., 7., 4.,
         8.,  4.,  7.,  1.,  1.,  5.,  4.,  2., 3.,  6.5, 6.,  6., 9., 2.2});

    gko::array<IndexType> perm(
        exec, I<IndexType>({10, 12, 17, 11, 5, 1, 9, 4, 3, 7, 13, 16, 0, 2, 14,
                            6, 8, 15}));
    gko::array<IndexType> inv_perm(exec, perm.get_num_elems());
    gko::array<IndexType> color_block_ptrs(exec, I<IndexType>({0, 8, 14, 18}));
    gko::array<IndexType> perm_after_2nd_ordering(
        exec, I<IndexType>({10, 17, 5, 12, 11, 1, 9, 4, 3, 13, 0, 7, 16, 2, 14,
                            6, 8, 15}));
    GKO_ASSERT(mtx->get_size()[0] == perm.get_num_elems());

    gko::array<ValueType> expected_l_diag_vals(
        exec, I<ValueType>({1.5, 2.2, 15., 7., 9., 0.5, 4., 1.3, 6.,
                            4.7, 12., 5.2, 3., 8., 10., 4., 0.,  2.,
                            2.,  6.5, 9.7, 1., 0., 2.,  3., 0.,  4.}));
    auto expected_l_diag_vals_vec = Vec::create(
        exec, gko::dim<2>{expected_l_diag_vals.get_num_elems(), 1},
        gko::make_array_view(exec, expected_l_diag_vals.get_num_elems(),
                             expected_l_diag_vals.get_data()),
        1);
    gko::array<IndexType> expected_l_diag_mtx_col_idxs(
        exec, I<IndexType>({0, 3, 1,  1, 2, 2, 2, 0, 1,  2, 1, 0,  0, 0,
                            0, 1, -1, 0, 2, 2, 1, 2, -1, 0, 1, -1, 2}));
    gko::array<IndexType> expected_l_diag_row(
        exec,
        I<IndexType>({0,  1,  2,  3,  4,  5,  3,  4,  5,  6,  7,  7,  8, 9,
                      10, 11, -1, 13, 11, 12, 13, 14, -1, 15, 16, -1, 17}));
    GKO_ASSERT(expected_l_diag_mtx_col_idxs.get_num_elems() ==
               expected_l_diag_vals.get_num_elems());

    gko::array<ValueType> expected_l_spmv_vals(
        exec, I<ValueType>({1., 5., 4., 2., 3., 6., 7., 4., 2., 6., 3., 5., 1.,
                            0., 0., 0., 0., 0., 0.}));
    auto expected_l_spmv_vals_vec = Vec::create(
        exec, gko::dim<2>{expected_l_spmv_vals.get_num_elems(), 1},
        gko::make_array_view(exec, expected_l_spmv_vals.get_num_elems(),
                             expected_l_spmv_vals.get_data()),
        1);
    gko::array<IndexType> expected_l_spmv_col_idxs(
        exec, I<IndexType>({6, 5, 3, 5, 4, 1, 0, 7, 12, 1, 11, 5, 10, 0, 0, 0,
                            0, 0, 0}));
    gko::array<IndexType> expected_l_spmv_row_ptrs(
        exec, I<IndexType>({0, 1, 1, 2, 4, 5, 6, 0, 2, 3, 5, 7}));
    gko::array<IndexType> expected_l_spmv_mtx_col_idxs(
        exec, I<IndexType>(
                  {2, 1, 4, 0, 1, 2, 1, 0, 1, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0}));

    IndexType max_color = 2;
    IndexType base_block_size = 2;
    IndexType lvl_2_block_size = 3;

    gko::preconditioner::storage_scheme dummy_storage_scheme(2 * max_color + 1);

    gko::kernels::reference::gauss_seidel::get_secondary_ordering(
        exec, perm.get_data(), dummy_storage_scheme, base_block_size,
        lvl_2_block_size, color_block_ptrs.get_const_data(), max_color);

    GKO_ASSERT_ARRAY_EQ(perm, perm_after_2nd_ordering);

    gko::kernels::reference::csr::invert_permutation(
        exec, perm.get_num_elems(), perm.get_const_data(), inv_perm.get_data());

    IndexType* dummyInd;
    ValueType* dummyVal;
    const auto diag_mem_requirement = 27;
    const auto l_spmv_val_col_mem_requirement = 19;
    const auto l_spmv_row_mem_requirement = 18 - 8 + max_color;
    gko::array<IndexType> l_diag_rows_(exec, diag_mem_requirement);
    gko::array<IndexType> l_diag_mtx_col_idxs_(exec, diag_mem_requirement);
    gko::array<ValueType> l_diag_vals_(exec, diag_mem_requirement);
    auto l_diag_vals_vec_ =
        Vec::create(exec, gko::dim<2>{l_diag_vals_.get_num_elems(), 1},
                    gko::make_array_view(exec, l_diag_vals_.get_num_elems(),
                                         l_diag_vals_.get_data()),
                    1);
    l_diag_vals_.fill(ValueType{0});
    l_diag_mtx_col_idxs_.fill(IndexType{-1});
    l_diag_rows_.fill(IndexType{-1});
    gko::array<IndexType> l_spmv_row_ptrs_(exec, l_spmv_row_mem_requirement);
    gko::array<IndexType> l_spmv_col_idxs_(exec,
                                           l_spmv_val_col_mem_requirement);
    gko::array<IndexType> l_spmv_mtx_col_idxs_(exec,
                                               l_spmv_val_col_mem_requirement);
    gko::array<ValueType> l_spmv_vals_(exec, l_spmv_val_col_mem_requirement);
    auto l_spmv_vals_vec_ =
        Vec::create(exec, gko::dim<2>{l_spmv_vals_.get_num_elems(), 1},
                    gko::make_array_view(exec, l_spmv_vals_.get_num_elems(),
                                         l_spmv_vals_.get_data()),
                    1);
    l_spmv_vals_.fill(ValueType{0});
    l_spmv_mtx_col_idxs_.fill(IndexType{0});
    l_spmv_col_idxs_.fill(IndexType{0});

    gko::kernels::reference::gauss_seidel::setup_blocks(
        exec, gko::lend(mtx), perm.get_const_data(), inv_perm.get_const_data(),
        dummy_storage_scheme, l_diag_rows_.get_data(),
        l_diag_mtx_col_idxs_.get_data(), l_diag_vals_.get_data(),
        l_spmv_row_ptrs_.get_data(), l_spmv_col_idxs_.get_data(),
        l_spmv_mtx_col_idxs_.get_data(), l_spmv_vals_.get_data(), dummyInd,
        dummyInd, dummyVal, dummyInd, dummyInd, dummyInd, dummyVal);

    GKO_ASSERT_MTX_NEAR(expected_l_diag_vals_vec, l_diag_vals_vec_,
                        r<ValueType>::value);
    GKO_ASSERT_ARRAY_EQ(expected_l_diag_mtx_col_idxs, l_diag_mtx_col_idxs_);
    GKO_ASSERT_ARRAY_EQ(expected_l_diag_row, l_diag_rows_);

    GKO_ASSERT_ARRAY_EQ(expected_l_spmv_row_ptrs, l_spmv_row_ptrs_);
    GKO_ASSERT_ARRAY_EQ(expected_l_spmv_vals, l_spmv_vals_);
    GKO_ASSERT_ARRAY_EQ(expected_l_spmv_mtx_col_idxs, l_spmv_mtx_col_idxs_);
    GKO_ASSERT_ARRAY_EQ(expected_l_spmv_col_idxs, l_spmv_col_idxs_);


    auto rhs = Vec::create(exec, gko::dim<2>{18, 1});
    this->template init_array<ValueType>(
        rhs->get_values(), {1.5, 2.2, 15., 11., 10.3, 6.5, 4.7, 17.2, 4., 8.,
                            15., 12., 9.5, 17.7, 12., 4., 12., 10.});
    auto x = Vec::create_with_config_of(gko::lend(rhs));
    x->fill(ValueType{0});
    auto ref_x = Vec::create_with_config_of(gko::lend(rhs));
    ref_x->fill(ValueType{1});
    gko::kernels::reference::gauss_seidel::simple_apply(
        exec, l_diag_rows_.get_const_data(), l_diag_vals_.get_const_data(),
        l_spmv_row_ptrs_.get_const_data(), l_spmv_col_idxs_.get_const_data(),
        l_spmv_vals_.get_const_data(), perm.get_const_data(),
        dummy_storage_scheme, lend(rhs), lend(x));

    GKO_ASSERT_MTX_NEAR(x, ref_x, r<ValueType>::value);


    gko::remove_complex<ValueType> a = 2.5;
    auto alpha = gko::initialize<Vec>({a}, exec);
    mtx->scale(lend(alpha));
    expected_l_diag_vals_vec->scale(gko::lend(alpha));
    expected_l_spmv_vals_vec->scale(gko::lend(alpha));
    gko::kernels::reference::gauss_seidel::fill_with_vals(
        exec, lend(mtx), perm.get_const_data(), dummy_storage_scheme,
        static_cast<IndexType>(diag_mem_requirement),
        l_diag_rows_.get_const_data(), l_diag_mtx_col_idxs_.get_const_data(),
        l_diag_vals_.get_data(), l_spmv_row_ptrs_.get_const_data(),
        l_spmv_col_idxs_.get_const_data(),
        l_spmv_mtx_col_idxs_.get_const_data(), l_spmv_vals_.get_data(),
        dummyInd, dummyInd, dummyVal, dummyInd, dummyInd, dummyInd, dummyVal);

    GKO_ASSERT_ARRAY_EQ(expected_l_diag_vals, l_diag_vals_);
    GKO_ASSERT_ARRAY_EQ(expected_l_spmv_vals, l_spmv_vals_);
}

TYPED_TEST(GaussSeidel, SimpleApplyHBMCKernel)
{
    using namespace gko::preconditioner;
    using IndexType = typename TestFixture::index_type;
    using ValueType = typename TestFixture::value_type;
    using Vec = typename TestFixture::Vec;
    auto exec = this->exec;
    const auto b_s = 2;
    const auto w = 2;
    auto rhs = Vec::create(exec, gko::dim<2>{9, 2});
    this->template init_array<ValueType>(
        rhs->get_values(), {2., 4., 2., 4., 3., 6., 3., 6., 3., 6., 6., 12.,
                            10., 20., 15., 30., 13., 26.});
    auto x = Vec::create(exec, gko::dim<2>{9, 2});
    auto exp_x = Vec::create(exec, gko::dim<2>{9, 2});
    this->template init_array<ValueType>(
        exp_x->get_values(), {1., 2., 1., 2., 1., 2., 1., 2., 1., 2., 1., 2.,
                              1., 2., 1., 2., 1., 2.});

    gko::array<IndexType> perm_idxs(exec,
                                    I<IndexType>({0, 1, 2, 3, 4, 5, 6, 7, 8}));
    gko::array<ValueType> l_diag_vals(
        exec,
        I<ValueType>({2., 2., 1., 1., 2., 2., 3., 3., 3., 4., 4., 4., 5.}));

    gko::array<IndexType> l_diag_rows(
        exec, I<IndexType>({0, 1, 2, 3, 2, 3, 4, 5, 5, 6, 7, 7, 8}));
    gko::array<IndexType> l_spmv_row_ptrs(exec, I<IndexType>({0, 1, 2, 3}));
    gko::array<IndexType> l_spmv_col_idxs(exec, I<IndexType>({1, 2, 3}));
    gko::array<ValueType> l_spmv_vals(exec, I<ValueType>({6., 7., 8.}));

    auto storage = storage_scheme(3);

    auto p_block_1 = parallel_block(0, 0, 6, 2, b_s, w, true);
    p_block_1.parallel_blocks_.emplace_back(
        std::make_shared<lvl_1_block>(lvl_1_block(0, 0, 4, b_s, w)));
    p_block_1.parallel_blocks_.emplace_back(
        std::make_shared<base_block_aggregation>(
            base_block_aggregation(6, 4, 6, 1, b_s)));
    storage.forward_solve_.emplace_back(
        std::make_shared<parallel_block>(p_block_1));

    storage.forward_solve_.emplace_back(
        std::make_shared<spmv_block>(spmv_block(0, 0, 6, 9, 0, 6)));

    auto p_block_2 = parallel_block(9, 6, 9, 1, b_s, w, true);
    p_block_2.parallel_blocks_.emplace_back(
        std::make_shared<base_block_aggregation>(
            base_block_aggregation(9, 6, 9, 2, b_s)));
    storage.forward_solve_.emplace_back(
        std::make_shared<parallel_block>(p_block_2));

    gko::kernels::reference::gauss_seidel::simple_apply(
        exec, l_diag_rows.get_const_data(), l_diag_vals.get_const_data(),
        l_spmv_row_ptrs.get_const_data(), l_spmv_col_idxs.get_const_data(),
        l_spmv_vals.get_const_data(), perm_idxs.get_const_data(), storage,
        gko::lend(rhs), gko::lend(x));

    GKO_ASSERT_MTX_NEAR(x, exp_x, r<ValueType>::value);
}

TYPED_TEST(GaussSeidel, SimpleApplyHBMC)
{
    using IndexType = typename TestFixture::index_type;
    using ValueType = typename TestFixture::value_type;
    using Csr = typename TestFixture::Csr;
    using Vec = typename TestFixture::Vec;
    using GS = typename TestFixture::GS;
    auto exec = this->exec;

    auto mtx = gko::share(Csr::create(exec, gko::dim<2>{15}, 63));
    this->template init_array<IndexType>(
        mtx->get_row_ptrs(),
        {0, 3, 8, 12, 18, 20, 25, 29, 34, 39, 43, 47, 52, 56, 60, 63});
    this->template init_array<IndexType>(
        mtx->get_col_idxs(),
        {0, 1, 3, 0,  1, 7,  10, 13, 2,  3,  5,  11, 0,  2,  3,  6,
         7, 8, 4, 6,  2, 5,  7,  9,  11, 3,  4,  6,  14, 1,  3,  5,
         7, 8, 3, 7,  8, 11, 14, 5,  9,  11, 12, 1,  10, 12, 13, 2,
         5, 8, 9, 11, 9, 10, 12, 13, 1,  10, 12, 13, 6,  8,  14});
    this->template init_array<ValueType>(
        mtx->get_values(),
        {2.,  5.,  1., 5.,  7., 1., 6., 10., 6.,  1., 4., 9.,  1., 1., 6.,  1.,
         11., 5.,  8., 6.,  4., 1., 1., 2.,  7.,  1., 6., 3.,  4., 1., 11., 1.,
         13., 10., 5., 10., 4., 1., 1., 2.,  3.,  8., 1., 6.,  4., 1., 9.,  9.,
         7.,  1.,  8., 10., 1., 1., 9., 1.,  10., 9., 1., 11., 4., 1., 5.});
    gko::array<IndexType> perm(
        exec, I<IndexType>({5, 9, 2, 11, 0, 10, 1, 13, 6, 14, 4, 8, 3, 12, 7}));
    gko::array<IndexType> perm_cpy{exec, 15};
    perm_cpy = perm;
    GKO_ASSERT_ARRAY_EQ(perm, perm_cpy);
    gko::array<IndexType> inv_perm(exec, perm.get_num_elems());
    gko::array<IndexType> color_block_ptrs(exec, I<IndexType>({0, 11, 15}));

    auto rhs = Vec::create(exec, gko::dim<2>{15, 1});
    this->template init_array<ValueType>(
        rhs->get_values(), {2., 85., 42., 80., 82., 6., 21., 246., 63., 42.,
                            44., 269., 152., 273., 103.});
    auto x = Vec::create_with_config_of(gko::lend(rhs));
    x->fill(ValueType{0});
    auto ref_x = Vec::create_with_config_of(gko::lend(rhs));
    std::iota(ref_x->get_values(), ref_x->get_values() + 15, 1);

    IndexType max_color = 1;
    IndexType base_block_size = 4;
    IndexType lvl_2_block_size = 3;

    gko::preconditioner::storage_scheme dummy_no_secondary(2 * max_color + 1);

    gko::kernels::reference::gauss_seidel::get_secondary_ordering(
        exec, perm.get_data(), dummy_no_secondary, base_block_size,
        lvl_2_block_size, color_block_ptrs.get_const_data(), max_color);

    GKO_ASSERT_ARRAY_EQ(perm, perm_cpy);  // cannot create a single lvl 1 block
                                          // -> no change in the perm array

    lvl_2_block_size = 2;
    gko::preconditioner::storage_scheme storage_scheme(2 * max_color + 1);
    gko::kernels::reference::gauss_seidel::get_secondary_ordering(
        exec, perm.get_data(), storage_scheme, base_block_size,
        lvl_2_block_size, color_block_ptrs.get_const_data(), max_color);

    gko::array<IndexType> perm_with_second_ordering(
        exec, I<IndexType>({5, 0, 9, 10, 2, 1, 11, 13, 6, 14, 4, 8, 3, 12, 7}));
    GKO_ASSERT_ARRAY_EQ(
        perm, perm_with_second_ordering);  // with smaller lvl 2 block size
                                           // second ordering successful

    gko::kernels::reference::csr::invert_permutation(
        exec, perm.get_num_elems(), perm.get_const_data(), inv_perm.get_data());

    IndexType* dummyInd;
    ValueType* dummyVal;
    const auto diag_mem_requirement = 40;
    const auto l_spmv_val_col_mem_requirement = 24;
    const auto l_spmv_row_mem_requirement = 15 - 11 + max_color;
    gko::array<IndexType> l_diag_rows_(exec, diag_mem_requirement);
    gko::array<IndexType> l_diag_mtx_col_idxs_(exec, diag_mem_requirement);
    gko::array<ValueType> l_diag_vals_(exec, diag_mem_requirement);
    l_diag_vals_.fill(ValueType{0});
    l_diag_mtx_col_idxs_.fill(IndexType{-1});
    l_diag_rows_.fill(IndexType{-1});
    gko::array<IndexType> l_spmv_row_ptrs_(exec, l_spmv_row_mem_requirement);
    gko::array<IndexType> l_spmv_col_idxs_(exec,
                                           l_spmv_val_col_mem_requirement);
    gko::array<IndexType> l_spmv_mtx_col_idxs_(exec,
                                               l_spmv_val_col_mem_requirement);
    gko::array<ValueType> l_spmv_vals_(exec, l_spmv_val_col_mem_requirement);

    l_spmv_vals_.fill(ValueType{0});
    l_spmv_mtx_col_idxs_.fill(IndexType{0});
    l_spmv_col_idxs_.fill(IndexType{0});

    gko::kernels::reference::gauss_seidel::setup_blocks(
        exec, gko::lend(mtx), perm.get_const_data(), inv_perm.get_const_data(),
        storage_scheme, l_diag_rows_.get_data(),
        l_diag_mtx_col_idxs_.get_data(), l_diag_vals_.get_data(),
        l_spmv_row_ptrs_.get_data(), l_spmv_col_idxs_.get_data(),
        l_spmv_mtx_col_idxs_.get_data(), l_spmv_vals_.get_data(), dummyInd,
        dummyInd, dummyVal, dummyInd, dummyInd, dummyInd, dummyVal);

    auto rhs_perm = gko::as<Vec>(rhs->row_permute(&perm));

    gko::kernels::reference::gauss_seidel::simple_apply(
        exec, l_diag_rows_.get_const_data(), l_diag_vals_.get_const_data(),
        l_spmv_row_ptrs_.get_const_data(), l_spmv_col_idxs_.get_const_data(),
        l_spmv_vals_.get_const_data(), perm.get_const_data(), storage_scheme,
        lend(rhs_perm), lend(x));

    GKO_ASSERT_MTX_NEAR(x, ref_x, r<ValueType>::value);
}

TYPED_TEST(GaussSeidel, SimpleApplyHBMC_RandMtx)
{
    using IndexType = typename TestFixture::index_type;
    using ValueType = typename TestFixture::value_type;
    using GS = typename TestFixture::GS;
    using Csr = typename TestFixture::Csr;
    using Vec = typename TestFixture::Vec;

    auto exec = this->exec;
    auto mtx = gko::share(this->generate_rand_matrix(
        IndexType{1003}, IndexType{5}, IndexType{10}, ValueType{0}));

    auto rhs =
        gko::share(this->generate_rand_dense(ValueType{0}, mtx->get_size()[0]));


    auto x = Vec::create_with_config_of(gko::lend(rhs));
    x->fill(ValueType{0});
    auto ref_x = Vec::create_with_config_of(gko::lend(rhs));
    ref_x->fill(ValueType{0});

    auto gs_HBMC_factory = GS::build()
                               .with_use_HBMC(true)
                               .with_base_block_size(2u)
                               .with_lvl_2_block_size(8u)
                               .on(exec);
    auto gs_HBMC = gs_HBMC_factory->generate(mtx);

    auto perm_idxs =
        gko::array<IndexType>(exec, gs_HBMC->get_permutation_idxs());

    auto mtx_perm = gko::as<Csr>(mtx->permute(&perm_idxs));
    gko::matrix_data<ValueType, IndexType> ref_data;
    mtx_perm->write(ref_data);
    gko::utils::make_lower_triangular(ref_data);
    ref_data.ensure_row_major_order();
    auto ref_mtx = gko::share(Csr::create(exec));
    ref_mtx->read(ref_data);
    const auto rhs_perm =
        gko::as<const Vec>(lend(rhs)->row_permute(&perm_idxs));

    auto ltrs_factory =
        gko::solver::LowerTrs<ValueType, IndexType>::build().on(exec);
    auto ref_ltrs = ltrs_factory->generate(ref_mtx);


    ref_ltrs->apply(gko::lend(rhs_perm), gko::lend(ref_x));

    auto ref_ans = Vec::create(exec);
    ref_ans->copy_from(
        std::move(gko::as<Vec>(ref_x->inverse_row_permute(&perm_idxs))));

    gs_HBMC->apply(gko::lend(rhs), gko::lend(x));

    GKO_ASSERT_MTX_NEAR(x, ref_ans, r<ValueType>::value);
}


}  // namespace