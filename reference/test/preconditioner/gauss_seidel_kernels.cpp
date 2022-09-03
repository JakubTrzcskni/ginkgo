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

#include "core/preconditioner/gauss_seidel_kernels.hpp"
#include "core/preconditioner/sparse_display.hpp"
#include "core/test/utils.hpp"
#include "core/utils/matrix_utils.hpp"
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
          gs_factory(GS::build().with_use_coloring(true).on(exec)),
          ref_gs_factory(
              GS::build().with_use_reference(true).with_use_coloring(false).on(
                  exec)),
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
                std::uniform_int_distribution<index_type>(5, 15),
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

    // auto v_colors = gs->get_vertex_colors();
    // auto p_idxs = gs->get_permutation_idxs();
    // auto col_ptrs = gs->get_color_ptrs();

    // this->print_array(v_colors);
    // this->print_array(p_idxs);
    // this->print_array(col_ptrs);
    // this->print_csr(lend(gs->get_ltr_matrix()));

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

    // auto v_colors = gs->get_vertex_colors();
    // auto p_idxs = gs->get_permutation_idxs();

    // this->print_array(v_colors);
    // this->print_array(p_idxs);

    // this->print_csr(lend(gs->get_ltr_matrix()));

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
    // auto ref_gs = this->ref_gs_factory->generate(mtx_rand);
    // comparing to ref_gs yields a small error every time ->effect of
    // reordering on the system?
    // mtx_rand isnt row_major sorted after gs_factory generation
    auto ltrs_factory =
        gko::solver::LowerTrs<ValueType, IndexType>::build().on(exec);

    gko::matrix_data<ValueType, IndexType> ref_data;
    gs->get_ltr_matrix()->write(ref_data);
    ref_data.ensure_row_major_order();  // without row major order ltrs won't
                                        // work correctly
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

    // auto colors = gs->get_color_ptrs();
    // std::cout << "sum of colors = "
    //           << gko::reduce_add(gs->get_vertex_colors(), IndexType{0})
    //           << "\nmax color = " << colors.get_num_elems() - 2 << std::endl;

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

    // auto v_colors = gs->get_vertex_colors();
    // auto p_idxs = gs->get_permutation_idxs();
    // auto col_ptrs = gs->get_color_ptrs();

    // this->print_array(v_colors);
    // this->print_array(p_idxs);
    // this->print_array(col_ptrs);
    // this->print_csr(lend(gs->get_ltr_matrix()));

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

// TODO
TYPED_TEST(GaussSeidel, SystemSolveGSPCG)
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
}

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

// TODO
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

    gko::kernels::reference::gauss_seidel::get_secondary_ordering(
        exec, block_ordering.get_data(), base_block_size, lvl_2_block_size,
        color_ptrs.get_const_data(), max_color);

    // auto standard_label = std::string("withoutOrdering-");
    // standard_label += typeid(ValueType).name() + std::string("-") +
    //                   typeid(IndexType).name() + std::string("-");
    // this->visualize(gko::lend(mtx), standard_label);

    // auto perm_mtx = Csr::create(exec);
    // perm_mtx->copy_from(gko::give(gko::as<Csr>(mtx->permute(&block_ordering))));

    // auto label = std::string("SecondaryOrdering-");
    // label += typeid(ValueType).name() + std::string("-") +
    //          typeid(IndexType).name() + std::string("-");
    // this->visualize(gko::lend(perm_mtx), label);

    GKO_ASSERT_ARRAY_EQ(block_ordering, I<IndexType>({0, 2, 4, 6, 1, 3, 5, 7}));
}

}  // namespace