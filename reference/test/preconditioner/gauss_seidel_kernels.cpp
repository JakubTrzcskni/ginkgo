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


    GaussSeidel()
        : exec{gko::ReferenceExecutor::create()},
          rand_engine{15},
          gs_factory(GS::build().with_use_coloring(true).on(exec)),
          ref_gs_factory(
              GS::build().with_use_reference(true).with_use_coloring(false).on(
                  exec)),
          iter_criterion_factory(Iter::build().with_max_iters(20u).on(exec)),
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

    template <typename value_type, typename index_type>
    std::unique_ptr<gko::matrix::Csr<value_type, index_type>>
    generate_rand_matrix(index_type size, index_type num_elems_lo,
                         index_type num_elems_hi, value_type deduction_help)
    {
        auto mtx = gko::matrix::Csr<value_type, index_type>::create(
            exec, gko::dim<2>(size));
        auto mat_data =
            gko::test::generate_random_matrix_data<value_type, index_type>(
                mtx->get_size()[0], mtx->get_size()[1],
                std::uniform_int_distribution<index_type>(num_elems_lo,
                                                          num_elems_hi),
                std::normal_distribution<gko::remove_complex<value_type>>(-1.0,
                                                                          1.0),
                rand_engine);

        gko::utils::make_hpd(mat_data, 2.0);
        mat_data.ensure_row_major_order();
        mtx->read(mat_data);

        return give(mtx);
    }

    // Source: jacobi_kernels.cpp (test)
    template <typename T>
    void init_array(T* arr, I<T> vals)
    {
        for (auto elem : vals) {
            *(arr++) = elem;
        }
    }

    template <typename value_type, typename index_type>
    void print_csr(const gko::matrix::Csr<value_type, index_type>* matrix)
    {
        const index_type* row_ptrs = matrix->get_const_row_ptrs();
        const index_type* col_idxs = matrix->get_const_col_idxs();
        const value_type* values = matrix->get_const_values();
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
    template <typename value_type, typename size_type>
    std::unique_ptr<Csr> generate_2D_regular_grid_matrix(
        size_type size, value_type deduction_help, bool nine_point = false)
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

    std::shared_ptr<const gko::ReferenceExecutor> exec;
    std::default_random_engine rand_engine;
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
    using value_type = typename TestFixture::value_type;

    auto ans = Vec::create_with_config_of(lend(this->rhs_4));
    ans->fill(value_type{0});
    auto ref_gs = this->ref_gs_factory->generate(this->mtx_csr_4);
    ref_gs->apply(lend(this->rhs_4), lend(ans));

    GKO_ASSERT_MTX_NEAR(ans, this->ltr_ans_4, r<value_type>::value);
}

TYPED_TEST(GaussSeidel, ReferenceSimpleApply_2)
{
    using Vec = typename TestFixture::Vec;
    using value_type = typename TestFixture::value_type;

    auto ans = Vec::create_with_config_of(lend(this->rhs_3));
    auto ref_gs = this->ref_gs_factory->generate(this->mtx_csr_3);
    ref_gs->apply(lend(this->rhs_3), lend(ans));

    GKO_ASSERT_MTX_NEAR(ans, this->ltr_ans_3, r<value_type>::value);
}

TYPED_TEST(GaussSeidel, ReferenceSimpleApplyKernel_rand_mat_spd)
{
    using Csr = typename TestFixture::Csr;
    using Vec = typename TestFixture::Vec;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    auto exec = this->exec;

    auto mtx_rand = this->mtx_rand;
    auto ref_mtx_rand = share(Csr::create(exec));
    gko::matrix_data<value_type, index_type> rand_mat_data;
    mtx_rand->write(rand_mat_data);
    gko::utils::make_lower_triangular(rand_mat_data);
    ref_mtx_rand->read(rand_mat_data);

    auto rhs_rand = this->rhs_rand;

    auto x = Vec::create_with_config_of(lend(rhs_rand));
    x->fill(value_type{0});
    auto ref_x = Vec::create_with_config_of(lend(rhs_rand));
    ref_x->fill(value_type{0});

    auto ref_gs = this->ref_gs_factory->generate(mtx_rand);
    auto ltrs_factory =
        gko::solver::LowerTrs<value_type, index_type>::build().on(exec);
    auto ref_ltrs = ltrs_factory->generate(ref_mtx_rand);

    ref_gs->apply(lend(rhs_rand), lend(x));
    ref_ltrs->apply(lend(rhs_rand), lend(ref_x));

    GKO_ASSERT_MTX_NEAR(x, ref_x, r<value_type>::value);
}

TYPED_TEST(GaussSeidel, SimpleApplyKernel)
{
    using Vec = typename TestFixture::Vec;
    using value_type = typename TestFixture::value_type;

    auto ans = Vec::create_with_config_of(lend(this->rhs_4));
    ans->fill(value_type{0});
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
                        l({value_type{6.0 / 10.0}, value_type{3598.0 / 1760.0},
                           value_type{-116.0 / 100.0}, value_type{15.0 / 8.0},
                           value_type{-1807.0 / 2800.0}}),
                        r<value_type>::value);
}

TYPED_TEST(GaussSeidel, SimpleApplyKernel_2)
{
    using Vec = typename TestFixture::Vec;
    using value_type = typename TestFixture::value_type;

    auto ans = Vec::create_with_config_of(lend(this->rhs_3));
    ans->fill(value_type{0});
    auto gs = this->gs_factory->generate(this->mtx_csr_3);
    gs->apply(lend(this->rhs_3), lend(ans));

    // auto v_colors = gs->get_vertex_colors();
    // auto p_idxs = gs->get_permutation_idxs();

    // this->print_array(v_colors);
    // this->print_array(p_idxs);

    // this->print_csr(lend(gs->get_ltr_matrix()));

    GKO_ASSERT_MTX_NEAR(
        ans,
        l({value_type{6.0 / 10.0}, value_type{799.0 / 440.0},
           value_type{-3744.0 / 4400.0}, value_type{15.0 / 8.0}}),
        r<value_type>::value);
}

TYPED_TEST(GaussSeidel, SimpleApplyKernel_rand_mat_spd)
{
    using Csr = typename TestFixture::Csr;
    using Vec = typename TestFixture::Vec;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    auto exec = this->exec;

    auto mtx_rand = this->mtx_rand;

    auto rhs_rand = this->rhs_rand;

    auto x = Vec::create_with_config_of(lend(rhs_rand));
    x->fill(value_type{0});
    auto ref_x = Vec::create_with_config_of(lend(rhs_rand));
    ref_x->fill(value_type{0});

    auto gs = this->gs_factory->generate(mtx_rand);
    // auto ref_gs = this->ref_gs_factory->generate(mtx_rand);
    // comparing to ref_gs yields a small error every time ->effect of
    // reordering on the system?
    auto ltrs_factory =
        gko::solver::LowerTrs<value_type, index_type>::build().on(exec);

    gko::matrix_data<value_type, index_type> ref_data;
    gs->get_ltr_matrix()->write(ref_data);
    ref_data.ensure_row_major_order();  // without row major order ltrs won't
                                        // work correctly
    auto ref_mtx = share(Csr::create(exec));
    ref_mtx->read(ref_data);
    auto ref_ltrs = ltrs_factory->generate(ref_mtx);

    auto perm_idxs_view =
        gko::array<index_type>(exec, gs->get_permutation_idxs());

    const auto rhs_rand_perm =
        gko::as<const Vec>(lend(rhs_rand)->row_permute(&perm_idxs_view));

    gs->apply(lend(rhs_rand), lend(x));
    ref_ltrs->apply(lend(rhs_rand_perm), lend(ref_x));

    auto ref_ans = Vec::create(exec);
    ref_ans->copy_from(
        std::move(gko::as<Vec>(ref_x->inverse_row_permute(&perm_idxs_view))));

    // auto colors = gs->get_color_ptrs();
    // std::cout << "sum of colors = "
    //           << gko::reduce_add(gs->get_vertex_colors(), index_type{0})
    //           << "\nmax color = " << colors.get_num_elems() - 2 << std::endl;

    GKO_ASSERT_MTX_NEAR(x, ref_ans, r<value_type>::value);
}

TYPED_TEST(GaussSeidel, SimpleApplyDiagonalMatrix)
{
    using Vec = typename TestFixture::Vec;
    using value_type = typename TestFixture::value_type;
    using Diagonal = typename TestFixture::Diagonal;

    auto diag_vals =
        gko::array<value_type>(this->exec, I<value_type>({1, 2, 3, 4, 5}));

    auto diag_mat = share(
        Diagonal::create(this->exec, diag_vals.get_num_elems(), diag_vals));

    auto ans = Vec::create_with_config_of(lend(this->rhs_4));
    ans->fill(value_type{0});
    auto gs = this->gs_factory->generate(diag_mat);
    gs->apply(lend(this->rhs_4), lend(ans));


    GKO_ASSERT_MTX_NEAR(ans,
                        l({value_type{6.0 / 1.0}, value_type{25.0 / 2.0},
                           value_type{-11.0 / 3.0}, value_type{15.0 / 4.0},
                           value_type{-3.0 / 5.0}}),
                        r<value_type>::value);
}

TYPED_TEST(GaussSeidel, SimpleApplyKernel_multi_rhs)
{
    using Vec = typename TestFixture::Vec;
    using value_type = typename TestFixture::value_type;

    auto mtx = this->mtx_csr_4;
    auto rhs = gko::initialize<Vec>(
        {I<value_type>{6.0, 3.0}, I<value_type>{25.0, 12.5},
         I<value_type>{-11.0, -5.5}, I<value_type>{15.0, 7.5},
         I<value_type>{-3.0, -1.5}},
        this->exec);
    auto ans = Vec::create_with_config_of(gko::lend(rhs));
    ans->fill(value_type{0});
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
        l({{value_type{6.0 / 10.0}, value_type{3.0 / 10.0}},
           {value_type{3598.0 / 1760.0}, value_type{1799.0 / 1760.0}},
           {value_type{-116.0 / 100.0}, value_type{-58.0 / 100.0}},
           {value_type{15.0 / 8.0}, value_type{7.5 / 8.0}},
           {value_type{-1807.0 / 2800.0}, value_type{-903.5 / 2800.0}}}),
        r<value_type>::value);
}

TYPED_TEST(GaussSeidel, SystemSolveIRGS)
{
    using Csr = typename TestFixture::Csr;
    using Vec = typename TestFixture::Vec;
    using GS = typename TestFixture::GS;
    using Ir = typename TestFixture::Ir;
    using value_type = typename TestFixture::value_type;

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

    GKO_ASSERT_MTX_NEAR(result, this->ans_3, r<value_type>::value);
}

TYPED_TEST(GaussSeidel, SystemSolveIRRefGS)
{
    using Ir = typename TestFixture::Ir;
    using Csr = typename TestFixture::Csr;
    using Vec = typename TestFixture::Vec;
    using value_type = typename TestFixture::value_type;
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

    GKO_ASSERT_MTX_NEAR(result, this->ans_3, r<value_type>::value);
}

// TODO
TYPED_TEST(GaussSeidel, SystemSolveGSPCG)
{
    using CG = typename TestFixture::CG;
    using Csr = typename TestFixture::Csr;
    using Vec = typename TestFixture::Vec;
    using GS = typename TestFixture::GS;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;

    auto mtx = this->mtx_rand;
    auto rhs = this->rhs_rand;

    // auto cg_factory =
}

TYPED_TEST(GaussSeidel, CorrectColoringRegularGrid)
{
    using index_type = typename TestFixture::index_type;
    using value_type = typename TestFixture::value_type;
    using SparsityCsr =
        typename gko::matrix::SparsityCsr<value_type, index_type>;
    using Csr = typename TestFixture::Csr;
    auto exec = this->exec;
    size_t grid_size = 10;
    auto regular_grid_matrix =
        share(this->generate_2D_regular_grid_matrix(grid_size, value_type{0}));

    gko::array<index_type> vertex_colors{exec, grid_size * grid_size};
    vertex_colors.fill(index_type{-1});
    gko::array<index_type> ans{exec, vertex_colors};
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
    index_type max_color{0};
    gko::kernels::reference::gauss_seidel::get_coloring(
        exec, lend(adjacency_matrix), vertex_colors, &max_color);

    GKO_ASSERT_ARRAY_EQ(vertex_colors, ans);
}

// TODO
// TYPED_TEST(GaussSeidel, CorrectReorderingRegularGrid)
// {
//     using index_type = typename TestFixture::index_type;
//     using value_type = typename TestFixture::value_type;
//     auto exec = this->exec;
//     size_t grid_size = 3;
//     auto regular_grid_matrix =
//         share(this->generate_2D_regular_grid_matrix(grid_size,
//         value_type{0}));
//     auto gs = this->gs_factory->generate(regular_grid_matrix);
//     auto perm_arr = gs->get_permutation_idxs();
//     GKO_ASSERT_EQ(0, 1);
// }

TYPED_TEST(GaussSeidel, RACE_1)
{
    using Csr = typename TestFixture::Csr;
    using Vec = typename TestFixture::Vec;
    using GS = typename TestFixture::GS;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;

    auto exec = this->exec;
    auto gs_race_factory = GS::build().with_use_RACE(true).on(exec);

    // auto grid_mtx =
    //     gko::share(this->generate_2D_regular_grid_matrix(100,
    //     value_type{0}));
    // auto gs_race = gs_race_factory->generate(grid_mtx);
    // auto gs = this->gs_factory->generate(grid_mtx);
    auto mtx = gko::share(this->generate_rand_matrix(
        index_type{20000}, index_type{10}, index_type{20},
        value_type{0}));  // initial random matrix has the selected nnz per
                          //     row,
                          // after turning the matrix hpd -> around 2x more
    auto gs_race = gs_race_factory->generate(mtx);
    auto gs = this->gs_factory->generate(mtx);

    auto lvl_ptrs = gs_race->get_level_ptrs();
    std::cout << lvl_ptrs.size() << std::endl;
    auto color_ptrs = gs->get_color_ptrs();
    std::cout << color_ptrs.get_num_elems() - 1 << std::endl;
}
}  // namespace