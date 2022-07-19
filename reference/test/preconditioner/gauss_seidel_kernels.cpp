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
#include <ginkgo/core/base/matrix_data.hpp>
#include <ginkgo/core/base/polymorphic_object.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>
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
    using Iter = gko::stop::Iteration;
    using ResNorm = gko::stop::ResidualNorm<value_type>;
    using Csr = gko::matrix::Csr<value_type, index_type>;
    using Vec = gko::matrix::Dense<value_type>;
    using MatData = gko::matrix_data<value_type, index_type>;


    GaussSeidel()
        : exec{gko::ReferenceExecutor::create()},
          gs_factory(GS::build().on(exec)),
          ref_gs_factory(GS::build().with_use_reference(true).on(exec)),
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
          mtx_dense_3{gko::initialize<Vec>({{10.0, -1.0, 2.0, 0.0},
                                            {-1.0, 11.0, -1.0, 3.0},
                                            {2.0, -1.0, 10.0, -1.0},
                                            {0.0, 3.0, -1.0, 8.0}},
                                           exec)},
          mtx_csr_3{gko::initialize<Csr>({{10.0, -1.0, 2.0, 0.0},
                                          {-1.0, 11.0, -1.0, 3.0},
                                          {2.0, -1.0, 10.0, -1.0},
                                          {0.0, 3.0, -1.0, 8.0}},
                                         exec)},
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
        // mtx_dense_3->convert_to(gko::lend(mtx_csr_3));
        init_array<index_type>(mtx_csr_4->get_row_ptrs(), {0, 1, 3, 5, 7, 10});
        init_array<index_type>(mtx_csr_4->get_col_idxs(),
                               {0, 0, 1, 0, 2, 1, 3, 2, 3, 4});
        init_array<value_type>(
            mtx_csr_4->get_values(),
            {10.0, -1.0, 11.0, 2.0, 10.0, 3.0, 8.0, -1.0, 1.0, 7.0});
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

    std::shared_ptr<const gko::ReferenceExecutor> exec;
    std::shared_ptr<typename GS::Factory> gs_factory;
    std::shared_ptr<typename GS::Factory> ref_gs_factory;
    std::shared_ptr<typename Iter::Factory> iter_criterion_factory;
    std::shared_ptr<typename ResNorm::Factory> res_norm_criterion_factory;
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
    gko::kernels::reference::gauss_seidel::get_coloring(
        exec, lend(adjacency_matrix), vertex_colors);

    GKO_ASSERT_ARRAY_EQ(vertex_colors, ans);
}

TYPED_TEST(GaussSeidel, CorrectReorderingRegularGrid)
{
    using index_type = typename TestFixture::index_type;
    using value_type = typename TestFixture::value_type;
    auto exec = this->exec;
    size_t grid_size = 4;

    auto regular_grid_matrix =
        share(this->generate_2D_regular_grid_matrix(grid_size, value_type{0}));

    auto gs = this->gs_factory->generate(regular_grid_matrix);

    auto perm_arr = gs->get_permutation_idxs();

    for (auto i = 0; i < grid_size * grid_size; i++) {
        std::cout << perm_arr.get_data()[i] << " ";
    }
}

}  // namespace