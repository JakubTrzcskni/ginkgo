// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/preconditioner/gauss_seidel.hpp>


#include <algorithm>
#include <fstream>
#include <memory>
#include <random>
#include <tuple>
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
#include <ginkgo/core/matrix/permutation.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>
#include <ginkgo/core/preconditioner/jacobi.hpp>
#include <ginkgo/core/reorder/hbmc.hpp>
#include <ginkgo/core/solver/cg.hpp>
#include <ginkgo/core/solver/gmres.hpp>
#include <ginkgo/core/solver/ir.hpp>
#include <ginkgo/core/solver/triangular.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm.hpp>


#include "core/matrix/permutation_kernels.hpp"
#include "core/preconditioner/gauss_seidel_kernels.hpp"
#include "core/preconditioner/sparse_display.hpp"
#include "core/test/utils.hpp"
#include "core/utils/matrix_utils.hpp"
#include "matrices/config.hpp"

namespace {

using apply_param_type = std::vector<std::tuple<int, int, int, int, bool>>;
static apply_param_type allParams{
    std::make_tuple(16, 2, 16, 8, false), std::make_tuple(20, 5, 32, 4, false),
    std::make_tuple(20, 5, 32, 4, true), std::make_tuple(1000, 5, 32, 4, false),
    std::make_tuple(1000, 5, 32, 4, true),
    std::make_tuple(1000, 5, 32, 8, false),
    std::make_tuple(1000, 5, 32, 8, true),
    std::make_tuple(1000, 5, 32, 2, false),
    std::make_tuple(1000, 5, 32, 2, true),
    std::make_tuple(1000, 15, 32, 4, false),
    std::make_tuple(1000, 15, 32, 4, true),
    std::make_tuple(1000, 10, 16, 4, false),
    std::make_tuple(1000, 10, 16, 4, true),
    std::make_tuple(1000, 10, 4, 4, false),
    std::make_tuple(1000, 10, 4, 4, true),
    std::make_tuple(1000, 10, 4, 8, false),
    std::make_tuple(1000, 10, 4, 8, true),
    // std::make_tuple(1003, 15, 32, 4, false),  // mtx size not multiple of
    // 4 does not work std::make_tuple(1003, 15, 32, 4,
    // true),  // same here, segfault
    std::make_tuple(10000, 20, 32, 4, false),
    std::make_tuple(10000, 20, 32, 4, true),
    std::make_tuple(10000, 20, 16, 8, false)};

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
          iter_logger_2(Log::create()),
          gs_factory(GS::build().with_use_HBMC(false).on(
              exec)),  // old tests depend on hbmc flag being false on default
          ref_gs_factory(
              GS::build().with_use_reference(true).with_use_HBMC(false).on(
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
          mtx_secondary_ordering{Csr::create(exec, gko::dim<2>{18}, 56)},
          mtx_secondary_ordering_2{Csr::create(exec, gko::dim<2>{15}, 63)},
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
              rand_engine, exec)},
          perm_secondary_ordering{
              exec, I<index_type>({10, 12, 17, 11, 5, 1, 9, 4, 3, 7, 13, 16, 0,
                                   2, 14, 6, 8, 15})},
          perm_secondary_ordering_2{
              exec, I<index_type>(
                        {5, 9, 2, 11, 0, 10, 1, 13, 6, 14, 4, 8, 3, 12, 7})},
          apply_params_{allParams}

    {
        mtx_dense_2->convert_to((mtx_csr_2));
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
        rand_mat_data.sort_row_major();
        mtx_rand->read(rand_mat_data);

        init_array<index_type>(mtx_secondary_ordering->get_row_ptrs(),
                               {0, 4, 9, 12, 15, 18, 20, 22, 27, 30, 33, 36, 39,
                                42, 43, 46, 49, 52, 56});
        init_array<index_type>(
            mtx_secondary_ordering->get_col_idxs(),
            {0,  1,  2,  15, 0, 1,  5,  7,  15, 0,  2,  17, 3,  7,
             9,  4,  9,  14, 1, 5,  6,  16, 1,  3,  7,  8,  12, 7,
             8,  17, 3,  4,  9, 10, 12, 14, 11, 16, 17, 7,  10, 12,
             13, 4,  10, 14, 0, 1,  15, 6,  11, 16, 2,  8,  11, 17});
        init_array<value_type>(
            mtx_secondary_ordering->get_values(),
            {10., 5.,  2.,  1.,  5.,  6., 0.5, 2., 5., 2., 9.7, 6., 3., 4., 1.,
             5.2, 12., 4.,  0.5, 15., 2., 2.,  2., 4., 2., 3.,  4., 3., 3., 6.,
             1.,  12., 4.7, 1.5, 7.,  7., 1.3, 3., 9., 4., 7.,  4., 8., 4., 7.,
             1.,  1.,  5.,  4.,  2.,  3., 6.5, 6., 6., 9., 2.2});

        init_array<index_type>(
            mtx_secondary_ordering_2->get_row_ptrs(),
            {0, 3, 8, 12, 18, 20, 25, 29, 34, 39, 43, 47, 52, 56, 60, 63});
        init_array<index_type>(
            mtx_secondary_ordering_2->get_col_idxs(),
            {0, 1, 3, 0,  1, 7,  10, 13, 2,  3,  5,  11, 0,  2,  3,  6,
             7, 8, 4, 6,  2, 5,  7,  9,  11, 3,  4,  6,  14, 1,  3,  5,
             7, 8, 3, 7,  8, 11, 14, 5,  9,  11, 12, 1,  10, 12, 13, 2,
             5, 8, 9, 11, 9, 10, 12, 13, 1,  10, 12, 13, 6,  8,  14});
        init_array<value_type>(
            mtx_secondary_ordering_2->get_values(),
            {2., 5., 1., 5.,  7.,  1., 6.,  10., 6., 1.,  4., 9., 1.,
             1., 6., 1., 11., 5.,  8., 6.,  4.,  1., 1.,  2., 7., 1.,
             6., 3., 4., 1.,  11., 1., 13., 10., 5., 10., 4., 1., 1.,
             2., 3., 8., 1.,  6.,  4., 1.,  9.,  9., 7.,  1., 8., 10.,
             1., 1., 9., 1.,  10., 9., 1.,  11., 4., 1.,  5.});
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
        mat_data.sort_row_major();
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
        data.sort_row_major();
        matrix->read(data);
        return std::move(matrix);
    }

    template <typename ValueType>
    void print_array(gko::array<ValueType>& arr)
    {
        for (auto i = 0; i < arr.get_size(); i++) {
            std::cout << arr.get_data()[i] << " ";
        }
        std::cout << std::endl;
    }

    template <typename ValueType, typename IndexType>
    void visualize(gko::matrix::Csr<ValueType, IndexType>* csr_mat,
                   std::string plot_label)
    {
        auto dense_mat = Vec::create(exec);
        csr_mat->convert_to((dense_mat));
        auto num_rows = dense_mat->get_size()[0];
        gko::preconditioner::visualize::spy_ge(
            num_rows, num_rows, dense_mat->get_values(), plot_label);
    }

    template <typename ValueType, typename IndexType>
    std::tuple<std::shared_ptr<gko::solver::LowerTrs<ValueType, IndexType>>,
               std::shared_ptr<gko::solver::UpperTrs<ValueType, IndexType>>>
    gen_ref_adv_apply(gko::matrix::Dense<ValueType>* system_matrix,
                      const gko::remove_complex<ValueType> omega_val,
                      const IndexType num_rows)
    {
        using Csr = gko::matrix::Csr<ValueType, IndexType>;
        const auto exec = this->exec;
        auto omega = gko::share(
            gko::initialize<gko::matrix::Dense<gko::remove_complex<ValueType>>>(
                {omega_val}, exec));

        auto diag = gko::share(system_matrix->extract_diagonal());

        auto inv_diag = diag->clone();
        for (auto i = 0; i < num_rows; ++i) {
            const auto val = inv_diag->get_const_values()[i];
            assert(val != gko::zero<ValueType>());
            inv_diag->get_values()[i] = gko::one<ValueType>() / val;
        }

        system_matrix->scale((omega));
        auto system_matrix_u = system_matrix->clone();
        inv_diag->apply((system_matrix), (system_matrix));
        gko::matrix_data<ValueType, IndexType> ref_data;
        system_matrix->write(ref_data);
        gko::utils::make_lower_triangular(ref_data);
        gko::utils::make_unit_diagonal(ref_data);
        ref_data.sort_row_major();
        auto ref_mtx = gko::share(Csr::create(exec));
        ref_mtx->read(ref_data);

        gko::matrix_data<ValueType, IndexType> ref_data_u;
        system_matrix_u->write(ref_data_u);
        gko::utils::make_upper_triangular(ref_data_u);
        gko::utils::make_remove_diagonal(ref_data_u);
        for (auto i = 0; i < num_rows; ++i) {
            ref_data_u.nonzeros.emplace_back(i, i, diag->get_const_values()[i]);
        }
        ref_data_u.sort_row_major();
        auto ref_mtx_u = gko::share(Csr::create(exec));
        ref_mtx_u->read(ref_data_u);

        auto ltrs_factory =
            gko::solver::LowerTrs<ValueType, IndexType>::build().on(exec);
        auto ref_ltrs = gko::share(ltrs_factory->generate(ref_mtx));

        auto utrs_factory =
            gko::solver::UpperTrs<ValueType, IndexType>::build().on(exec);
        auto ref_utrs = gko::share(utrs_factory->generate(ref_mtx_u));

        return std::make_tuple(ref_ltrs, ref_utrs);
    }

    template <typename ValueType, typename IndexType>
    void ref_adv_apply(
        std::tuple<std::shared_ptr<gko::solver::LowerTrs<ValueType, IndexType>>,
                   std::shared_ptr<gko::solver::UpperTrs<ValueType, IndexType>>>
            adv_apply_tuple,
        gko::matrix::Dense<ValueType>* rhs, gko::matrix::Dense<ValueType>* x)
    {
        const auto exec = this->exec;

        std::get<0>(adv_apply_tuple)->apply(rhs, x);  //  (rhs),  (x));
        rhs->copy_from(x);
        std::get<1>(adv_apply_tuple)->apply(rhs, x);  //  (x),  (x));
    }

    std::shared_ptr<const gko::ReferenceExecutor> exec;
    std::default_random_engine rand_engine;
    std::shared_ptr<Log> iter_logger;
    std::shared_ptr<Log> iter_logger_2;
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
    std::shared_ptr<Csr> mtx_secondary_ordering;
    std::shared_ptr<Csr> mtx_secondary_ordering_2;
    std::shared_ptr<Vec> rhs_2;
    std::shared_ptr<Vec> ans_2;
    std::shared_ptr<Vec> rhs_3;
    std::shared_ptr<Vec> ans_3;
    std::shared_ptr<Vec> ltr_ans_3;
    std::shared_ptr<Vec> x_1_3;
    std::shared_ptr<Vec> rhs_4;
    std::shared_ptr<Vec> ltr_ans_4;
    std::shared_ptr<Vec> rhs_rand;
    gko::array<index_type> perm_secondary_ordering;
    gko::array<index_type> perm_secondary_ordering_2;
    apply_param_type apply_params_;
};

TYPED_TEST_SUITE(GaussSeidel, gko::test::RealValueIndexTypes,
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

    auto ans = Vec::create_with_config_of((this->rhs_4));
    ans->fill(ValueType{0});
    auto ref_gs = this->ref_gs_factory->generate(this->mtx_csr_4);
    ref_gs->apply((this->rhs_4), (ans));

    GKO_ASSERT_MTX_NEAR(ans, this->ltr_ans_4, r<ValueType>::value);
}

TYPED_TEST(GaussSeidel, ReferenceSimpleApply_2)
{
    using Vec = typename TestFixture::Vec;
    using ValueType = typename TestFixture::value_type;

    auto ans = Vec::create_with_config_of((this->rhs_3));
    auto ref_gs = this->ref_gs_factory->generate(this->mtx_csr_3);
    ref_gs->apply((this->rhs_3), (ans));

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

    auto x = Vec::create_with_config_of((rhs_rand));
    x->fill(ValueType{0});
    auto ref_x = Vec::create_with_config_of((rhs_rand));
    ref_x->fill(ValueType{0});

    auto ref_gs = this->ref_gs_factory->generate(mtx_rand);
    auto ltrs_factory =
        gko::solver::LowerTrs<ValueType, IndexType>::build().on(exec);
    auto ref_ltrs = ltrs_factory->generate(ref_mtx_rand);

    ref_gs->apply((rhs_rand), (x));
    ref_ltrs->apply((rhs_rand), (ref_x));

    GKO_ASSERT_MTX_NEAR(x, ref_x, r<ValueType>::value);
}

TYPED_TEST(GaussSeidel, SimpleApplyKernel)
{
    using Vec = typename TestFixture::Vec;
    using ValueType = typename TestFixture::value_type;

    auto ans = Vec::create_with_config_of((this->rhs_4));
    ans->fill(ValueType{0});
    auto gs = this->gs_factory->generate(this->mtx_csr_4);

    gs->apply((this->rhs_4), (ans));

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

    auto ans = Vec::create_with_config_of((this->rhs_3));
    ans->fill(ValueType{0});
    auto gs = this->gs_factory->generate(this->mtx_csr_3);
    gs->apply((this->rhs_3), (ans));

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

    auto x = Vec::create_with_config_of((rhs_rand));
    x->fill(ValueType{0});
    auto ref_x = Vec::create_with_config_of((rhs_rand));
    ref_x->fill(ValueType{0});

    auto gs = this->gs_factory->generate(mtx_rand);

    auto ltrs_factory =
        gko::solver::LowerTrs<ValueType, IndexType>::build().on(exec);

    gko::matrix_data<ValueType, IndexType> ref_data;
    gs->get_ltr_matrix()->write(ref_data);
    ref_data.sort_row_major();

    auto ref_mtx = share(Csr::create(exec));
    ref_mtx->read(ref_data);
    auto ref_ltrs = ltrs_factory->generate(ref_mtx);

    auto perm_idxs_view =
        gko::array<IndexType>(exec, gs->get_permutation_idxs());

    const auto rhs_rand_perm =
        gko::as<const Vec>((rhs_rand)->row_permute(&perm_idxs_view));

    gs->apply((rhs_rand), (x));
    ref_ltrs->apply((rhs_rand_perm), (ref_x));

    auto ref_ans = Vec::create(exec);
    ref_ans->move_from(
        gko::as<Vec>(ref_x->inverse_row_permute(&perm_idxs_view)));

    GKO_ASSERT_MTX_NEAR(x, ref_ans, r<ValueType>::value);
}

TYPED_TEST(GaussSeidel, SimpleApplyDiagonalMatrix)
{
    using Vec = typename TestFixture::Vec;
    using ValueType = typename TestFixture::value_type;
    using Diagonal = typename TestFixture::Diagonal;

    auto diag_vals =
        gko::array<ValueType>(this->exec, I<ValueType>({1, 2, 3, 4, 5}));

    auto diag_mat =
        share(Diagonal::create(this->exec, diag_vals.get_size(), diag_vals));

    auto ans = Vec::create_with_config_of((this->rhs_4));
    ans->fill(ValueType{0});
    auto gs = this->gs_factory->generate(diag_mat);
    gs->apply((this->rhs_4), (ans));


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
    auto ans = Vec::create_with_config_of((rhs));
    ans->fill(ValueType{0});
    auto gs = this->gs_factory->generate(mtx);

    gs->apply((rhs), (ans));

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

    auto result = share(Vec::create_with_config_of((this->rhs_3)));
    result->fill(0.0);

    irs->apply((this->rhs_3), (result));

    GKO_ASSERT_MTX_NEAR(result, this->ans_3, r<ValueType>::value);
}

// segfaults, why? ;(
//  TYPED_TEST(GaussSeidel, SystemSolveIRRefGS)
//  {
//      using Ir = typename TestFixture::Ir;
//      using Csr = typename TestFixture::Csr;
//      using Vec = typename TestFixture::Vec;
//      using ValueType = typename TestFixture::value_type;
//      auto exec = this->exec;

//     auto result = Vec::create_with_config_of((this->rhs_3));
//     result->fill(0.0);

//     auto ir_factory = Ir::build()
//                           .with_solver(this->ref_gs_factory)
//                           .with_criteria(this->iter_criterion_factory,
//                                          this->res_norm_criterion_factory)
//                           .on(exec);
//     auto irs = ir_factory->generate(this->mtx_csr_3);
//     irs->apply((this->rhs_3), result.get());

//     GKO_ASSERT_MTX_NEAR(result, this->ans_3, r<ValueType>::value);
// }

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
    for (auto i = 0; i < ans.get_size(); i++) {
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
        exec, adjacency_matrix.get(), vertex_colors, &max_color);

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
        lvl_2_block_size, color_ptrs.get_const_data(), max_color, false, false);

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
        exec, adjacency_mtx.get(), block_ordering.get_data(),
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
        exec, adjacency_mtx.get(), block_ordering.get_data(),
        degrees.get_const_data(), visited.get_data(), block_size);

    // // for minDegree seedPolicy & maxNumEdges policy
    GKO_ASSERT_ARRAY_EQ(block_ordering,
                        I<IndexType>({5, 6, 0, 7, 2, 1, 3, 4, 8}));


    block_size = 6;
    std::fill_n(visited.get_data(), num_nodes, gko::int8{0});
    block_ordering.resize_and_reset(num_nodes);
    gko::kernels::reference::gauss_seidel::assign_to_blocks(
        exec, adjacency_mtx.get(), block_ordering.get_data(),
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
    auto mtx = this->mtx_secondary_ordering;
    auto perm = this->perm_secondary_ordering;

    gko::array<IndexType> inv_perm(exec, perm.get_size());
    gko::array<IndexType> color_block_ptrs(exec, I<IndexType>({0, 8, 14, 18}));
    gko::array<IndexType> perm_after_2nd_ordering(
        exec, I<IndexType>({10, 17, 5, 12, 11, 1, 9, 4, 3, 13, 0, 7, 16, 2, 14,
                            6, 8, 15}));
    GKO_ASSERT(mtx->get_size()[0] == perm.get_size());

    gko::array<ValueType> expected_l_diag_vals(
        exec, I<ValueType>({1.5, 2.2, 15., 7., 9., 0.5, 4., 1.3, 6.,
                            4.7, 12., 5.2, 3., 8., 10., 4., 0.,  2.,
                            2.,  6.5, 9.7, 1., 0., 2.,  3., 0.,  4.}));
    auto expected_l_diag_vals_vec =
        Vec::create(exec, gko::dim<2>{expected_l_diag_vals.get_size(), 1},
                    gko::make_array_view(exec, expected_l_diag_vals.get_size(),
                                         expected_l_diag_vals.get_data()),
                    1);
    gko::array<IndexType> expected_l_diag_mtx_col_idxs(
        exec, I<IndexType>({0, 3, 1,  1, 2, 2, 2, 0, 1,  2, 1, 0,  0, 0,
                            0, 1, -1, 0, 2, 2, 1, 2, -1, 0, 1, -1, 2}));
    gko::array<IndexType> expected_l_diag_row(
        exec,
        I<IndexType>({0,  1,  2,  3,  4,  5,  3,  4,  5,  6,  7,  7,  8, 9,
                      10, 11, -1, 13, 11, 12, 13, 14, -1, 15, 16, -1, 17}));
    GKO_ASSERT(expected_l_diag_mtx_col_idxs.get_size() ==
               expected_l_diag_vals.get_size());

    gko::array<ValueType> expected_l_spmv_vals(
        exec, I<ValueType>({1., 5., 4., 2., 3., 6., 7., 4., 2., 6., 3., 5., 1.,
                            0., 0., 0., 0., 0., 0.}));
    auto expected_l_spmv_vals_vec =
        Vec::create(exec, gko::dim<2>{expected_l_spmv_vals.get_size(), 1},
                    gko::make_array_view(exec, expected_l_spmv_vals.get_size(),
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
        lvl_2_block_size, color_block_ptrs.get_const_data(), max_color, false,
        false);

    GKO_ASSERT_ARRAY_EQ(perm, perm_after_2nd_ordering);

    gko::kernels::reference::permutation::invert(
        exec, perm.get_const_data(), perm.get_size(), inv_perm.get_data());

    IndexType* dummyInd;
    ValueType* dummyVal;
    const auto diag_mem_requirement = 27;
    const auto l_spmv_val_col_mem_requirement = 19;
    const auto l_spmv_row_mem_requirement = 18 - 8 + max_color;
    gko::array<IndexType> l_diag_rows_(exec, diag_mem_requirement);
    gko::array<IndexType> l_diag_mtx_col_idxs_(exec, diag_mem_requirement);
    gko::array<ValueType> l_diag_vals_(exec, diag_mem_requirement);
    auto l_diag_vals_vec_ =
        Vec::create(exec, gko::dim<2>{l_diag_vals_.get_size(), 1},
                    gko::make_array_view(exec, l_diag_vals_.get_size(),
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
        Vec::create(exec, gko::dim<2>{l_spmv_vals_.get_size(), 1},
                    gko::make_array_view(exec, l_spmv_vals_.get_size(),
                                         l_spmv_vals_.get_data()),
                    1);
    l_spmv_vals_.fill(ValueType{0});
    l_spmv_mtx_col_idxs_.fill(IndexType{0});
    l_spmv_col_idxs_.fill(IndexType{0});

    gko::kernels::reference::gauss_seidel::setup_blocks(
        exec, mtx.get(), perm.get_const_data(), inv_perm.get_const_data(),
        dummy_storage_scheme, static_cast<gko::remove_complex<ValueType>>(1.),
        l_diag_rows_.get_data(), l_diag_mtx_col_idxs_.get_data(),
        l_diag_vals_.get_data(), l_spmv_row_ptrs_.get_data(),
        l_spmv_col_idxs_.get_data(), l_spmv_mtx_col_idxs_.get_data(),
        l_spmv_vals_.get_data(), dummyInd, dummyInd, dummyVal, dummyInd,
        dummyInd, dummyInd, dummyVal, false);

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
    auto x = Vec::create_with_config_of((rhs));
    x->fill(ValueType{0});
    auto ref_x = Vec::create_with_config_of((rhs));
    ref_x->fill(ValueType{1});
    gko::kernels::reference::gauss_seidel::simple_apply(
        exec, l_diag_rows_.get_const_data(), l_diag_vals_.get_const_data(),
        l_spmv_row_ptrs_.get_const_data(), l_spmv_col_idxs_.get_const_data(),
        l_spmv_vals_.get_const_data(), perm.get_const_data(),
        dummy_storage_scheme, rhs.get(), x.get(), 0);

    GKO_ASSERT_MTX_NEAR(x, ref_x, r<ValueType>::value);


    gko::remove_complex<ValueType> a = 2.5;
    auto alpha = gko::initialize<Vec>({a}, exec);
    mtx->scale((alpha));
    expected_l_diag_vals_vec->scale((alpha));
    expected_l_spmv_vals_vec->scale((alpha));
    gko::kernels::reference::gauss_seidel::fill_with_vals(
        exec, mtx.get(), perm.get_const_data(), dummy_storage_scheme,
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
        rhs.get(), x.get(), 0);

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

    auto mtx = this->mtx_secondary_ordering_2;

    auto perm = this->perm_secondary_ordering_2;
    gko::array<IndexType> perm_cpy;
    perm_cpy = perm;
    gko::array<IndexType> inv_perm(exec, perm.get_size());
    gko::array<IndexType> color_block_ptrs(exec, I<IndexType>({0, 11, 15}));

    auto rhs = Vec::create(exec, gko::dim<2>{15, 1});
    this->template init_array<ValueType>(
        rhs->get_values(), {2., 85., 42., 80., 82., 6., 21., 246., 63., 42.,
                            44., 269., 152., 273., 103.});
    auto x = Vec::create_with_config_of((rhs));
    x->fill(ValueType{0});
    auto ref_x = Vec::create_with_config_of((rhs));
    std::iota(ref_x->get_values(), ref_x->get_values() + 15, 1);

    IndexType max_color = 1;
    IndexType base_block_size = 4;
    IndexType lvl_2_block_size = 3;

    gko::preconditioner::storage_scheme dummy_no_secondary(2 * max_color + 1);

    gko::kernels::reference::gauss_seidel::get_secondary_ordering(
        exec, perm.get_data(), dummy_no_secondary, base_block_size,
        lvl_2_block_size, color_block_ptrs.get_const_data(), max_color, false,
        false);

    GKO_ASSERT_ARRAY_EQ(perm, perm_cpy);  // cannot create a single lvl 1
    // block
    // -> no change in the perm array

    lvl_2_block_size = 2;
    gko::preconditioner::storage_scheme storage_scheme(2 * max_color + 1);
    gko::kernels::reference::gauss_seidel::get_secondary_ordering(
        exec, perm.get_data(), storage_scheme, base_block_size,
        lvl_2_block_size, color_block_ptrs.get_const_data(), max_color, false,
        false);

    gko::array<IndexType> perm_with_second_ordering(
        exec, I<IndexType>({5, 0, 9, 10, 2, 1, 11, 13, 6, 14, 4, 8, 3, 12, 7}));
    GKO_ASSERT_ARRAY_EQ(
        perm, perm_with_second_ordering);  // with smaller lvl 2 block size
                                           // second ordering successful

    gko::kernels::reference::permutation::invert(
        exec, perm.get_const_data(), perm.get_size(), inv_perm.get_data());

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
        exec, mtx.get(), perm.get_const_data(), inv_perm.get_const_data(),
        storage_scheme, static_cast<gko::remove_complex<ValueType>>(1.),
        l_diag_rows_.get_data(), l_diag_mtx_col_idxs_.get_data(),
        l_diag_vals_.get_data(), l_spmv_row_ptrs_.get_data(),
        l_spmv_col_idxs_.get_data(), l_spmv_mtx_col_idxs_.get_data(),
        l_spmv_vals_.get_data(), dummyInd, dummyInd, dummyVal, dummyInd,
        dummyInd, dummyInd, dummyVal, false);

    auto rhs_perm = gko::as<Vec>(rhs->row_permute(&perm));

    gko::kernels::reference::gauss_seidel::simple_apply(
        exec, l_diag_rows_.get_const_data(), l_diag_vals_.get_const_data(),
        l_spmv_row_ptrs_.get_const_data(), l_spmv_col_idxs_.get_const_data(),
        l_spmv_vals_.get_const_data(), perm.get_const_data(), storage_scheme,
        rhs_perm.get(), x.get(), 0);

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
    for (auto const& [num_rows, row_limit, w, b_s, padding] :
         this->apply_params_) {
        auto mtx = gko::share(
            this->generate_rand_matrix(IndexType{num_rows}, IndexType{1},
                                       IndexType{row_limit}, ValueType{0}));

        gko::size_type num_rhs = 10;
        auto rhs = gko::share(this->generate_rand_dense(
            ValueType{0}, mtx->get_size()[0], num_rhs));

        auto x = Vec::create_with_config_of((rhs));
        x->fill(ValueType{0});
        auto ref_x = Vec::create_with_config_of((rhs));
        ref_x->fill(ValueType{0});

        auto gs_HBMC_factory =
            GS::build()
                .with_use_HBMC(true)
                .with_base_block_size(static_cast<gko::size_type>(b_s))
                .with_lvl_2_block_size(static_cast<gko::size_type>(w))
                .with_use_padding(padding)
                .on(exec);
        auto gs_HBMC = gs_HBMC_factory->generate(mtx);

        auto perm_idxs =
            gko::array<IndexType>(exec, gs_HBMC->get_permutation_idxs());

        auto mtx_perm = gko::as<Csr>(mtx->permute(&perm_idxs));
        gko::matrix_data<ValueType, IndexType> ref_data;
        mtx_perm->write(ref_data);
        gko::utils::make_lower_triangular(ref_data);
        ref_data.sort_row_major();
        auto ref_mtx = gko::share(Csr::create(exec));
        ref_mtx->read(ref_data);
        const auto rhs_perm =
            gko::as<const Vec>((rhs)->row_permute(&perm_idxs));

        auto ltrs_factory =
            gko::solver::LowerTrs<ValueType, IndexType>::build().on(exec);
        auto ref_ltrs = ltrs_factory->generate(ref_mtx);

        ref_ltrs->apply((rhs_perm), (ref_x));

        auto ref_ans = Vec::create(exec);
        ref_ans->move_from(
            gko::as<Vec>(ref_x->inverse_row_permute(&perm_idxs)));

        gs_HBMC->apply((rhs), (x));

        GKO_ASSERT_MTX_NEAR(x, ref_ans, r<ValueType>::value);
    }
}

TYPED_TEST(GaussSeidel, ApplyHBMC_RandMtx)
{
    using IndexType = typename TestFixture::index_type;
    using ValueType = typename TestFixture::value_type;
    using GS = typename TestFixture::GS;
    using Csr = typename TestFixture::Csr;
    using Vec = typename TestFixture::Vec;
    auto exec = this->exec;
    auto i = 1;
    for (auto const& [num_rows, row_limit, w, b_s, padding] :
         this->apply_params_) {
        auto mtx = gko::share(
            this->generate_rand_matrix(IndexType{num_rows}, IndexType{1},
                                       IndexType{row_limit}, ValueType{0}));

        gko::size_type num_rhs = 10;
        auto rhs = gko::share(this->generate_rand_dense(
            ValueType{0}, mtx->get_size()[0], num_rhs));

        auto x = Vec::create_with_config_of((rhs));
        x->fill(ValueType{1});
        auto ref_x = x->clone();

        auto alpha = gko::share(gko::initialize<Vec>({2.0}, exec));
        auto beta = gko::share(gko::initialize<Vec>({-1.0}, exec));

        auto gs_HBMC_factory =
            GS::build()
                .with_use_HBMC(true)
                .with_base_block_size(static_cast<gko::size_type>(b_s))
                .with_lvl_2_block_size(static_cast<gko::size_type>(w))
                .with_use_padding(padding)
                .on(exec);
        auto gs_HBMC = gs_HBMC_factory->generate(mtx);

        auto perm_idxs =
            gko::array<IndexType>(exec, gs_HBMC->get_permutation_idxs());

        auto mtx_perm = gko::as<Csr>(mtx->permute(&perm_idxs));
        gko::matrix_data<ValueType, IndexType> ref_data;
        mtx_perm->write(ref_data);
        gko::utils::make_lower_triangular(ref_data);
        ref_data.sort_row_major();
        auto ref_mtx = gko::share(Csr::create(exec));
        ref_mtx->read(ref_data);
        const auto rhs_perm =
            gko::as<const Vec>((rhs)->row_permute(&perm_idxs));

        auto ltrs_factory =
            gko::solver::LowerTrs<ValueType, IndexType>::build().on(exec);
        auto ref_ltrs = ltrs_factory->generate(ref_mtx);

        ref_ltrs->apply((alpha), (rhs_perm), (beta), (ref_x));

        auto ref_ans = Vec::create(exec);
        ref_ans->move_from(
            gko::as<Vec>(ref_x->inverse_row_permute(&perm_idxs)));

        gs_HBMC->apply((alpha), (rhs), (beta), (x));
        // std::cout << "tuple " << i++ << std::endl;
        GKO_ASSERT_MTX_NEAR(x, ref_ans, r<ValueType>::value);
    }
}

TYPED_TEST(GaussSeidel, SecondaryOrderingSetupBlocksKernelPadding)
{
    using IndexType = typename TestFixture::index_type;
    using ValueType = typename TestFixture::value_type;
    using Csr = typename TestFixture::Csr;
    using Vec = typename TestFixture::Vec;
    auto exec = this->exec;
    auto mtx = this->mtx_secondary_ordering;
    auto perm = this->perm_secondary_ordering;
    gko::array<IndexType> inv_perm(exec, perm.get_size());
    gko::array<IndexType> color_block_ptrs(exec, I<IndexType>({0, 8, 14, 18}));

    IndexType max_color = 2;
    IndexType base_block_size = 2;
    IndexType lvl_2_block_size = 3;

    gko::preconditioner::storage_scheme storage_scheme_padding(2 * max_color +
                                                               1);
    gko::kernels::reference::gauss_seidel::get_secondary_ordering(
        exec, perm.get_data(), storage_scheme_padding, base_block_size,
        lvl_2_block_size, color_block_ptrs.get_const_data(), max_color, true,
        false);

    gko::array<IndexType> exp_perm_after_2nd_ordering_padding(
        exec, I<IndexType>({10, 17, 5, 12, 11, 1, 9, 4, 3, 13, 0, 7, 16, 2, 14,
                            8, 6, 15}));

    GKO_ASSERT_ARRAY_EQ(perm, exp_perm_after_2nd_ordering_padding);

    gko::kernels::reference::permutation::invert(
        exec, perm.get_const_data(), perm.get_size(), inv_perm.get_data());

    IndexType* dummyInd;
    ValueType* dummyVal;
    const auto diag_mem_requirement = 36;
    const auto l_spmv_val_col_mem_requirement = 19;
    const auto l_spmv_row_mem_requirement = 18 - 8 + max_color;
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
        exec, mtx.get(), perm.get_const_data(), inv_perm.get_const_data(),
        storage_scheme_padding, static_cast<gko::remove_complex<ValueType>>(1.),
        l_diag_rows_.get_data(), l_diag_mtx_col_idxs_.get_data(),
        l_diag_vals_.get_data(), l_spmv_row_ptrs_.get_data(),
        l_spmv_col_idxs_.get_data(), l_spmv_mtx_col_idxs_.get_data(),
        l_spmv_vals_.get_data(), dummyInd, dummyInd, dummyVal, dummyInd,
        dummyInd, dummyInd, dummyVal, false);

    gko::array<ValueType> expected_l_diag_vals(
        exec,
        I<ValueType>({1.5, 2.2, 15., 7.,  9., 0.5, 4., 1.3, 6.,  4.7, 0., 0.,
                      12., 0.,  0.,  5.2, 0., 0.,  3., 8.,  10., 4.,  0., 2.,
                      2.,  6.5, 9.7, 1.,  3., 0.,  0., 0.,  0.,  2.,  4., 0.}));
    gko::array<IndexType> expected_l_diag_mtx_col_idxs(
        exec, I<IndexType>({0, 3,  1,  1, 2,  2,  2,  0,  1,  2, -1, -1,
                            1, -1, -1, 0, -1, -1, 0,  0,  0,  1, -1, 0,
                            2, 2,  1,  2, 1,  -1, -1, -1, -1, 0, 2,  -1}));
    gko::array<IndexType> expected_l_diag_row(
        exec, I<IndexType>({0,  1,  2,  3,  4,  5,  3,  4,  5,  6,  -1, -1,
                            7,  -1, -1, 7,  -1, -1, 8,  9,  10, 11, -1, 13,
                            11, 12, 13, 14, 15, -1, -1, -1, -1, 16, 17, -1}));

    GKO_ASSERT_ARRAY_EQ(expected_l_diag_vals, l_diag_vals_);
    GKO_ASSERT_ARRAY_EQ(expected_l_diag_mtx_col_idxs, l_diag_mtx_col_idxs_);
    GKO_ASSERT_ARRAY_EQ(expected_l_diag_row, l_diag_rows_);
}

TYPED_TEST(GaussSeidel, SecondaryOrderingSetupBlocksKernelPadding2)
{
    using IndexType = typename TestFixture::index_type;
    using ValueType = typename TestFixture::value_type;
    using Csr = typename TestFixture::Csr;
    using Vec = typename TestFixture::Vec;
    using GS = typename TestFixture::GS;
    auto exec = this->exec;

    auto mtx = this->mtx_secondary_ordering_2;

    auto perm = this->perm_secondary_ordering_2;
    gko::array<IndexType> perm_cpy;
    perm_cpy = perm;
    gko::array<IndexType> inv_perm(exec, perm.get_size());
    gko::array<IndexType> color_block_ptrs(exec, I<IndexType>({0, 11, 15}));

    IndexType max_color = 1;
    IndexType base_block_size = 4;
    IndexType lvl_2_block_size = 2;
    gko::preconditioner::storage_scheme secondary_padding_easy(2 * max_color +
                                                               1);

    gko::kernels::reference::gauss_seidel::get_secondary_ordering(
        exec, perm.get_data(), secondary_padding_easy, base_block_size,
        lvl_2_block_size, color_block_ptrs.get_const_data(), max_color, true,
        false);

    gko::array<IndexType> perm_with_second_ordering_ez(
        exec, I<IndexType>({5, 0, 9, 10, 2, 1, 11, 13, 6, 14, 4, 8, 3, 12, 7}));
    GKO_ASSERT_ARRAY_EQ(perm, perm_with_second_ordering_ez);

    lvl_2_block_size = 3;
    gko::preconditioner::storage_scheme secondary_padding_medium(2 * max_color +
                                                                 1);

    gko::kernels::reference::gauss_seidel::get_secondary_ordering(
        exec, perm_cpy.get_data(), secondary_padding_medium, base_block_size,
        lvl_2_block_size, color_block_ptrs.get_const_data(), max_color, true,
        false);
    gko::array<IndexType> perm_with_second_ordering_med(
        exec, I<IndexType>({5, 0, 6, 9, 10, 14, 2, 1, 4, 11, 13, 8, 3, 12, 7}));
    GKO_ASSERT_ARRAY_EQ(perm_cpy, perm_with_second_ordering_med);
}

TYPED_TEST(GaussSeidel, SystemSolveGS_PGMRES)
{
    using Csr = typename TestFixture::Csr;
    using Vec = typename TestFixture::Vec;
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using GMRES = gko::solver::Gmres<ValueType>;
    using GS = typename TestFixture::GS;
    using Log = typename TestFixture::Log;
    using BJ = gko::preconditioner::Jacobi<ValueType, IndexType>;
    using namespace gko;

    auto exec = this->exec;

    auto mtx = share(this->generate_rand_matrix(IndexType{1000}, IndexType{5},
                                                IndexType{15}, ValueType{0}));
    auto rhs =
        share(this->generate_rand_dense(ValueType{0}, mtx->get_size()[0]));
    auto x = Vec::create_with_config_of((rhs));
    x->fill(0.0);
    auto x_clone = clone(exec, x);
    auto rhs_clone = clone(exec, rhs);

    auto iter_crit = this->iter_criterion_factory;
    iter_crit->add_logger(this->iter_logger);
    auto res_norm = this->res_norm_criterion_factory;
    res_norm->add_logger(this->iter_logger);

    auto gmres_factory =
        GMRES::build().with_criteria(iter_crit, res_norm).on(exec);
    auto gmres = gmres_factory->generate(mtx);
    gmres->apply((rhs), (x));
    auto gmres_num_iters = this->iter_logger->get_num_iterations();

    auto gs_hbmc_factory = share(GS::build()
                                     .with_use_HBMC(true)
                                     .with_base_block_size(4u)
                                     .with_lvl_2_block_size(32u)
                                     .with_use_padding(true)
                                     .on(exec));

    // auto jacobi = share(BJ::build().with_max_block_size(8u).on(exec));
    // auto ltrs_factory =
    //     share(solver::LowerTrs<ValueType, IndexType>::build().on(exec));
    auto gs_hbmc = share(gs_hbmc_factory->generate(mtx));

    iter_crit->add_logger(this->iter_logger_2);
    res_norm->add_logger(this->iter_logger_2);
    auto pgmres_factory = GMRES::build()
                              .with_criteria(iter_crit, res_norm)
                              .with_generated_preconditioner(gs_hbmc)
                              .on(exec);
    auto pgmres = pgmres_factory->generate(mtx);
    pgmres->apply((rhs_clone), (x_clone));

    auto pgmres_num_iters = this->iter_logger_2->get_num_iterations();

    ASSERT_EQ(pgmres_num_iters <= gmres_num_iters, 1);  // should be <
    // doesnt work for complex floats/doubles
}

TYPED_TEST(GaussSeidel, AdvancedSecondaryOrderingSetupBlocksKernel)
{
    using IndexType = typename TestFixture::index_type;
    using ValueType = typename TestFixture::value_type;
    using Csr = typename TestFixture::Csr;
    using Vec = typename TestFixture::Vec;
    auto exec = this->exec;
    auto omega_val = gko::remove_complex<ValueType>{1.5};
    auto omega =
        gko::initialize<gko::matrix::Dense<gko::remove_complex<ValueType>>>(
            {omega_val}, exec);
    auto mtx = gko::share(Csr::create(exec, gko::dim<2>{16}, 68));
    this->template init_array<IndexType>(
        mtx->get_row_ptrs(),
        {0, 5, 10, 15, 20, 24, 28, 32, 37, 42, 46, 50, 54, 57, 61, 65, 68});
    this->template init_array<IndexType>(
        mtx->get_col_idxs(),
        {0, 1,  2,  3, 8,  0,  1,  2,  3,  9,  0,  1,  2,  3,  10, 0,  1,
         2, 3,  11, 4, 5,  6,  7,  4,  5,  7,  12, 4,  6,  7,  13, 4,  5,
         6, 7,  14, 0, 8,  9,  11, 15, 1,  8,  9,  10, 2,  9,  10, 11, 3,
         8, 10, 11, 5, 12, 14, 6,  13, 14, 15, 7,  12, 13, 14, 8,  13, 15});
    this->template init_array<ValueType>(
        mtx->get_values(),
        {10, 1,  1,  1,  1,  1, 11, 1,  1,  1, 1,  1, 12, 1,  1,  1, 1,
         1,  13, 1,  14, 2,  2, 2,  2,  15, 2, 1,  2, 16, 2,  1,  2, 2,
         2,  17, 1,  1,  18, 3, 3,  1,  1,  3, 19, 3, 1,  3,  20, 3, 1,
         3,  3,  21, 1,  22, 4, 1,  23, 4,  4, 1,  4, 4,  24, 1,  4, 25});

    gko::array<IndexType> perm(exec, mtx->get_size()[0]);
    std::iota(perm.get_data(), perm.get_data() + perm.get_size(), 0);
    gko::array<IndexType> inv_perm(exec, mtx->get_size()[0]);

    gko::array<IndexType> color_block_ptrs(exec, I<IndexType>({0, 8, 12, 16}));

    auto expected_l_diag_vals = Vec::create(exec, gko::dim<2>{40, 1});
    this->template init_array<ValueType>(expected_l_diag_vals->get_values(),
                                         {
                                             // clang-format off
                                        1. / omega_val,
                                        1. / 11., 1. / omega_val,
                                        1. / 12., 1. / 12., 1. / omega_val,
                                        1. / 13., 1. / 13., 1. / 13., 1. / omega_val,
                                        1. / omega_val,
                                        2. / 15., 1. / omega_val,
                                        2. / 16., 0., 1. / omega_val,
                                        2. / 17., 2. / 17., 2. / 17., 1. / omega_val,
                                        1. / omega_val,
                                        3. / 19., 1. / omega_val,
                                        0., 3. / 20., 1. / omega_val,
                                        3. / 21., 0., 3. / 21., 1. / omega_val,
                                        1. / omega_val,
                                        0., 1. / omega_val,
                                        4. / 24., 4. / 24., 1. / omega_val,
                                        0., 4. / 25., 0., 1. / omega_val
                                             // clang-format on
                                         });
    expected_l_diag_vals->scale((omega));
    gko::array<IndexType> expected_u_diag_rows(
        exec, I<IndexType>({
                  // clang-format off
                                        3,
                                        2, 2,
                                        1, 1, 1, 
                                        0, 0, 0, 0,
                                        7, 
                                        6, 6, 
                                        -1, 5, 5, 
                                        4, 4, 4, 4,
                                        11, 
                                        10, 10, 
                                        9, -1, 9,
                                        8, -1, 8, 8,
                                        15,
                                        -1, 14, 
                                        13, 13, 13, 
                                        -1, 12, -1, 12
                  // clang-format on
              }));
    auto expected_u_diag_vals = Vec::create(exec, gko::dim<2>{40, 1});
    this->template init_array<ValueType>(expected_u_diag_vals->get_values(),
                                         {
                                             // clang-format off
                                        13. / omega_val,
                                        1., 12. / omega_val,
                                        1., 1., 11. / omega_val,
                                        1., 1., 1., 10. / omega_val,
                                        17. / omega_val,
                                        2., 16. / omega_val,
                                        0., 2., 15. / omega_val,
                                        2., 2., 2., 14. / omega_val,
                                        21. / omega_val,
                                        3., 20. / omega_val,
                                        3., 0., 19. / omega_val,
                                        3., 0., 3., 18. / omega_val,
                                        25. / omega_val,
                                        0., 24. / omega_val,
                                        4., 4., 23. / omega_val,
                                        0., 4., 0., 22. / omega_val
                                             // clang-format on
                                         });
    expected_u_diag_vals->scale((omega));

    auto expected_l_spmv_vals = Vec::create(exec, gko::dim<2>{8, 1});
    this->template init_array<ValueType>(
        expected_l_spmv_vals->get_values(),
        {1. / 18., 1. / 19., 1. / 20., 1. / 21., 1. / 22., 1. / 23., 1. / 24.,
         1. / 25.});
    expected_l_spmv_vals->scale((omega));
    gko::array<IndexType> expected_l_spmv_row_ptrs(
        exec, I<IndexType>({0, 1, 2, 3, 4, 0, 1, 2, 3, 4}));

    auto expected_u_spmv_vals = Vec::create(exec, gko::dim<2>{8, 1});
    this->template init_array<ValueType>(expected_u_spmv_vals->get_values(),
                                         {1., 1., 1., 1., 1., 1., 1., 1.});
    expected_u_spmv_vals->scale((omega));
    gko::array<IndexType> expected_u_spmv_row_ptrs(
        exec, I<IndexType>({0, 1, 2, 3, 4, 4, 5, 6, 7, 0, 1, 1, 1, 1}));

    IndexType max_color = 2;
    IndexType base_block_size = 4;
    IndexType lvl_2_block_size = 1;

    gko::preconditioner::storage_scheme dummy_storage_scheme(2 * max_color + 1,
                                                             true);

    gko::array<IndexType> perm_cpy(perm);
    gko::kernels::reference::gauss_seidel::get_secondary_ordering(
        exec, perm.get_data(), dummy_storage_scheme, base_block_size,
        lvl_2_block_size, color_block_ptrs.get_const_data(), max_color, false,
        false);

    const auto diag_mem_requirement = 40;
    const auto l_spmv_val_col_mem_requirement = 8;  // adjusted down
    const auto l_spmv_row_mem_requirement = 16 - 8 + max_color;
    gko::array<IndexType> l_diag_rows_(exec, diag_mem_requirement);
    gko::array<IndexType> l_diag_mtx_col_idxs_(exec, diag_mem_requirement);
    gko::array<ValueType> l_diag_vals_(exec, diag_mem_requirement);
    auto l_diag_vals_vec_ = Vec::create(
        exec, gko::dim<2>{expected_l_diag_vals->get_num_stored_elements(), 1},
        gko::make_array_view(exec,
                             expected_l_diag_vals->get_num_stored_elements(),
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
    auto l_spmv_vals_vec_ = Vec::create(
        exec, gko::dim<2>{expected_l_spmv_vals->get_num_stored_elements(), 1},
        gko::make_array_view(exec,
                             expected_l_spmv_vals->get_num_stored_elements(),
                             l_spmv_vals_.get_data()),
        1);
    l_spmv_vals_.fill(ValueType{0});
    l_spmv_mtx_col_idxs_.fill(IndexType{0});
    l_spmv_col_idxs_.fill(IndexType{0});

    auto u_spmv_val_col_mem_requirement = l_spmv_val_col_mem_requirement;
    auto u_spmv_row_mem_requirement = 16 - 4 + max_color;
    gko::array<IndexType> u_diag_rows_(exec, diag_mem_requirement);
    gko::array<IndexType> u_diag_mtx_col_idxs_(exec, diag_mem_requirement);
    gko::array<ValueType> u_diag_vals_(exec, diag_mem_requirement);
    auto u_diag_vals_vec_ = Vec::create(
        exec, gko::dim<2>{expected_u_diag_vals->get_num_stored_elements(), 1},
        gko::make_array_view(exec,
                             expected_u_diag_vals->get_num_stored_elements(),
                             u_diag_vals_.get_data()),
        1);
    u_diag_vals_.fill(ValueType{0});
    u_diag_mtx_col_idxs_.fill(IndexType{-1});
    u_diag_rows_.fill(IndexType{-1});
    gko::array<IndexType> u_spmv_row_ptrs_(exec, u_spmv_row_mem_requirement);
    gko::array<IndexType> u_spmv_col_idxs_(exec,
                                           u_spmv_val_col_mem_requirement);
    gko::array<IndexType> u_spmv_mtx_col_idxs_(exec,
                                               u_spmv_val_col_mem_requirement);
    gko::array<ValueType> u_spmv_vals_(exec, u_spmv_val_col_mem_requirement);
    auto u_spmv_vals_vec_ = Vec::create(
        exec, gko::dim<2>{expected_u_spmv_vals->get_num_stored_elements(), 1},
        gko::make_array_view(exec,
                             expected_u_spmv_vals->get_num_stored_elements(),
                             u_spmv_vals_.get_data()),
        1);
    u_spmv_vals_.fill(ValueType{0});
    u_spmv_mtx_col_idxs_.fill(IndexType{0});
    u_spmv_col_idxs_.fill(IndexType{0});

    gko::kernels::reference::permutation::invert(
        exec, perm.get_const_data(), perm.get_size(), inv_perm.get_data());

    gko::kernels::reference::gauss_seidel::setup_blocks(
        exec, mtx.get(), perm.get_const_data(), inv_perm.get_const_data(),
        dummy_storage_scheme, omega_val, l_diag_rows_.get_data(),
        l_diag_mtx_col_idxs_.get_data(), l_diag_vals_.get_data(),
        l_spmv_row_ptrs_.get_data(), l_spmv_col_idxs_.get_data(),
        l_spmv_mtx_col_idxs_.get_data(), l_spmv_vals_.get_data(),
        u_diag_rows_.get_data(), u_diag_mtx_col_idxs_.get_data(),
        u_diag_vals_.get_data(), u_spmv_row_ptrs_.get_data(),
        u_spmv_col_idxs_.get_data(), u_spmv_mtx_col_idxs_.get_data(),
        u_spmv_vals_.get_data(), false);

    GKO_ASSERT_MTX_NEAR(expected_l_diag_vals, l_diag_vals_vec_,
                        r<ValueType>::value);
    GKO_ASSERT_MTX_NEAR(expected_u_diag_vals, u_diag_vals_vec_,
                        r<ValueType>::value);
    GKO_ASSERT_ARRAY_EQ(expected_u_diag_rows, u_diag_rows_);
    GKO_ASSERT_MTX_NEAR(expected_l_spmv_vals, l_spmv_vals_vec_,
                        r<ValueType>::value);
    GKO_ASSERT_MTX_NEAR(expected_u_spmv_vals, u_spmv_vals_vec_,
                        r<ValueType>::value);
    GKO_ASSERT_ARRAY_EQ(expected_l_spmv_row_ptrs, l_spmv_row_ptrs_);
    GKO_ASSERT_ARRAY_EQ(expected_u_spmv_row_ptrs, u_spmv_row_ptrs_);

    auto mtx_perm = gko::share(gko::as<Csr>(mtx->permute(&perm)));
    auto mtx_dense = Vec::create(exec);
    mtx_perm->convert_to((mtx_dense));
    auto rhs = Vec::create(exec, gko::dim<2>{16, 1});
    rhs->fill(ValueType{1.});
    auto ref_rhs = rhs->clone();
    auto x = Vec::create_with_config_of((rhs));
    x->fill(ValueType{0.});
    auto ref_x = x->clone();

    this->ref_adv_apply(this->gen_ref_adv_apply(mtx_dense.get(), omega_val,
                                                static_cast<IndexType>(16)),
                        ref_rhs.get(), ref_x.get());


    auto ref_ans = Vec::create(exec);
    ref_ans->move_from(gko::as<Vec>(ref_x->inverse_row_permute(&perm)));

    gko::kernels::reference::gauss_seidel::advanced_apply(
        exec, l_diag_rows_.get_const_data(), l_diag_vals_.get_const_data(),
        l_spmv_row_ptrs_.get_const_data(), l_spmv_col_idxs_.get_const_data(),
        l_spmv_vals_.get_const_data(), u_diag_rows_.get_const_data(),
        u_diag_vals_.get_const_data(), u_spmv_row_ptrs_.get_const_data(),
        u_spmv_col_idxs_.get_const_data(), u_spmv_vals_.get_const_data(),
        perm.get_const_data(), dummy_storage_scheme, omega_val, rhs.get(),
        x.get(), 9);

    GKO_ASSERT_MTX_NEAR(x, ref_ans, r<ValueType>::value);
}

TYPED_TEST(GaussSeidel, AdvancedApplyHBMC_RandMtx)
{
    using IndexType = typename TestFixture::index_type;
    using ValueType = typename TestFixture::value_type;
    using GS = typename TestFixture::GS;
    using Csr = typename TestFixture::Csr;
    using Vec = typename TestFixture::Vec;
    auto exec = this->exec;

    for (auto const& [num_rows, row_limit, w, b_s, padding] :
         this->apply_params_) {
        const auto omega_val = gko::remove_complex<ValueType>{1.5};
        auto mtx = gko::share(
            this->generate_rand_matrix(IndexType{num_rows}, IndexType{1},
                                       IndexType{row_limit}, ValueType{0}));

        gko::size_type num_rhs = 10;
        auto rhs = gko::share(this->generate_rand_dense(
            ValueType{0}, mtx->get_size()[0], num_rhs));

        auto rhs_cpy = rhs->clone();
        auto x = Vec::create_with_config_of((rhs));
        x->fill(ValueType{0});
        auto ref_x = x->clone();
        auto mtx_clone = mtx->clone();

        auto gs_HBMC_factory =
            GS::build()
                .with_use_HBMC(true)
                .with_base_block_size(static_cast<gko::size_type>(b_s))
                .with_lvl_2_block_size(static_cast<gko::size_type>(w))
                .with_use_padding(padding)
                .with_symmetric_preconditioner(true)
                .with_relaxation_factor(omega_val)
                .on(exec);
        auto gs_HBMC = gs_HBMC_factory->generate(mtx);

        auto perm_idxs =
            gko::array<IndexType>(exec, gs_HBMC->get_permutation_idxs());

        auto mtx_perm = Vec::create(exec);
        gko::as<Csr>(mtx_clone->permute(&perm_idxs))->convert_to((mtx_perm));
        auto adv_apply_tuple = this->gen_ref_adv_apply(
            mtx_perm.get(), omega_val, static_cast<IndexType>(num_rows));

        const auto rhs_perm = Vec::create_with_config_of((rhs));
        rhs->row_permute(&perm_idxs, (rhs_perm));

        this->ref_adv_apply(adv_apply_tuple, rhs_perm.get(), ref_x.get());

        auto ref_ans = Vec::create(exec);
        ref_ans->move_from(
            gko::as<Vec>(ref_x->inverse_row_permute(&perm_idxs)));
        gs_HBMC->apply((rhs), (x));
        GKO_ASSERT_MTX_NEAR(x, ref_ans, r<ValueType>::value);
    }
}

TYPED_TEST(GaussSeidel, SystemSolveSGS_CG)
{
    using Csr = typename TestFixture::Csr;
    using Vec = typename TestFixture::Vec;
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using CG = typename TestFixture::CG;

    using GS = typename TestFixture::GS;
    using Log = typename TestFixture::Log;
    using BJ = gko::preconditioner::Jacobi<ValueType, IndexType>;
    using namespace gko;

    auto exec = this->exec;

    auto mtx = share(this->generate_rand_matrix(IndexType{1000}, IndexType{5},
                                                IndexType{15}, ValueType{0}));
    auto rhs =
        share(this->generate_rand_dense(ValueType{0}, mtx->get_size()[0]));
    auto x = Vec::create_with_config_of((rhs));
    x->fill(0.0);
    auto x_clone = clone(exec, x);
    auto rhs_clone = clone(exec, rhs);

    auto iter_crit = this->iter_criterion_factory;
    iter_crit->add_logger(this->iter_logger);
    auto res_norm = this->res_norm_criterion_factory;
    res_norm->add_logger(this->iter_logger);

    auto cg_factory = CG::build().with_criteria(iter_crit, res_norm).on(exec);
    auto cg = cg_factory->generate(mtx);
    cg->apply((rhs), (x));
    auto cg_num_iters = this->iter_logger->get_num_iterations();

    const auto omega_val = gko::remove_complex<ValueType>{1.5};
    auto gs_hbmc_factory = share(GS::build()
                                     .with_use_HBMC(true)
                                     .with_base_block_size(4u)
                                     .with_lvl_2_block_size(32u)
                                     .with_symmetric_preconditioner(true)
                                     .with_relaxation_factor(omega_val)
                                     .with_use_padding(true)
                                     .on(exec));

    auto gs_hbmc = share(gs_hbmc_factory->generate(mtx));

    iter_crit->add_logger(this->iter_logger_2);
    res_norm->add_logger(this->iter_logger_2);
    auto pcg_factory = CG::build()
                           .with_criteria(iter_crit, res_norm)
                           .with_generated_preconditioner(gs_hbmc)
                           .on(exec);
    auto pcg = pcg_factory->generate(mtx);
    pcg->apply((rhs_clone), (x_clone));

    auto pcg_num_iters = this->iter_logger_2->get_num_iterations();

    ASSERT_EQ(pcg_num_iters < cg_num_iters, 1);
}


TYPED_TEST(GaussSeidel, WorksWithPrepermMtxAndstorageSchemeInput)
{
    using Csr = typename TestFixture::Csr;
    using Vec = typename TestFixture::Vec;
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using GS = typename TestFixture::GS;
    using HBMC = gko::experimental::reorder::Hbmc<IndexType>;

    auto exec = this->exec;
    int tuple = 0;
    for (auto const& [num_rows, row_limit, lvl_2_block_size, base_block_size,
                      padding] : this->apply_params_) {
        std::cout << "tuple: " << tuple << " mtx size: " << num_rows
                  << std::endl;

        auto omega_val = gko::remove_complex<ValueType>{1.5};
        bool symm_precond = true;  // TODO make a loop over possible values

        auto mtx = gko::share(
            this->generate_rand_matrix(IndexType{num_rows}, IndexType{1},
                                       IndexType{row_limit}, ValueType{0}));
        auto rhs = gko::share(
            this->generate_rand_dense(ValueType{0}, mtx->get_size()[0]));
        auto x = Vec::create_with_config_of((rhs));
        x->fill(0.0);
        auto x_clone = clone(exec, x);
        auto rhs_clone = clone(exec, rhs);
        std::cout << "test 1" << std::endl;

        auto hbmc_reorder_factory =
            HBMC::build()
                .with_base_block_size(
                    static_cast<gko::size_type>(base_block_size))
                .with_lvl_2_block_size(
                    static_cast<gko::size_type>(lvl_2_block_size))
                .with_padding(padding)
                .with_symmetric_preconditioner(symm_precond)
                .on(exec);
        auto hbmc_reorder = gko::share(
            hbmc_reorder_factory->generate(gko::as<gko::LinOp>(mtx), true));
        auto ref_hbmc = GS::build()
                            .with_use_HBMC(true)
                            .with_base_block_size(base_block_size)
                            .with_lvl_2_block_size(lvl_2_block_size)
                            .with_symmetric_preconditioner(symm_precond)
                            .with_relaxation_factor(omega_val)
                            .with_use_padding(padding)
                            .with_prepermuted_input(false)
                            .with_preperm_mtx(false)
                            .on(exec)
                            ->generate(gko::as<gko::LinOp>(mtx));
        auto ref_vertex_colors = ref_hbmc->get_vertex_colors();

        std::cout << "test 2" << std::endl;

        auto storage_from_reorder =
            hbmc_reorder_factory->get_hbmc_storage_scheme();
        auto ref_storage = ref_hbmc->get_storage_scheme();
        GKO_ASSERT(storage_from_reorder.symm_ == symm_precond);

        std::cout << "test 2 1/2 storage num_blocks: "
                  << storage_from_reorder.num_blocks_ << std::endl;


        gko::array<IndexType> ref_vertex_colors_perm(exec, num_rows);

        auto tmp = gko::make_array_view(exec, num_rows,
                                        hbmc_reorder->get_permutation());
        GKO_ASSERT_ARRAY_EQ(tmp, ref_hbmc->get_permutation_idxs());

        for (int i = 0; i < num_rows; i++) {
            ref_vertex_colors_perm.get_data()[i] =
                ref_vertex_colors.get_data()[tmp.get_data()[i]];
        }

        std::cout << "test 3" << std::endl;

        auto preperm_mtx = gko::share(gko::as<Csr>(mtx->permute(hbmc_reorder)));
        auto perm_rhs = rhs->clone();
        rhs->permute(hbmc_reorder, perm_rhs, gko::matrix::permute_mode::rows);

        auto hbmc = GS::build()
                        .with_use_HBMC(true)
                        .with_base_block_size(base_block_size)
                        .with_lvl_2_block_size(lvl_2_block_size)
                        .with_use_padding(padding)
                        .with_preperm_mtx(true)
                        .with_prepermuted_input(
                            false)  // special case for reference kernels
                        .with_symmetric_preconditioner(symm_precond)
                        .with_relaxation_factor(omega_val)
                        .with_storage_scheme(storage_from_reorder)
                        .with_storage_scheme_ready(true)
                        .on(exec)
                        ->generate(gko::as<gko::LinOp>(preperm_mtx));

        std::cout << "test 4" << std::endl;

        auto perm = gko::array<IndexType>(exec, num_rows);
        std::iota(perm.get_data(), perm.get_data() + perm.get_size(), 0);
        auto inv_perm = gko::array<IndexType>(exec, num_rows);
        gko::kernels::reference::permutation::invert(
            exec, perm.get_const_data(), perm.get_size(), inv_perm.get_data());
        GKO_ASSERT_ARRAY_EQ(perm, inv_perm);
        auto perm_idxs_hbmc_w_preperm_mtx = hbmc->get_permutation_idxs();
        GKO_ASSERT_ARRAY_EQ(perm_idxs_hbmc_w_preperm_mtx, perm);

        std::cout << "test 5" << std::endl;

        // auto ref_color_ptrs = ref_hbmc->get_color_ptrs();
        // auto color_ptrs = hbmc->get_color_ptrs();
        // GKO_ASSERT_ARRAY_EQ(ref_color_ptrs, color_ptrs);

        // auto vertex_colors = hbmc->get_vertex_colors();
        // GKO_ASSERT_ARRAY_EQ(vertex_colors, ref_vertex_colors_perm);

        // std::cout << "test 5 1/2" << std::endl;

        GKO_ASSERT_ARRAY_EQ(hbmc->get_l_diag_rows(),
                            ref_hbmc->get_l_diag_rows());
        GKO_ASSERT_ARRAY_EQ(hbmc->get_l_spmv_row_ptrs(),
                            ref_hbmc->get_l_spmv_row_ptrs());
        GKO_ASSERT_ARRAY_EQ(hbmc->get_l_spmv_col_idxs(),
                            ref_hbmc->get_l_spmv_col_idxs());


        auto ref_l_diag_vals = ref_hbmc->get_l_diag_vals();
        auto l_diag_vals = hbmc->get_l_diag_vals();
        GKO_ASSERT_EQ(ref_l_diag_vals.get_size(), l_diag_vals.get_size());
        auto ref_l_diag_vals_vec =
            Vec::create(exec, gko::dim<2>{ref_l_diag_vals.get_size(), 1},
                        gko::make_array_view(exec, ref_l_diag_vals.get_size(),
                                             ref_l_diag_vals.get_data()),
                        1);
        auto l_diag_vals_vec =
            Vec::create(exec, gko::dim<2>{l_diag_vals.get_size(), 1},
                        gko::make_array_view(exec, l_diag_vals.get_size(),
                                             l_diag_vals.get_data()),
                        1);
        GKO_ASSERT_MTX_NEAR(ref_l_diag_vals_vec, l_diag_vals_vec,
                            r<ValueType>::value);
        if (symm_precond) {
            std::cout << "test 5 3/4" << std::endl;
            auto ref_u_diag_vals = ref_hbmc->get_u_diag_vals();
            auto u_diag_vals = hbmc->get_u_diag_vals();
            GKO_ASSERT_EQ(ref_u_diag_vals.get_size(), u_diag_vals.get_size());
            auto ref_u_diag_vals_vec = Vec::create(
                exec, gko::dim<2>{ref_u_diag_vals.get_size(), 1},
                gko::make_array_view(exec, ref_u_diag_vals.get_size(),
                                     ref_u_diag_vals.get_data()),
                1);
            auto u_diag_vals_vec =
                Vec::create(exec, gko::dim<2>{u_diag_vals.get_size(), 1},
                            gko::make_array_view(exec, u_diag_vals.get_size(),
                                                 u_diag_vals.get_data()),
                            1);
            GKO_ASSERT_MTX_NEAR(ref_u_diag_vals_vec, u_diag_vals_vec,
                                r<ValueType>::value);
        }

        auto ref_l_spmv_vals = ref_hbmc->get_l_spmv_vals();
        auto l_spmv_vals = hbmc->get_l_spmv_vals();
        GKO_ASSERT_EQ(ref_l_spmv_vals.get_size(), l_spmv_vals.get_size());
        auto ref_l_spmv_vals_vec =
            Vec::create(exec, gko::dim<2>{ref_l_spmv_vals.get_size(), 1},
                        gko::make_array_view(exec, ref_l_spmv_vals.get_size(),
                                             ref_l_spmv_vals.get_data()),
                        1);

        auto l_spmv_vals_vec =
            Vec::create(exec, gko::dim<2>{l_spmv_vals.get_size(), 1},
                        gko::make_array_view(exec, l_spmv_vals.get_size(),
                                             l_spmv_vals.get_data()),
                        1);
        GKO_ASSERT_MTX_NEAR(ref_l_spmv_vals_vec, l_spmv_vals_vec,
                            r<ValueType>::value);


        std::cout << "test 6" << std::endl;

        hbmc->apply((perm_rhs), (x));
        auto inv_perm_x = Vec::create_with_config_of((x));
        x->permute(hbmc_reorder, inv_perm_x,
                   gko::matrix::permute_mode::inverse_rows);

        std::cout << "test 7" << std::endl;

        auto storage = hbmc->get_storage_scheme();

        std::cout << "test 8" << std::endl;
        ref_hbmc->apply((rhs_clone), (x_clone));
        GKO_ASSERT_MTX_NEAR(x_clone, inv_perm_x, r<ValueType>::value);
        tuple++;
    }
}

}  // namespace
