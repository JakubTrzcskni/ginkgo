// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/preconditioner/gauss_seidel.hpp>


#include <random>


#include <gtest/gtest.h>


// #include <ginkgo/core/matrix/csr.hpp>
// #include <ginkgo/core/matrix/dense.hpp>
// #include <ginkgo/core/reorder/hbmc.hpp>
#include <ginkgo/ginkgo.hpp>

#include "core/preconditioner/gauss_seidel_kernels.hpp"
#include "core/test/utils.hpp"
#include "core/test/utils/matrix_generator.hpp"
#include "core/utils/matrix_utils.hpp"
#include "hip/test/utils.hip.hpp"

namespace {

using apply_param_type = std::vector<std::tuple<int, int, int, int, bool>>;
static apply_param_type allParams{
    std::make_tuple(16, 2, 16, 8, false), std::make_tuple(20, 5, 32, 4, false),
    std::make_tuple(20, 5, 32, 4, true), std::make_tuple(1000, 5, 32, 4, true),
    std::make_tuple(1000, 5, 32, 4, false),
    std::make_tuple(1000, 5, 32, 8, false),
    std::make_tuple(1000, 5, 32, 8, true),
    std::make_tuple(1000, 5, 32, 2, false),
    std::make_tuple(1000, 5, 32, 2, true),
    std::make_tuple(1000, 5, 32, 2, false),
    std::make_tuple(1000, 5, 32, 2, true),
    std::make_tuple(1000, 10, 16, 4, false),
    std::make_tuple(1000, 10, 16, 4, true),
    std::make_tuple(1000, 10, 4, 4, false),
    std::make_tuple(1000, 10, 4, 4, true),
    // std::make_tuple(1000, 10, 32, 16, false),//problem
    // std::make_tuple(1000, 10, 32, 16, true),//segafult hehe
    // std::make_tuple(1003, 15, 32, 4, false),//problem
    // std::make_tuple(1003, 15, 32, 4, true),//problem
    std::make_tuple(1000, 10, 4, 8, false),
    std::make_tuple(1000, 10, 4, 8, true)};

template <typename ValueIndexType>
class GaussSeidel : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using GS = gko::preconditioner::GaussSeidel<value_type, index_type>;
    GaussSeidel() : rand_engine(42), apply_params_{allParams} {}

    void SetUp()
    {
        ASSERT_GT(gko::HipExecutor::get_num_devices(), 0);
        ref = gko::ReferenceExecutor::create();
        hip = gko::HipExecutor::create(0, ref);
    }

    void TearDown()
    {
        if (hip != nullptr) {
            ASSERT_NO_THROW(hip->synchronize());
        }
    }

    std::default_random_engine rand_engine;
    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<gko::HipExecutor> hip;
    apply_param_type apply_params_;
};

TYPED_TEST_SUITE(GaussSeidel, gko::test::RealValueIndexTypes,
                 PairTypenameNameGenerator);


TYPED_TEST(GaussSeidel, GetDegreeOfNodesKernel)
{
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using Csr = gko::matrix::Csr<ValueType, IndexType>;
    auto hip_exec = this->hip;

    auto mtx = gko::initialize<Csr>({{1., 0., 0., 0.},
                                     {1., 2., 0., 0.},
                                     {1., 2., 3., 0.},
                                     {1., 2., 3., 4.}},
                                    hip_exec);
    const IndexType num_nodes = mtx->get_size()[0];
    gko::array<IndexType> degrees(hip_exec, num_nodes);

    gko::kernels::hip::gauss_seidel::get_degree_of_nodes(
        hip_exec, num_nodes, mtx->get_const_row_ptrs(), degrees.get_data());

    GKO_ASSERT_ARRAY_EQ(degrees, I<IndexType>({1, 2, 3, 4}));
}

TYPED_TEST(GaussSeidel, SimpleApplyKernelFromRef)
{
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using GS = typename TestFixture::GS;
    using Csr = gko::matrix::Csr<ValueType, IndexType>;
    using Vec = gko::matrix::Dense<ValueType>;

    auto ref_exec = this->ref;
    auto hip_exec = this->hip;

    gko::size_type num_rows = 35;
    gko::size_type row_limit = 3;
    gko::size_type num_rhs = 5;
    auto nz_dist = std::uniform_int_distribution<IndexType>(1, row_limit);
    auto val_dist =
        std::uniform_real_distribution<gko::remove_complex<ValueType>>(-1., 1.);

    auto mat_data =
        gko::test::generate_random_matrix_data<ValueType, IndexType>(
            num_rows, num_rows, nz_dist, val_dist, this->rand_engine);
    gko::utils::make_hpd(mat_data, 2.0);
    mat_data.sort_row_major();
    auto rhs = gko::test::generate_random_matrix<Vec>(
        num_rows, num_rhs,
        std::uniform_int_distribution<IndexType>(num_rows * num_rhs,
                                                 num_rows * num_rhs),
        val_dist, this->rand_engine, ref_exec, gko::dim<2>{num_rows, num_rhs});

    auto x = Vec::create_with_config_of(rhs.get());

    auto mtx = gko::share(Csr::create(ref_exec, gko::dim<2>(num_rows)));
    mtx->read(mat_data);
    auto d_mtx = gko::clone(hip_exec, mtx);

    auto ref_gs_factory = GS::build()
                              .with_use_HBMC(true)
                              .with_base_block_size(4u)
                              .with_lvl_2_block_size(32u)
                              .with_use_padding(true)
                              .on(ref_exec);

    auto ref_gs = ref_gs_factory->generate(mtx);
    auto perm_idxs =
        gko::array<IndexType>(ref_exec, ref_gs->get_permutation_idxs());
    auto rhs_perm = gko::as<Vec>(rhs.get()->row_permute(&perm_idxs));

    auto storage_scheme = ref_gs->get_storage_scheme();
    auto l_diag_rows = ref_gs->get_l_diag_rows();
    auto d_l_diag_rows = make_temporary_clone(hip_exec, &l_diag_rows);
    auto l_diag_vals = ref_gs->get_l_diag_vals();
    auto d_l_diag_vals = make_temporary_clone(hip_exec, &l_diag_vals);
    auto l_spmv_row_ptrs = ref_gs->get_l_spmv_row_ptrs();
    auto d_l_spmv_row_ptrs = make_temporary_clone(hip_exec, &l_spmv_row_ptrs);
    auto l_spmv_col_idxs = ref_gs->get_l_spmv_col_idxs();
    auto d_l_spmv_col_idxs = make_temporary_clone(hip_exec, &l_spmv_col_idxs);
    auto l_spmv_vals = ref_gs->get_l_spmv_vals();
    auto d_l_spmv_vals = make_temporary_clone(hip_exec, &l_spmv_vals);

    for (int kernel_version = 1; kernel_version <= 9; ++kernel_version) {
        x->fill(ValueType{0});
        auto d_x = gko::clone(hip_exec, x);
        auto d_rhs_perm = gko::clone(hip_exec, rhs_perm);
        auto d_perm_idxs = make_temporary_clone(hip_exec, &perm_idxs);
        gko::kernels::reference::gauss_seidel::simple_apply(
            ref_exec, l_diag_rows.get_const_data(),
            l_diag_vals.get_const_data(), l_spmv_row_ptrs.get_const_data(),
            l_spmv_col_idxs.get_const_data(), l_spmv_vals.get_const_data(),
            perm_idxs.get_const_data(), storage_scheme, rhs_perm.get(), x.get(),
            1);

        hip_exec->synchronize();
        gko::kernels::hip::gauss_seidel::simple_apply(
            hip_exec, d_l_diag_rows->get_const_data(),
            d_l_diag_vals->get_const_data(),
            d_l_spmv_row_ptrs->get_const_data(),
            d_l_spmv_col_idxs->get_const_data(),
            d_l_spmv_vals->get_const_data(), d_perm_idxs->get_const_data(),
            storage_scheme, d_rhs_perm.get(), d_x.get(), kernel_version);
        hip_exec->synchronize();

        GKO_ASSERT_MTX_NEAR(x, d_x, r<ValueType>::value);
    }
}

TYPED_TEST(GaussSeidel, SimpleApply)
{
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using GS = typename TestFixture::GS;
    using Csr = gko::matrix::Csr<ValueType, IndexType>;
    using Vec = gko::matrix::Dense<ValueType>;
    auto ref_exec = this->ref;
    auto hip_exec = this->hip;

    for (auto const& [num_rows, row_limit, w, b_s, padding] :
         this->apply_params_) {
        gko::size_type num_rhs = 10;
        auto nz_dist = std::uniform_int_distribution<IndexType>(
            1, static_cast<gko::size_type>(row_limit));
        auto val_dist =
            std::uniform_real_distribution<gko::remove_complex<ValueType>>(-1.,
                                                                           1.);
        auto mat_data =
            gko::test::generate_random_matrix_data<ValueType, IndexType>(
                static_cast<gko::size_type>(num_rows),
                static_cast<gko::size_type>(num_rows), nz_dist, val_dist,
                this->rand_engine);
        gko::utils::make_hpd(mat_data, 2.0);
        mat_data.sort_row_major();
        auto rhs = gko::test::generate_random_matrix<Vec>(
            static_cast<gko::size_type>(num_rows), num_rhs,
            std::uniform_int_distribution<IndexType>(
                static_cast<gko::size_type>(num_rows) * num_rhs,
                static_cast<gko::size_type>(num_rows) * num_rhs),
            val_dist, this->rand_engine, ref_exec,
            gko::dim<2>{static_cast<gko::size_type>(num_rows), num_rhs});
        auto d_rhs = gko::clone(hip_exec, rhs);

        auto x = Vec::create_with_config_of(rhs.get());
        x->fill(ValueType{0});
        auto d_x = gko::clone(hip_exec, x);

        auto mtx = gko::share(Csr::create(
            ref_exec, gko::dim<2>(static_cast<gko::size_type>(num_rows))));
        mtx->read(mat_data);

        auto ref_gs_factory =
            GS::build()
                .with_use_HBMC(true)
                .with_base_block_size(static_cast<gko::size_type>(b_s))
                .with_lvl_2_block_size(static_cast<gko::size_type>(w))
                .with_use_padding(padding)
                .on(ref_exec);
        auto ref_gs = ref_gs_factory->generate(mtx);

        auto device_gs_factory =
            GS::build()
                .with_use_HBMC(true)
                .with_base_block_size(static_cast<gko::size_type>(b_s))
                .with_lvl_2_block_size(static_cast<gko::size_type>(w))
                .with_use_padding(padding)
                .with_kernel_version(2)
                .on(hip_exec);
        auto device_gs = device_gs_factory->generate(mtx);

        ref_gs->apply(rhs.get(), x.get());

        device_gs->apply(d_rhs.get(), d_x.get());

        GKO_ASSERT_MTX_NEAR(x, d_x, r<ValueType>::value);
    }
}

TYPED_TEST(GaussSeidel, PrepermutedSimpleApply)
{
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using GS = typename TestFixture::GS;
    using Csr = gko::matrix::Csr<ValueType, IndexType>;
    using Vec = gko::matrix::Dense<ValueType>;
    auto ref_exec = this->ref;
    auto hip_exec = this->hip;
    for (auto const& [num_rows, row_limit, w, b_s, padding] :
         this->apply_params_) {
        gko::size_type num_rhs = 10;
        auto nz_dist = std::uniform_int_distribution<IndexType>(
            1, static_cast<gko::size_type>(row_limit));
        auto val_dist =
            std::uniform_real_distribution<gko::remove_complex<ValueType>>(-1.,
                                                                           1.);
        auto mat_data =
            gko::test::generate_random_matrix_data<ValueType, IndexType>(
                static_cast<gko::size_type>(num_rows),
                static_cast<gko::size_type>(num_rows), nz_dist, val_dist,
                this->rand_engine);
        gko::utils::make_hpd(mat_data, 2.0);
        mat_data.sort_row_major();
        auto rhs = gko::test::generate_random_matrix<Vec>(
            static_cast<gko::size_type>(num_rows), num_rhs,
            std::uniform_int_distribution<IndexType>(
                static_cast<gko::size_type>(num_rows) * num_rhs,
                static_cast<gko::size_type>(num_rows) * num_rhs),
            val_dist, this->rand_engine, ref_exec,
            gko::dim<2>{static_cast<gko::size_type>(num_rows), num_rhs});

        auto mtx = gko::share(Csr::create(
            ref_exec, gko::dim<2>(static_cast<gko::size_type>(num_rows))));
        mtx->read(mat_data);

        auto perm_gs_factory =
            GS::build()
                .with_use_HBMC(true)
                .with_base_block_size(static_cast<gko::size_type>(b_s))
                .with_lvl_2_block_size(static_cast<gko::size_type>(w))
                .with_use_padding(padding)
                .with_prepermuted_input(false)
                .on(hip_exec);
        auto perm_gs = perm_gs_factory->generate(mtx);
        for (int kernel_version = 2; kernel_version <= 9; ++kernel_version) {
            auto preperm_gs_factory =
                GS::build()
                    .with_use_HBMC(true)
                    .with_base_block_size(static_cast<gko::size_type>(b_s))
                    .with_lvl_2_block_size(static_cast<gko::size_type>(w))
                    .with_use_padding(padding)
                    .with_prepermuted_input(true)
                    .with_kernel_version(kernel_version)
                    .on(hip_exec);
            auto preperm_gs = preperm_gs_factory->generate(mtx);

            auto perm_idxs = gko::array<IndexType>(
                hip_exec, preperm_gs->get_permutation_idxs());
            auto d_rhs = gko::clone(hip_exec, rhs);
            auto d_permuted_rhs =
                gko::as<Vec>(d_rhs.get()->row_permute(&perm_idxs));
            auto d_x = Vec::create_with_config_of(d_rhs.get());
            d_x->fill(ValueType{0});
            auto d_permuted_x = gko::clone(hip_exec, d_x);

            perm_gs->apply(d_rhs.get(), d_x.get());
            // std::cout << "kernel version: " << kernel_version << std::endl;
            preperm_gs->apply(d_permuted_rhs.get(), d_permuted_x.get());

            auto ans = gko::as<Vec>(
                d_permuted_x.get()->inverse_row_permute(&perm_idxs));
            GKO_ASSERT_MTX_NEAR(ans, d_x, r<ValueType>::value);
        }
    }
}

TYPED_TEST(GaussSeidel, AdvancedApply)
{
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using GS = typename TestFixture::GS;
    using Csr = gko::matrix::Csr<ValueType, IndexType>;
    using Vec = gko::matrix::Dense<ValueType>;
    auto ref_exec = this->ref;
    auto hip_exec = this->hip;
    int i = 1;
    for (auto const& [num_rows, row_limit, w, b_s, padding] :
         this->apply_params_) {
        const auto omega_val = gko::remove_complex<ValueType>{1.5};
        gko::size_type num_rhs = 1;
        auto nz_dist = std::uniform_int_distribution<IndexType>(
            1, static_cast<gko::size_type>(row_limit));
        auto val_dist =
            std::uniform_real_distribution<gko::remove_complex<ValueType>>(-1.,
                                                                           1.);
        auto mat_data =
            gko::test::generate_random_matrix_data<ValueType, IndexType>(
                static_cast<gko::size_type>(num_rows),
                static_cast<gko::size_type>(num_rows), nz_dist, val_dist,
                this->rand_engine);
        gko::utils::make_hpd(mat_data, 2.0);
        mat_data.sort_row_major();
        auto rhs = gko::test::generate_random_matrix<Vec>(
            static_cast<gko::size_type>(num_rows), num_rhs,
            std::uniform_int_distribution<IndexType>(
                static_cast<gko::size_type>(num_rows) * num_rhs,
                static_cast<gko::size_type>(num_rows) * num_rhs),
            val_dist, this->rand_engine, ref_exec,
            gko::dim<2>{static_cast<gko::size_type>(num_rows), num_rhs});
        auto d_rhs = gko::clone(hip_exec, rhs);

        auto x = Vec::create_with_config_of(rhs.get());
        x->fill(ValueType{0});
        auto d_x = gko::clone(hip_exec, x);

        auto mtx = gko::share(Csr::create(
            ref_exec, gko::dim<2>(static_cast<gko::size_type>(num_rows))));
        mtx->read(mat_data);

        auto ref_gs_factory =
            GS::build()
                .with_use_HBMC(true)
                .with_base_block_size(static_cast<gko::size_type>(b_s))
                .with_lvl_2_block_size(static_cast<gko::size_type>(w))
                .with_use_padding(padding)
                .with_symmetric_preconditioner(true)
                .with_prepermuted_input(false)
                .with_relaxation_factor(omega_val)
                .on(ref_exec);
        auto ref_gs = ref_gs_factory->generate(mtx);

        auto device_gs_factory =
            GS::build()
                .with_use_HBMC(true)
                .with_base_block_size(static_cast<gko::size_type>(b_s))
                .with_lvl_2_block_size(static_cast<gko::size_type>(w))
                .with_use_padding(padding)
                .with_prepermuted_input(false)
                .with_symmetric_preconditioner(true)
                .with_relaxation_factor(omega_val)
                .with_kernel_version(9)
                .on(hip_exec);
        auto device_gs = device_gs_factory->generate(mtx);

        ref_gs->apply(rhs.get(), x.get());

        device_gs->apply(d_rhs.get(), d_x.get());
        // std::cout << "tuple: " << i++ << std::endl;
        GKO_ASSERT_MTX_NEAR(x, d_x, r<ValueType>::value);
    }
}

TYPED_TEST(GaussSeidel, PrepermutedAdvancedApply)
{
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using GS = typename TestFixture::GS;
    using Csr = gko::matrix::Csr<ValueType, IndexType>;
    using Vec = gko::matrix::Dense<ValueType>;
    auto ref_exec = this->ref;
    auto hip_exec = this->hip;
    auto i = 1;
    for (auto const& [num_rows, row_limit, w, b_s, padding] :
         this->apply_params_) {
        const auto omega_val = gko::remove_complex<ValueType>{1.5};
        gko::size_type num_rhs = 1;
        auto nz_dist = std::uniform_int_distribution<IndexType>(
            1, static_cast<gko::size_type>(row_limit));
        auto val_dist =
            std::uniform_real_distribution<gko::remove_complex<ValueType>>(-1.,
                                                                           1.);
        auto mat_data =
            gko::test::generate_random_matrix_data<ValueType, IndexType>(
                static_cast<gko::size_type>(num_rows),
                static_cast<gko::size_type>(num_rows), nz_dist, val_dist,
                this->rand_engine);
        gko::utils::make_hpd(mat_data, 2.0);
        mat_data.sort_row_major();
        auto rhs = gko::test::generate_random_matrix<Vec>(
            static_cast<gko::size_type>(num_rows), num_rhs,
            std::uniform_int_distribution<IndexType>(
                static_cast<gko::size_type>(num_rows) * num_rhs,
                static_cast<gko::size_type>(num_rows) * num_rhs),
            val_dist, this->rand_engine, ref_exec,
            gko::dim<2>{static_cast<gko::size_type>(num_rows), num_rhs});

        auto mtx = gko::share(Csr::create(
            ref_exec, gko::dim<2>(static_cast<gko::size_type>(num_rows))));
        mtx->read(mat_data);

        auto perm_gs_factory =
            GS::build()
                .with_use_HBMC(true)
                .with_base_block_size(static_cast<gko::size_type>(b_s))
                .with_lvl_2_block_size(static_cast<gko::size_type>(w))
                .with_use_padding(padding)
                .with_symmetric_preconditioner(true)
                .with_relaxation_factor(omega_val)
                .with_prepermuted_input(false)
                .on(ref_exec);
        auto ref_gs = perm_gs_factory->generate(mtx);

        auto preperm_gs_factory =
            GS::build()
                .with_use_HBMC(true)
                .with_base_block_size(static_cast<gko::size_type>(b_s))
                .with_lvl_2_block_size(static_cast<gko::size_type>(w))
                .with_use_padding(padding)
                .with_symmetric_preconditioner(true)
                .with_relaxation_factor(omega_val)
                .with_prepermuted_input(true)
                .on(hip_exec);
        auto preperm_gs = preperm_gs_factory->generate(mtx);

        auto perm_idxs =
            gko::array<IndexType>(hip_exec, preperm_gs->get_permutation_idxs());
        auto ref_perm_idxs =
            gko::array<IndexType>(ref_exec, ref_gs->get_permutation_idxs());
        GKO_ASSERT_ARRAY_EQ(perm_idxs, ref_perm_idxs);
        auto d_rhs = gko::clone(hip_exec, rhs);
        auto d_permuted_rhs =
            gko::as<Vec>(d_rhs.get()->row_permute(&perm_idxs));
        auto x = Vec::create_with_config_of(rhs.get());
        x->fill(ValueType{0});
        auto d_permuted_x = gko::clone(hip_exec, x);

        ref_gs->apply(rhs.get(), x.get());

        preperm_gs->apply(d_permuted_rhs.get(), d_permuted_x.get());

        auto ans =
            gko::as<Vec>(d_permuted_x.get()->inverse_row_permute(&perm_idxs));
        std::cout << "tuple: " << i++ << std::endl;
        GKO_ASSERT_MTX_NEAR(ans, x, r<ValueType>::value);
    }
}


TYPED_TEST(GaussSeidel, WorksWithPrepermMtxInput)
{
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using GS = typename TestFixture::GS;
    using Csr = gko::matrix::Csr<ValueType, IndexType>;
    using Vec = gko::matrix::Dense<ValueType>;
    using HBMC = gko::experimental::reorder::Hbmc<IndexType>;
   
    auto ref_exec = this->ref;
    auto hip_exec = this->hip;

    auto num_rows = 1000;
    const auto omega_val = gko::remove_complex<ValueType>{1.5};
    gko::size_type num_rhs = 1;
    auto padding = false;
    bool symm_precond = false;
    auto b_s = 4u;
    auto w = 32u;
    auto nz_dist = std::uniform_int_distribution<IndexType>(
        1, static_cast<gko::size_type>(15));
    auto val_dist =
        std::uniform_real_distribution<gko::remove_complex<ValueType>>(-1., 1.);
    auto mat_data =
        gko::test::generate_random_matrix_data<ValueType, IndexType>(
            static_cast<gko::size_type>(num_rows),
            static_cast<gko::size_type>(num_rows), nz_dist, val_dist,
            this->rand_engine);
    gko::utils::make_hpd(mat_data, 2.0);
    mat_data.sort_row_major();
    auto rhs = gko::test::generate_random_matrix<Vec>(
        static_cast<gko::size_type>(num_rows), num_rhs,
        std::uniform_int_distribution<IndexType>(
            static_cast<gko::size_type>(num_rows) * num_rhs,
            static_cast<gko::size_type>(num_rows) * num_rhs),
        val_dist, this->rand_engine, ref_exec,
        gko::dim<2>{static_cast<gko::size_type>(num_rows), num_rhs});

    auto mtx = gko::share(Csr::create(
        ref_exec, gko::dim<2>(static_cast<gko::size_type>(num_rows))));
    mtx->read(mat_data);
    auto hbmc_reorder_factory = HBMC::build()
                                    .with_base_block_size(b_s)
                                    .with_lvl_2_block_size(w)
                                    .with_padding(padding)
                                    .with_symmetric_preconditioner(symm_precond)
                                    .on(ref_exec);
    auto hbmc_reorder =
        hbmc_reorder_factory->generate(gko::as<gko::LinOp>(mtx), true);
    auto storage_from_reorder = hbmc_reorder_factory->get_hbmc_storage_scheme();
    auto preperm_mtx = gko::share(mtx->permute(hbmc_reorder));
    auto preperm_rhs =
        gko::share(rhs->permute(hbmc_reorder, gko::matrix::permute_mode::rows));

    auto gs_factory =
        GS::build()
            .with_use_HBMC(true)
            .with_base_block_size(static_cast<gko::size_type>(b_s))
            .with_lvl_2_block_size(static_cast<gko::size_type>(w))
            .with_use_padding(padding)
            .with_symmetric_preconditioner(symm_precond)
            .with_relaxation_factor(omega_val)
            .with_prepermuted_input(false)
            .with_preperm_mtx(true)
            .with_storage_scheme(storage_from_reorder)
            .with_storage_scheme_ready(true)
            .on(ref_exec);
    auto ref_gs = gko::share(gs_factory->generate(preperm_mtx));

    auto d_gs_factory =
        GS::build()
            .with_use_HBMC(true)
            .with_base_block_size(static_cast<gko::size_type>(b_s))
            .with_lvl_2_block_size(static_cast<gko::size_type>(w))
            .with_use_padding(padding)
            .with_symmetric_preconditioner(symm_precond)
            .with_relaxation_factor(omega_val)
            .with_prepermuted_input(false)
            .with_storage_scheme(storage_from_reorder)
            .with_storage_scheme_ready(true)
            .with_preperm_mtx(true)
            .with_kernel_version(9)
            .on(hip_exec);
    auto d_gs = gko::share(d_gs_factory->generate(preperm_mtx));

    auto d_rhs = gko::clone(hip_exec, preperm_rhs);
    auto x = Vec::create_with_config_of(preperm_rhs.get());
    x->fill(ValueType{0});
    auto d_x = gko::clone(hip_exec, x);

    ref_gs->apply(preperm_rhs.get(), x.get());

    d_gs->apply(d_rhs.get(), d_x.get());

    GKO_ASSERT_MTX_NEAR(x, d_x, r<ValueType>::value);
}

}  // namespace
