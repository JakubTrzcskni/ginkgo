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


#include <random>


#include <gtest/gtest.h>


#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>

#include "core/preconditioner/gauss_seidel_kernels.hpp"
#include "core/test/utils.hpp"
#include "core/test/utils/matrix_generator.hpp"
#include "core/utils/matrix_utils.hpp"
#include "cuda/test/utils.hpp"

namespace {

using apply_param_type = std::vector<std::tuple<int, int, int, int, bool>>;
static apply_param_type allParams{std::make_tuple(1000, 5, 32, 4, false),
                                  std::make_tuple(1000, 5, 32, 4, true),
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
                                  std::make_tuple(1000, 10, 4, 8, false),
                                  std::make_tuple(1000, 10, 4, 8, true),
                                  //   std::make_tuple(1003, 15, 32, 4, false),
                                  //   std::make_tuple(1003, 15, 32, 4, true),
                                  std::make_tuple(20, 5, 32, 4, false),
                                  std::make_tuple(20, 5, 32, 4, true)};

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
        ASSERT_GT(gko::CudaExecutor::get_num_devices(), 0);
        ref = gko::ReferenceExecutor::create();
        cuda = gko::CudaExecutor::create(0, ref);
    }

    void TearDown()
    {
        if (cuda != nullptr) {
            ASSERT_NO_THROW(cuda->synchronize());
        }
    }

    std::default_random_engine rand_engine;
    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<gko::CudaExecutor> cuda;
    apply_param_type apply_params_;
};

TYPED_TEST_SUITE(GaussSeidel, gko::test::ValueIndexTypes,
                 PairTypenameNameGenerator);

// TYPED_TEST(GaussSeidel, SimpleTest)
// {
//     using ValueType = typename TestFixture::value_type;
//     using IndexType = typename TestFixture::index_type;
//     auto cuda_exec = this->cuda;

//     const IndexType num_nodes = 12;
//     gko::array<IndexType> coloring(
//         cuda_exec, I<IndexType>({1, 1, 2, 2, 1, 1, 0, 0, 2, 2, 0, 0}));
//     const IndexType max_color = 2;
//     gko::array<IndexType> color_ptrs(cuda_exec, max_color + 2);
//     gko::array<IndexType> permutation_idxs(cuda_exec, num_nodes);
//     gko::array<IndexType> block_ordering(
//         cuda_exec, I<IndexType>({0, 1, 4, 5, 2, 3, 8, 9, 6, 7, 10, 11}));

//     gko::kernels::cuda::gauss_seidel::get_permutation_from_coloring(
//         cuda_exec, num_nodes, coloring.get_data(), max_color,
//         color_ptrs.get_data(), permutation_idxs.get_data(),
//         block_ordering.get_const_data());

//     GKO_ASSERT_ARRAY_EQ(permutation_idxs,
//                         I<IndexType>({6, 7, 10, 11, 0, 1, 4, 5, 2, 3, 8,
//                         9}));
//     GKO_ASSERT_ARRAY_EQ(color_ptrs, I<IndexType>({0, 4, 8, 12}));
// }

TYPED_TEST(GaussSeidel, GetDegreeOfNodesKernel)
{
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using Csr = gko::matrix::Csr<ValueType, IndexType>;
    auto cuda_exec = this->cuda;

    auto mtx = gko::initialize<Csr>({{1., 0., 0., 0.},
                                     {1., 2., 0., 0.},
                                     {1., 2., 3., 0.},
                                     {1., 2., 3., 4.}},
                                    cuda_exec);
    const IndexType num_nodes = mtx->get_size()[0];
    gko::array<IndexType> degrees(cuda_exec, num_nodes);

    gko::kernels::cuda::gauss_seidel::get_degree_of_nodes(
        cuda_exec, num_nodes, mtx->get_const_row_ptrs(), degrees.get_data());

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
    auto cuda_exec = this->cuda;

    gko::size_type num_rows = 1000;
    gko::size_type row_limit = 10;
    gko::size_type num_rhs = 5;
    auto nz_dist = std::uniform_int_distribution<IndexType>(1, row_limit);
    auto val_dist =
        std::uniform_real_distribution<gko::remove_complex<ValueType>>(-1., 1.);
    auto mat_data =
        gko::test::generate_random_matrix_data<ValueType, IndexType>(
            num_rows, num_rows, nz_dist, val_dist, this->rand_engine);
    gko::utils::make_hpd(mat_data, 2.0);
    mat_data.ensure_row_major_order();
    auto rhs = gko::test::generate_random_matrix<Vec>(
        num_rows, num_rhs,
        std::uniform_int_distribution<IndexType>(num_rows * num_rhs,
                                                 num_rows * num_rhs),
        val_dist, this->rand_engine, ref_exec, gko::dim<2>{num_rows, num_rhs});

    auto x = Vec::create_with_config_of(gko::lend(rhs));
    x->fill(ValueType{0});
    auto d_x = gko::clone(cuda_exec, x);

    auto mtx = gko::share(Csr::create(ref_exec, gko::dim<2>(num_rows)));
    mtx->read(mat_data);

    auto ref_gs_factory = GS::build()
                              .with_use_HBMC(true)
                              .with_base_block_size(4u)
                              .with_lvl_2_block_size(32u)
                              .on(ref_exec);

    auto ref_gs = ref_gs_factory->generate(mtx);

    auto perm_idxs =
        gko::array<IndexType>(ref_exec, ref_gs->get_permutation_idxs());
    auto rhs_perm = gko::as<Vec>(gko::lend(rhs)->row_permute(&perm_idxs));

    auto storage_scheme = ref_gs->get_storage_scheme();
    auto l_diag_rows = ref_gs->get_l_diag_rows();
    auto d_l_diag_rows = make_temporary_clone(cuda_exec, &l_diag_rows);
    auto l_diag_vals = ref_gs->get_l_diag_vals();
    auto d_l_diag_vals = make_temporary_clone(cuda_exec, &l_diag_vals);
    auto l_spmv_row_ptrs = ref_gs->get_l_spmv_row_ptrs();
    auto d_l_spmv_row_ptrs = make_temporary_clone(cuda_exec, &l_spmv_row_ptrs);
    auto l_spmv_col_idxs = ref_gs->get_l_spmv_col_idxs();
    auto d_l_spmv_col_idxs = make_temporary_clone(cuda_exec, &l_spmv_col_idxs);
    auto l_spmv_vals = ref_gs->get_l_spmv_vals();
    auto d_l_spmv_vals = make_temporary_clone(cuda_exec, &l_spmv_vals);

    auto d_perm_idxs = make_temporary_clone(cuda_exec, &perm_idxs);
    auto d_rhs_perm = gko::clone(cuda_exec, rhs_perm);

    gko::kernels::reference::gauss_seidel::simple_apply(
        ref_exec, l_diag_rows.get_const_data(), l_diag_vals.get_const_data(),
        l_spmv_row_ptrs.get_const_data(), l_spmv_col_idxs.get_const_data(),
        l_spmv_vals.get_const_data(), perm_idxs.get_const_data(),
        storage_scheme, gko::lend(rhs_perm), gko::lend(x), 0);

    cuda_exec->synchronize();

    gko::kernels::cuda::gauss_seidel::simple_apply(
        cuda_exec, d_l_diag_rows->get_const_data(),
        d_l_diag_vals->get_const_data(), d_l_spmv_row_ptrs->get_const_data(),
        d_l_spmv_col_idxs->get_const_data(), d_l_spmv_vals->get_const_data(),
        d_perm_idxs->get_const_data(), storage_scheme, gko::lend(d_rhs_perm),
        gko::lend(d_x), 1);
    cuda_exec->synchronize();

    GKO_ASSERT_MTX_NEAR(x, d_x, r<ValueType>::value);
}

TYPED_TEST(GaussSeidel, SimpleApply)
{
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using GS = typename TestFixture::GS;
    using Csr = gko::matrix::Csr<ValueType, IndexType>;
    using Vec = gko::matrix::Dense<ValueType>;

    auto ref_exec = this->ref;
    auto cuda_exec = this->cuda;

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
        mat_data.ensure_row_major_order();
        auto rhs = gko::test::generate_random_matrix<Vec>(
            static_cast<gko::size_type>(num_rows), num_rhs,
            std::uniform_int_distribution<IndexType>(
                static_cast<gko::size_type>(num_rows) * num_rhs,
                static_cast<gko::size_type>(num_rows) * num_rhs),
            val_dist, this->rand_engine, ref_exec,
            gko::dim<2>{static_cast<gko::size_type>(num_rows), num_rhs});
        auto d_rhs = gko::clone(cuda_exec, rhs);

        auto x = Vec::create_with_config_of(gko::lend(rhs));
        x->fill(ValueType{0});
        auto d_x = gko::clone(cuda_exec, x);

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
                .on(cuda_exec);
        auto device_gs = device_gs_factory->generate(mtx);

        ref_gs->apply(gko::lend(rhs), gko::lend(x));

        device_gs->apply(gko::lend(d_rhs), gko::lend(d_x));

        GKO_ASSERT_MTX_NEAR(x, d_x, r<ValueType>::value);
    }
}

TYPED_TEST(GaussSeidel, PrepermutedApply)
{
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using GS = typename TestFixture::GS;
    using Csr = gko::matrix::Csr<ValueType, IndexType>;
    using Vec = gko::matrix::Dense<ValueType>;
    auto ref_exec = this->ref;
    auto cuda_exec = this->cuda;
    int i = 1;
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
        mat_data.ensure_row_major_order();
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
                .on(cuda_exec);
        auto perm_gs = perm_gs_factory->generate(mtx);

        auto preperm_gs_factory =
            GS::build()
                .with_use_HBMC(true)
                .with_base_block_size(static_cast<gko::size_type>(b_s))
                .with_lvl_2_block_size(static_cast<gko::size_type>(w))
                .with_use_padding(padding)
                .with_prepermuted_input(true)
                .with_kernel_version(3)
                .on(cuda_exec);
        auto preperm_gs = preperm_gs_factory->generate(mtx);

        auto perm_idxs = gko::array<IndexType>(
            cuda_exec, preperm_gs->get_permutation_idxs());
        auto d_rhs = gko::clone(cuda_exec, rhs);
        auto d_permuted_rhs =
            gko::as<Vec>(gko::lend(d_rhs)->row_permute(&perm_idxs));
        auto d_x = Vec::create_with_config_of(gko::lend(d_rhs));
        d_x->fill(ValueType{0});
        auto d_permuted_x = gko::clone(cuda_exec, d_x);

        perm_gs->apply(gko::lend(d_rhs), gko::lend(d_x));

        // std::cout << "tuple: " << i++ << std::endl;
        preperm_gs->apply(gko::lend(d_permuted_rhs), gko::lend(d_permuted_x));

        auto ans = gko::as<Vec>(
            gko::lend(d_permuted_x)->inverse_row_permute(&perm_idxs));

        GKO_ASSERT_MTX_NEAR(ans, d_x, r<ValueType>::value);
    }
}

}  // namespace