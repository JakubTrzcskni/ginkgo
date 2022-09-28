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


#include <ginkgo/ginkgo.hpp>


#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

// #include <gtest/gtest.h>

// #include <ginkgo/core/base/array.hpp>
// #include <ginkgo/core/base/executor.hpp>
// #include <ginkgo/core/matrix/csr.hpp>
// #include <ginkgo/core/matrix/dense.hpp>
// #include <ginkgo/core/preconditioner/gauss_seidel.hpp>

#include "core/preconditioner/sparse_display.hpp"
// #include "core/test/utils.hpp"
#include "core/test/utils/matrix_generator.hpp"
#include "core/utils/matrix_utils.hpp"

template <typename ValueType, typename IndexType>
void visualize(std::shared_ptr<const gko::Executor> exec,
               gko::matrix::Csr<ValueType, IndexType>* csr_mat,
               std::string plot_label)
{
    auto dense_mat = gko::matrix::Dense<ValueType>::create(exec);
    csr_mat->convert_to(lend(dense_mat));
    auto num_rows = dense_mat->get_size()[0];
    gko::preconditioner::visualize::spy_ge(num_rows, num_rows,
                                           dense_mat->get_values(), plot_label);
}

template <typename ValueType, typename IndexType>
std::unique_ptr<gko::matrix::Csr<ValueType, IndexType>>
generate_2D_regular_grid_matrix(std::shared_ptr<const gko::Executor> exec,
                                size_t size, ValueType deduction_help,
                                IndexType deduction_help_ind,
                                bool nine_point = false)
{
    gko::dim<2>::dimension_type grid_points = size * size;
    gko::matrix_data<ValueType, IndexType> data(gko::dim<2>{grid_points});

    auto matrix = gko::matrix::Csr<ValueType, IndexType>::create(
        exec, gko::dim<2>{grid_points});

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

template <typename ValueType, typename IndexType>
std::unique_ptr<gko::matrix::Csr<ValueType, IndexType>> generate_rand_matrix(
    std::shared_ptr<const gko::Executor> exec, IndexType size,
    IndexType num_elems_lo, IndexType num_elems_hi, ValueType deduction_help,
    gko::remove_complex<ValueType> values_lo = -1.0,
    gko::remove_complex<ValueType> values_hi = 1.0)
{
    std::default_random_engine rand_engine{15};
    auto mtx =
        gko::matrix::Csr<ValueType, IndexType>::create(exec, gko::dim<2>(size));
    auto mat_data =
        gko::test::generate_random_matrix_data<ValueType, IndexType>(
            mtx->get_size()[0], mtx->get_size()[1],
            std::uniform_int_distribution<IndexType>(num_elems_lo,
                                                     num_elems_hi),
            std::normal_distribution<gko::remove_complex<ValueType>>(values_lo,
                                                                     values_hi),
            rand_engine);

    gko::utils::make_hpd(mat_data, 2.0);
    mat_data.ensure_row_major_order();
    mtx->read(mat_data);

    return give(mtx);
}
template <typename ValueType>
std::unique_ptr<gko::matrix::Dense<ValueType>> generate_rand_dense(
    std::shared_ptr<const gko::Executor> exec, ValueType deduction_help,
    size_t num_rows, size_t num_cols = 1,
    gko::remove_complex<ValueType> values_lo = -1.0,
    gko::remove_complex<ValueType> values_hi = 1.0)
{
    std::default_random_engine rand_engine{15};
    auto rhs_rand{
        gko::test::generate_random_matrix<gko::matrix::Dense<ValueType>>(
            num_rows, num_cols,
            std::uniform_int_distribution<size_t>(num_rows, num_rows),
            std::normal_distribution<gko::remove_complex<ValueType>>(values_lo,
                                                                     values_hi),
            rand_engine, exec)};

    return give(rhs_rand);
}

int main(int argc, char* argv[])
{
    // Some shortcuts
    using ValueType = double;
    using IndexType = int;
    using vec = gko::matrix::Dense<ValueType>;
    using Csr = gko::matrix::Csr<ValueType, IndexType>;
    using GS = gko::preconditioner::GaussSeidel<ValueType, IndexType>;


    // Print version information
    std::cout << gko::version_info::get() << std::endl;

    if (argc == 2 && (std::string(argv[1]) == "--help")) {
        std::cerr << "Usage: " << argv[0] << " [executor]" << std::endl;
        std::exit(-1);
    }

    // TODO benchmark measure the time to solution + time to generate against
    // reference

    const auto executor_string = argc >= 2 ? argv[1] : "reference";
    const auto base_block_size_arg = argc >= 3 ? argv[2] : "1";
    const auto lvl_2_block_size_arg = argc >= 4 ? argv[3] : "0";
    const auto rand_size_arg = argc >= 5 ? argv[4] : "100";
    const auto rand_nnz_row_lo_arg = argc >= 6 ? argv[5] : "1";
    const auto rand_nnz_row_hi_arg = argc >= 7 ? argv[6] : "10";
    IndexType base_block_size;
    std::stringstream ss_1(base_block_size_arg);
    if (!(ss_1 >> base_block_size)) GKO_NOT_SUPPORTED(base_block_size_arg);
    IndexType lvl_2_block_size;
    std::stringstream ss_2(lvl_2_block_size_arg);
    if (!(ss_2 >> lvl_2_block_size)) GKO_NOT_SUPPORTED(lvl_2_block_size_arg);
    IndexType rand_size;
    IndexType rand_nnz_row_lo;
    IndexType rand_nnz_row_hi;
    std::stringstream ss_3(rand_size_arg);
    std::stringstream ss_4(rand_nnz_row_lo_arg);
    std::stringstream ss_5(rand_nnz_row_hi_arg);
    if (!(ss_3 >> rand_size)) GKO_NOT_SUPPORTED(rand_size_arg);
    if (!(ss_4 >> rand_nnz_row_lo)) GKO_NOT_SUPPORTED(rand_nnz_row_lo_arg);
    if (!(ss_5 >> rand_nnz_row_hi)) GKO_NOT_SUPPORTED(rand_nnz_row_hi_arg);


    // Figure out where to run the code
    std::map<std::string, std::function<std::shared_ptr<gko::Executor>()>>
        exec_map{
            {"omp", [] { return gko::OmpExecutor::create(); }},
            {"cuda",
             [] {
                 return gko::CudaExecutor::create(0, gko::OmpExecutor::create(),
                                                  true);
             }},
            {"hip",
             [] {
                 return gko::HipExecutor::create(0, gko::OmpExecutor::create(),
                                                 true);
             }},
            {"dpcpp",
             [] {
                 return gko::DpcppExecutor::create(0,
                                                   gko::OmpExecutor::create());
             }},
            {"reference", [] { return gko::ReferenceExecutor::create(); }}};

    // executor where Ginkgo will perform the computation
    const auto exec = exec_map.at(executor_string)();  // throws if not valid

    auto mtx_rand = gko::share(generate_rand_matrix(
        exec, rand_size, rand_nnz_row_lo, rand_nnz_row_hi, ValueType{0}));
    auto mtx_rand_grid = gko::share(generate_2D_regular_grid_matrix(
        exec, size_t{std::sqrt(rand_size)}, ValueType{}, IndexType{}, true));

    visualize(exec, gko::lend(mtx_rand), std::string("mtxRand"));
    visualize(exec, gko::lend(mtx_rand_grid), std::string("mtxRandGrid"));

    auto HBMC_gs_factory = GS::build()
                               .with_use_HBMC(true)
                               .with_base_block_size(base_block_size)
                               .with_lvl_2_block_size(lvl_2_block_size)
                               .on(exec);


    auto gs_rand = HBMC_gs_factory->generate(gko::as<gko::LinOp>(mtx_rand));
    auto gs_rand_grid =
        HBMC_gs_factory->generate(gko::as<gko::LinOp>(mtx_rand_grid));
    auto label = std::string("mtxRandPermuted-");
    // label += typeid(ValueType).name() + std::string("-") +
    //                   typeid(IndexType).name() + std::string("-");
    // label += std::to_string(base_block_size) + std::string("-") +
    //          std::to_string(lvl_2_block_size);
    visualize(exec, gko::lend(gs_rand->get_ltr_matrix()), label);

    // auto label2 = std::string("mtxRandBlockOrdering-");
    // label2 += std::to_string(base_block_size) + std::string("-") +
    //           std::to_string(lvl_2_block_size);
    // visualize(exec, gko::lend(gs_rand->get_utr_matrix()), label2);
    auto label_grid = std::string("mtxRandGridPermuted-");
    visualize(exec, gko::lend(gs_rand_grid->get_ltr_matrix()), label_grid);
}
