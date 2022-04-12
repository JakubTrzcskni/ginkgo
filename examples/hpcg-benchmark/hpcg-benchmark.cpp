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

#include <omp.h>
#include <chrono>
#include <fstream>
#include <ginkgo/ginkgo.hpp>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>


#include "include/geometric-multigrid.hpp"

#include "include/utils.hpp"
#include "test/problem_generation_test.hpp"
#include "test/prolong_restrict_test.hpp"


template <typename ValueType, typename IndexType>
void cg_without_preconditioner(const std::shared_ptr<gko::Executor> exec,
                               gko::multigrid::problem_geometry& geometry,
                               ValueType value_help, IndexType index_help)
{
    using vec = gko::matrix::Dense<ValueType>;
    using mtx = gko::matrix::Csr<ValueType, IndexType>;
    using cg = gko::solver::Cg<ValueType>;
    using bj = gko::preconditioner::Jacobi<ValueType, IndexType>;
    using geo = gko::multigrid::problem_geometry;

    const unsigned int dp_3D = get_dp_3D(geometry.nx, geometry.ny, geometry.nz);

    std::cout << "Reference CG - no preconditioner" << std::endl;
    // initialize matrix and vectors
    auto matrix = share(mtx::create(exec, gko::dim<2>(dp_3D)));
    auto rhs = vec::create(exec, gko::dim<2>(dp_3D, 1));
    auto x = vec::create(exec, gko::dim<2>(dp_3D, 1));
    auto x_exact = vec::create(exec, gko::dim<2>(dp_3D, 1));
    std::cout << "matrices and vectors initialized" << std::endl;

    auto one = gko::initialize<vec>({1.0}, exec);
    auto neg_one = gko::initialize<vec>({-1.0}, exec);
    auto initres = gko::initialize<vec>({0.0}, exec);
    auto res = gko::initialize<vec>({0.0}, exec);

    generate_problem(exec, geometry, lend(matrix), lend(rhs), lend(x),
                     lend(x_exact));
    std::cout << "problem generated" << std::endl;


    const gko::remove_complex<ValueType> reduction_factor = 1e-7;
    // Generate solver and solve the system
    std::shared_ptr<const gko::log::Convergence<ValueType>> logger =
        gko::log::Convergence<ValueType>::create(exec);
    auto iter_stop =
        gko::stop::Iteration::build().with_max_iters(dp_3D).on(exec);
    auto tol_stop = gko::stop::ResidualNorm<ValueType>::build()
                        .with_reduction_factor(reduction_factor)
                        .on(exec);
    iter_stop->add_logger(logger);
    tol_stop->add_logger(logger);
    auto cg_factory =
        cg::build()
            .with_criteria(gko::share(iter_stop), gko::share(tol_stop))
            .on(exec);

    auto solver = cg_factory->generate(
        clone(exec, matrix));  // copy the matrix to the executor


    auto tic = std::chrono::steady_clock::now();

    solver->apply(lend(rhs), lend(x));
    exec->synchronize();

    auto tac = std::chrono::steady_clock::now();

    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(tac - tic);

    matrix->apply(lend(one), lend(x), lend(neg_one), lend(rhs));
    rhs->compute_norm2(lend(res));

    std::cout << "Solve complete.\nThe average relative error is "
              << calculate_error_device(dp_3D, lend(x), lend(x_exact)) /
                     static_cast<gko::remove_complex<ValueType>>(dp_3D)
              << std::endl;

    std::cout << "CG iteration count:     " << logger->get_num_iterations()
              << std::endl;
    std::cout << "CG execution time [ms]: "
              << static_cast<double>(time.count()) / 1000000.0 << std::endl;
    std::cout << "CG execution time per iteraion[ms]: "
              << static_cast<double>(time.count()) / 1000000.0 /
                     logger->get_num_iterations()
              << std::endl;
}


template <typename ValueType, typename IndexType>
void cg_with_mg(const std::shared_ptr<gko::Executor> exec,
                gko::multigrid::problem_geometry& geometry,
                ValueType value_help, IndexType index_help)
{
    using vec = gko::matrix::Dense<ValueType>;
    using mtx = gko::matrix::Csr<ValueType, IndexType>;
    using cg = gko::solver::Cg<ValueType>;
    using bj = gko::preconditioner::Jacobi<ValueType, IndexType>;
    using geo = gko::multigrid::problem_geometry;
    using ir = gko::solver::Ir<ValueType>;
    using mg = gko::solver::Multigrid;
    using gmg = gko::multigrid::Gmg<ValueType, IndexType>;

    const unsigned int dp_3D = get_dp_3D(geometry.nx, geometry.ny, geometry.nz);


    std::cout << "CG with MG preconditioner" << std::endl;
    // initialize matrix and vectors
    auto matrix = share(mtx::create(exec, gko::dim<2>(dp_3D)));
    auto rhs = vec::create(exec, gko::dim<2>(dp_3D, 1));
    auto x = vec::create(exec, gko::dim<2>(dp_3D, 1));
    auto x_exact = vec::create(exec, gko::dim<2>(dp_3D, 1));
    std::cout << "matrices and vectors initialized" << std::endl;

    auto one = gko::initialize<vec>({1.0}, exec);
    auto neg_one = gko::initialize<vec>({-1.0}, exec);
    auto initres = gko::initialize<vec>({0.0}, exec);
    auto res = gko::initialize<vec>({0.0}, exec);

    // generate matrix, rhs and solution
    generate_problem(exec, geometry, lend(matrix), lend(rhs), lend(x),
                     lend(x_exact));
    std::cout << "problem generated" << std::endl;

    const gko::remove_complex<ValueType> reduction_factor = 1e-7;
    auto iter_stop = gko::stop::Iteration::build().with_max_iters(100u).on(
        exec);  // dp_3D).on(exec);
    auto tol_stop = gko::stop::ResidualNorm<ValueType>::build()
                        .with_reduction_factor(reduction_factor)
                        .on(exec);

    std::shared_ptr<const gko::log::Convergence<ValueType>> logger =
        gko::log::Convergence<ValueType>::create(exec);
    iter_stop->add_logger(logger);
    tol_stop->add_logger(logger);

    auto inner_solver_gen =
        gko::share(bj::build().with_max_block_size(1u).on(exec));

    auto smoother_gen = gko::share(
        ir::build()
            .with_solver(inner_solver_gen)
            .with_relaxation_factor(0.9)
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(2u).on(exec))
            .on(exec));

    auto mg_level_gen = gmg::build().with_fine_geo(geometry).on(exec);

    auto coarsest_gen = gko::share(
        ir::build()
            .with_solver(inner_solver_gen)
            .with_relaxation_factor(0.9)
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(4u).on(exec))
            .on(exec));

    auto multigrid_gen =
        mg::build()
            .with_max_levels(9u)
            .with_min_coarse_rows(28u)
            .with_pre_smoother(smoother_gen)
            .with_post_uses_pre(true)
            .with_mg_level(gko::share(mg_level_gen))
            .with_coarsest_solver(coarsest_gen)
            .with_zero_guess(true)
            .with_criteria(gko::stop::Iteration::build().with_max_iters(1u).on(
                exec))  // max_iters - how many mg cycles are used for the
                        // preconditioning/solving
            .on(exec);


    auto solver_gen =
        cg::build()
            .with_criteria(gko::share(iter_stop), gko::share(tol_stop))
            .with_preconditioner(gko::share(multigrid_gen))
            .on(exec);

    std::chrono::nanoseconds gen_time(0);
    auto gen_tic = std::chrono::steady_clock::now();
    auto solver = solver_gen->generate(matrix);
    // auto solver = multigrid_gen->generate(matrix);
    exec->synchronize();
    auto gen_toc = std::chrono::steady_clock::now();

    gen_time +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(gen_toc - gen_tic);


    exec->synchronize();
    std::chrono::nanoseconds time(0);
    auto tic = std::chrono::steady_clock::now();
    solver->apply(lend(rhs), lend(x));
    exec->synchronize();
    auto toc = std::chrono::steady_clock::now();
    time += std::chrono::duration_cast<std::chrono::nanoseconds>(toc - tic);

    matrix->apply(lend(one), lend(x), lend(neg_one), lend(rhs));
    rhs->compute_norm2(lend(res));

    std::cout << "Solve complete.\nThe average relative error is "
              << calculate_error_device(dp_3D, lend(x), lend(x_exact)) /
                     static_cast<gko::remove_complex<ValueType>>(dp_3D)
              << std::endl;

    // Print solver statistics
    std::cout << "CG iteration count:     " << logger->get_num_iterations()
              << std::endl;
    std::cout << "CG generation time [ms]: "
              << static_cast<double>(gen_time.count()) / 1000000.0 << std::endl;
    std::cout << "CG execution time [ms]: "
              << static_cast<double>(time.count()) / 1000000.0 << std::endl;
    std::cout << "CG execution time per iteraion[ms]: "
              << static_cast<double>(time.count()) / 1000000.0 /
                     logger->get_num_iterations()
              << std::endl;
}

int main(int argc, char* argv[])
{
    // Some shortcuts
    using ValueType = double;
    using IndexType = int;

    using geo = gko::multigrid::problem_geometry;

    // Print version information
    std::cout << gko::version_info::get() << std::endl;

    if (argc == 2 && (std::string(argv[1]) == "--help")) {
        std::cerr << "Usage: " << argv[0]
                  << " [executor] [DISCRETIZATION_POINTS in one dimension of a "
                     "qube (default = 32, minimum = 16)]\n"
                  << "Alternative: " << argv[0]
                  << " [executor] [DP_x] [DP_y] [DP_z]" << std::endl;
        std::exit(-1);
    }

    // Get number of discretization points
    const auto executor_string = argc >= 2 ? argv[1] : "reference";
    geo geometry = {};
    if (argc == 3) {
        const unsigned int dp_1D =
            std::atoi(argv[2]);  //>= 16 ? std::atoi(argv[2]) : 32;
        geometry = geo{dp_1D, dp_1D, dp_1D};
    } else if (argc == 5) {
        const unsigned int dp_x =
            std::atoi(argv[2]);  // >= 16 ? std::atoi(argv[2]) : 32;
        const unsigned int dp_y =
            std::atoi(argv[3]);  // >= 16 ? std::atoi(argv[3]) : 32;
        const unsigned int dp_z =
            std::atoi(argv[4]);  // >= 16 ? std::atoi(argv[4]) : 32;
        geometry = geo{dp_x, dp_y, dp_z};
    } else {
        geometry = geo{32, 32, 32};
    }

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


    // explicit restrict
    {
        test_restriction(exec, geometry, ValueType{}, IndexType{});
    }

    // explicit prolong
    {
        test_prolongation(exec, geometry, ValueType{}, IndexType{});
    }

    // test matrix generation cuda
    if (strcmp(executor_string, "cuda") == 0) {
        test_matrix_generation(exec, geometry, ValueType{}, IndexType{});
    }

    // test problem generation cuda
    if (strcmp(executor_string, "cuda") == 0) {
        test_rhs_and_x_generation(exec, geometry, ValueType{}, IndexType{});
    }

    // Reference CG solve
    {
        cg_without_preconditioner(exec, geometry, ValueType{}, IndexType{});
    }

    // MG preconditioned CG
    {
        cg_with_mg(exec, geometry, ValueType{}, IndexType{});
    }
}
