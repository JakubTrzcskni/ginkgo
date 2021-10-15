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


#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>


int main(int argc, char* argv[])
{
    // Some shortcuts
    using ValueType = double;
    using MixedType = float;
    using IndexType = int;
    using vec = gko::matrix::Dense<ValueType>;
    using mtx = gko::matrix::Csr<ValueType, IndexType>;
    using fcg = gko::solver::Fcg<ValueType>;
    using cg = gko::solver::Cg<ValueType>;
    using ir = gko::solver::Ir<ValueType>;
    using mg = gko::solver::Multigrid;
    using bj = gko::preconditioner::Jacobi<ValueType, IndexType>;
    using amgx_pgm = gko::multigrid::AmgxPgm<ValueType, IndexType>;

    using cg_f = gko::solver::Cg<MixedType>;
    using ir_f = gko::solver::Ir<MixedType>;
    using bj_f = gko::preconditioner::Jacobi<MixedType, IndexType>;
    using amgx_pgm_f = gko::multigrid::AmgxPgm<MixedType, IndexType>;

    // Print version information
    std::cout << gko::version_info::get() << std::endl;

    const auto executor_string = argc >= 2 ? argv[1] : "reference";
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

    // const auto fileA = argc >= 3 ? argv[2] : "data/A.mtx";
    const bool use_mixed = argc >= 3;
    std::cout << "Use Mixed: " << use_mixed << std::endl;
    // Read data
    auto A = share(gko::read<mtx>(std::ifstream("data/A.mtx"), exec));
    // Create RHS as 1 and initial guess as 0
    gko::size_type size = A->get_size()[0];
    auto host_x = vec::create(exec->get_master(), gko::dim<2>(size, 1));
    auto host_b = vec::create(exec->get_master(), gko::dim<2>(size, 1));
    for (auto i = 0; i < size; i++) {
        host_x->at(i, 0) = 0.;
        host_b->at(i, 0) = 1.;
    }
    auto x = vec::create(exec);
    auto b = vec::create(exec);
    x->copy_from(host_x.get());
    b->copy_from(host_b.get());

    // Calculate initial residual by overwriting b
    auto one = gko::initialize<vec>({1.0}, exec);
    auto neg_one = gko::initialize<vec>({-1.0}, exec);
    auto initres = gko::initialize<vec>({0.0}, exec);
    A->apply(lend(one), lend(x), lend(neg_one), lend(b));
    b->compute_norm2(lend(initres));

    // copy b again
    b->copy_from(host_b.get());

    // Prepare the stopping criteria
    const gko::remove_complex<ValueType> tolerance = 1e-8;
    auto iter_stop =
        gko::stop::Iteration::build().with_max_iters(100u).on(exec);
    auto tol_stop = gko::stop::AbsoluteResidualNorm<ValueType>::build()
                        .with_tolerance(tolerance)
                        .on(exec);

    std::shared_ptr<const gko::log::Convergence<ValueType>> logger =
        gko::log::Convergence<ValueType>::create(exec);
    iter_stop->add_logger(logger);
    tol_stop->add_logger(logger);

    // Create smoother factory (ir with bj)
    auto inner_solver_gen =
        gko::share(bj::build().with_max_block_size(1u).on(exec));
    auto inner_solver_gen_f =
        gko::share(bj_f::build().with_max_block_size(1u).on(exec));
    auto smoother_gen = gko::share(
        ir::build()
            .with_solver(inner_solver_gen)
            .with_relaxation_factor(0.9)
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(2u).on(exec))
            .on(exec));
    auto smoother_gen_f = gko::share(
        ir_f::build()
            .with_solver(inner_solver_gen_f)
            .with_relaxation_factor(0.9)
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(2u).on(exec))
            .on(exec));
    // Create MultigridLevel factory
    auto mg_level_gen = amgx_pgm::build().with_deterministic(true).on(exec);
    auto mg_level_gen_f = amgx_pgm_f::build().with_deterministic(true).on(exec);
    // Create CoarsestSolver factory
    auto coarsest_gen = gko::share(
        ir::build()
            .with_solver(inner_solver_gen)
            .with_relaxation_factor(0.9)
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(4u).on(exec))
            .on(exec));
    auto coarsest_gen_f = gko::share(
        ir_f::build()
            .with_solver(inner_solver_gen_f)
            .with_relaxation_factor(0.9)
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(4u).on(exec))
            .on(exec));
    // Create multigrid factory
    std::shared_ptr<gko::LinOpFactory> multigrid_gen;
    if (use_mixed) {
        std::cout << "USE" << std::endl;
        multigrid_gen =
            mg::build()
                .with_max_levels(9u)
                .with_min_coarse_rows(2u)
                .with_pre_smoother(smoother_gen, smoother_gen_f)
                .with_post_uses_pre(true)
                .with_mg_level(gko::share(mg_level_gen),
                               gko::share(mg_level_gen_f))
                .with_level_selector([](const gko::size_type level,
                                        const gko::LinOp*) -> gko::size_type {
                    std::cout << "level " << level << std::endl;
                    return level >= 1 ? 1 : 0;
                })
                .with_coarsest_solver(coarsest_gen_f)
                .with_zero_guess(true)
                .with_criteria(
                    gko::stop::Iteration::build().with_max_iters(1u).on(exec))
                .on(exec);
    } else {
        multigrid_gen =
            mg::build()
                .with_max_levels(9u)
                .with_min_coarse_rows(2u)
                .with_pre_smoother(smoother_gen)
                .with_post_uses_pre(true)
                .with_mg_level(gko::share(mg_level_gen))
                .with_coarsest_solver(coarsest_gen)
                .with_zero_guess(true)
                .with_criteria(
                    gko::stop::Iteration::build().with_max_iters(1u).on(exec))
                .on(exec);
    }
    std::cout << "multigrid_gen " << multigrid_gen.get() << std::endl;
    // Create solver factory
    auto solver_gen =
        cg::build()
            .with_criteria(gko::share(iter_stop), gko::share(tol_stop))
            .with_preconditioner(gko::share(multigrid_gen))
            .on(exec);
    // Create solver
    std::chrono::nanoseconds gen_time(0);
    auto gen_tic = std::chrono::steady_clock::now();
    auto solver = solver_gen->generate(A);
    exec->synchronize();
    auto gen_toc = std::chrono::steady_clock::now();
    gen_time +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(gen_toc - gen_tic);

    // Solve system
    exec->synchronize();
    std::chrono::nanoseconds time(0);
    auto tic = std::chrono::steady_clock::now();
    solver->apply(lend(b), lend(x));
    exec->synchronize();
    auto toc = std::chrono::steady_clock::now();
    time += std::chrono::duration_cast<std::chrono::nanoseconds>(toc - tic);

    // Calculate residual
    auto res = gko::initialize<vec>({0.0}, exec);
    A->apply(lend(one), lend(x), lend(neg_one), lend(b));
    b->compute_norm2(lend(res));

    std::cout << "Initial residual norm sqrt(r^T r): \n";
    write(std::cout, lend(initres));
    std::cout << "Final residual norm sqrt(r^T r): \n";
    write(std::cout, lend(res));

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
