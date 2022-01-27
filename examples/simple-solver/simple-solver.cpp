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

// @sect3{Include files}

// This is the main ginkgo header file.
#include <ginkgo/ginkgo.hpp>

// Add the fstream header to read from data from files.
#include <fstream>
// Add the C++ iostream header to output information to the console.
#include <iostream>
// Add the STL map header for the executor selection
#include <map>
// Add the string manipulation header to handle strings.
#include <string>


template <typename ValueType>
using vec = gko::matrix::Dense<ValueType>;


template <typename ValueType>
using real_vec = gko::matrix::Dense<gko::remove_complex<ValueType>>;


namespace utils {


// creates a zero vector
template <typename ValueType>
std::unique_ptr<vec<ValueType>> create_vector(
    std::shared_ptr<const gko::Executor> exec, gko::size_type size,
    ValueType value)
{
    auto res = vec<ValueType>::create(exec);
    res->read(gko::matrix_data<ValueType>(gko::dim<2>{size, 1}, value));
    return res;
}


// utilities for computing norms and residuals
template <typename ValueType>
ValueType get_first_element(const vec<ValueType>* norm)
{
    return norm->get_executor()->copy_val_to_host(norm->get_const_values());
}


template <typename ValueType>
gko::remove_complex<ValueType> compute_norm(const vec<ValueType>* b)
{
    auto exec = b->get_executor();
    auto b_norm = gko::initialize<real_vec<ValueType>>({0.0}, exec);
    b->compute_norm2(gko::lend(b_norm));
    return get_first_element(gko::lend(b_norm));
}


template <typename ValueType>
gko::remove_complex<ValueType> compute_residual_norm(
    const gko::LinOp* system_matrix, const vec<ValueType>* b,
    const vec<ValueType>* x)
{
    auto exec = system_matrix->get_executor();
    auto one = gko::initialize<vec<ValueType>>({1.0}, exec);
    auto neg_one = gko::initialize<vec<ValueType>>({-1.0}, exec);
    auto res = gko::clone(b);
    system_matrix->apply(gko::lend(one), gko::lend(x), gko::lend(neg_one),
                         gko::lend(res));
    return compute_norm(gko::lend(res));
}


}  // namespace utils


int main(int argc, char* argv[])
{
    // Use some shortcuts. In Ginkgo, vectors are seen as a gko::matrix::Dense
    // with one column/one row. The advantage of this concept is that using
    // multiple vectors is a now a natural extension of adding columns/rows are
    // necessary.
    using ValueType = double;
    using RealValueType = gko::remove_complex<ValueType>;
    using IndexType = int;
    using vec = gko::matrix::Dense<ValueType>;
    using real_vec = gko::matrix::Dense<RealValueType>;
    // The gko::matrix::Csr class is used here, but any other matrix class such
    // as gko::matrix::Coo, gko::matrix::Hybrid, gko::matrix::Ell or
    // gko::matrix::Sellp could also be used.
    using mtx = gko::matrix::Csr<ValueType, IndexType>;
    // The gko::solver::Cg is used here, but any other solver class can also be
    // used.
    using cg = gko::solver::Cg<ValueType>;

    // Print the ginkgo version information.
    std::cout << gko::version_info::get() << std::endl;

    if (argc == 2 && (std::string(argv[1]) == "--help")) {
        std::cerr << "Usage: " << argv[0] << " [executor] " << std::endl;
        std::exit(-1);
    }

    // @sect3{Where do you want to run your solver ?}
    // The gko::Executor class is one of the cornerstones of Ginkgo. Currently,
    // we have support for
    // an gko::OmpExecutor, which uses OpenMP multi-threading in most of its
    // kernels, a gko::ReferenceExecutor, a single threaded specialization of
    // the OpenMP executor and a gko::CudaExecutor which runs the code on a
    // NVIDIA GPU if available.
    // @note With the help of C++, you see that you only ever need to change the
    // executor and all the other functions/ routines within Ginkgo should
    // automatically work and run on the executor with any other changes.
    const auto executor_string = argc >= 2 ? argv[1] : "reference";
    const auto grid_dim =
        static_cast<gko::size_type>(argc >= 3 ? std::atoi(argv[2]) : 20);
    const auto max_iters =
        static_cast<gko::size_type>(argc >= 4 ? std::atoi(argv[3]) : 1000);
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
    const auto num_rows = grid_dim * grid_dim * grid_dim;

    gko::matrix_data<ValueType, IndexType> A_data;
    A_data.size = {num_rows, num_rows};
    for (int i = 0; i < grid_dim; i++) {
        for (int j = 0; j < grid_dim; j++) {
            for (int k = 0; k < grid_dim; k++) {
                auto idx = i * grid_dim * grid_dim + j * grid_dim + k;
                if (i > 0)
                    A_data.nonzeros.emplace_back(idx, idx - grid_dim * grid_dim,
                                                 -1);
                if (j > 0)
                    A_data.nonzeros.emplace_back(idx, idx - grid_dim, -1);
                if (k > 0) A_data.nonzeros.emplace_back(idx, idx - 1, -1);
                A_data.nonzeros.emplace_back(idx, idx, 8);
                if (k < grid_dim - 1)
                    A_data.nonzeros.emplace_back(idx, idx + 1, -1);
                if (j < grid_dim - 1)
                    A_data.nonzeros.emplace_back(idx, idx + grid_dim, -1);
                if (i < grid_dim - 1)
                    A_data.nonzeros.emplace_back(idx, idx + grid_dim * grid_dim,
                                                 -1);
            }
        }
    }
    // Generate b and x vectors
    auto A = gko::share(mtx::create(exec, A_data.size));
    auto b = utils::create_vector<ValueType>(exec, A->get_size()[0], 1.0);
    auto x = utils::create_vector<ValueType>(exec, A->get_size()[0], 0.0);
    A->read(A_data);

    const RealValueType reduction_factor{1e-7};
    auto solver_gen =
        cg::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(max_iters).on(
                    exec),
                gko::stop::ResidualNorm<ValueType>::build()
                    .with_reduction_factor(reduction_factor)
                    .on(exec))
            .on(exec);
    auto solver = solver_gen->generate(A);

    auto x_clone = gko::clone(x);
    {
        solver->apply(lend(b), lend(x));
    }

    {
        solver->apply(lend(b), lend(x_clone));
    }
    auto one = gko::initialize<vec>({1.0}, exec);
    auto neg_one = gko::initialize<vec>({-1.0}, exec);
    auto res = gko::initialize<real_vec>({0.0}, exec);
    A->apply(lend(one), lend(x), lend(neg_one), lend(b));
    b->compute_norm2(lend(res));

    std::cout << "Matrix size: " << A->get_size()
              << "\nResidual norm sqrt(r^T r):\n";
    write(std::cout, lend(res));
}
