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
#include <iostream>
#include <map>
#include <string>

struct problem_geometry {
    unsigned int nx;
    unsigned int ny;
    unsigned int nz;
};
typedef struct problem_geometry geo;

void generate_geometry(unsigned int discretization_points_1D, geo* geometry)
{
    geometry->nx = discretization_points_1D;
    geometry->ny = discretization_points_1D;
    geometry->nz = discretization_points_1D;
}

// Creates a stencil matrix in CSR format, rhs, x, and the corresponding exact
// solution
template <typename ValueType, typename IndexType>
void generate_problem(gko::matrix::Csr<ValueType, IndexType>* matrix,
                      gko::matrix::Dense<ValueType>* rhs,
                      gko::matrix::Dense<ValueType>* x,
                      gko::matrix::Dense<ValueType>* x_exact, geo& geometry)
{
    auto nnz = 0;
    const auto nx = geometry.nx;
    const auto ny = geometry.ny;
    const auto nz = geometry.nz;
    const auto discretization_points = matrix->get_size()[0];

    auto rhs_values = rhs->get_values();
    auto x_values = x->get_values();
    auto x_exact_values = x_exact->get_values();

    gko::matrix_data<ValueType, IndexType> data{
        gko::dim<2>{discretization_points, discretization_points}, {}};
    auto pos = 0;

    for (auto iz = 0; iz < nz; iz++) {
        for (auto iy = 0; iy < ny; iy++) {
            for (auto ix = 0; ix < nx; ix++) {
                auto current_row = iz * nx * ny + iy * nx + ix;
                auto nnz_in_row = 0;
                for (auto ofs_z : {-1, 0, 1}) {
                    if (iz + ofs_z > -1 && iz + ofs_z < nz) {
                        for (auto ofs_y : {-1, 0, 1}) {
                            if (iy + ofs_y > -1 && iy + ofs_y < ny) {
                                for (auto ofs_x : {-1, 0, 1}) {
                                    if (ix + ofs_x > -1 && ix + ofs_x < nx) {
                                        auto current_col = current_row +
                                                           ofs_z * ny * nx +
                                                           ofs_y * nx + ofs_x;
                                        if (current_col == current_row) {
                                            data.nonzeros.emplace_back(
                                                current_row, current_col, 26.0);
                                        } else {
                                            data.nonzeros.emplace_back(
                                                current_row, current_col, -1.0);
                                        }
                                        nnz_in_row++;
                                    }
                                }
                            }
                        }
                    }
                }
                nnz += nnz_in_row;
                rhs_values[current_row] = 26.0 - ValueType(nnz_in_row - 1);
                x_values[current_row] = 0.0;
                x_exact_values[current_row] = 1.0;
            }
        }
    }
    matrix->read(data);
    assert(matrix->get_num_stored_elements() == nnz);
}

// Prints the solution `u`.
template <typename ValueType>
void print_solution(ValueType u0, ValueType u1,
                    const gko::matrix::Dense<ValueType>* u)
{
    std::cout << u0 << '\n';
    for (int i = 0; i < u->get_size()[0]; ++i) {
        std::cout << u->get_const_values()[i] << '\n';
    }
    std::cout << u1 << std::endl;
}

// Computes the 1-norm of the error given the computed `u` and the correct
// solution function `correct_u`.
template <typename ValueType>
gko::remove_complex<ValueType> calculate_error(
    int discretization_points, const gko::matrix::Dense<ValueType>* u,
    const gko::matrix::Dense<ValueType>* u_exact)
{
    const ValueType h = 1.0 / static_cast<ValueType>(discretization_points + 1);
    gko::remove_complex<ValueType> error = 0.0;
    for (int i = 0; i < discretization_points; ++i) {
        using std::abs;
        error +=
            abs(u->get_const_values()[i] - u_exact->get_const_values()[i]) /
            abs(u_exact->get_const_values()[i]);
    }
    return error;
}

int main(int argc, char* argv[])
{
    // Some shortcuts
    using ValueType = double;
    using IndexType = int;

    using vec = gko::matrix::Dense<ValueType>;
    using mtx = gko::matrix::Csr<ValueType, IndexType>;
    using cg = gko::solver::Cg<ValueType>;
    using bj = gko::preconditioner::Jacobi<ValueType, IndexType>;

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
            std::atoi(argv[2]) > 16 ? std::atoi(argv[2]) : 32;
        geometry = geo{dp_1D, dp_1D, dp_1D};
    } else if (argc == 5) {
        const unsigned int dp_x =
            std::atoi(argv[2]) > 16 ? std::atoi(argv[2]) : 32;
        const unsigned int dp_y =
            std::atoi(argv[3]) > 16 ? std::atoi(argv[3]) : 32;
        const unsigned int dp_z =
            std::atoi(argv[4]) > 16 ? std::atoi(argv[4]) : 32;
        geometry = geo{dp_x, dp_y, dp_z};
    } else {
        geometry = geo{32, 32, 32};
    }


    const unsigned int dp_3D = geometry.nx * geometry.ny * geometry.nz;

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
    // executor used by the application
    const auto app_exec = exec->get_master();

    // initialize matrix and vectors
    auto matrix = mtx::create(app_exec, gko::dim<2>(dp_3D));
    auto rhs = vec::create(app_exec, gko::dim<2>(dp_3D, 1));
    auto x = vec::create(app_exec, gko::dim<2>(dp_3D, 1));
    auto x_exact = vec::create(app_exec, gko::dim<2>(dp_3D, 1));
    std::cout << "matrices and vectors initialized" << std::endl;

    // generate matrix, rhs and solution
    generate_problem(lend(matrix), lend(rhs), lend(x), lend(x_exact), geometry);
    std::cout << "problem generated" << std::endl;

    // Reference CG solve
    const gko::remove_complex<ValueType> reduction_factor = 1e-7;
    // Generate solver and solve the system
    cg::build()
        .with_criteria(
            gko::stop::Iteration::build().with_max_iters(dp_3D).on(exec),
            gko::stop::ResidualNorm<ValueType>::build()
                .with_reduction_factor(reduction_factor)
                .on(exec))
        .with_preconditioner(bj::build().on(exec))
        .on(exec)
        ->generate(clone(exec, matrix))  // copy the matrix to the executor
        ->apply(lend(rhs), lend(x));

    std::cout << "Solve complete.\nThe average relative error is "
              << calculate_error(dp_3D, lend(x), lend(x_exact)) /
                     static_cast<gko::remove_complex<ValueType>>(dp_3D)
              << std::endl;
}