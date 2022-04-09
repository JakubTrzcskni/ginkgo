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

#ifndef GKO_PUBLIC_CORE_MULTIGRID_GMG_HPP_
#define GKO_PUBLIC_CORE_MULTIGRID_GMG_HPP_


#include <array>
#include <vector>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/composition.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/polymorphic_object.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/identity.hpp>
#include <ginkgo/core/matrix/row_gatherer.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>
#include <ginkgo/core/multigrid/multigrid_level.hpp>
#include "core/components/fill_array_kernels.hpp"
#include "core/matrix/csr_builder.hpp"


template <typename ValueType>
void prolongation_kernel(int nx, int ny, int nz, const ValueType* coeffs,
                         const ValueType* rhs, const int rhs_size, ValueType* x,
                         const int x_size);
template <typename ValueType>
void restriction_kernel(int nx, int ny, int nz, const ValueType* coeffs,
                        const ValueType* rhs, const int rhs_size, ValueType* x,
                        const int x_size);
template <typename ValueType, typename IndexType>
std::array<std::shared_ptr<gko::matrix::Dense<ValueType>>, 3u>
rhs_and_x_generation_kernel(
    std::shared_ptr<const gko::Executor> exec,
    gko::matrix::Csr<ValueType, IndexType>* system_matrix);

template <typename ValueType, typename IndexType>
std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>>
matrix_generation_kernel(std::shared_ptr<const gko::Executor> exec,
                         const int nx, const int ny, const int nz,
                         ValueType value_help, IndexType index_help);


namespace gko {
namespace multigrid {

struct problem_geometry {
    size_type nx;
    size_type ny;
    size_type nz;
};

static size_t grid2index(size_t x, size_t y, size_t z, size_t nx, size_t ny,
                         size_t offset = 0)
{
    return offset + z * (ny + 1) * (nx + 1) + y * (nx + 1) + x;
};
static int integerPower(int base, int exponent)
{
    if (exponent == 0) return 1;

    int result = integerPower(base, exponent / 2);
    result *= result;

    if (exponent & 1) result *= base;

    return result;
};
template <typename ValueType, typename IndexType>
void generate_problem_matrix(std::shared_ptr<const gko::Executor> exec,
                             problem_geometry& geo,
                             matrix::Csr<ValueType, IndexType>* mat)
{
    struct matrix_generation : gko::Operation {
        matrix_generation(std::shared_ptr<const gko::Executor> exec,
                          problem_geometry geo,
                          matrix::Csr<ValueType, IndexType>* mat)
            : exec{exec}, geo{geo}, mat{mat}
        {}
        void run(std::shared_ptr<const gko::CudaExecutor>) const override
        {
            // matrix_generation_kernel(exec, geo.nx, geo.ny, geo.nz, mat);
        }

        void run(std::shared_ptr<const gko::OmpExecutor>) const override
        {
            auto nx = geo.nx;
            auto ny = geo.ny;
            auto nz = geo.nz;
            const auto discretization_points = mat->get_size()[0];
            GKO_ASSERT((nx + 1) * (ny + 1) * (nz + 1) == discretization_points);
            gko::matrix_data<ValueType, IndexType> data{
                gko::dim<2>(discretization_points), {}};

#pragma omp for
            for (auto iz = 0; iz <= nz; iz++) {
                for (auto iy = 0; iy <= ny; iy++) {
                    for (auto ix = 0; ix <= nx; ix++) {
                        auto current_row = grid2index(ix, iy, iz, nx, ny);
                        for (auto ofs_z : {-1, 0, 1}) {
                            if (iz + ofs_z > -1 && iz + ofs_z <= nz) {
                                for (auto ofs_y : {-1, 0, 1}) {
                                    if (iy + ofs_y > -1 && iy + ofs_y <= ny) {
                                        for (auto ofs_x : {-1, 0, 1}) {
                                            if (ix + ofs_x > -1 &&
                                                ix + ofs_x <= nx) {
                                                auto current_col = grid2index(
                                                    ofs_x, ofs_y, ofs_z, nx, ny,
                                                    current_row);
                                                if (current_col ==
                                                    current_row) {
                                                    data.nonzeros.emplace_back(
                                                        current_row,
                                                        current_col, 26.0);
                                                } else {
                                                    data.nonzeros.emplace_back(
                                                        current_row,
                                                        current_col, -1.0);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            mat->read(data);
        }
        std::shared_ptr<const gko::Executor> exec;
        problem_geometry geo;
        matrix::Csr<ValueType, IndexType>* mat;
    };
    exec->run(matrix_generation(exec, geo, mat));
}

// Creates a stencil matrix in CSR format, rhs, x, and the corresponding exact
// solution
template <typename ValueType, typename IndexType>
void generate_problem(std::shared_ptr<const gko::Executor> exec,
                      gko::matrix::Csr<ValueType, IndexType>* matrix,
                      gko::matrix::Dense<ValueType>* rhs,
                      gko::matrix::Dense<ValueType>* x,
                      gko::matrix::Dense<ValueType>* x_exact,
                      gko::multigrid::problem_geometry& geometry)
{
    const auto nx = geometry.nx;
    const auto ny = geometry.ny;
    const auto nz = geometry.nz;

    auto rhs_values = rhs->get_values();
    auto x_values = x->get_values();
    auto x_exact_values = x_exact->get_values();

    // gko::multigrid::
    generate_problem_matrix(exec, geometry, matrix);
#pragma omp for
    for (auto iz = 0; iz <= nz; iz++) {
        for (auto iy = 0; iy <= ny; iy++) {
            for (auto ix = 0; ix <= nx; ix++) {
                auto current_row =
                    gko::multigrid::grid2index(ix, iy, iz, nx, ny);
                auto nnz_in_row = 0;
#pragma omp simd  // probably it doesn't help here
                {
                    for (auto ofs_z : {-1, 0, 1}) {
                        if (iz + ofs_z > -1 && iz + ofs_z <= nz) {
                            for (auto ofs_y : {-1, 0, 1}) {
                                if (iy + ofs_y > -1 && iy + ofs_y <= ny) {
                                    for (auto ofs_x : {-1, 0, 1}) {
                                        if (ix + ofs_x > -1 &&
                                            ix + ofs_x <= nx) {
                                            nnz_in_row++;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    rhs_values[current_row] = 26.0 - ValueType(nnz_in_row - 1);
                    x_values[current_row] = 0.0;
                    x_exact_values[current_row] = 1.0;
                }
            }
        }
    }
}


template <typename ValueType = default_precision, typename IndexType = int32>
class gmg_restriction
    : public gko::EnableLinOp<gmg_restriction<ValueType, IndexType>>,
      public gko::EnableCreateMethod<gmg_restriction<ValueType, IndexType>> {
public:
    // uses a 3D stencil (27-Point) to restrict the rhs vector. It halves the
    // number of discretization points n respect to every axis
    //  -> rhs vector length is increased 8-fold
    //  the stencil is generated symmetrically
    // passed geometry has to be the coarse problem geometry
    gmg_restriction(std::shared_ptr<const gko::Executor> exec,
                    gko::size_type dp_x = 16, gko::size_type dp_y = 16,
                    gko::size_type dp_z = 16, gko::size_type coarse_size = 4096,
                    gko::size_type fine_size = 32768, ValueType left = 0.25,
                    ValueType center = 0.5, ValueType right = 0.25)
        : gko::EnableLinOp<gmg_restriction>(
              exec, gko::dim<2>{coarse_size, fine_size}),
          coefficients(exec, {left, center, right}),
          geo{dp_x, dp_y, dp_z}

    {}

protected:
    using vec = gko::matrix::Dense<ValueType>;
    using coef_type = gko::Array<ValueType>;

    void apply_impl(const gko::LinOp* b, gko::LinOp* x) const override
    {
        auto dense_b = gko::as<vec>(b);
        auto dense_x = gko::as<vec>(x);

        struct restriction_operation : gko::Operation {
            restriction_operation(const coef_type& coefficients, const vec* b,
                                  vec* x, problem_geometry geo)
                : coefficients{coefficients},
                  fine_rhs{b},
                  coarse_x{x},
                  coarse_geo{geo}
            {}
            void run(std::shared_ptr<const gko::OmpExecutor>) const override
            {
                auto c_values = coarse_x->get_values();

                const auto f_values = fine_rhs->get_const_values();

                auto c_nz = coarse_geo.nz;
                auto c_ny = coarse_geo.ny;
                auto c_nx = coarse_geo.nx;

                auto coeffs = coefficients.get_const_data();

#pragma omp parallel for
                for (auto c_z = 0; c_z <= c_nz; c_z++) {
                    auto f_z = 2 * c_z;
                    for (auto c_y = 0; c_y <= c_ny; c_y++) {
                        auto f_y = 2 * c_y;
                        for (auto c_x = 0; c_x <= c_nx; c_x++) {
                            auto f_x = 2 * c_x;
                            auto c_row = grid2index(c_x, c_y, c_z, c_nx, c_ny);
                            auto f_row =
                                grid2index(f_x, f_y, f_z, 2 * c_nx, 2 * c_ny);
                            c_values[c_row] = 0;
                            const auto on_border =
                                (c_x == 0) | (c_y == 0) | (c_z == 0) |
                                (c_x == c_nx) | (c_y == c_ny) | (c_z == c_nz);

                            if (!on_border) {
                                for (auto ofs_z : {-1, 0, 1}) {
                                    if (f_z + ofs_z > -1 &&
                                        f_z + ofs_z <= 2 * c_nz) {
                                        for (auto ofs_y : {-1, 0, 1}) {
                                            if (f_y + ofs_y > -1 &&
                                                f_y + ofs_y <= 2 * c_ny) {
                                                for (auto ofs_x : {-1, 0, 1}) {
                                                    if (f_x + ofs_x > -1 &&
                                                        f_x + ofs_x <=
                                                            2 * c_nx) {
                                                        auto f_id = grid2index(
                                                            ofs_x, ofs_y, ofs_z,
                                                            2 * c_nx, 2 * c_ny,
                                                            f_row);

                                                        c_values[c_row] +=
                                                            coeffs[ofs_z + 1] *
                                                            coeffs[ofs_y + 1] *
                                                            coeffs[ofs_x + 1] *
                                                            f_values[f_id];
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            } else {
                                c_values[c_row] = f_values[f_row];
                            }
                        }
                    }
                }
            }
            // cuda impl
            void run(std::shared_ptr<const gko::CudaExecutor>) const override
            {
                restriction_kernel(
                    coarse_geo.nx, coarse_geo.ny, coarse_geo.nz,
                    coefficients.get_const_data(), fine_rhs->get_const_values(),
                    fine_rhs->get_size()[0], coarse_x->get_values(),
                    coarse_x->get_size()[0]);
            }
            void run(
                std::shared_ptr<const gko::ReferenceExecutor>) const override
            {}
            const coef_type& coefficients;
            const vec* fine_rhs;
            vec* coarse_x;
            problem_geometry coarse_geo;
        };
        this->get_executor()->run(
            restriction_operation(coefficients, dense_b, dense_x, geo));
    }

    void apply_impl(const gko::LinOp* alpha, const gko::LinOp* b,
                    const gko::LinOp* beta, gko::LinOp* x) const override
    {
        auto dense_b = gko::as<vec>(b);
        auto dense_x = gko::as<vec>(x);
        auto tmp_x = dense_x->clone();
        this->apply_impl(b, lend(tmp_x));
        dense_x->scale(beta);
        dense_x->add_scaled(alpha, lend(tmp_x));
    }

private:
    coef_type coefficients;
    problem_geometry geo;
};

template <typename ValueType = default_precision, typename IndexType = int32>
class gmg_prolongation
    : public gko::EnableLinOp<gmg_prolongation<ValueType, IndexType>>,
      public gko::EnableCreateMethod<gmg_prolongation<ValueType, IndexType>> {
public:
    // uses a 3D stencil (27-Point) to interpolate (prolong) the rhs vector
    // it doubles the number of discretization points in respect to every
    // axis
    // -> rhs vector length is increased 8-folg
    // the stencil is generated symmetrically
    // passed geometry is the coarse problem geometry
    gmg_prolongation(std::shared_ptr<const gko::Executor> exec,
                     gko::size_type dp_x = 16, gko::size_type dp_y = 16,
                     gko::size_type dp_z = 16,
                     gko::size_type coarse_size = 4096,
                     gko::size_type fine_size = 32768, ValueType left = 0.5,
                     ValueType center = 1.0, ValueType right = 0.5)
        : gko::EnableLinOp<gmg_prolongation>(
              exec, gko::dim<2>{fine_size, coarse_size}),
          coefficients(exec, {left, center, right}),
          geo{dp_x, dp_y, dp_z}
    {}

protected:
    using vec = gko::matrix::Dense<ValueType>;
    using coef_type = gko::Array<ValueType>;

    void apply_impl(const gko::LinOp* b, gko::LinOp* x) const override
    {
        auto dense_b = gko::as<vec>(b);
        auto dense_x = gko::as<vec>(x);

        struct prolongation_operation : gko::Operation {
            prolongation_operation(const coef_type& coefficients, const vec* b,
                                   vec* x, problem_geometry geo)
                : coefficients{coefficients},
                  coarse_rhs{b},
                  fine_x{x},
                  coarse_geo{geo}
            {}
            void run(std::shared_ptr<const gko::OmpExecutor>) const override
            {
                const auto c_values = coarse_rhs->get_const_values();

                auto f_values = fine_x->get_values();

                const auto c_nz = coarse_geo.nz;
                const auto c_ny = coarse_geo.ny;
                const auto c_nx = coarse_geo.nx;

                const auto coeffs = coefficients.get_const_data();
#pragma omp parallel for
                for (auto c_z = 0; c_z <= c_nz; c_z++) {
                    const auto f_z = 2 * c_z;
                    for (auto c_y = 0; c_y <= c_ny; c_y++) {
                        const auto f_y = 2 * c_y;
                        for (auto c_x = 0; c_x <= c_nx; c_x++) {
                            const auto f_x = 2 * c_x;
                            const auto c_row =
                                grid2index(c_x, c_y, c_z, c_nx, c_ny);
                            const auto f_row =
                                grid2index(f_x, f_y, f_z, 2 * c_nx, 2 * c_ny);
                            for (auto ofs_z : {-1, 0, 1}) {
                                if (f_z + ofs_z > -1 &&
                                    f_z + ofs_z <= 2 * c_nz) {
                                    for (auto ofs_y : {-1, 0, 1}) {
                                        if (f_y + ofs_y > -1 &&
                                            f_y + ofs_y <= 2 * c_ny) {
                                            for (auto ofs_x : {-1, 0, 1}) {
                                                if (f_x + ofs_x > -1 &&
                                                    f_x + ofs_x <= 2 * c_nx) {
                                                    const auto f_id =
                                                        grid2index(
                                                            ofs_x, ofs_y, ofs_z,
                                                            2 * c_nx, 2 * c_ny,
                                                            f_row);

                                                    f_values[f_id] +=
                                                        coeffs[ofs_z + 1] *
                                                        coeffs[ofs_y + 1] *
                                                        coeffs[ofs_x + 1] *
                                                        c_values[c_row];
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            // cuda impl
            void run(std::shared_ptr<const gko::CudaExecutor>) const override
            {
                prolongation_kernel(
                    coarse_geo.nx, coarse_geo.ny, coarse_geo.nz,
                    coefficients.get_const_data(),
                    coarse_rhs->get_const_values(), coarse_rhs->get_size()[0],
                    fine_x->get_values(), fine_x->get_size()[0]);
            }
            void run(
                std::shared_ptr<const gko::ReferenceExecutor>) const override
            {}
            const coef_type& coefficients;
            const vec* coarse_rhs;
            vec* fine_x;
            problem_geometry coarse_geo;
        };
        this->get_executor()->run(
            prolongation_operation(coefficients, dense_b, dense_x, geo));
    }

    void apply_impl(const gko::LinOp* alpha, const gko::LinOp* b,
                    const gko::LinOp* beta, gko::LinOp* x) const override
    {
        auto dense_b = gko::as<vec>(b);
        auto dense_x = gko::as<vec>(x);
        auto tmp_x = dense_x->clone();
        this->apply_impl(b, lend(tmp_x));
        dense_x->scale(beta);
        dense_x->add_scaled(alpha, lend(tmp_x));
    }

private:
    coef_type coefficients;
    problem_geometry geo;
};

template <typename ValueType = default_precision, typename IndexType = int32>
class Gmg : public EnableLinOp<Gmg<ValueType, IndexType>>,
            public EnableMultigridLevel<ValueType> {
    friend class EnableLinOp<Gmg>;
    friend class EnablePolymorphicObject<Gmg, LinOp>;

public:
    using value_type = ValueType;
    using index_type = IndexType;

    std::shared_ptr<const LinOp> get_system_matrix() const
    {
        return system_matrix_;
    }

    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
    {
        problem_geometry GKO_FACTORY_PARAMETER(fine_geo,
                                               problem_geometry{32, 32, 32});
    };
    GKO_ENABLE_LIN_OP_FACTORY(Gmg, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

protected:
    void apply_impl(const LinOp* b, LinOp* x) const override
    {
        this->get_composition()->apply(b, x);
    }

    void apply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                    LinOp* x) const override
    {
        this->get_composition()->apply(alpha, b, beta, x);
    }

    explicit Gmg(std::shared_ptr<const Executor> exec)
        : EnableLinOp<Gmg>(std::move(exec))
    {}

    explicit Gmg(const Factory* factory,
                 std::shared_ptr<const LinOp> system_matrix)
        : EnableLinOp<Gmg>(factory->get_executor(), system_matrix->get_size()),
          EnableMultigridLevel<ValueType>(system_matrix),
          parameters_{factory->get_parameters()},
          system_matrix_{system_matrix}
    {
        if (system_matrix_->get_size()[0] != 0) {
            this->geo = parameters_.fine_geo;
            this->generate();
        }
    }

    // void generate();  // not sure if needed with the generate() defined below
    void generate()
    {
        using csr_type = matrix::Csr<ValueType, IndexType>;
        using real_type = remove_complex<ValueType>;
        auto exec = this->get_executor();
        const auto num_rows = this->system_matrix_->get_size()[0];
        std::shared_ptr<const csr_type> gmg_op_shared_ptr{};
        const csr_type* gmg_op =
            dynamic_cast<const csr_type*>(system_matrix_.get());
        if (!gmg_op) {
            gmg_op_shared_ptr =
                copy_and_convert_to<csr_type>(exec, system_matrix_);
            gmg_op = gmg_op_shared_ptr.get();
            this->set_fine_op(gmg_op_shared_ptr);
        }

        problem_geometry coarse_geo;
        generate_coarse_geo(this->geo, coarse_geo);
        auto discretization_points =
            (coarse_geo.nx + 1) * (coarse_geo.ny + 1) * (coarse_geo.nz + 1);

        // which executor ?
        auto coarse_mat =
            share(csr_type::create(exec, gko::dim<2>(discretization_points)));

        auto fine_dim = this->system_matrix_->get_size()[0];
        generate_problem_matrix(exec, coarse_geo, lend(coarse_mat.get()));

        this->set_multigrid_level(
            gmg_prolongation<ValueType, IndexType>::create(
                exec, coarse_geo.nx, coarse_geo.ny, coarse_geo.nz,
                discretization_points, fine_dim),
            coarse_mat,
            gmg_restriction<ValueType, IndexType>::create(
                exec, coarse_geo.nx, coarse_geo.ny, coarse_geo.nz,
                discretization_points, fine_dim));
    }

private:
    std::shared_ptr<const LinOp> system_matrix_{};
    problem_geometry geo;

    void generate_coarse_geo(problem_geometry& fine_geo,
                             problem_geometry& coarse_geo)
    {
        coarse_geo.nx = fine_geo.nx / 2;
        coarse_geo.ny = fine_geo.ny / 2;
        coarse_geo.nz = fine_geo.nz / 2;
    }
};

}  // namespace multigrid
}  // namespace gko

#endif  // GKO_PUBLIC_CORE_MULTIGRID_GMG_HPP_